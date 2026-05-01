"""
PointNet-based Local Objectness Predictor (LOP).

Reimplements the PointNet-style LOP backbone from Xiao et al., USENIX
Security '23 ("Exorcising Wraith"). The model consumes a fixed-size pillar
point tensor of shape (N, 7), where the 7 features are:
    (dx, dy, x, y, z, intensity, depth)

and outputs a single binary logit indicating whether the pillar intersects a
real object's bounding box.

Notes:
- The depth channel is critical; it allows the model to implicitly learn the
  depth-density relationship emphasized in the paper.
- The network is PointNet-style: shared per-point MLPs + symmetric max pool.
- Fixed-size sampled/padded point sets are standard for PointNet pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class PointNetLOP(nn.Module):
#     """
#     Minimal PointNet for per-pillar binary objectness classification.

#     Architecture:
#       - Shared per-point MLP: 7 -> 64 -> 128 -> 512
#       - Global max-pool over points
#       - Classification head: 512 -> 256 -> 1

#     We intentionally omit the T-Net from vanilla PointNet. Pillars are already
#     in a useful canonical frame because dx/dy are defined relative to the
#     pillar center.
#     """

#     def __init__(self, input_dim=7, num_points=1024, dropout=0.3):
#         super().__init__()
#         self.input_dim = input_dim
#         # Stored for bookkeeping / checkpoint metadata compatibility.
#         # The forward pass itself is agnostic to the exact point count because
#         # it uses a symmetric max-pool over the point dimension.
#         self.num_points = num_points

#         # Shared per-point MLP implemented as 1x1 convolutions.
#         self.conv1 = nn.Conv1d(input_dim, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 512, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(512)

#         # Classification head.
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 1)
#         self.bn_fc1 = nn.BatchNorm1d(256)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: (B, N, input_dim)
#         mask = (x.abs().sum(dim=-1) > 0)  # (B, N), True for real points

#         x = x.transpose(1, 2).contiguous()  # (B, D, N)

#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))  # (B, 512, N)

#         # Prevent padded rows from participating in PointNet max-pooling.
#         x = x.masked_fill(~mask.unsqueeze(1), -1e6)
#         x = torch.max(x, dim=2)[0]  # (B, 512)

#         # Safety for empty pillars; should not occur in occupied-only mode.
#         x = torch.where(x < -1e5, torch.zeros_like(x), x)

#         x = F.relu(self.bn_fc1(self.fc1(x)))
#         x = self.dropout(x)
#         logits = self.fc2(x).squeeze(-1)
#         return logits

# class PointNetLOP(nn.Module):
#     def __init__(self, input_dim=7, num_points=1024, dropout=0.3):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_points = num_points

#         # Per-point MLP (unchanged architecture)
#         self.conv1 = nn.Conv1d(input_dim, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 512, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(512)

#         # +1 for the explicit point-count feature
#         self.fc1 = nn.Linear(512 + 1, 256)
#         self.fc2 = nn.Linear(256, 1)
#         self.bn_fc1 = nn.BatchNorm1d(256)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: (B, N, input_dim)
#         mask = (x.abs().sum(dim=-1) > 0)           # (B, N)
#         point_count = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
#         # Normalize: fraction of occupied slots
#         density_feat = point_count / self.num_points          # (B, 1)

#         x = x.transpose(1, 2).contiguous()          # (B, D, N)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))          # (B, 512, N)

#         # Masked max-pool: padding positions don't participate
#         x = x.masked_fill(~mask.unsqueeze(1), -1e6)
#         x = torch.max(x, dim=2)[0]                  # (B, 512)
#         x = torch.where(x < -1e5, torch.zeros_like(x), x)

#         # Concatenate explicit density signal
#         x = torch.cat([x, density_feat], dim=1)      # (B, 513)

#         x = F.relu(self.bn_fc1(self.fc1(x)))
#         x = self.dropout(x)
#         return self.fc2(x).squeeze(-1)

class PointNetLOP(nn.Module):
    def __init__(
        self,
        input_dim=7,
        num_points=1024,
        dropout=0.35,
        pillar_size=1.0,
        range_norm=75.0,
        z_norm=5.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.pillar_size = pillar_size
        self.range_norm = range_norm
        self.z_norm = z_norm

        # Shared per-point MLP.
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)

        # Global feature = max pool + mean pool + log_count + mean_depth
        # 512 max + 512 mean + 2 auxiliary scalar features.
        self.fc1 = nn.Linear(512 + 512 + 2, 256)
        self.fc2 = nn.Linear(256, 1)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def _normalize_features(self, x):
        """
        x feature order:
            0 dx
            1 dy
            2 x
            3 y
            4 z
            5 intensity
            6 depth

        Only scale features; do not subtract means. This keeps padded all-zero
        rows at zero and preserves the padding mask semantics.
        """
        scale = x.new_tensor([
            0.5 * self.pillar_size,  # dx roughly [-0.5, 0.5]
            0.5 * self.pillar_size,  # dy roughly [-0.5, 0.5]
            self.range_norm,         # x
            self.range_norm,         # y
            self.z_norm,             # z
            1.0,                     # intensity, leave as-is for now
            self.range_norm,         # depth
        ])
        return x / scale

    def forward(self, x):
        # x: (B, N, 7)
        mask = (x.abs().sum(dim=-1) > 0)  # (B, N), True for real points
        count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

        # Auxiliary density/depth features computed before normalization.
        log_count = torch.log1p(count) / np.log1p(self.num_points)  # (B, 1)
        mean_depth = (x[..., 6] * mask.float()).sum(dim=1, keepdim=True) / count
        mean_depth = mean_depth / self.range_norm                   # (B, 1)

        x = self._normalize_features(x)

        # Shared PointNet MLP.
        x = x.transpose(1, 2).contiguous()  # (B, D, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 512, N)

        mask_3d = mask.unsqueeze(1)  # (B, 1, N)

        # Masked max pooling.
        x_max = x.masked_fill(~mask_3d, -1e6).max(dim=2)[0]
        x_max = torch.where(x_max < -1e5, torch.zeros_like(x_max), x_max)

        # Masked mean pooling. This is important for density-sensitive learning.
        x_mean = (x * mask_3d.float()).sum(dim=2) / count

        aux = torch.cat([log_count, mean_depth], dim=1)  # (B, 2)

        x_global = torch.cat([x_max, x_mean, aux], dim=1)

        x_global = F.relu(self.bn_fc1(self.fc1(x_global)))
        x_global = self.dropout(x_global)

        logits = self.fc2(x_global).squeeze(-1)
        return logits

class FocalLoss(nn.Module):
    """
    Binary focal loss on logits.

    alpha:
        Positive-class weight. Use 0.5 for symmetric focal loss.
        Use 0.55-0.65 if recall is too low.
    gamma:
        Focusing parameter. gamma=2 is the standard focal-loss setting.
    """

    def __init__(self, alpha=0.60, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)

        return (focal_weight * bce).mean()

if __name__ == "__main__":
    model = PointNetLOP(input_dim=7, num_points=1024)
    x = torch.randn(8, 1024, 7)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # expected: (8,)
    print(f"Num params: {sum(p.numel() for p in model.parameters()):,}")
