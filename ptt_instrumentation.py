"""
ptt_instrumentation.py
======================
Hook-based instrumentation for PTTHead (Point Trajectory Transformer).

Captures intermediate activations and attention weights during inference
without modifying trained model weights.

The custom Attention class in ptt_head.py computes but discards the attention
weight matrix A; this module monkey-patches individual Attention instances to
store A alongside the normal forward output.

USAGE
-----
    from ptt_instrumentation import PTTInstrumentationContext

    with PTTInstrumentationContext(model.roi_head, capture_attn=True) as ctx:
        with torch.no_grad():
            model(batch_dict)
        acts = ctx.activations   # dict[str -> Tensor/list] on CPU
        attn = ctx.attn_weights  # dict[str -> Tensor] on CPU

CAPTURED activations keys (all detached CPU tensors):
  "ptt_inputs"         list[box_seq, feat, cur_feat]  pre-hook on PTTransformer
  "long_mem_out"       [B*R, T_long, 96]    M-hat_l after attn1_long
  "short_mem_out"      [B*R, T_short, 96]   M-hat_s after attn1_short
  "future_feat_out"    [B*R, T_fut+1, 96]   M-hat_f after attn1_future
  "enhanced_pt_feat"   [B*R, T_total, 128]  full-seq enhanced P-T after attn2
  "cur_feat_after_sa"  [B*R, N/2, D]        current-point after attn3_point
  "cross_long_out"     [B*R, 1, 256]        M-hat_lp (aggregator cross-attn)
  "cross_short_out"    [B*R, 1, 256]        M-hat_sp (aggregator cross-attn)
  "cross_future_out"   [B*R, 1, 256]        M-hat_fp (aggregator cross-attn)
  "ptt_outputs"        list[box_reg, feat_box, feat_traj, cur_feat_final]
  "cls_logits"         [K*B*R, 1]           raw cls logits before sigmoid

attn_weights keys (only when capture_attn=True):
  "attn_long"     [B*R, T_long, T_long]    long self-attn matrix
  "attn_short"    [B*R, T_short, T_long]   short cross-attn matrix
  "attn_future"   [B*R, T_fut+1, T_sl]    future cross-attn matrix
  "attn2"         [B*R, T_total, T_total]  full-seq self-attn matrix
  "attn3_point"   [B*R, N/2, N/2]         current-point self-attn matrix
  "cross_long"    [B*R, 1, T_long]         aggregator: point queries long
  "cross_short"   [B*R, 1, T_short]        aggregator: point queries short
  "cross_future"  [B*R, 1, T_fut+1]        aggregator: point queries future

For PTT 32-frame config: T_long=24, T_short=8, T_fut=16,
T_total=48, N=256 lidar pts, B=1 at inference.
"""

import types
import math
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Attention monkey-patching helpers
# ---------------------------------------------------------------------------

#run original attention,
#then additionally compute and store attention weights for inspection
def _make_instrumented_attn_forward(orig_forward, store: dict, key: str):
    """
    input
    orig_forward: original attention function
    store: dict where attention matrices will be saved
    key: name under which we store result (e.g "attn_long")
    
    Returns a new bound-method that:
      1. Runs the original Attention forward (returns O unchanged).
      2. Recomputes attention matrix A under no_grad and stores in store[key].

    We recompute A rather than intercepting mid-original-forward so the
    original autograd graph is not affected.
    """
    
    #replacement for original forward
    def new_forward(self, Q, K):
        """
        Q: queries (what is asking for information)
        K: keys (what contains information)
        """
        O = orig_forward(self, Q, K)
        
        #recompute attention weights for logging
        with torch.no_grad():
            
            # transform Q and K into feature space
            #feature space: pass raw inputs through learned linear layers so their 
            #numerical representatin becomes useful for comparison
            
            #model is learning how to represent each proposal
            #taking dot between Q and K us cinoarubg kearbed features that encode things like 
            #trajectory consistenty or object identity
            Q_proj = self.fc_q(Q)
            K_proj = self.fc_k(K)
            
            #split features into smaller chunks, then stack along batch dimension
            # model computes several attention patterns in parallel and then recombines them
            dim_split = self.dim_LIN // self.num_heads
            Q_s = torch.cat(Q_proj.split(dim_split, 2), 0)
            K_s = torch.cat(K_proj.split(dim_split, 2), 0)
            
            #compute attention matrix
            #A[i,j] = how much query i attends to key j 
            A = torch.softmax(
                Q_s.bmm(K_s.transpose(1, 2)) / math.sqrt(self.dim_LIN), 2
            )
            
            #average across heads
            #combine multiple heads into one matrix, average across heads
            if self.num_heads >= 2:
                A = torch.mean(
                    torch.stack(list(A.split(Q_proj.size(0), dim=0)), dim=0),
                    dim=0,
                )
            #save attention matrix; remove gradients, move off GPU
            store[key] = A.detach().cpu()
        return O

    return new_forward


def patch_attention_weight_capture(module, store: dict, key: str):
    """
    Takes one attention module and swap out its normal forward pass with instrumented version
    
    Monkey-patch one Attention instance in-place to also record attn weights.
    Returns the original forward function (unbound) for later restoration.
    """
    # save original forward function
    orig = module.forward.__func__ 
    
    # create new version of forward
    # runs original computation and stores attention matrix 
    module.forward = types.MethodType(
        _make_instrumented_attn_forward(orig, store, key), module
    )
    return orig


def unpatch_attention(module, orig_func):
    """
    reassigns modules forward method to original implementation
    
    Restore original forward on a previously-patched Attention instance."""
    module.forward = types.MethodType(orig_func, module)


# ---------------------------------------------------------------------------
# Main context manager
# ---------------------------------------------------------------------------

class PTTInstrumentationContext:
    """
    Context manager that instruments a PTTHead model for one or more forward
    passes.  Registers forward hooks and optionally patches Attention modules.

    Parameters
    ----------
    roi_head : PTTHead
        The roi_head attribute of the MPPNet model (i.e., model.roi_head).
    capture_attn : bool
        If True, monkey-patch Attention instances to also record A matrices.
        Adds a small overhead (one no_grad recomputation per Attention call).
    """

    def __init__(self, roi_head, capture_attn: bool = True):
        """
        sets up state needed for instrumentation
        
        initialize container to keep track of hook
        intiialize dict to store original attention forward passes
        initialize dicts to store captured data
        """
        self.rh = roi_head
        self.capture_attn = capture_attn
        self._hooks = []
        self._attn_originals = {}
        self.activations: dict = {}
        self.attn_weights: dict = {}

    # ------------------------------------------------------------------
    # Context protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        
        """
        attach forward hooks to various parts of PTT model 
        
        replaces attention forward functions with instrumented versions
        """
        self._register_hooks()
        if self.capture_attn:
            self._patch_attention_modules()
        return self

    def __exit__(self, *args):
        
        """
        remove all hooks
        restores original forward functions
        """
        self._remove_hooks()
        if self.capture_attn:
            self._unpatch_attention_modules()
        return False

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _post_hook(self, key):
        """
        captures outputs and saves it for later inspection
        """
        def hook(mod, inputs, output):
            if isinstance(output, torch.Tensor):
                self.activations[key] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                self.activations[key] = [
                    t.detach().cpu() if isinstance(t, torch.Tensor) else t
                    for t in output
                ]
            else:
                self.activations[key] = output
        return hook

    def _pre_hook(self, key):
        """
        captures inputs going into module
        """
        def hook(mod, inputs):
            saved = []
            for t in inputs:
                if isinstance(t, torch.Tensor):
                    saved.append(t.detach().cpu())
                else:
                    saved.append(t)
            self.activations[key] = saved
        return hook

    def _register_hooks(self):
        
        """
        what parts of model you want to observe
        builds map of pipeline and says where to record outputs
        """
        rh  = self.rh
        ptt = rh.ptt   # PTTransformer

        # Post-hooks (capture module output)
        post_targets = [
            (ptt.attn1_long,             "long_mem_out"),
            (ptt.attn1_short,            "short_mem_out"),
            (ptt.attn1_future,           "future_feat_out"),
            (ptt.attn2,                  "enhanced_pt_feat"),
            (ptt.attn3_point,            "cur_feat_after_sa"),
            (ptt.cross_long,             "cross_long_out"),
            (ptt.cross_short,            "cross_short_out"),
            (ptt.cross_future,           "cross_future_out"),
            (ptt,                        "ptt_outputs"),
            (rh.class_embed[0],          "cls_logits"),
        ]
        for mod, key in post_targets:
            self._hooks.append(mod.register_forward_hook(self._post_hook(key)))

        # Pre-hook on PTTransformer to capture (box_seq, feat, cur_feat)
        self._hooks.append(
            ptt.register_forward_pre_hook(self._pre_hook("ptt_inputs"))
        )

    def _remove_hooks(self):
        """detaches hooks from model"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Attention patching
    # ------------------------------------------------------------------

    def _patch_attention_modules(self):
        
        """ 
        replace forward functions inside all attention layers with instrumented versions
        """
        ptt = self.rh.ptt
        attn_map = {
            "attn_long":    ptt.attn1_long,
            "attn_short":   ptt.attn1_short,
            "attn_future":  ptt.attn1_future,
            "attn2":        ptt.attn2,
            "attn3_point":  ptt.attn3_point,
            "cross_long":   ptt.cross_long,
            "cross_short":  ptt.cross_short,
            "cross_future": ptt.cross_future,
        }
        for key, mod in attn_map.items():
            orig = patch_attention_weight_capture(mod, self.attn_weights, key)
            self._attn_originals[key] = (mod, orig)

    def _unpatch_attention_modules(self):
        """
        restores each modules original forward patch using unpatch_attention
        """
        for _key, (mod, orig) in self._attn_originals.items():
            unpatch_attention(mod, orig)
        self._attn_originals.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def flush(self):
        """
        Retrieve current captures and reset storage for the next forward pass.
        Returns (activations_dict, attn_weights_dict).
        """
        acts = dict(self.activations)
        attn = dict(self.attn_weights)
        self.activations.clear()
        self.attn_weights.clear()
        return acts, attn


# ---------------------------------------------------------------------------
# Per-proposal feature extractor
# ---------------------------------------------------------------------------

def collect_proposal_features(
    acts,
    attn,
    batch_dict,
    spoof_gt,
    real_gt_boxes,
    iou_fn,
    tp_thresh=0.7,
    spoof_thresh=0.7,
    score_thresh=0.1,
):
    """
    Build a list of per-proposal feature dicts from one instrumented forward pass.

    Parameters
    ----------
    acts, attn       : dicts returned by PTTInstrumentationContext (CPU tensors)
    batch_dict       : the batch dict after model(batch_dict).
                       Expects 'rois' (B,R,7+) and 'batch_cls_preds' (B,R) or (B,R,1).
    spoof_gt         : (7+,) float array or None - the injected phantom box
    real_gt_boxes    : (M,7+) float array or None - ground-truth real boxes
    iou_fn           : callable((N,7) cuda tensor, (M,7) cuda tensor) -> (N,M) tensor
                       e.g. iou3d_nms_utils.boxes_iou3d_gpu
    tp_thresh        : IoU >= this -> 'real_tp' (car/class-1 default; ped/cyc use 0.5)
    spoof_thresh     : IoU >= this with spoof_gt -> 'spoofed_fp'
    score_thresh     : confidence < this -> 'background'

    Returns
    -------
    List of dicts, one per proposal. Each dict contains:
      proposal_idx, category, confidence, roi_box (7,), iou_real, iou_spoof,
      long_mem, short_mem, future_feat, enhanced_pt,
      cur_feat_sa, cross_long, cross_short, cross_future,
      cur_feat_final, feat_traj, cls_logit, box_seq,
      attn_long, attn_short, attn_future, attn2,
      cross_long_w, cross_short_w, cross_future_w
    All feature values are numpy arrays or None.
    """

    def _np(key, default=None):
        """convert values to numpy arrays"""
        v = acts.get(key, default)
        if isinstance(v, torch.Tensor):
            return v.numpy()
        return v

    def _np_attn(key):
        """ convert values in attn dictionary into numpy"""
        v = attn.get(key, None)
        if isinstance(v, torch.Tensor):
            return v.numpy()
        return v

    def _slice(arr, idx):
        """if an array exists, pull out one proposal's slice from it"""
        return arr[idx].copy() if arr is not None else None

    # Proposal boxes and scores (batch_size=1 at inference)
    #
    rois = batch_dict.get('rois', None)
    if rois is None:
        return []
    rois_np = rois[0].detach().cpu().numpy()   # (R, 7+)
    num_rois = rois_np.shape[0]

    # extract proposal scores
    cls_p = batch_dict.get('batch_cls_preds', None)
    if cls_p is not None:
        scores_np = (
            cls_p[0, :, 0] if cls_p.dim() == 3 else cls_p[0]
        ).detach().cpu().numpy().astype(np.float32)
    else:
        scores_np = np.zeros(num_rois, dtype=np.float32)

    # Per-proposal TP threshold: 0.7 for cars (label==1), 0.5 for ped/cyc (label==2,3)
    roi_labels_raw = batch_dict.get('roi_labels', None)
    if roi_labels_raw is not None:
        roi_labels_np = roi_labels_raw[0].detach().cpu().numpy().astype(np.int32)  # (R,)
        tp_thresh_arr = np.where(roi_labels_np == 1, 0.7, 0.5).astype(np.float32)
    else:
        tp_thresh_arr = np.full(num_rois, tp_thresh, dtype=np.float32)

    # IoU with real GT and spoof GT
    iou_real_all  = np.zeros(num_rois, dtype=np.float32)
    iou_spoof_all = np.zeros(num_rois, dtype=np.float32)
    pred_t = torch.tensor(rois_np[:, :7], dtype=torch.float32).cuda()

    # compute proposal-vs-real IoU for all proposal
    # then keep maximum IoU each proposal achieves with any real object
    #
    if real_gt_boxes is not None and len(real_gt_boxes) > 0:
        gt_t = torch.tensor(
            np.asarray(real_gt_boxes)[:, :7], dtype=torch.float32
        ).cuda()
        iou_real_all = iou_fn(pred_t, gt_t).max(dim=1)[0].cpu().numpy()
    #if spoof gt box exists, compute each proposals IoU with single spoof box
    if spoof_gt is not None:
        sp_t = torch.tensor(
            np.asarray(spoof_gt)[:7].reshape(1, 7), dtype=torch.float32
        ).cuda()
        iou_spoof_all = iou_fn(pred_t, sp_t)[:, 0].cpu().numpy()

    # Feature arrays
    long_m  = _np("long_mem_out")          # (R, T_long, 96)
    short_m = _np("short_mem_out")         # (R, T_short, 96)
    fut_m   = _np("future_feat_out")       # (R, T_fut+1, 96)
    enh_pt  = _np("enhanced_pt_feat")      # (R, T_total, 128)
    cur_sa  = _np("cur_feat_after_sa")     # (R, N/2, D) -> max over point dim
    
    #reduce current point feature along point dimension
    if cur_sa is not None and cur_sa.ndim == 3:
        cur_sa = cur_sa.max(axis=1)        # (R, D)

    # load cross attention ouputs
    # outputs after current frame point feature query long term, short term, and cuture memory
    clong   = _np("cross_long_out")        # (R, 1, 256)
    cshort  = _np("cross_short_out")
    cfuture = _np("cross_future_out")

    # PTT outputs: list[box_reg, feat_box, feat_traj, cur_feat_final]
    #extract trajectory feature tensor and final current frame feature tensor--> attached to each proposal record
    ptt_out = acts.get("ptt_outputs", None)
    cur_feat_final = None
    feat_traj_arr  = None
    if isinstance(ptt_out, list) and len(ptt_out) == 4:
        ft = ptt_out[2]
        if isinstance(ft, torch.Tensor):
            ft = ft.numpy()
        feat_traj_arr = ft    # (256, R, T) or similar
        cf = ptt_out[3]
        if isinstance(cf, torch.Tensor):
            cf = cf.numpy()
        cur_feat_final = cf   # (R, 256)

    # cls logits: [K*R, 1] - take last R rows (final encoder layer)
    #extract logits
    # raw classification logits before sigmoid
    logits_raw = _np("cls_logits")
    cls_logits = None
    if logits_raw is not None:
        if logits_raw.ndim == 2:
            cls_logits = logits_raw[-num_rois:, 0]
        else:
            cls_logits = logits_raw[-num_rois:]

    # PTT inputs pre-hook: list[box_seq, feat, cur_feat]
    # load transformer inputs
    #pull out box_seq (proposal sequence input to PTT)
    # lets you inspect trajectory history associated with each proposal
    ptt_inp = acts.get("ptt_inputs", None)
    box_seq_arr = None
    if ptt_inp is not None and len(ptt_inp) >= 1:
        bsq = ptt_inp[0]
        if isinstance(bsq, torch.Tensor):
            bsq = bsq.numpy()
        box_seq_arr = bsq   # (R, 8, T_total)

    # Attention weights
    #load in attention matrices
    aw_long    = _np_attn("attn_long")     # (R, T_long, T_long)
    aw_short   = _np_attn("attn_short")    # (R, T_short, T_long)
    aw_future  = _np_attn("attn_future")   # (R, T_fut+1, T_sl)
    aw_attn2   = _np_attn("attn2")         # (R, T_total, T_total)
    aw_clong   = _np_attn("cross_long")    # (R, 1, T_long)
    aw_cshort  = _np_attn("cross_short")   # (R, 1, T_short)
    aw_cfuture = _np_attn("cross_future")  # (R, 1, T_fut+1)

    def _slice_traj(arr, idx):
        """
        check which axis corresponds to proposals, extracts trajectory feature for one proposal
        feat_traj from PTT forward is x_ori permuted as (2,0,1) -> (T,R,256).
        In ptt_outputs[2] it is passed as traj_feat before that permute,
        so shape is (256, R, T_total). Detect by checking which axis == num_rois."""
        if arr is None or arr.ndim != 3:
            return None
        if arr.shape[1] == num_rois:
            return arr[:, idx, :].copy()   # (256, T_total)
        if arr.shape[0] == num_rois:
            return arr[idx].copy()
        return None

    #iterate through every proposal and builds one dictionary per proposal
    records = []
    for r in range(num_rois):
        
        #read scalar metadata
        score = float(scores_np[r])
        iou_r = float(iou_real_all[r])
        iou_s = float(iou_spoof_all[r])

        #assign the proposal to a category
        if iou_r >= tp_thresh_arr[r]:
            cat = 'real_tp'
        elif iou_s >= spoof_thresh:
            cat = 'spoofed_fp'
        elif score >= score_thresh:
            cat = 'clean_fp'
        else:
            cat = 'background'

        #append full record
        records.append({
            'proposal_idx':   r,
            'category':       cat,
            'confidence':     score,
            'roi_box':        rois_np[r, :7].copy(),
            'iou_real':       iou_r,
            'iou_spoof':      iou_s,
            # Feature arrays (numpy or None)
            'long_mem':       _slice(long_m, r),
            'short_mem':      _slice(short_m, r),
            'future_feat':    _slice(fut_m, r),
            'enhanced_pt':    _slice(enh_pt, r),
            'cur_feat_sa':    _slice(cur_sa, r),
            'cross_long':     _slice(clong, r),
            'cross_short':    _slice(cshort, r),
            'cross_future':   _slice(cfuture, r),
            'cur_feat_final': _slice(cur_feat_final, r),
            'feat_traj':      _slice_traj(feat_traj_arr, r),
            'cls_logit':      float(cls_logits[r]) if cls_logits is not None else None,
            'box_seq':        _slice(box_seq_arr, r),
            # Attention weight matrices
            'attn_long':      _slice(aw_long, r),
            'attn_short':     _slice(aw_short, r),
            'attn_future':    _slice(aw_future, r),
            'attn2':          _slice(aw_attn2, r),
            'cross_long_w':   _slice(aw_clong, r),
            'cross_short_w':  _slice(aw_cshort, r),
            'cross_future_w': _slice(aw_cfuture, r),
        })

    return records
