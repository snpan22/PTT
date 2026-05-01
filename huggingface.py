import os, glob, time
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub import create_repo
import sys
repo_id = sys.argv[1]  # e.g. "spanse30/global_medium"
data_dir = sys.argv[2]
batch_size = 5

api = HfApi()
files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
files = [f for f in files if os.path.basename(f) not in existing]
print(f"Skipping {len(existing)} already-uploaded files, {len(files)} remaining")
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    ops = [
        CommitOperationAdd(
            path_in_repo=os.path.basename(f),
            path_or_fileobj=f
        ) for f in batch
    ]

    for attempt in range(5):
        try:
            print(f"Uploading batch {i//batch_size + 1}")
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=ops,
                commit_message=f"batch {i//batch_size + 1}"
            )
            break
        except Exception as e:
            print("Retry:", e)
            time.sleep(10 * (attempt + 1))