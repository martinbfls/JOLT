#!/usr/bin/env python3
"""Backs up experiment results to a private Hugging Face dataset repo (or restores them).

Incremental and resumable: re-running only uploads new/changed files, and an interrupted upload
picks up where it stopped. Run from the repo root, on a machine with internet access (a login
node, not a compute node), after `hf auth login` with a write token.

  python scripts/backup_results.py                  # experiments/ -> <you>/jolt-results
  python scripts/backup_results.py --dry-run        # what would be uploaded
  python scripts/backup_results.py --checkpoints    # also out/checkpoints/ -> <you>/jolt-checkpoints
  python scripts/backup_results.py --restore        # <you>/jolt-results -> experiments/

Skipped by default: select_flips' per-worker index/label files (tens of thousands of small files,
regenerated in minutes on CPU from gen labels.npy -- pass --keep-workers to keep them) and Python
caches. Everything the results are computed from is kept: caccs/paccs, poison stats, labels,
triggers, metrics, configs, plots.
"""

import argparse
import fnmatch
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

WORKER_PATTERNS = ["*/worker*/*"]
ALWAYS_IGNORED = ["*__pycache__*", "*.pyc", "*.done"]


def selected_files(folder, ignore_patterns):
    for path in sorted(Path(folder).rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and not any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
            yield path


def upload(api, repo_id, folder, ignore_patterns, dry_run):
    files = list(selected_files(folder, ignore_patterns))
    size_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"{folder}/ -> {repo_id}: {len(files)} files, {size_gb:.2f} GB (before deduplication against the Hub)")
    if len(files) > 100_000:
        print("[WARNING] > 100k files: above what the Hub recommends per repo; consider excluding more.")
    if dry_run:
        return
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    # huggingface_hub < 1.0 has a dedicated resumable uploader; >= 1.0 folded that behavior
    # (multi-commit, resumable) into upload_folder itself.
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            repo_id=repo_id, folder_path=folder, repo_type="dataset",
            ignore_patterns=ignore_patterns, num_workers=min(8, os.cpu_count() or 1),
        )
    else:
        api.upload_folder(
            repo_id=repo_id, folder_path=folder, repo_type="dataset",
            ignore_patterns=ignore_patterns, commit_message=f"Backup {folder}/",
        )
    print(f"[DONE] https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO"),
                        help="results dataset repo id (default: $HF_REPO, else <hf user>/jolt-results)")
    parser.add_argument("--checkpoints-repo", default=os.environ.get("HF_CKPT_REPO"),
                        help="checkpoints dataset repo id (default: $HF_CKPT_REPO, else <hf user>/jolt-checkpoints)")
    parser.add_argument("--folder", default="experiments")
    parser.add_argument("--checkpoints", action="store_true", help="also back up out/checkpoints/")
    parser.add_argument("--keep-workers", action="store_true", help="also upload select_flips' worker*/ files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true", help="download the results repo into --folder instead")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        if not args.dry_run:
            raise SystemExit(f"Not logged in to the Hugging Face Hub ({e}) -- run `hf auth login` with a write token.")
        user = "<hf-user>"
    repo = args.repo or f"{user}/jolt-results"

    if args.restore:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=args.folder)
        print(f"[DONE] {repo} -> {args.folder}/")
        return

    ignore = ALWAYS_IGNORED + ([] if args.keep_workers else WORKER_PATTERNS)
    upload(api, repo, args.folder, ignore, args.dry_run)
    if args.checkpoints:
        upload(api, args.checkpoints_repo or f"{user}/jolt-checkpoints", "out/checkpoints", ALWAYS_IGNORED, args.dry_run)


if __name__ == "__main__":
    main()
