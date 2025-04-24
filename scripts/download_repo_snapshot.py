import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_repo_snapshot(repo_id, token=None, ignore_patterns=None):
    """
    Download a snapshot of a repository from Hugging Face Hub.

    Args:
        repo_id (str): The ID of the repository to download.
        revision (str, optional): The revision to download. Defaults to None.

    Returns:
        str: The path to the downloaded snapshot.
    """
    local_dir = Path(repo_id.split("/")[-1])
    local_dir = "snapshots" / local_dir
    local_dir.mkdir(exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
        ignore_patterns=ignore_patterns,
    )

    return local_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a snapshot of a repository from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id", type=str, help="The ID of the repository to download."
    )

    parser.add_argument(
        "--token", type=str, default=None, help="The token to access the repository."
    )

    parser.add_argument(
        "--ignore-patterns",
        type=str,
        default=None,
        help="Patterns to ignore when downloading the snapshot.",
    )

    args = parser.parse_args()

    local_dir = download_repo_snapshot(
        repo_id=args.repo_id,
        token=args.token,
        ignore_patterns=args.ignore_patterns,
    )

    print(f"Downloaded snapshot to {local_dir}")
