from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi

# unset HF_ENDPOINT
# python upload_huggingface.py --repo-id "BoiWanKenobi/myNovelGPT"

DEFAULT_MODEL_PATH = Path("tinygpt.onnx")
DEFAULT_TOKENIZER_PATH = Path("params/tokenizer2.json")
DEFAULT_METADATA_PATH = Path("tinygpt.onnx.json")


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")


def upload_one(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    local_path: Path,
    repo_path: str,
    commit_message: str,
) -> str:
    require_file(local_path)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(repo_path)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload TinyGPT ONNX assets to a Hugging Face Hub repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example username/tinygpt-onnx")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    parser.add_argument("--private", action="store_true", help="create the repo as private if it does not exist")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--commit-message", default="upload TinyGPT ONNX assets")
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; defaults to HF_TOKEN or the cached huggingface-cli login token",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    uploaded: list[tuple[str, str]] = []
    uploaded.append(
        (
            "model",
            upload_one(
                api=api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                local_path=args.model,
                repo_path=args.model.name,
                commit_message=args.commit_message,
            ),
        )
    )
    uploaded.append(
        (
            "tokenizer",
            upload_one(
                api=api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                local_path=args.tokenizer,
                repo_path="params/tokenizer2.json",
                commit_message=args.commit_message,
            ),
        )
    )
    if not args.skip_metadata and args.metadata.exists():
        uploaded.append(
            (
                "metadata",
                upload_one(
                    api=api,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    local_path=args.metadata,
                    repo_path=args.metadata.name,
                    commit_message=args.commit_message,
                ),
            )
        )

    urls = dict(uploaded)
    print("uploaded:")
    for name, url in uploaded:
        print(f"  {name}: {url}")
    print()
    print("write.html URL parameters:")
    print(f"  ?model={quote(urls['model'], safe='')}&tokenizer={quote(urls['tokenizer'], safe='')}")


if __name__ == "__main__":
    main()
