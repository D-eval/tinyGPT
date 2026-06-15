from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path
from typing import BinaryIO


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset2.read import DEFAULT_CONTEXT_LENGTH, token_preprocess_dir  # noqa: E402
from read import BPETokenizer  # noqa: E402
from train_tokenizer import iter_dataset2_raw_texts  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent / "data"
TOKENIZER_PATH = ROOT_DIR / "params" / "tokenizer2.json"
VOCAB_SIZE = 16_000
TOKEN_DTYPE = "uint32"
INDEX_DTYPE = "uint64"
PREVIEW_CHARS = 160
PREVIEW_TOKENS = 40


def compact_preview(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def tokenizer_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def token_shard_prefix(
    source_path: Path,
    context_length: int,
    dataset_dir: Path,
    preprocess_dir: Path,
) -> Path:
    relative = source_path.relative_to(dataset_dir)
    parent = relative.parent.name
    name = safe_stem(source_path)
    if parent:
        name = f"{parent}__{name}"
    return preprocess_dir / name


def complete_path(prefix: Path) -> Path:
    return prefix.with_suffix(".complete")


def source_is_complete(prefix: Path, tokenizer_hash: str, force: bool) -> bool:
    if force:
        return False
    done_path = complete_path(prefix)
    meta_path = prefix.with_suffix(".meta.json")
    if not done_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return meta.get("tokenizer_sha256") == tokenizer_hash


def print_tokenizer_preview(
    row_index: int,
    source_path: Path,
    text: str,
    token_ids: list[int],
    tokenizer: BPETokenizer,
    preview_chars: int,
    preview_tokens: int,
) -> None:
    decoded = tokenizer.decode(token_ids)
    print(
        json.dumps(
            {
                "row": row_index,
                "path": str(source_path),
                "text_preview": compact_preview(text, preview_chars),
                "token_count": len(token_ids),
                "token_ids_preview": token_ids[:preview_tokens],
                "decoded_preview": compact_preview(decoded, preview_chars),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


class FixedContextTokenShardWriter:
    def __init__(
        self,
        prefix: Path,
        context_length: int,
        source_path: Path,
        data_name: str,
        tokenizer_path: Path,
        tokenizer_hash: str,
    ) -> None:
        self.prefix = prefix
        self.context_length = context_length
        self.source_path = source_path
        self.data_name = data_name
        self.tokenizer_path = tokenizer_path
        self.tokenizer_hash = tokenizer_hash
        self.bin_path = prefix.with_suffix(".bin")
        self.idx_path = prefix.with_suffix(".idx")
        self.meta_path = prefix.with_suffix(".meta.json")
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.bin_file: BinaryIO = self.bin_path.open("wb")
        self.offsets = array("Q")
        self.buffer: list[int] = []
        self.token_count = 0
        self.sample_count = 0
        self.input_sample_count = 0
        self.input_token_count = 0
        self.dropped_tail_tokens = 0
        self.closed = False

    def write_tokens(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.input_sample_count += 1
        self.input_token_count += len(token_ids)
        self.buffer.extend(token_ids)
        while len(self.buffer) >= self.context_length:
            chunk = self.buffer[: self.context_length]
            del self.buffer[: self.context_length]
            self.offsets.append(self.token_count)
            tokens = array("I", chunk)
            tokens.tofile(self.bin_file)
            self.token_count += len(tokens)
            self.sample_count += 1

    def close(self) -> None:
        if self.closed:
            return
        self.dropped_tail_tokens = len(self.buffer)
        self.buffer.clear()
        self.bin_file.close()
        with self.idx_path.open("wb") as file:
            self.offsets.tofile(file)
        self.meta_path.write_text(
            json.dumps(
                {
                    "dataset": "dataset2",
                    "data_name": self.data_name,
                    "source_path": str(self.source_path),
                    "bin_path": str(self.bin_path),
                    "idx_path": str(self.idx_path),
                    "tokenizer_path": str(self.tokenizer_path),
                    "tokenizer_sha256": self.tokenizer_hash,
                    "context_length": self.context_length,
                    "token_dtype": TOKEN_DTYPE,
                    "index_dtype": INDEX_DTYPE,
                    "sample_count": self.sample_count,
                    "token_count": self.token_count,
                    "input_sample_count": self.input_sample_count,
                    "input_token_count": self.input_token_count,
                    "dropped_tail_tokens": self.dropped_tail_tokens,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.closed = True

    def mark_complete(self) -> None:
        if not self.closed:
            self.close()
        complete_path(self.prefix).write_text(
            json.dumps(
                {
                    "dataset": "dataset2",
                    "data_name": self.data_name,
                    "source_path": str(self.source_path),
                    "bin_path": str(self.bin_path),
                    "idx_path": str(self.idx_path),
                    "meta_path": str(self.meta_path),
                    "sample_count": self.sample_count,
                    "token_count": self.token_count,
                    "context_length": self.context_length,
                    "tokenizer_sha256": self.tokenizer_hash,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def preprocess_by_file(
    dataset_dir: Path,
    tokenizer: BPETokenizer,
    tokenizer_path: Path,
    tokenizer_hash: str,
    context_length: int,
    preprocess_dir: Path,
    max_samples: int | None = None,
    progress_every: int = 10_000,
    preview_chars: int = PREVIEW_CHARS,
    preview_tokens: int = PREVIEW_TOKENS,
    force: bool = False,
) -> dict[str, int]:
    writers: list[FixedContextTokenShardWriter] = []
    writer: FixedContextTokenShardWriter | None = None
    current_prefix: Path | None = None
    current_source_path: Path | None = None
    rows_seen = 0
    skipped = 0
    skipped_completed = 0
    stopped_early = False

    try:
        for _, source_path, text in iter_dataset2_raw_texts(dataset_dir, max_samples=max_samples):
            rows_seen += 1
            prefix = token_shard_prefix(
                source_path=source_path,
                context_length=context_length,
                dataset_dir=dataset_dir,
                preprocess_dir=preprocess_dir,
            )

            if current_prefix is not None and current_prefix != prefix and writer is not None:
                writer.mark_complete()
                print(f"complete data_name={writer.data_name} path={complete_path(writer.prefix)}", flush=True)
                writer = None
                current_prefix = None
                current_source_path = None

            if source_is_complete(prefix, tokenizer_hash, force):
                skipped_completed += 1
                continue

            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if not token_ids:
                skipped += 1
                continue

            if writer is None:
                writer = FixedContextTokenShardWriter(
                    prefix=prefix,
                    context_length=context_length,
                    source_path=source_path,
                    data_name=prefix.name,
                    tokenizer_path=tokenizer_path,
                    tokenizer_hash=tokenizer_hash,
                )
                writers.append(writer)
                current_prefix = prefix
                current_source_path = source_path

            writer.write_tokens(token_ids)
            if progress_every > 0 and rows_seen % progress_every == 0:
                sample_count = sum(item.sample_count for item in writers)
                print(
                    f"rows={rows_seen} shards={len(writers)} skipped_complete_rows={skipped_completed} "
                    f"full_context_samples={sample_count}",
                    flush=True,
                )
                print_tokenizer_preview(
                    rows_seen,
                    source_path,
                    text,
                    token_ids,
                    tokenizer,
                    preview_chars,
                    preview_tokens,
                )

        if max_samples is not None and rows_seen >= max_samples:
            stopped_early = True
    except Exception:
        if writer is not None:
            writer.close()
        raise
    else:
        if writer is not None:
            if stopped_early:
                writer.close()
            else:
                writer.mark_complete()
                print(f"complete data_name={writer.data_name} path={complete_path(writer.prefix)}", flush=True)

    manifest = {
        "dataset": "dataset2",
        "context_length": context_length,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "shard_count": len(writers),
        "input_rows": rows_seen,
        "skipped_rows": skipped,
        "skipped_completed_rows": skipped_completed,
        "sample_count": sum(writer.sample_count for writer in writers),
        "token_count": sum(writer.token_count for writer in writers),
        "shards": [
            {
                "data_name": writer.data_name,
                "source_path": str(writer.source_path),
                "bin_path": str(writer.bin_path),
                "idx_path": str(writer.idx_path),
                "complete_path": str(complete_path(writer.prefix)),
                "sample_count": writer.sample_count,
                "token_count": writer.token_count,
                "dropped_tail_tokens": writer.dropped_tail_tokens,
            }
            for writer in sorted(writers, key=lambda item: item.data_name)
        ],
    }
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    (preprocess_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "rows": rows_seen,
        "skipped": skipped,
        "skipped_completed": skipped_completed,
        "shards": len(writers),
        "samples": int(manifest["sample_count"]),
        "tokens": int(manifest["token_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset2 raw JSON/JSONL files into bin+idx token shards.")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to dataset2/preprocess{context_length}")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--preview-chars", type=int, default=PREVIEW_CHARS)
    parser.add_argument("--preview-tokens", type=int, default=PREVIEW_TOKENS)
    parser.add_argument("--force", action="store_true", help="rewrite shards even when tokenizer hash matches")
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f"missing tokenizer: {args.tokenizer}. Run train_tokenizer.py first.")

    out_dir = args.out_dir or token_preprocess_dir(args.context_length)
    tokenizer_hash = tokenizer_sha256(args.tokenizer)
    tokenizer = BPETokenizer.load(args.tokenizer)
    counts = preprocess_by_file(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        tokenizer_hash=tokenizer_hash,
        context_length=args.context_length,
        preprocess_dir=out_dir,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
        preview_chars=args.preview_chars,
        preview_tokens=args.preview_tokens,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "counts": counts,
                "out_dir": str(out_dir),
                "tokenizer_path": str(args.tokenizer),
                "tokenizer_sha256": tokenizer_hash,
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
