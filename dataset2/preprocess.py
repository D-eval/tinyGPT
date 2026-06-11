from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset2.read import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CONTEXT_LENGTH,
    Dataset,
    load_indexed_text,
    token_preprocess_dir,
    token_shard_prefix,
)
from read import BPETokenizer, SPECIAL_TOKENS  # noqa: E402


TOKENIZER_PATH = ROOT_DIR / "params" / "tokenizer2.json"
VOCAB_SIZE = 16_000
TOKEN_DTYPE = "uint32"
INDEX_DTYPE = "uint64"
PREVIEW_CHARS = 160
PREVIEW_TOKENS = 40


def dataset2_text_iter(dataset_dir: str | Path, rebuild_index: bool = False) -> Iterator[tuple[Path, str]]:
    dataset = Dataset(dataset_dir, rebuild_index=rebuild_index, lazy=True)
    for path, text in dataset:
        text = text.strip()
        if text:
            yield path, text


def train_streaming_tokenizer(
    sample_iter: Iterable[tuple[Path, str]],
    tokenizer_path: str | Path = TOKENIZER_PATH,
    vocab_size: int = VOCAB_SIZE,
) -> BPETokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator((text for _, text in sample_iter if text), trainer=trainer)
    wrapped = BPETokenizer(tokenizer, target_vocab_size=vocab_size)
    wrapped.save(tokenizer_path)
    return wrapped


def load_or_train_tokenizer(
    tokenizer_path: str | Path,
    dataset_dir: str | Path,
    vocab_size: int,
    rebuild_index: bool = False,
) -> BPETokenizer:
    path = Path(tokenizer_path)
    if path.exists():
        return BPETokenizer.load(path)
    return train_streaming_tokenizer(
        dataset2_text_iter(dataset_dir, rebuild_index=rebuild_index),
        path,
        vocab_size,
    )


def compact_preview(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


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


def complete_path(prefix: Path) -> Path:
    return prefix.with_suffix(".complete")


class FixedContextTokenShardWriter:
    def __init__(
        self,
        prefix: Path,
        context_length: int,
        source_path: Path,
        data_name: str,
        tokenizer_path: Path,
    ) -> None:
        self.prefix = prefix
        self.context_length = context_length
        self.source_path = source_path
        self.data_name = data_name
        self.tokenizer_path = tokenizer_path
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
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def preprocess_by_file(
    dataset: Dataset,
    tokenizer: BPETokenizer,
    tokenizer_path: Path,
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
    current_source_path: Path | None = None
    rows_seen = 0
    skipped = 0
    skipped_completed = 0
    stopped_early = False
    try:
        for row in dataset.iter_rows():
            if max_samples is not None and rows_seen >= max_samples:
                stopped_early = True
                break
            rows_seen += 1
            source_path = Path(row["path"])

            if current_source_path is not None and source_path != current_source_path and writer is not None:
                writer.mark_complete()
                print(f"complete data_name={writer.data_name} path={complete_path(writer.prefix)}", flush=True)
                writer = None
                current_source_path = None

            prefix = token_shard_prefix(
                source_path,
                context_length=context_length,
                dataset_dir=dataset.dataset_dir,
                preprocess_dir=preprocess_dir,
            )
            done_path = complete_path(prefix)
            if done_path.exists() and not force:
                skipped_completed += 1
                continue

            text = load_indexed_text(row).strip()
            if not text:
                skipped += 1
                continue

            if writer is None:
                writer = FixedContextTokenShardWriter(
                    prefix=prefix,
                    context_length=context_length,
                    source_path=source_path,
                    data_name=prefix.name,
                    tokenizer_path=tokenizer_path,
                )
                writers.append(writer)
                current_source_path = source_path
            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
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
    parser = argparse.ArgumentParser(description="Preprocess dataset2 into one bin/idx pair per source data file.")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to dataset2/preprocess{context_length}")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--preview-chars", type=int, default=PREVIEW_CHARS)
    parser.add_argument("--preview-tokens", type=int, default=PREVIEW_TOKENS)
    parser.add_argument("--force", action="store_true", help="rewrite shards even when {data_name}.complete exists")
    args = parser.parse_args()

    out_dir = args.out_dir or token_preprocess_dir(args.context_length)
    tokenizer = load_or_train_tokenizer(
        args.tokenizer,
        args.dataset_dir,
        args.vocab_size,
        rebuild_index=args.rebuild_index,
    )
    dataset = Dataset(args.dataset_dir, rebuild_index=args.rebuild_index, lazy=True)
    counts = preprocess_by_file(
        dataset,
        tokenizer,
        args.tokenizer,
        args.context_length,
        out_dir,
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
