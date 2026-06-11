from __future__ import annotations

import argparse
import json
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from dataset2.read import Dataset as Dataset2
from read import BPETokenizer, SPECIAL_TOKENS, read_texts as read_dataset1_texts


TOKENIZER_PATH = Path("params/tokenizer2.json")
DEFAULT_OUT_DIR = Path("preprocess")
VOCAB_SIZE = 16_000
VALID_RATIO = 0.01
TOKEN_DTYPE = "uint32"
INDEX_DTYPE = "uint64"
PREVIEW_CHARS = 160
PREVIEW_TOKENS = 40


def dataset1_iter(dataset_dir: str | Path) -> Iterator[tuple[Path, str]]:
    for path, text in read_dataset1_texts(dataset_dir):
        text = text.strip()
        if text:
            yield path, text


def dataset2_iter(dataset_dir: str | Path, rebuild_index: bool = False) -> Iterator[tuple[Path, str]]:
    dataset = Dataset2(dataset_dir, rebuild_index=rebuild_index, lazy=True)
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
    dataset2_dir: str | Path,
    vocab_size: int,
    rebuild_index: bool = False,
) -> BPETokenizer:
    path = Path(tokenizer_path)
    if path.exists():
        return BPETokenizer.load(path)

    def all_samples() -> Iterator[tuple[Path, str]]:
        yield from dataset1_iter(dataset_dir)
        yield from dataset2_iter(dataset2_dir, rebuild_index=rebuild_index)

    return train_streaming_tokenizer(all_samples(), path, vocab_size)


def split_name(
    sample_index: int,
    valid_ratio: float = VALID_RATIO,
    total_samples: int | None = None,
) -> str:
    if valid_ratio <= 0:
        return "train"
    if total_samples is not None:
        valid_count = max(int(total_samples * valid_ratio), 1 if total_samples > 1 else 0)
        return "valid" if sample_index >= total_samples - valid_count else "train"
    interval = max(round(1.0 / valid_ratio), 1)
    return "valid" if sample_index % interval == interval - 1 else "train"


class TokenShardWriter:
    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.bin_path = prefix.with_suffix(".bin")
        self.idx_path = prefix.with_suffix(".idx")
        self.meta_path = prefix.with_suffix(".meta.json")
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.bin_file: BinaryIO = self.bin_path.open("wb")
        self.offsets = array("Q")
        self.token_count = 0
        self.sample_count = 0

    def write(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.offsets.append(self.token_count)
        tokens = array("I", token_ids)
        tokens.tofile(self.bin_file)
        self.token_count += len(tokens)
        self.sample_count += 1

    def close(self) -> None:
        self.bin_file.close()
        with self.idx_path.open("wb") as file:
            self.offsets.tofile(file)

    def write_meta(self, extra: dict[str, object]) -> None:
        payload = {
            **extra,
            "bin_path": str(self.bin_path),
            "idx_path": str(self.idx_path),
            "token_dtype": TOKEN_DTYPE,
            "index_dtype": INDEX_DTYPE,
            "sample_count": self.sample_count,
            "token_count": self.token_count,
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_writers(out_dir: Path) -> dict[tuple[str, str], TokenShardWriter]:
    return {
        ("dataset1", "train"): TokenShardWriter(out_dir / "tokens_dataset1_train"),
        ("dataset1", "valid"): TokenShardWriter(out_dir / "tokens_dataset1_valid"),
        ("dataset2", "train"): TokenShardWriter(out_dir / "tokens_dataset2_train"),
        ("dataset2", "valid"): TokenShardWriter(out_dir / "tokens_dataset2_valid"),
    }


def compact_preview(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def print_sample_preview(
    name: str,
    sample_index: int,
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
                "dataset": name,
                "sample": sample_index + 1,
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


def encode_samples(
    name: str,
    samples: Iterable[tuple[Path, str]],
    tokenizer: BPETokenizer,
    writers: dict[tuple[str, str], TokenShardWriter],
    valid_ratio: float,
    total_samples: int | None = None,
    max_samples: int | None = None,
    progress_every: int = 10_000,
    preview_chars: int = PREVIEW_CHARS,
    preview_tokens: int = PREVIEW_TOKENS,
) -> dict[str, int]:
    if total_samples is not None and max_samples is not None:
        total_samples = min(total_samples, max_samples)
    counts = {"train": 0, "valid": 0, "skipped": 0}
    for index, (path, text) in enumerate(samples):
        if max_samples is not None and index >= max_samples:
            break
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if not token_ids:
            counts["skipped"] += 1
            continue
        split = split_name(index, valid_ratio, total_samples=total_samples)
        writers[(name, split)].write(token_ids)
        counts[split] += 1
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                f"{name}: samples={index + 1} train={counts['train']} valid={counts['valid']}",
                flush=True,
            )
            print_sample_preview(
                name,
                index,
                path,
                text,
                token_ids,
                tokenizer,
                preview_chars,
                preview_tokens,
            )
    return counts


def load_token_sample(prefix: Path, sample_index: int) -> list[int]:
    idx_path = prefix.with_suffix(".idx")
    bin_path = prefix.with_suffix(".bin")
    offsets = array("Q")
    with idx_path.open("rb") as file:
        offsets.fromfile(file, idx_path.stat().st_size // offsets.itemsize)
    if sample_index < 0 or sample_index >= len(offsets):
        raise IndexError(sample_index)

    start = offsets[sample_index]
    if sample_index + 1 < len(offsets):
        end = offsets[sample_index + 1]
    else:
        end = bin_path.stat().st_size // array("I").itemsize
    tokens = array("I")
    with bin_path.open("rb") as file:
        file.seek(start * tokens.itemsize)
        tokens.fromfile(file, end - start)
    return list(tokens)


def verify_outputs(out_dir: Path, tokenizer: BPETokenizer) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for name in (
        "tokens_dataset1_train",
        "tokens_dataset1_valid",
        "tokens_dataset2_train",
        "tokens_dataset2_valid",
    ):
        prefix = out_dir / name
        meta_path = prefix.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta["sample_count"]) == 0:
            decoded[name] = ""
            continue
        tokens = load_token_sample(prefix, 0)
        text = tokenizer.decode(tokens)
        if not text.strip():
            raise RuntimeError(f"decoded empty text from {prefix}")
        decoded[name] = text[:120].replace("\n", " ")
    return decoded


def close_writers(writers: dict[tuple[str, str], TokenShardWriter]) -> None:
    for writer in writers.values():
        writer.close()


def write_metadata(
    writers: dict[tuple[str, str], TokenShardWriter],
    tokenizer_path: Path,
    valid_ratio: float,
    counts: dict[str, dict[str, int]],
) -> None:
    for (dataset_name, split), writer in writers.items():
        writer.write_meta(
            {
                "dataset": dataset_name,
                "split": split,
                "tokenizer_path": str(tokenizer_path),
                "valid_ratio": valid_ratio,
                "source_counts": counts.get(dataset_name, {}),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize read_union dataset1/dataset2 into bin+idx shards.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset2-dir", type=Path, default=Path("dataset2/data"))
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-dataset1-samples", type=int, default=None)
    parser.add_argument("--max-dataset2-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--preview-chars", type=int, default=PREVIEW_CHARS)
    parser.add_argument("--preview-tokens", type=int, default=PREVIEW_TOKENS)
    args = parser.parse_args()

    tokenizer = load_or_train_tokenizer(
        args.tokenizer,
        args.dataset_dir,
        args.dataset2_dir,
        args.vocab_size,
        rebuild_index=args.rebuild_index,
    )
    writers = make_writers(args.out_dir)
    try:
        dataset1_samples = list(dataset1_iter(args.dataset_dir))
        dataset2 = Dataset2(args.dataset2_dir, rebuild_index=args.rebuild_index, lazy=True)
        counts = {
            "dataset1": encode_samples(
                "dataset1",
                dataset1_samples,
                tokenizer,
                writers,
                args.valid_ratio,
                total_samples=len(dataset1_samples),
                max_samples=args.max_dataset1_samples,
                progress_every=args.progress_every,
                preview_chars=args.preview_chars,
                preview_tokens=args.preview_tokens,
            ),
            "dataset2": encode_samples(
                "dataset2",
                dataset2,
                tokenizer,
                writers,
                args.valid_ratio,
                total_samples=len(dataset2),
                max_samples=args.max_dataset2_samples,
                progress_every=args.progress_every,
                preview_chars=args.preview_chars,
                preview_tokens=args.preview_tokens,
            ),
        }
    finally:
        close_writers(writers)

    write_metadata(writers, args.tokenizer, args.valid_ratio, counts)
    decoded = verify_outputs(args.out_dir, tokenizer)
    print(json.dumps({"counts": counts, "decoded_preview": decoded}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
