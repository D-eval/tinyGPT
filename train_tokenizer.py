from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from dataset2.read import DEFAULT_CONTEXT_LENGTH
from read import BPETokenizer, SPECIAL_TOKENS
from read_union import (
    DATASET_CHOICES,
    load_dataset2,
    read_dataset_emotion_sft_part,
    read_dataset_part,
)


TOKENIZER_PATH = Path("params/tokenizer2.json")
VOCAB_SIZE = 16_000
STATS_PATH = Path("params/tokenizer2_stats.json")
DATASET2_RAW_DIR = Path("dataset2/data")
TEXT_KEYS = (
    "instruction",
    "input",
    "output",
    "question",
    "answer",
    "query",
    "response",
    "prompt",
    "completion",
    "text",
    "content",
    "article",
    "passage",
    "title",
)
CHAT_KEYS = ("messages", "conversations", "dialogue", "history", "turns")


@dataclass
class SourceStats:
    samples: int = 0
    chars: int = 0
    longest_chars: int = 0
    longest_path: str = ""

    def add(self, path: Path, text: str) -> None:
        self.samples += 1
        self.chars += len(text)
        if len(text) > self.longest_chars:
            self.longest_chars = len(text)
            self.longest_path = str(path)


def normalize_text(text: str) -> str:
    return text.strip()


def iter_dataset_texts(dataset_dir: Path) -> Iterator[tuple[str, Path, str]]:
    for path, text in read_dataset_part(dataset_dir):
        text = normalize_text(text)
        if text:
            yield "dataset", path, text


def iter_emotion_sft_texts() -> Iterator[tuple[str, Path, str]]:
    for path, text in read_dataset_emotion_sft_part():
        text = normalize_text(text)
        if text:
            yield "dataset_emotion_sft", path, text


def collect_text(value: Any, fallback: bool = False) -> str:
    parts: list[str] = []
    if isinstance(value, str):
        text = normalize_text(value)
        if text and (fallback or len(text) >= 2):
            parts.append(text)
    elif isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value:
                text = collect_text(value[key], fallback=True)
                if text:
                    parts.append(text)
        for key in CHAT_KEYS:
            if key in value:
                text = collect_text(value[key], fallback=True)
                if text:
                    parts.append(text)
        if not parts and fallback:
            for item in value.values():
                text = collect_text(item, fallback=False)
                if text:
                    parts.append(text)
    elif isinstance(value, list):
        for item in value:
            text = collect_text(item, fallback=fallback)
            if text:
                parts.append(text)
    return "\n".join(parts)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        buffer = ""
        reached_array = False
        eof = False
        while True:
            if not eof:
                chunk = file.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            index = 0
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if not reached_array:
                    if index >= len(buffer):
                        break
                    if buffer[index] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array.")
                    reached_array = True
                    index += 1
                    continue
                while index < len(buffer) and (buffer[index].isspace() or buffer[index] == ","):
                    index += 1
                if index < len(buffer) and buffer[index] == "]":
                    return
                if index >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError:
                    break
                yield value
                index = end

            buffer = buffer[index:]
            if eof:
                if buffer.strip() not in {"", "]"}:
                    raise ValueError(f"Could not fully parse {path}.")
                return


def iter_json_stream(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        buffer = ""
        eof = False
        while True:
            if not eof:
                chunk = file.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            index = 0
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError:
                    break
                yield value
                index = end

            buffer = buffer[index:]
            if eof:
                if buffer.strip():
                    raise ValueError(f"Could not fully parse {path}.")
                return


def iter_json_values(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        while True:
            char = file.read(1)
            if not char:
                return
            if char.isspace():
                continue
            first = char
            break
    if first == "[":
        yield from iter_json_array(path)
    else:
        yield from iter_json_stream(path)


def iter_dataset2_raw_texts(dataset2_dir: Path, max_samples: int | None) -> Iterator[tuple[str, Path, str]]:
    count = 0
    for path in sorted(dataset2_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in {".json", ".jsonl", ".txt", ".csv"}:
            continue
        if path.suffix.lower() == ".txt":
            text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                yield "dataset2", path, text
                count += 1
        else:
            for value in iter_json_values(path):
                text = normalize_text(collect_text(value, fallback=True))
                if text:
                    yield "dataset2", path, text
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        return


def iter_dataset2_shard_texts(
    source_tokenizer_path: Path,
    context_length: int,
    preprocess_dir: Path | None,
    max_samples: int | None,
) -> Iterator[tuple[str, Path, str]]:
    if not source_tokenizer_path.exists():
        raise FileNotFoundError(
            f"{source_tokenizer_path} is required to decode existing dataset2 token shards. "
            "After training the new tokenizer, rerun dataset2 preprocessing with it."
        )

    source_tokenizer = BPETokenizer.load(source_tokenizer_path)
    dataset2 = load_dataset2(context_length=context_length, preprocess_dir=preprocess_dir)
    for index, (path, tokens) in enumerate(dataset2):
        if max_samples is not None and index >= max_samples:
            break
        text = normalize_text(source_tokenizer.decode(tokens))
        if text:
            yield "dataset2", path, text


def iter_union_texts(
    datasets: set[str],
    dataset_dir: Path,
    dataset2_source_tokenizer: Path,
    context_length: int,
    preprocess_dir: Path | None,
    max_dataset2_samples: int | None,
    dataset2_dir: Path,
    dataset2_from_shards: bool,
    progress_every: int,
    stats: dict[str, SourceStats],
) -> Iterator[str]:
    iterators: list[Iterator[tuple[str, Path, str]]] = []
    if "dataset" in datasets:
        iterators.append(iter_dataset_texts(dataset_dir))
    if "dataset_emotion_sft" in datasets:
        iterators.append(iter_emotion_sft_texts())
    if "dataset2" in datasets:
        if dataset2_from_shards:
            iterators.append(
                iter_dataset2_shard_texts(
                    dataset2_source_tokenizer,
                    context_length=context_length,
                    preprocess_dir=preprocess_dir,
                    max_samples=max_dataset2_samples,
                )
            )
        else:
            iterators.append(
                iter_dataset2_raw_texts(
                    dataset2_dir=dataset2_dir,
                    max_samples=max_dataset2_samples,
                )
            )

    total = 0
    for sample_iter in iterators:
        for source, path, text in sample_iter:
            stats[source].add(path, text)
            total += 1
            if progress_every > 0 and total % progress_every == 0:
                print(
                    f"tokenizer data: samples={total} "
                    f"dataset={stats['dataset'].samples} "
                    f"dataset_emotion_sft={stats['dataset_emotion_sft'].samples} "
                    f"dataset2={stats['dataset2'].samples}",
                    flush=True,
                )
            yield text


def train_tokenizer(text_iter: Iterator[str], vocab_size: int) -> BPETokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    return BPETokenizer(tokenizer, target_vocab_size=vocab_size)


def tokenizer_token_stats(tokenizer: BPETokenizer, stats: dict[str, SourceStats]) -> dict[str, object]:
    return {
        "vocab_size": tokenizer.vocab_size,
        "target_vocab_size": tokenizer.target_vocab_size,
        "special_tokens": {token: tokenizer.special_id(token) for token in SPECIAL_TOKENS},
        "sources": {source: asdict(source_stats) for source, source_stats in stats.items()},
        "total_samples": sum(source_stats.samples for source_stats in stats.values()),
        "total_chars": sum(source_stats.chars for source_stats in stats.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train params/tokenizer2.json from read_union data without preserving the previous tokenizer ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--stats-out", type=Path, default=STATS_PATH)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset2-dir", type=Path, default=DATASET2_RAW_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=list(DATASET_CHOICES))
    parser.add_argument(
        "--dataset2-source-tokenizer",
        type=Path,
        default=TOKENIZER_PATH,
        help="tokenizer used only to decode existing dataset2 token shards before overwriting --out",
    )
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--preprocess-dir", type=Path, default=None)
    parser.add_argument("--max-dataset2-samples", type=int, default=None)
    parser.add_argument("--dataset2-from-shards", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    selected = set(args.datasets)
    stats = {source: SourceStats() for source in DATASET_CHOICES}
    text_iter = iter_union_texts(
        selected,
        dataset_dir=args.dataset_dir,
        dataset2_source_tokenizer=args.dataset2_source_tokenizer,
        context_length=args.context_length,
        preprocess_dir=args.preprocess_dir,
        max_dataset2_samples=args.max_dataset2_samples,
        dataset2_dir=args.dataset2_dir,
        dataset2_from_shards=args.dataset2_from_shards,
        progress_every=args.progress_every,
        stats=stats,
    )

    tokenizer = train_tokenizer(text_iter, vocab_size=args.vocab_size)
    tokenizer.save(args.out)

    payload = tokenizer_token_stats(tokenizer, stats)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"saved tokenizer: {args.out}", flush=True)
    print(f"saved stats: {args.stats_out}", flush=True)


if __name__ == "__main__":
    main()
