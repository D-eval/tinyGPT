from __future__ import annotations

import argparse
import json
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dataset2.read import DEFAULT_CONTEXT_LENGTH, Dataset as Dataset2
from read import (
    BPETokenizer,
    corpus_stats,
    ids_corpus_stats,
    load_or_train_tokenizer,
    read_ids as read_dataset_ids,
    read_texts as read_dataset_texts,
)
from sft.emotion.download_data import parse_parlai_personachat
from sft.emotion.read import Dataset as EmotionDataset
from sft.emotion.read import EmotionSample
from sft.emotion.read import format_dialogue


DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_TOKENIZER_PATH = Path("params/tokenizer2.json")
IGNORE_INDEX = -100
EMOTION_SFT_PERSONACHAT_TARGET = 64_670
EMOTION_SFT_SOURCE_ORDER = ("dailydialog", "personachat", "livechat", "naturalconv", "cped")
DATASET_CHOICES = ("dataset", "dataset2", "dataset_emotion_sft")
_EMOTION_SFT_CACHE: list[EmotionSample] | None = None


@dataclass(frozen=True)
class TokenSample:
    source: str
    path: Path
    tokens: list[int]
    labels: list[int] | None = None


def _clean_texts(texts: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    cleaned: list[tuple[Path, str]] = []
    for path, text in texts:
        text = text.strip()
        if text:
            cleaned.append((path, text))
    return cleaned


def read_dataset_part(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> list[tuple[Path, str]]:
    return _clean_texts(read_dataset_texts(dataset_dir))


def read_dataset_part_ids(
    tokenizer: BPETokenizer,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> list[tuple[Path, list[int]]]:
    return read_dataset_ids(tokenizer, dataset_dir)


def load_dataset2(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Dataset2:
    return Dataset2(context_length=context_length, preprocess_dir=preprocess_dir)


def load_dataset_emotion_sft(
    data_dir: str | Path | None = None,
    split: str | None = None,
    source: str | None = None,
    no_names: bool = True,
) -> EmotionDataset:
    return EmotionDataset(data_dir=data_dir, split=split, source=source, no_names=no_names)


def _personachat_raw_paths(data_dir: str | Path | None = None) -> list[Path]:
    if data_dir is None:
        root = Path(__file__).resolve().parent / "sft" / "emotion" / "data" / "raw" / "personachat"
    else:
        root = Path(data_dir) / "raw" / "personachat"
    paths = [path for path in root.rglob("*.txt") if path.is_file() and not path.name.startswith("._")]
    split_rank = {"train": 0, "valid": 1, "test": 2}

    def key(path: Path) -> tuple[int, str]:
        split = path.name.split("_", 1)[0]
        return split_rank.get(split, 99), path.name

    return sorted(paths, key=key)


def load_dataset_emotion_sft_samples(
    data_dir: str | Path | None = None,
    no_names: bool = True,
) -> list[EmotionSample]:
    global _EMOTION_SFT_CACHE
    if data_dir is None and no_names and _EMOTION_SFT_CACHE is not None:
        return _EMOTION_SFT_CACHE

    samples: list[EmotionSample] = []
    for source in EMOTION_SFT_SOURCE_ORDER:
        if source == "personachat":
            personachat_samples = []
            for path in _personachat_raw_paths(data_dir):
                personachat_samples.extend(parse_parlai_personachat(path))
                if len(personachat_samples) >= EMOTION_SFT_PERSONACHAT_TARGET:
                    break
            samples.extend(personachat_samples[:EMOTION_SFT_PERSONACHAT_TARGET])
        else:
            samples.extend(
                load_dataset_emotion_sft(data_dir=data_dir, split="train", source=source, no_names=no_names).samples
            )
    if data_dir is None and no_names:
        _EMOTION_SFT_CACHE = samples
    return samples


def read_dataset_emotion_sft_part(
    data_dir: str | Path | None = None,
    split: str | None = "train",
    source: str | None = None,
    no_names: bool = True,
) -> list[tuple[Path, str]]:
    if split == "train" and source is None:
        return [
            (Path(sample.id), format_dialogue(sample.system, sample.turns))
            for sample in load_dataset_emotion_sft_samples(data_dir, no_names)
        ]
    return [
        (Path(sample.id), format_dialogue(sample.system, sample.turns))
        for sample in load_dataset_emotion_sft(data_dir=data_dir, split=split, source=source, no_names=no_names).samples
    ]


def _append_encoded_segment(
    tokenizer: BPETokenizer,
    tokens: list[int],
    labels: list[int],
    text: str,
    predict: bool,
) -> None:
    segment_ids = tokenizer.encode(text)
    tokens.extend(segment_ids)
    labels.extend(segment_ids if predict else [IGNORE_INDEX] * len(segment_ids))


def _append_special_token(
    tokens: list[int],
    labels: list[int],
    token_id: int,
    predict: bool,
) -> None:
    tokens.append(token_id)
    labels.append(token_id if predict else IGNORE_INDEX)


def _append_agent_output(
    tokenizer: BPETokenizer,
    tokens: list[int],
    labels: list[int],
    text: str,
) -> None:
    _append_special_token(tokens, labels, tokenizer.bos_id, predict=True)
    _append_encoded_segment(tokenizer, tokens, labels, text, predict=True)
    _append_special_token(tokens, labels, tokenizer.eos_id, predict=True)


def _append_role_prefix(
    tokenizer: BPETokenizer,
    tokens: list[int],
    labels: list[int],
    role_token: str,
) -> None:
    _append_special_token(tokens, labels, tokenizer.special_id(role_token), predict=False)
    _append_encoded_segment(tokenizer, tokens, labels, ": ", predict=False)


def encode_emotion_sft_sample(tokenizer: BPETokenizer, sample: EmotionSample) -> TokenSample:
    tokens = [tokenizer.bos_id]
    labels = [IGNORE_INDEX]

    _append_role_prefix(tokenizer, tokens, labels, "<system>")
    _append_encoded_segment(tokenizer, tokens, labels, sample.system.rstrip(), predict=False)
    for turn in sample.turns:
        role = str(turn.get("role", "")).strip()
        text = str(turn.get("text", "")).rstrip()
        if role == "agent":
            _append_encoded_segment(tokenizer, tokens, labels, "\n", predict=False)
            _append_role_prefix(tokenizer, tokens, labels, "<agent>")
            _append_agent_output(tokenizer, tokens, labels, text)
        elif role == "usr":
            _append_encoded_segment(tokenizer, tokens, labels, "\n", predict=False)
            _append_role_prefix(tokenizer, tokens, labels, "<usr>")
            _append_encoded_segment(tokenizer, tokens, labels, text, predict=False)
    return TokenSample("dataset_emotion_sft", Path(sample.id), tokens, labels)


def iter_dataset_emotion_sft_token_samples(
    tokenizer: BPETokenizer,
    data_dir: str | Path | None = None,
    split: str | None = "train",
    source: str | None = None,
    no_names: bool = True,
) -> Iterator[TokenSample]:
    if split == "train" and source is None:
        samples = load_dataset_emotion_sft_samples(data_dir, no_names)
    else:
        samples = load_dataset_emotion_sft(data_dir=data_dir, split=split, source=source, no_names=no_names).samples
    for sample in samples:
        token_sample = encode_emotion_sft_sample(tokenizer, sample)
        if token_sample.labels is not None and any(label != IGNORE_INDEX for label in token_sample.labels[1:]):
            yield token_sample


def read_dataset_emotion_sft_part_ids(
    tokenizer: BPETokenizer,
    data_dir: str | Path | None = None,
    split: str | None = None,
    source: str | None = None,
    no_names: bool = True,
) -> list[tuple[Path, list[int]]]:
    return [
        (sample.path, sample.tokens)
        for sample in iter_dataset_emotion_sft_token_samples(
            tokenizer,
            data_dir=data_dir,
            split=split,
            source=source,
            no_names=no_names,
        )
    ]


def iter_dataset2_tokens(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Iterator[tuple[Path, list[int]]]:
    yield from load_dataset2(context_length=context_length, preprocess_dir=preprocess_dir)


def read_dataset2_tokens(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> list[tuple[Path, list[int]]]:
    return list(iter_dataset2_tokens(context_length=context_length, preprocess_dir=preprocess_dir))


def read_dataset2_part_ids(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> list[tuple[Path, list[int]]]:
    if max_samples is None:
        raise ValueError("dataset2 ids are huge; use iter_dataset2_tokens() or pass max_samples")
    samples: list[tuple[Path, list[int]]] = []
    for index, sample in enumerate(iter_dataset2_tokens(context_length=context_length, preprocess_dir=preprocess_dir)):
        if index >= max_samples:
            break
        samples.append(sample)
    return samples


def read_dataset2_part(
    dataset2_dir: str | Path | None = None,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_samples: int | None = None,
) -> list[tuple[Path, str]]:
    preprocess_dir = Path(dataset2_dir) if dataset2_dir is not None and Path(dataset2_dir).name.startswith("preprocess") else None
    dataset = load_dataset2(context_length=context_length, preprocess_dir=preprocess_dir)
    tokenizer = BPETokenizer.load(tokenizer_path)
    texts: list[tuple[Path, str]] = []
    for index, (path, tokens) in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        text = tokenizer.decode(tokens).strip()
        if text:
            texts.append((path, text))
    return texts


def read_union_texts(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    dataset2_dir: str | Path | None = None,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_dataset2_samples: int | None = None,
) -> list[tuple[Path, str]]:
    texts = read_dataset_part(dataset_dir)
    texts.extend(read_dataset_emotion_sft_part())
    texts.extend(
        read_dataset2_part(
            dataset2_dir,
            tokenizer_path=tokenizer_path,
            context_length=context_length,
            max_samples=max_dataset2_samples,
        )
    )
    return texts


def iter_dataset1_token_samples(
    tokenizer: BPETokenizer,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> Iterator[TokenSample]:
    for path, text in read_dataset_part(dataset_dir):
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        if tokens:
            yield TokenSample("dataset", path, tokens)


def iter_dataset2_token_samples(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Iterator[TokenSample]:
    for path, tokens in iter_dataset2_tokens(context_length=context_length, preprocess_dir=preprocess_dir):
        if tokens:
            yield TokenSample("dataset2", path, tokens)


def iter_union_token_samples(
    tokenizer: BPETokenizer,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Iterator[TokenSample]:
    yield from iter_dataset1_token_samples(tokenizer, dataset_dir)
    yield from iter_dataset_emotion_sft_token_samples(tokenizer)
    yield from iter_dataset2_token_samples(context_length=context_length, preprocess_dir=preprocess_dir)


def iter_selected_token_samples(
    tokenizer: BPETokenizer,
    datasets: set[str] | None = None,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Iterator[TokenSample]:
    selected = set(DATASET_CHOICES) if datasets is None else datasets
    if "dataset" in selected:
        yield from iter_dataset1_token_samples(tokenizer, dataset_dir)
    if "dataset_emotion_sft" in selected:
        yield from iter_dataset_emotion_sft_token_samples(tokenizer)
    if "dataset2" in selected:
        yield from iter_dataset2_token_samples(context_length=context_length, preprocess_dir=preprocess_dir)


def iter_union_ids(
    tokenizer_or_path: BPETokenizer | str | Path = DEFAULT_TOKENIZER_PATH,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
) -> Iterator[tuple[Path, list[int]]]:
    tokenizer = (
        tokenizer_or_path
        if isinstance(tokenizer_or_path, BPETokenizer)
        else BPETokenizer.load(tokenizer_or_path)
    )
    for sample in iter_union_token_samples(
        tokenizer,
        dataset_dir=dataset_dir,
        context_length=context_length,
        preprocess_dir=preprocess_dir,
    ):
        yield sample.path, sample.tokens


def read_union_ids(
    tokenizer_or_path: BPETokenizer | str | Path = DEFAULT_TOKENIZER_PATH,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> list[tuple[Path, list[int]]]:
    if max_samples is None:
        raise ValueError("union ids are huge; use iter_union_ids() or pass max_samples")
    samples: list[tuple[Path, list[int]]] = []
    for index, sample in enumerate(
        iter_union_ids(
            tokenizer_or_path,
            dataset_dir=dataset_dir,
            context_length=context_length,
            preprocess_dir=preprocess_dir,
        )
    ):
        if max_samples is not None and index >= max_samples:
            break
        samples.append(sample)
    return samples


def _dataset2_token_count(dataset: Dataset2) -> int:
    return sum(sample.token_count for sample in dataset.samples)


def verify_union(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    preprocess_dir: str | Path | None = None,
    datasets: set[str] | None = None,
) -> dict[str, object]:
    selected = set(DATASET_CHOICES) if datasets is None else datasets
    dataset_texts = read_dataset_part(dataset_dir) if "dataset" in selected else []
    emotion_sft = load_dataset_emotion_sft_samples() if "dataset_emotion_sft" in selected else []
    emotion_sft_source_counts: dict[str, int] = {}
    for sample in emotion_sft:
        emotion_sft_source_counts[sample.source] = emotion_sft_source_counts.get(sample.source, 0) + 1

    dataset2_samples = 0
    dataset2_tokens = 0
    dataset2_preprocess_dir = str(preprocess_dir) if preprocess_dir is not None else None
    dataset2_shards = 0
    dataset2_bin_tokens = 0
    shard_sample_counts: dict[str, int] = {}
    missing_dataset2_samples = 0
    dataset2_ok = True
    if "dataset2" in selected:
        dataset2 = load_dataset2(context_length=context_length, preprocess_dir=preprocess_dir)
        shard_sample_counts = {shard.name: len(shard.offsets) for shard in dataset2.shards}
        shard_token_counts = {shard.name: shard.token_count for shard in dataset2.shards}
        expected_dataset2_samples = sum(shard_sample_counts.values())
        dataset2_samples = len(dataset2)
        dataset2_tokens = _dataset2_token_count(dataset2)
        dataset2_preprocess_dir = str(dataset2.preprocess_dir)
        dataset2_shards = len(dataset2.shards)
        dataset2_bin_tokens = sum(shard_token_counts.values())
        missing_dataset2_samples = expected_dataset2_samples - len(dataset2)
        dataset2_ok = len(dataset2) == expected_dataset2_samples

    return {
        "ok": dataset2_ok,
        "selected_datasets": sorted(selected),
        "dataset_samples": len(dataset_texts),
        "dataset_emotion_sft_samples": len(emotion_sft),
        "dataset_emotion_sft_source_counts": emotion_sft_source_counts,
        "dataset2_samples": dataset2_samples,
        "union_samples": len(dataset_texts) + len(emotion_sft) + dataset2_samples,
        "dataset2_preprocess_dir": dataset2_preprocess_dir,
        "dataset2_shards": dataset2_shards,
        "dataset2_tokens": dataset2_tokens,
        "dataset2_bin_tokens": dataset2_bin_tokens,
        "dataset2_shard_sample_counts": shard_sample_counts,
        "missing_dataset_samples": 0,
        "missing_dataset2_samples": missing_dataset2_samples,
    }


def verify_union_texts(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    dataset2_dir: str | Path | None = None,
) -> dict[str, object]:
    preprocess_dir = Path(dataset2_dir) if dataset2_dir is not None and Path(dataset2_dir).name.startswith("preprocess") else None
    return verify_union(dataset_dir=dataset_dir, preprocess_dir=preprocess_dir)


def load_or_train_union_tokenizer(
    texts: list[tuple[Path, str]],
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    vocab_size: int = 16000,
) -> BPETokenizer:
    return load_or_train_tokenizer(texts, tokenizer_path=tokenizer_path, vocab_size=vocab_size)


def decode_preview(tokenizer: BPETokenizer, tokens: list[int], limit: int = 240) -> str:
    text = tokenizer.tokenizer.decode(tokens, skip_special_tokens=False)
    return " ".join(text.split())[:limit]


def sample_preview(tokenizer: BPETokenizer, sample: TokenSample, token_limit: int = 200) -> str:
    if sample.labels is not None:
        for index, label in enumerate(sample.labels):
            if label != IGNORE_INDEX:
                start = max(0, index - 12)
                end = min(len(sample.tokens), index + token_limit)
                return decode_preview(tokenizer, sample.tokens[start:end])
    return decode_preview(tokenizer, sample.tokens[:token_limit])


def _write_tokens_bin(tokens: Iterator[int], bin_path: Path) -> int:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    buffer = array("I")
    with bin_path.open("wb") as file:
        for token in tokens:
            buffer.append(int(token))
            if len(buffer) >= 1_000_000:
                buffer.tofile(file)
                count += len(buffer)
                buffer = array("I")
        if buffer:
            buffer.tofile(file)
            count += len(buffer)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the union of dataset text samples and dataset2 token shards.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
        help="datasets to check: dataset dataset2 dataset_emotion_sft",
    )
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--preprocess-dir", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--token-stats", action="store_true")
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--max-iterate", type=int, default=-1, help="iterate this many union id samples; -1 means all, 0 means skip")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    selected_datasets = set(args.datasets)

    verification = verify_union(args.dataset_dir, args.context_length, args.preprocess_dir, selected_datasets)
    output: dict[str, object] = {"verification": verification}

    if args.token_stats:
        tokenizer = BPETokenizer.load(args.tokenizer)
        if "dataset" in selected_datasets:
            dataset_ids = read_dataset_part_ids(tokenizer, args.dataset_dir)
            output["dataset_ids_stats"] = ids_corpus_stats(dataset_ids)
        if "dataset_emotion_sft" in selected_datasets:
            emotion_sft_samples = list(iter_dataset_emotion_sft_token_samples(tokenizer))
            output["dataset_emotion_sft_ids_stats"] = ids_corpus_stats(
                [(sample.path, sample.tokens) for sample in emotion_sft_samples]
            )
            output["dataset_emotion_sft_labeled_tokens"] = sum(
                1
                for sample in emotion_sft_samples
                for label in (sample.labels or [])
                if label != IGNORE_INDEX
            )
        if "dataset2" in selected_datasets:
            output["dataset2_ids_stats"] = {
                "files": verification["dataset2_samples"],
                "tokens": verification["dataset2_tokens"],
                "longest_context": args.context_length,
                "token_memory_mb": round(int(verification["dataset2_tokens"]) * 8 / (1024 * 1024), 3),
            }

    if args.decode_samples > 0 and "dataset2" in selected_datasets:
        tokenizer = BPETokenizer.load(args.tokenizer)
        previews = []
        dataset2 = load_dataset2(args.context_length, args.preprocess_dir)
        for index, (path, tokens) in enumerate(dataset2):
            if index >= args.decode_samples:
                break
            previews.append(
                {
                    "index": index,
                    "path": str(path),
                    "tokens": len(tokens),
                    "preview": decode_preview(tokenizer, tokens[:1000]),
                }
            )
        output["dataset2_previews"] = previews

    if args.max_iterate != 0:
        tokenizer = BPETokenizer.load(args.tokenizer)
        iterated = 0
        total_tokens = 0
        previews = []
        for sample in iter_selected_token_samples(
            tokenizer,
            datasets=selected_datasets,
            dataset_dir=args.dataset_dir,
            context_length=args.context_length,
            preprocess_dir=args.preprocess_dir,
        ):
            if args.max_iterate > 0 and iterated >= args.max_iterate:
                break
            path = sample.path
            tokens = sample.tokens
            iterated += 1
            total_tokens += len(tokens)
            should_print = (
                iterated == 1
                or (args.progress_every > 0 and iterated % args.progress_every == 0)
                or (args.max_iterate > 0 and iterated == args.max_iterate)
                or (args.max_iterate < 0 and iterated == int(verification["union_samples"]))
            )
            if should_print:
                preview = sample_preview(tokenizer, sample)
                message = {
                    "index": iterated - 1,
                    "source": sample.source,
                    "path": str(path),
                    "tokens": len(tokens),
                    "total_tokens": total_tokens,
                    "preview": preview,
                }
                if not args.json:
                    print(json.dumps(message, ensure_ascii=False), flush=True)
                if len(previews) < 5:
                    previews.append(message)
        output["iter_union_ids"] = {
            "samples": iterated,
            "tokens": total_tokens,
            "previews": previews,
        }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(
            "union verify: "
            f"ok={verification['ok']} "
            f"selected={verification['selected_datasets']} "
            f"dataset={verification['dataset_samples']} "
            f"dataset_emotion_sft={verification['dataset_emotion_sft_samples']} "
            f"dataset_emotion_sft_sources={verification['dataset_emotion_sft_source_counts']} "
            f"dataset2={verification['dataset2_samples']} "
            f"union={verification['union_samples']} "
            f"dataset2_shards={verification['dataset2_shards']} "
            f"dataset2_tokens={verification['dataset2_tokens']}"
        )
        if "dataset_ids_stats" in output:
            print("dataset ids stats:", output["dataset_ids_stats"])
        if "dataset_emotion_sft_ids_stats" in output:
            print("dataset_emotion_sft ids stats:", output["dataset_emotion_sft_ids_stats"])
            print("dataset_emotion_sft labeled tokens:", output["dataset_emotion_sft_labeled_tokens"])
        if "dataset2_ids_stats" in output:
            print("dataset2 ids stats:", output["dataset2_ids_stats"])
        # for preview in output.get("dataset2_previews", []):
        #     print(json.dumps(preview, ensure_ascii=False))
        # if "iter_union_ids" in output:
        #     print("iter union ids:", json.dumps(output["iter_union_ids"], ensure_ascii=False))

    if not verification["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
