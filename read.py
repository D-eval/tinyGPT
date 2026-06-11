from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def find_text_files(dataset_dir: str | Path = "dataset") -> list[Path]:
    root = Path(dataset_dir)
    return sorted(
        path
        for path in root.rglob("*.txt")
        if path.is_file() and not path.name.startswith("._")
    )


def read_texts(dataset_dir: str | Path = "dataset") -> list[tuple[Path, str]]:
    texts: list[tuple[Path, str]] = []
    for path in find_text_files(dataset_dir):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            texts.append((path, text))
    return texts


def read_ids(
    tokenizer: "BPETokenizer",
    dataset_dir: str | Path = "dataset",
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[tuple[Path, list[int]]]:
    samples: list[tuple[Path, list[int]]] = []
    for path, text in read_texts(dataset_dir):
        ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        if ids:
            samples.append((path, ids))
    return samples


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer, target_vocab_size: int = 16000) -> None:
        self.tokenizer = tokenizer
        self.target_vocab_size = target_vocab_size
        vocab = tokenizer.get_vocab()
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 16000) -> "BPETokenizer":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train_from_iterator((text for text in texts if text), trainer=trainer)
        return cls(tokenizer, target_vocab_size=vocab_size)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [
            int(idx)
            for idx in ids
            if int(idx) not in {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        ]
        return self.tokenizer.decode(filtered, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        meta_path(path).write_text(
            json.dumps({"target_vocab_size": self.target_vocab_size}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        target_vocab_size = 16000
        if meta_path(path).exists():
            payload = json.loads(meta_path(path).read_text(encoding="utf-8"))
            target_vocab_size = int(payload.get("target_vocab_size", target_vocab_size))
        return cls(Tokenizer.from_file(str(path)), target_vocab_size=target_vocab_size)


def meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta")


def corpus_stats(tokenizer: BPETokenizer, texts: list[tuple[Path, str]]) -> dict[str, int | float | str]:
    encoded_lengths = [len(tokenizer.encode(text, add_bos=True, add_eos=True)) for _, text in texts]
    total_tokens = sum(encoded_lengths)
    longest_index = max(range(len(encoded_lengths)), key=encoded_lengths.__getitem__, default=-1)
    longest_context = encoded_lengths[longest_index] if longest_index >= 0 else 0
    token_memory_mb = total_tokens * 8 / (1024 * 1024)
    return {
        "files": len(texts),
        "chars": sum(len(text) for _, text in texts),
        "tokens": total_tokens,
        "longest_context": longest_context,
        "longest_file": str(texts[longest_index][0]) if longest_index >= 0 else "",
        "token_memory_mb": round(token_memory_mb, 3),
    }


def ids_corpus_stats(samples: list[tuple[Path, list[int]]]) -> dict[str, int | float | str]:
    lengths = [len(ids) for _, ids in samples]
    total_tokens = sum(lengths)
    longest_index = max(range(len(lengths)), key=lengths.__getitem__, default=-1)
    longest_context = lengths[longest_index] if longest_index >= 0 else 0
    token_memory_mb = total_tokens * 8 / (1024 * 1024)
    return {
        "files": len(samples),
        "tokens": total_tokens,
        "longest_context": longest_context,
        "longest_file": str(samples[longest_index][0]) if longest_index >= 0 else "",
        "token_memory_mb": round(token_memory_mb, 3),
    }


def load_or_train_tokenizer(
    texts: list[tuple[Path, str]],
    tokenizer_path: str | Path = "params/tokenizer.json",
    vocab_size: int = 16000,
) -> BPETokenizer:
    path = Path(tokenizer_path)
    if path.exists():
        return BPETokenizer.load(path)

    tokenizer = BPETokenizer.train([text for _, text in texts], vocab_size=vocab_size)
    tokenizer.save(path)
    return tokenizer
