from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


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


class BPETokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: list[tuple[str, str]],
        target_vocab_size: int = 16000,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.merges = merges
        self.target_vocab_size = target_vocab_size
        self.pad_id = token_to_id["<pad>"]
        self.unk_id = token_to_id["<unk>"]
        self.bos_id = token_to_id["<bos>"]
        self.eos_id = token_to_id["<eos>"]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 16000,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        chars = sorted({char for text in texts for char in text})
        token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS + chars)}
        corpus = [list(text) + ["<eos>"] for text in texts if text]
        merges: list[tuple[str, str]] = []

        while len(token_to_id) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for tokens in corpus:
                pair_counts.update(zip(tokens, tokens[1:]))
            if not pair_counts:
                break

            (left, right), count = pair_counts.most_common(1)[0]
            if count < min_pair_freq:
                break

            merged = left + right
            if merged in token_to_id:
                break

            token_to_id[merged] = len(token_to_id)
            merges.append((left, right))
            for index, tokens in enumerate(corpus):
                corpus[index] = _merge_pair(tokens, left, right, merged)

        while len(token_to_id) < vocab_size:
            token_to_id[f"<unused_{len(token_to_id)}>"] = len(token_to_id)

        return cls(token_to_id, merges, vocab_size)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = list(text)
        for left, right in self.merges:
            tokens = _merge_pair(tokens, left, right, left + right)

        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token in SPECIAL_TOKENS or token.startswith("<unused_"):
                continue
            parts.append(token)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        payload = {
            "target_vocab_size": self.target_vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_id={str(key): int(value) for key, value in payload["token_to_id"].items()},
            merges=[tuple(pair) for pair in payload["merges"]],
            target_vocab_size=int(payload.get("target_vocab_size", 16000)),
        )


def _merge_pair(tokens: list[str], left: str, right: str, merged: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
            result.append(merged)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return result


def corpus_stats(tokenizer: BPETokenizer, texts: list[tuple[Path, str]]) -> dict[str, int | float]:
    encoded_lengths = [len(tokenizer.encode(text, add_bos=True, add_eos=True)) for _, text in texts]
    total_tokens = sum(encoded_lengths)
    longest_context = max(encoded_lengths, default=0)
    token_memory_mb = total_tokens * 8 / (1024 * 1024)
    return {
        "files": len(texts),
        "chars": sum(len(text) for _, text in texts),
        "tokens": total_tokens,
        "longest_context": longest_context,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    return tokenizer
