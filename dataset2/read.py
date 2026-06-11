from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONTEXT_LENGTH = 10_000
TOKEN_ARRAY_TYPE = "I"  # uint32, written by dataset2/preprocess.py
INDEX_ARRAY_TYPE = "Q"  # uint64, written by dataset2/preprocess.py


def token_preprocess_dir(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    base_dir: str | Path = BASE_DIR,
) -> Path:
    return Path(base_dir) / f"preprocess{context_length}"


@dataclass(frozen=True)
class TokenShard:
    name: str
    bin_path: Path
    idx_path: Path
    offsets: array
    token_count: int


@dataclass(frozen=True)
class TokenSampleRef:
    shard_index: int
    sample_index: int
    start_token: int
    end_token: int

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token


def _load_offsets(idx_path: Path) -> array:
    offsets = array(INDEX_ARRAY_TYPE)
    with idx_path.open("rb") as file:
        offsets.fromfile(file, idx_path.stat().st_size // offsets.itemsize)
    return offsets


def _load_shard(idx_path: Path) -> TokenShard:
    bin_path = idx_path.with_suffix(".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"missing bin file for {idx_path}: {bin_path}")

    return TokenShard(
        name=idx_path.stem,
        bin_path=bin_path,
        idx_path=idx_path,
        offsets=_load_offsets(idx_path),
        token_count=bin_path.stat().st_size // array(TOKEN_ARRAY_TYPE).itemsize,
    )


def load_shards(preprocess_dir: str | Path) -> list[TokenShard]:
    root = Path(preprocess_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    shards = [_load_shard(idx_path) for idx_path in sorted(root.glob("*.idx"))]
    if not shards:
        raise FileNotFoundError(f"no .idx files found in {root}")
    return shards


def merge_sample_refs(shards: Sequence[TokenShard]) -> list[TokenSampleRef]:
    refs: list[TokenSampleRef] = []
    for shard_index, shard in enumerate(shards):
        for sample_index, start in enumerate(shard.offsets):
            end = shard.offsets[sample_index + 1] if sample_index + 1 < len(shard.offsets) else shard.token_count
            start_token = int(start)
            end_token = int(end)
            if end_token <= start_token:
                continue
            refs.append(
                TokenSampleRef(
                    shard_index=shard_index,
                    sample_index=sample_index,
                    start_token=start_token,
                    end_token=end_token,
                )
            )
    return refs


def read_tokens(shard: TokenShard, ref: TokenSampleRef) -> list[int]:
    tokens = array(TOKEN_ARRAY_TYPE)
    with shard.bin_path.open("rb") as file:
        file.seek(ref.start_token * tokens.itemsize)
        tokens.fromfile(file, ref.token_count)
    return list(tokens)


class Dataset:
    def __init__(
        self,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        preprocess_dir: str | Path | None = None,
    ) -> None:
        self.context_length = context_length
        self.preprocess_dir = Path(preprocess_dir) if preprocess_dir is not None else token_preprocess_dir(context_length)
        self.shards = load_shards(self.preprocess_dir)
        self.samples = merge_sample_refs(self.shards)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Path, list[int]]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)

        ref = self.samples[index]
        shard = self.shards[ref.shard_index]
        sample_path = Path(f"{shard.name}#{ref.sample_index}")
        return sample_path, read_tokens(shard, ref)

    def __iter__(self) -> Iterator[tuple[Path, list[int]]]:
        for index in range(len(self)):
            yield self[index]


if __name__ == "__main__":
    import sys

    ROOT_DIR = BASE_DIR.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from read import BPETokenizer

    tokenizer = BPETokenizer.load(ROOT_DIR / "params" / "tokenizer2.json")
    dataset = Dataset()
    total_tokens = 0
    print(
        f"dataset2 token dataset: preprocess_dir={dataset.preprocess_dir} "
        f"shards={len(dataset.shards)} samples={len(dataset)} "
        f"tokenizer_vocab={tokenizer.vocab_size}"
    )

    for index, (path, tokens) in enumerate(dataset):
        if not tokens:
            raise RuntimeError(f"empty sample at index={index} path={path}")
        total_tokens += len(tokens)
        if index == 0 or (index + 1) % 10_000 == 0 or index + 1 == len(dataset):
            preview = " ".join(tokenizer.decode(tokens[:200]).split())[:160]
            print(
                f"index={index} path={path} tokens={len(tokens)} "
                f"min={min(tokens)} max={max(tokens)} total_tokens={total_tokens} "
                f"preview={preview}",
                flush=True,
            )

    print(f"done samples={len(dataset)} total_tokens={total_tokens}")
