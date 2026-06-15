from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
NO_NAMES_DATA_DIR = BASE_DIR / "data_no_names"
DEFAULT_SAMPLES_PATH = NO_NAMES_DATA_DIR / "samples.jsonl"


@dataclass(frozen=True)
class EmotionSample:
    id: str
    source: str
    split: str
    system: str
    turns: list[dict[str, Any]]
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        return format_dialogue(self.system, self.turns)

    def as_prompt_completion(self) -> tuple[str, str]:
        prompt_turns, completion_turn = split_prompt_completion(self.turns)
        completion = str(completion_turn.get("text", "")).rstrip()
        return format_dialogue(self.system, prompt_turns) + "\n<agent>:", f"<bos>{completion}<eos>"


def normalize_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_turns = record.get("turns")
    if isinstance(raw_turns, list):
        turns = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if role in {"usr", "agent"} and text:
                turns.append({"role": role, "text": text, "metadata": dict(item.get("metadata") or {})})
        return turns

    user = str(record.get("user", "")).strip()
    agent = str(record.get("agent", "")).strip()
    turns = []
    if user:
        turns.append({"role": "usr", "text": user, "metadata": {}})
    if agent:
        turns.append({"role": "agent", "text": agent, "metadata": {}})
    return turns


def format_dialogue(system: str, turns: list[dict[str, Any]]) -> str:
    lines = [f"<system>: {system.rstrip()}"]
    for item in turns:
        role = item["role"]
        text = str(item["text"]).rstrip()
        if role == "agent":
            lines.append(f"<agent>: <bos>{text}<eos>")
        else:
            lines.append(f"<{role}>: {text}")
    return "\n".join(lines)


def split_prompt_completion(turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for index in range(len(turns) - 1, -1, -1):
        if turns[index].get("role") == "agent":
            return turns[:index], turns[index]
    return turns, {"role": "agent", "text": "", "metadata": {}}


def load_samples(path: str | Path = DEFAULT_SAMPLES_PATH) -> list[EmotionSample]:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"{sample_path} does not exist. Run `python sft/emotion/download_data.py` first."
        )

    samples: list[EmotionSample] = []
    with sample_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                EmotionSample(
                    id=str(record["id"]),
                    source=str(record["source"]),
                    split=str(record["split"]),
                    system=str(record["system"]),
                    turns=normalize_turns(record),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
    return samples


def split_matches(sample: EmotionSample, split: str) -> bool:
    if sample.split == split:
        return True
    if split == "train":
        return sample.split.startswith("train") or (sample.source == "livechat" and sample.split == "subset")
    if split in {"valid", "validation", "dev"}:
        return sample.split.startswith("valid") or sample.split in {"validation", "dev"}
    if split == "test":
        return sample.split.startswith("test")
    return False


class Dataset:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str | None = None,
        source: str | None = None,
        no_names: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else (NO_NAMES_DATA_DIR if no_names else DEFAULT_DATA_DIR)
        self.samples_path = self.data_dir / "samples.jsonl"
        samples = load_samples(self.samples_path)
        if split is not None:
            samples = [sample for sample in samples if split_matches(sample, split)]
        if source is not None:
            samples = [sample for sample in samples if sample.source == source]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        return self.samples[index].text

    def __iter__(self) -> Iterator[str]:
        for sample in self.samples:
            yield sample.text

    def get_sample(self, index: int) -> EmotionSample:
        return self.samples[index]

    def texts(self) -> list[tuple[Path, str]]:
        return [(Path(sample.id), sample.text) for sample in self.samples]

    def prompt_completions(self) -> list[tuple[str, str]]:
        return [sample.as_prompt_completion() for sample in self.samples]


if __name__ == "__main__":
    dataset = Dataset()
    counts: dict[str, int] = {}
    for sample in dataset.samples:
        counts[sample.source] = counts.get(sample.source, 0) + 1
    print(f"emotion dataset: samples={len(dataset)} sources={counts}")
    if dataset:
        print(dataset[0])
