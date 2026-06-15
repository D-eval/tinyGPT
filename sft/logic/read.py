from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_SAMPLES_PATH = DEFAULT_DATA_DIR / "samples.jsonl"
REASONING_KEYS = (
    "proof",
    "proofs",
    "reasoning",
    "reasoning_chain",
    "reasoning chains",
    "rationale",
    "explanation",
    "solution",
    "chain",
    "cot",
    "derivation",
)
FORMAL_KEYS = ("premises-FOL", "conclusion-FOL")


@dataclass(frozen=True)
class LogicSample:
    id: str
    source: str
    split: str
    task: str
    prompt: str
    completion: str
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        return format_dialogue(self.prompt, self.agent_reply)

    @property
    def agent_reply(self) -> str:
        return build_agent_reply(self.completion, self.metadata)

    def as_prompt_completion(self) -> tuple[str, str]:
        return format_user(self.prompt) + "\n<agent>:", self.agent_reply


def format_user(prompt: str) -> str:
    return f"<usr>: {prompt.rstrip()}"


def format_dialogue(prompt: str, completion: str) -> str:
    return f"{format_user(prompt)}\n<agent>: {completion.rstrip()}"


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(stringify(item) for item in value if stringify(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def metadata_value(metadata: dict[str, Any], wanted_keys: tuple[str, ...]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    lowered = {key.lower(): key for key in metadata}
    for wanted_key in wanted_keys:
        key = lowered.get(wanted_key.lower())
        if key is None:
            continue
        value = stringify(metadata.get(key))
        if value:
            values.append((key, value))
    return values


def contains_text(haystack: str, needle: str) -> bool:
    return " ".join(needle.split()) in " ".join(haystack.split())


def build_agent_reply(completion: str, metadata: dict[str, Any]) -> str:
    completion = completion.strip()
    parts: list[str] = []

    reasoning_values = [
        value
        for _, value in metadata_value(metadata, REASONING_KEYS)
        if value and not contains_text(completion, value)
    ]
    if reasoning_values:
        parts.append("Reasoning:\n" + "\n".join(reasoning_values))

    formal_values = metadata_value(metadata, FORMAL_KEYS)
    if formal_values:
        formal_lines = [f"{key}:\n{value}" for key, value in formal_values]
        formal_text = "Formal logic:\n" + "\n".join(formal_lines)
        if not contains_text(completion, formal_text):
            parts.append(formal_text)

    answer = expand_answer(completion, metadata)
    if answer:
        label = "Answer" if parts else ""
        parts.append(f"{label}:\n{answer}" if label else answer)

    return "\n\n".join(parts).strip()


def expand_answer(answer: str, metadata: dict[str, Any]) -> str:
    options = metadata.get("options") or metadata.get("choices")
    if not isinstance(options, list) or not answer:
        return answer

    selected = selected_option(answer, options)
    if not selected or contains_text(answer, selected):
        return answer
    return f"{answer}\nSelected option: {selected}"


def selected_option(answer: str, options: list[Any]) -> str:
    answer = answer.strip()
    normalized = answer.rstrip(".。)").upper()
    if len(normalized) == 1 and "A" <= normalized <= "Z":
        index = ord(normalized) - ord("A")
        if 0 <= index < len(options):
            return stringify(options[index])
    if normalized.isdigit():
        number = int(normalized)
        for index in (number, number - 1):
            if 0 <= index < len(options):
                return stringify(options[index])
    for option in options:
        text = stringify(option)
        if text.upper().startswith(normalized):
            return text
    return ""


def load_samples(path: str | Path = DEFAULT_SAMPLES_PATH) -> list[LogicSample]:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"{sample_path} does not exist. Run `python sft/logic/download_data.py` first."
        )

    samples: list[LogicSample] = []
    with sample_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                LogicSample(
                    id=str(record["id"]),
                    source=str(record["source"]),
                    split=str(record["split"]),
                    task=str(record["task"]),
                    prompt=str(record["prompt"]),
                    completion=str(record["completion"]),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
    return samples


class Dataset:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        split: str | None = None,
        source: str | None = None,
        task: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.samples_path = self.data_dir / "samples.jsonl"
        samples = load_samples(self.samples_path)
        if split is not None:
            samples = [sample for sample in samples if sample.split == split]
        if source is not None:
            samples = [sample for sample in samples if sample.source == source]
        if task is not None:
            samples = [sample for sample in samples if sample.task == task]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        return self.samples[index].text

    def __iter__(self) -> Iterator[str]:
        for sample in self.samples:
            yield sample.text

    def get_sample(self, index: int) -> LogicSample:
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
    print(f"logic dataset: samples={len(dataset)} sources={counts}")
    if dataset:
        first = dataset.get_sample(0)
        preview = " ".join(dataset[0].split())[:240]
        # print(f"first id={first.id} source={first.source} task={first.task} preview={preview}")
        print(first)