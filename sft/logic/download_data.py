from __future__ import annotations

import csv
import importlib
import json
import os
import random
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

FOLIO_URLS = {
    "train": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl",
    "validation": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl",
}
DETERMLR_TREE_URL = "https://api.github.com/repos/XiaoMi/DetermLR/git/trees/main?recursive=1"
DETERMLR_RAW_PREFIX = "https://raw.githubusercontent.com/XiaoMi/DetermLR/main/"
PFOLIO_URLS = {
    "folio": "https://huggingface.co/datasets/yale-nlp/P-FOLIO/resolve/main/FOLIO.csv",
    "pfolio": "https://huggingface.co/datasets/yale-nlp/P-FOLIO/resolve/main/P-FOLIO.csv",
}
LOGIC_INFERENCE_FILES = [
    "example_generation.py",
    "inference_methods.py",
    "inference_problems.py",
    "logic_inference_lib.py",
    "rules.py",
    "splits.py",
]
LOGIC_INFERENCE_RAW_PREFIX = (
    "https://raw.githubusercontent.com/google-research/google-research/master/"
    "logic_inference_dataset/"
)


@dataclass(frozen=True)
class Sample:
    id: str
    source: str
    split: str
    task: str
    prompt: str
    completion: str
    metadata: dict[str, Any]


def download(url: str, path: Path, token: str | None = None) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        print(f"skip {url}: HTTP {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        print(f"skip {url}: {exc.reason}")
        return False
    return True


def read_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return [payload]


def clean(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(clean(item) for item in text if clean(item))
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").split())


def option_text(options: Any) -> str:
    if not options:
        return ""
    return "\n".join(clean(option) for option in options)


def build_prompt(context: str, question: str, options: Any = None) -> str:
    parts = []
    if context:
        parts.append(f"Context:\n{context}")
    if question:
        parts.append(f"Question:\n{question}")
    options_text = option_text(options)
    if options_text:
        parts.append(f"Options:\n{options_text}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def normalize_folio(record: dict[str, Any], source_id: str, split: str, index: int) -> Sample:
    premises = clean(record.get("premises") or record.get("context"))
    conclusion = clean(record.get("conclusion") or record.get("question"))
    label = clean(record.get("label") or record.get("answer"))
    question = (
        f"Does the conclusion follow from the premises? Conclusion: {conclusion}"
        if "conclusion" in record
        else conclusion
    )
    return Sample(
        id=f"{source_id}-{split}-{index}",
        source=source_id,
        split=split,
        task="first_order_logic_entailment",
        prompt=build_prompt(premises, question, ["True", "False", "Uncertain"]),
        completion=label,
        metadata=record,
    )


def normalize_choice_record(record: dict[str, Any], source: str, split: str, index: int) -> Sample | None:
    context = clean(record.get("context") or record.get("text") or record.get("premises"))
    question = clean(record.get("question") or record.get("query") or record.get("conclusion"))
    answer = clean(record.get("answer") or record.get("label") or record.get("target"))
    if not question or not answer:
        return None
    return Sample(
        id=clean(record.get("id") or record.get("example_id") or f"{source}-{split}-{index}"),
        source=source,
        split=split,
        task="logical_reasoning_choice",
        prompt=build_prompt(context, question, record.get("options") or record.get("choices")),
        completion=answer,
        metadata=record,
    )


def normalize_csv_row(row: dict[str, Any], source: str, index: int) -> Sample | None:
    context = clean(
        row.get("premises")
        or row.get("context")
        or row.get("story")
        or row.get("input")
    )
    question = clean(row.get("conclusion") or row.get("question") or row.get("query"))
    proof = clean(row.get("proof") or row.get("reasoning") or row.get("reasoning_chain") or row.get("chain"))
    label = clean(row.get("label") or row.get("answer") or row.get("target"))
    completion = "\n".join(part for part in [proof, label] if part)
    if not question or not completion:
        return None
    return Sample(
        id=clean(row.get("id") or row.get("example_id") or f"{source}-{index}"),
        source=source,
        split=clean(row.get("split") or "train"),
        task="proof_generation" if proof else "logical_reasoning_choice",
        prompt=build_prompt(context, question),
        completion=completion,
        metadata=row,
    )


def collect_folio() -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    files: list[str] = []
    for split, url in FOLIO_URLS.items():
        path = RAW_DIR / "folio" / f"{split}.jsonl"
        if not download(url, path):
            continue
        files.append(str(path.relative_to(DATA_DIR)))
        records = read_json_or_jsonl(path)
        samples.extend(normalize_folio(record, "folio", split, i) for i, record in enumerate(records))
    return samples, {"files": files, "samples": len(samples), "urls": FOLIO_URLS}


def collect_determlr() -> tuple[list[Sample], dict[str, Any]]:
    tree_path = RAW_DIR / "determlr" / "tree.json"
    if not download(DETERMLR_TREE_URL, tree_path):
        return [], {"samples": 0, "files": [], "url": DETERMLR_TREE_URL}
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    data_paths = [
        item["path"]
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").startswith(("FOLIO/data/", "LogiQA/data/", "LogicalDeduction/data/", "ProofWriter/data/"))
        and Path(item["path"]).suffix in {".json", ".jsonl"}
        and "visited_nodes" not in item["path"]
    ]

    samples: list[Sample] = []
    files: list[str] = []
    for data_path in data_paths:
        local_path = RAW_DIR / "determlr" / data_path
        if not download(DETERMLR_RAW_PREFIX + data_path, local_path):
            continue
        files.append(str(local_path.relative_to(DATA_DIR)))
        source = "determlr_" + data_path.split("/", 1)[0].lower()
        split = "dev" if "dev" in data_path.lower() else "train"
        for index, record in enumerate(read_json_or_jsonl(local_path)):
            if not isinstance(record, dict):
                continue
            sample = normalize_choice_record(record, source, split, index)
            if sample is not None:
                samples.append(sample)
    return samples, {"files": files, "samples": len(samples), "url": DETERMLR_TREE_URL}


def collect_pfolio() -> tuple[list[Sample], dict[str, Any]]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    samples: list[Sample] = []
    files: list[str] = []
    skipped: list[str] = []
    for name, url in PFOLIO_URLS.items():
        path = RAW_DIR / "pfolio" / f"{name}.csv"
        if not download(url, path, token=token):
            skipped.append(name)
            continue
        files.append(str(path.relative_to(DATA_DIR)))
        with path.open(newline="", encoding="utf-8") as file:
            for index, row in enumerate(csv.DictReader(file)):
                sample = normalize_csv_row(row, f"pfolio_{name}", index)
                if sample is not None:
                    samples.append(sample)
    return samples, {
        "files": files,
        "samples": len(samples),
        "urls": PFOLIO_URLS,
        "skipped": skipped,
        "note": "P-FOLIO is gated on Hugging Face; set HF_TOKEN to download it." if skipped else "",
    }


def collect_logic_inference() -> tuple[list[Sample], dict[str, Any]]:
    module_dir = RAW_DIR / "logic_inference_dataset"
    for file_name in LOGIC_INFERENCE_FILES:
        download(LOGIC_INFERENCE_RAW_PREFIX + file_name, module_dir / file_name)

    sys.path.insert(0, str(module_dir))
    try:
        rules = importlib.import_module("rules")
        splits = importlib.import_module("splits")
        random.seed(0)
        rules.precompute_rules()
        train, test = splits.generate_training_and_test_sets_iid(
            50,
            20,
            600,
            0.9,
            answer_at_the_end=True,
        )
    finally:
        if sys.path and sys.path[0] == str(module_dir):
            sys.path.pop(0)

    samples = []
    for split, examples in [("train", train), ("test", test)]:
        for index, example in enumerate(examples):
            samples.append(
                Sample(
                    id=f"logic_inference-{split}-{index}",
                    source="logic_inference",
                    split=split,
                    task="symbolic_logic_inference",
                    prompt=build_prompt("", clean(example.inputs)),
                    completion=clean(example.targets),
                    metadata={"inputs": example.inputs, "targets": example.targets},
                )
            )
    raw_path = RAW_DIR / "logic_inference_dataset" / "generated_sample.jsonl"
    write_jsonl(raw_path, samples)
    return samples, {
        "files": [str(raw_path.relative_to(DATA_DIR))],
        "samples": len(samples),
        "url": "https://github.com/google-research/google-research/tree/master/logic_inference_dataset",
    }


def dedupe(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[Sample] = []
    for sample in samples:
        key = (sample.source, sample.prompt, sample.completion)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def write_jsonl(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_samples: list[Sample] = []
    sources: dict[str, Any] = {}
    for name, collector in [
        ("folio", collect_folio),
        ("determlr", collect_determlr),
        ("pfolio", collect_pfolio),
        ("logic_inference", collect_logic_inference),
    ]:
        samples, info = collector()
        all_samples.extend(samples)
        sources[name] = info
        print(f"{name}: {len(samples)} samples")

    unique_samples = dedupe(all_samples)
    write_jsonl(SAMPLES_PATH, unique_samples)
    manifest = {
        "samples_path": str(SAMPLES_PATH.relative_to(DATA_DIR)),
        "total_samples": len(unique_samples),
        "deduped_from": len(all_samples),
        "sources": sources,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {SAMPLES_PATH} samples={len(unique_samples)}")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
