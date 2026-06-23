from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re


DATASET_WBY_SFT_DIR = Path(__file__).resolve().parent / "dataset_wby_sft"
SPEAKER_PATTERN = re.compile(r"`([^`]+)`\s*:")
LEAF_LINK_PATTERN = re.compile(r"\[\[.*?\]\]", flags=re.S)


spec = spec_from_file_location("dataset_wby_sft_myparse", DATASET_WBY_SFT_DIR / "myParse.py")
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load {DATASET_WBY_SFT_DIR / 'myParse.py'}")
module = module_from_spec(spec)
spec.loader.exec_module(module)
choice = module.choice
randomChar = module.randomChar


safe_globals = {
    "__builtins__": {
        "__import__": __import__,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "set": set,
        "dict": dict,
        "range": range,
    },
    "choice": choice,
    "randomChar": randomChar,
}


def normalize_speaker_name(name: str) -> str:
    return name.strip()


def speaker_token(name: str) -> str:
    return f"<{normalize_speaker_name(name)}>"


def replace_speaker_tokens(text: str) -> str:
    return SPEAKER_PATTERN.sub(lambda match: f"{speaker_token(match.group(1))}:", text)


def list_leaf_samples(dataset_dir: str | Path = DATASET_WBY_SFT_DIR) -> list[Path]:
    root = Path(dataset_dir)
    leaves: list[Path] = []
    for sample in sorted(root.glob("*.md")):
        if sample.name == "data.md":
            continue
        content = sample.read_text(encoding="utf-8", errors="ignore")
        if not LEAF_LINK_PATTERN.search(content):
            leaves.append(sample)
    return leaves


def speaker_tokens(dataset_dir: str | Path = DATASET_WBY_SFT_DIR) -> list[str]:
    tokens: set[str] = set()
    for sample in list_leaf_samples(dataset_dir):
        content = sample.read_text(encoding="utf-8", errors="ignore")
        for name in SPEAKER_PATTERN.findall(content):
            tokens.add(speaker_token(name))
    return sorted(tokens)


def getitem(idx: int, dataset_dir: str | Path = DATASET_WBY_SFT_DIR) -> str:
    sample_name = list_leaf_samples(dataset_dir)[idx]
    content = sample_name.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"%%.*?%%\n", "", content, flags=re.S)
    match = re.match(r"```python\n(.*?)\n```", content, flags=re.S)
    if match:
        prefix = match.group(1)
        content = content[match.end() :].lstrip()
        context: dict[str, object] = {}
        exec(prefix, safe_globals, context)
    else:
        context = {}

    try:
        result = content.format(**context)
    except (KeyError, ValueError):
        result = content
    end = result.find("`thatsAll`")
    if end != -1:
        result = result[:end]
    return replace_speaker_tokens(result).strip()


def read_texts(dataset_dir: str | Path = DATASET_WBY_SFT_DIR) -> list[tuple[Path, str]]:
    texts: list[tuple[Path, str]] = []
    samples = list_leaf_samples(dataset_dir)
    for idx, sample in enumerate(samples):
        text = getitem(idx, dataset_dir).strip()
        if text:
            texts.append((sample, text))
    return texts
