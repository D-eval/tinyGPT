from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

DAILYDIALOG_URL = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
LIVECHAT_FILES = {
    "subset": "Dataset/Subset/subset.json",
    "basic_profile": "Dataset/Subset/basic_profile.json",
    "text_profile": "Dataset/Subset/text_profile.json",
}
LIVECHAT_RAW_PREFIX = "https://raw.githubusercontent.com/gaojingsheng/LiveChat/master/"
PERSONACHAT_KAGGLE_DATASET = "atharvjairath/personachat"
PERSONACHAT_PARLAI_URL = "http://parl.ai/downloads/personachat/personachat.tgz"
PERSONACHAT_HF_FILES = {
    "train": "https://huggingface.co/datasets/AlekseyKorshuk/persona-chat/resolve/main/data/train-00000-of-00001.parquet",
    "validation": "https://huggingface.co/datasets/AlekseyKorshuk/persona-chat/resolve/main/data/validation-00000-of-00001.parquet",
}
PERSONACHAT_DIALOGUE_FILES = {
    "train_both_original.txt",
    "valid_both_original.txt",
    "test_both_original.txt",
}
NATURALCONV_URL = "https://ailab.tencent.com/ailab/nlp/dialogue/datasets/NaturalConv_Release_20210318.zip"
CPED_RAW_PREFIX = "https://raw.githubusercontent.com/scutcyr/CPED/main/data/CPED/"
CPED_FILES = {
    "train": "train_split.csv",
    "valid": "valid_split.csv",
    "test": "test_split.csv",
    "speakers": "speakers.txt",
}
_CPED_SPEAKER_NAMES: list[str] | None = None

DAILYDIALOG_EMOTIONS = {
    0: "no emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


@dataclass(frozen=True)
class Sample:
    id: str
    source: str
    split: str
    system: str
    turns: list[dict[str, Any]]
    metadata: dict[str, Any]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(part for part in (clean(item) for item in value) if part)
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tinyGPT-data-loader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        print(f"skip {url}: HTTP {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        print(f"skip {url}: {exc.reason}")
        return False
    return True


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(root: Path, name: str) -> Path | None:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    return matches[0] if matches else None


def turn(role: str, text: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"role": role, "text": clean(text), "metadata": metadata or {}}


def valid_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in turns if clean(item.get("text")) and item.get("role") in {"usr", "agent"}]


def cped_speaker_names() -> list[str]:
    global _CPED_SPEAKER_NAMES
    if _CPED_SPEAKER_NAMES is not None:
        return _CPED_SPEAKER_NAMES

    path = RAW_DIR / "cped" / CPED_FILES["speakers"]
    if not path.exists():
        _CPED_SPEAKER_NAMES = []
        return _CPED_SPEAKER_NAMES

    names = [clean(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _CPED_SPEAKER_NAMES = sorted((name for name in names if name), key=len, reverse=True)
    return _CPED_SPEAKER_NAMES


def remove_cped_speaker_names(text: Any) -> str:
    result = clean(text)
    for name in cped_speaker_names():
        result = result.replace(name, "")
    return clean(result)


def collect_dailydialog() -> tuple[list[Sample], dict[str, Any]]:
    archive_path = RAW_DIR / "dailydialog" / "data.zip"
    if not download(DAILYDIALOG_URL, archive_path):
        return [], {"samples": 0, "files": [], "url": DAILYDIALOG_URL}

    extract_dir = RAW_DIR / "dailydialog" / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    samples: list[Sample] = []
    dialogues_json = first_existing(extract_dir, "dialogues.json")
    if dialogues_json is not None:
        for dialogue_index, dialogue in enumerate(read_json(dialogues_json)):
            split = clean(dialogue.get("data_split") or "train")
            raw_turns = dialogue.get("turns") or []
            dialogue_turns = []
            emotions = []
            for turn_index, raw_turn in enumerate(raw_turns):
                if not isinstance(raw_turn, dict):
                    continue
                role = "agent" if raw_turn.get("speaker") == "system" else "usr"
                text = clean(raw_turn.get("utterance"))
                if not text:
                    continue
                emotion = clean(raw_turn.get("emotion") or "unknown")
                emotions.append(emotion)
                dialogue_turns.append(
                    turn(
                        role,
                        text,
                        {
                            "turn_id": turn_index,
                            "speaker": raw_turn.get("speaker"),
                            "emotion": emotion,
                            "dialogue_acts": raw_turn.get("dialogue_acts", {}),
                        },
                    )
                )
            dialogue_turns = valid_turns(dialogue_turns)
            if len(dialogue_turns) < 2:
                continue
            samples.append(
                Sample(
                    id=f"dailydialog-{split}-{dialogue_index}",
                    source="dailydialog",
                    split=split,
                    system="You are a helpful dialogue agent. Respond naturally across the full conversation.",
                    turns=dialogue_turns,
                    metadata={
                        "dialogue_id": dialogue.get("dialogue_id", dialogue_index),
                        "domains": dialogue.get("domains", []),
                        "emotions": emotions,
                    },
                )
            )
        return samples, {
            "samples": len(samples),
            "files": [str(archive_path.relative_to(DATA_DIR)), str(dialogues_json.relative_to(DATA_DIR))],
            "url": DAILYDIALOG_URL,
            "paper": "https://arxiv.org/abs/1710.03957",
        }

    for split in ("train", "validation", "test"):
        split_dir = extract_dir / split
        text_path = first_existing(split_dir, "dialogues_text.txt")
        emotion_path = first_existing(split_dir, "dialogues_emotion.txt")
        act_path = first_existing(split_dir, "dialogues_act.txt")
        if text_path is None:
            continue

        dialogues = text_path.read_text(encoding="utf-8").splitlines()
        emotions = (
            emotion_path.read_text(encoding="utf-8").splitlines()
            if emotion_path and emotion_path.exists()
            else ["" for _ in dialogues]
        )
        acts = (
            act_path.read_text(encoding="utf-8").splitlines()
            if act_path and act_path.exists()
            else ["" for _ in dialogues]
        )

        for dialogue_index, line in enumerate(dialogues):
            utterances = [clean(part) for part in line.split("__eou__") if clean(part)]
            emotion_ids = [int(item) for item in emotions[dialogue_index].split() if item.isdigit()]
            act_ids = [int(item) for item in acts[dialogue_index].split() if item.isdigit()]
            dialogue_turns = []
            for turn_index, utterance in enumerate(utterances):
                emotion_id = emotion_ids[turn_index] if turn_index < len(emotion_ids) else None
                act_id = act_ids[turn_index] if turn_index < len(act_ids) else None
                dialogue_turns.append(
                    turn(
                        "usr" if turn_index % 2 == 0 else "agent",
                        utterance,
                        {
                            "turn_id": turn_index,
                            "emotion_id": emotion_id,
                            "emotion": DAILYDIALOG_EMOTIONS.get(emotion_id, "unknown"),
                            "act_id": act_id,
                        },
                    )
                )
            dialogue_turns = valid_turns(dialogue_turns)
            if len(dialogue_turns) >= 2:
                samples.append(
                    Sample(
                        id=f"dailydialog-{split}-{dialogue_index}",
                        source="dailydialog",
                        split=split,
                        system="You are a helpful dialogue agent. Respond naturally across the full conversation.",
                        turns=dialogue_turns,
                        metadata={
                            "dialogue_id": dialogue_index,
                        },
                    )
                )

    return samples, {
        "samples": len(samples),
        "files": [str(archive_path.relative_to(DATA_DIR))],
        "url": DAILYDIALOG_URL,
        "paper": "https://arxiv.org/abs/1710.03957",
    }


def find_personachat_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".json", ".jsonl", ".txt"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def download_personachat_with_kagglehub(target_dir: Path) -> tuple[bool, str]:
    if importlib.util.find_spec("kagglehub") is None:
        return False, "kagglehub is not installed"
    try:
        import kagglehub  # type: ignore

        cache_path = Path(kagglehub.dataset_download(PERSONACHAT_KAGGLE_DATASET))
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in find_personachat_files(cache_path):
            rel = path.relative_to(cache_path)
            out = target_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    except Exception as exc:  # pragma: no cover - depends on local Kaggle setup.
        return False, f"kagglehub failed: {exc}"
    return True, "downloaded with kagglehub"


def download_personachat_with_cli(target_dir: Path) -> tuple[bool, str]:
    if shutil.which("kaggle") is None:
        return False, "kaggle CLI is not installed"
    archive_path = target_dir / "personachat.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            PERSONACHAT_KAGGLE_DATASET,
            "-p",
            str(target_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "kaggle CLI failed").strip()
    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    return True, "downloaded with kaggle CLI"


def collect_personachat_from_hf(target_dir: Path) -> tuple[list[Sample], dict[str, Any]]:
    skipped: list[str] = []
    if importlib.util.find_spec("pyarrow") is None:
        return [], {
            "samples": 0,
            "files": [],
            "urls": PERSONACHAT_HF_FILES,
            "skipped": ["pyarrow is not installed; cannot parse Hugging Face parquet files"],
        }

    import pandas as pd

    samples: list[Sample] = []
    files: list[str] = []
    for split, url in PERSONACHAT_HF_FILES.items():
        path = target_dir / f"{split}.parquet"
        if not download(url, path):
            skipped.append(split)
            continue
        files.append(str(path.relative_to(DATA_DIR)))
        frame = pd.read_parquet(path)
        for index, row in frame.iterrows():
            row_dict = row.to_dict()
            sample = normalize_personachat_record(row_dict, split, index)
            if sample is not None:
                samples.append(sample)

    return samples, {"samples": len(samples), "files": files, "urls": PERSONACHAT_HF_FILES, "skipped": skipped}


def normalize_personachat_record(record: dict[str, Any], split: str, index: int) -> Sample | None:
    history = record.get("history") or record.get("dialogue") or record.get("utterances")
    personas = record.get("personality") or record.get("persona") or record.get("personas")
    candidates = record.get("candidates") or []
    if hasattr(history, "tolist"):
        history = history.tolist()
    if hasattr(personas, "tolist"):
        personas = personas.tolist()
    if hasattr(candidates, "tolist"):
        candidates = candidates.tolist()

    utterances = [clean(item) for item in history] if isinstance(history, list) else []
    if len(utterances) < 2:
        return None
    persona_lines = [clean(item) for item in personas] if isinstance(personas, list) else []
    system = "You are a persona-consistent dialogue agent."
    if persona_lines:
        system += "\nPersona:\n" + "\n".join(f"- {line}" for line in persona_lines if line)
    dialogue_turns = [
        turn("usr" if turn_index % 2 == 0 else "agent", utterance, {"turn_id": turn_index})
        for turn_index, utterance in enumerate(utterances)
    ]
    if isinstance(candidates, list) and candidates:
        dialogue_turns.append(turn("agent", candidates[-1], {"turn_id": len(dialogue_turns), "from_candidates": True}))
    dialogue_turns = valid_turns(dialogue_turns)
    if len(dialogue_turns) < 2:
        return None
    return Sample(
        id=f"personachat-{split}-{index}",
        source="personachat",
        split=split,
        system=system,
        turns=dialogue_turns,
        metadata={"persona": persona_lines},
    )


def collect_personachat_from_files(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in find_personachat_files(root):
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() == ".txt":
            if path.name not in PERSONACHAT_DIALOGUE_FILES:
                continue
            samples.extend(parse_parlai_personachat(path))
        if path.suffix.lower() == ".jsonl":
            for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = normalize_personachat_record(record, path.stem, index)
                if sample is not None:
                    samples.append(sample)
        elif path.suffix.lower() == ".json":
            payload = read_json(path)
            records = payload if isinstance(payload, list) else payload.get("train", []) if isinstance(payload, dict) else []
            for index, record in enumerate(records):
                if isinstance(record, dict):
                    sample = normalize_personachat_record(record, path.stem, index)
                    if sample is not None:
                        samples.append(sample)
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8", errors="ignore") as file:
                for index, row in enumerate(csv.DictReader(file)):
                    sample = normalize_personachat_record(dict(row), path.stem, index)
                    if sample is not None:
                        samples.append(sample)
    return samples


def parse_parlai_personachat(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    personas: list[str] = []
    dialogue_turns: list[dict[str, Any]] = []
    split = path.stem
    conversation_index = 0

    def flush() -> None:
        nonlocal conversation_index, dialogue_turns
        dialogue_turns = valid_turns(dialogue_turns)
        if len(dialogue_turns) < 2:
            dialogue_turns = []
            return
        system = "You are a persona-consistent dialogue agent."
        if personas:
            system += "\nPersona:\n" + "\n".join(f"- {persona}" for persona in personas if persona)
        samples.append(
            Sample(
                id=f"personachat-{split}-{conversation_index}",
                source="personachat",
                split=split,
                system=system,
                turns=list(dialogue_turns),
                metadata={"persona": list(personas)},
            )
        )
        conversation_index += 1
        dialogue_turns = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = line.strip()
        if not line:
            continue
        if " " not in line:
            continue
        turn_no, payload = line.split(" ", 1)
        if turn_no == "1":
            flush()
            personas = []
        if payload.startswith("your persona:"):
            personas.append(clean(payload.removeprefix("your persona:")))
            continue

        fields = payload.split("\t")
        if len(fields) < 2:
            continue
        user, agent = clean(fields[0]), clean(fields[1])
        if not user or not agent:
            continue
        dialogue_turns.append(turn("usr", user, {"line_number": line_number}))
        dialogue_turns.append(turn("agent", agent, {"line_number": line_number}))
    flush()
    return samples


def download_personachat_from_parlai(target_dir: Path) -> tuple[bool, str]:
    archive_path = target_dir / "personachat.tgz"
    if not download(PERSONACHAT_PARLAI_URL, archive_path):
        return False, "ParlAI tarball download failed"
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
            archive.extractall(target_dir, **kwargs)
    except tarfile.TarError as exc:
        return False, f"ParlAI tarball extraction failed: {exc}"
    return True, "downloaded from official ParlAI tarball"


def collect_personachat() -> tuple[list[Sample], dict[str, Any]]:
    raw_dir = RAW_DIR / "personachat"
    attempts = []
    for downloader in (
        download_personachat_with_kagglehub,
        download_personachat_with_cli,
        download_personachat_from_parlai,
    ):
        ok, message = downloader(raw_dir)
        attempts.append(message)
        if ok:
            samples = collect_personachat_from_files(raw_dir)
            if samples:
                return samples, {
                    "samples": len(samples),
                    "files": [str(path.relative_to(DATA_DIR)) for path in find_personachat_files(raw_dir)],
                    "kaggle": f"https://www.kaggle.com/datasets/{PERSONACHAT_KAGGLE_DATASET}",
                    "parlai": PERSONACHAT_PARLAI_URL,
                    "attempts": attempts,
                }

    samples, info = collect_personachat_from_hf(raw_dir / "hf")
    info["kaggle"] = f"https://www.kaggle.com/datasets/{PERSONACHAT_KAGGLE_DATASET}"
    info["parlai"] = PERSONACHAT_PARLAI_URL
    info["attempts"] = attempts
    return samples, info


def collect_livechat() -> tuple[list[Sample], dict[str, Any]]:
    paths: dict[str, Path] = {}
    for name, remote_path in LIVECHAT_FILES.items():
        local_path = RAW_DIR / "livechat" / remote_path
        if download(LIVECHAT_RAW_PREFIX + remote_path, local_path):
            paths[name] = local_path

    if "subset" not in paths:
        return [], {"samples": 0, "files": [], "url": "https://github.com/gaojingsheng/LiveChat"}

    subset = read_json(paths["subset"])
    basic_profile = read_json(paths["basic_profile"]) if "basic_profile" in paths else {}
    text_profile = read_json(paths["text_profile"]) if "text_profile" in paths else {}

    samples: list[Sample] = []
    for streamer_id, pairs in subset.items():
        profile = basic_profile.get(streamer_id, {})
        profile_text = text_profile.get(streamer_id, [])
        system_parts = ["You are a Chinese live-streaming dialogue agent. Reply as the streamer."]
        if profile:
            system_parts.append("Basic profile: " + json.dumps(profile, ensure_ascii=False, sort_keys=True))
        if profile_text:
            system_parts.append("Text profile:\n" + "\n".join(clean(item) for item in profile_text[:8]))
        for index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            user, agent = clean(pair[0]), clean(pair[1])
            if not user or not agent:
                continue
            samples.append(
                Sample(
                    id=f"livechat-{streamer_id}-{index}",
                    source="livechat",
                    split="subset",
                    system="\n".join(system_parts),
                    turns=[
                        turn("usr", user, {"turn_id": 0}),
                        turn("agent", agent, {"turn_id": 1}),
                    ],
                    metadata={"streamer_id": streamer_id, "profile": profile},
                )
            )

    return samples, {
        "samples": len(samples),
        "files": [str(path.relative_to(DATA_DIR)) for path in paths.values()],
        "url": "https://github.com/gaojingsheng/LiveChat",
        "paper": "https://arxiv.org/abs/2306.08401",
    }


def collect_naturalconv() -> tuple[list[Sample], dict[str, Any]]:
    archive_path = RAW_DIR / "naturalconv" / "NaturalConv_Release_20210318.zip"
    if not download(NATURALCONV_URL, archive_path):
        return [], {
            "samples": 0,
            "files": [],
            "url": NATURALCONV_URL,
            "paper": "https://arxiv.org/abs/2103.02548",
        }

    extract_dir = RAW_DIR / "naturalconv" / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    dialogues_path = first_existing(extract_dir, "dialog_release.json")
    if dialogues_path is None:
        return [], {
            "samples": 0,
            "files": [str(archive_path.relative_to(DATA_DIR))],
            "url": NATURALCONV_URL,
            "paper": "https://arxiv.org/abs/2103.02548",
            "error": "dialog_release.json not found",
        }

    split_by_id: dict[str, str] = {}
    for split_name, file_name in (("train", "train.txt"), ("dev", "dev.txt"), ("test", "test.txt")):
        split_path = first_existing(extract_dir, file_name)
        if split_path is None:
            continue
        for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            dialog_id = clean(line)
            if dialog_id:
                split_by_id[dialog_id] = split_name

    samples: list[Sample] = []
    for dialogue_index, dialogue in enumerate(read_json(dialogues_path)):
        dialog_id = clean(dialogue.get("dialog_id") or dialogue_index)
        split = split_by_id.get(dialog_id, "train")
        content = dialogue.get("content") or []
        utterances = [clean(item) for item in content if clean(item)]
        dialogue_turns = [
            turn("usr" if turn_index % 2 == 0 else "agent", utterance, {"turn_id": turn_index})
            for turn_index, utterance in enumerate(utterances)
        ]
        dialogue_turns = valid_turns(dialogue_turns)
        if len(dialogue_turns) >= 2:
            samples.append(
                Sample(
                    id=f"naturalconv-{dialog_id}",
                    source="naturalconv",
                    split=split,
                    system=(
                        "You are a Chinese multi-turn topic-driven dialogue agent. "
                        "Reply naturally and keep the topic transition smooth."
                    ),
                    turns=dialogue_turns,
                    metadata={
                        "dialog_id": dialog_id,
                        "document_id": dialogue.get("document_id"),
                    },
                )
            )

    files = [str(archive_path.relative_to(DATA_DIR)), str(dialogues_path.relative_to(DATA_DIR))]
    return samples, {
        "samples": len(samples),
        "files": files,
        "url": NATURALCONV_URL,
        "paper": "https://arxiv.org/abs/2103.02548",
    }


def cped_system(row: dict[str, str], target_speaker: str | None = None, hide_speaker_names: bool = False) -> str:
    scene = clean(row.get("Scene"))
    speaker = clean(target_speaker or row.get("Speaker"))
    traits = {
        "gender": clean(row.get("Gender")),
        "age": clean(row.get("Age")),
        "neuroticism": clean(row.get("Neuroticism")),
        "extraversion": clean(row.get("Extraversion")),
        "openness": clean(row.get("Openness")),
        "agreeableness": clean(row.get("Agreeableness")),
        "conscientiousness": clean(row.get("Conscientiousness")),
    }

    parts = [
        "You are a Chinese personalized and emotional dialogue agent.",
        "Continue the multi-party conversation as the target speaker.",
    ]
    if speaker and not hide_speaker_names:
        parts.append(f"Target speaker: {speaker}.")
    profile = {key: value for key, value in traits.items() if value}
    if profile:
        parts.append("Target speaker profile: " + json.dumps(profile, ensure_ascii=False, sort_keys=True))
    conditions = []
    if scene:
        conditions.append(f"scene={scene}")
    if conditions:
        parts.append("Conversation conditions: " + ", ".join(conditions) + ".")
    return "\n".join(parts)


def cped_turn_text(row: dict[str, str], hide_speaker_names: bool = False) -> str:
    speaker = clean(row.get("Speaker"))
    utterance = remove_cped_speaker_names(row.get("Utterance")) if hide_speaker_names else clean(row.get("Utterance"))
    return utterance if hide_speaker_names or not speaker else f"{speaker}: {utterance}"


def cped_samples_for_dialogue(
    split: str,
    dialogue_id: str,
    rows: list[dict[str, str]],
    hide_speaker_names: bool = False,
) -> list[Sample]:
    samples: list[Sample] = []
    speakers = []
    for row in rows:
        speaker = clean(row.get("Speaker"))
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    for speaker in speakers:
        speaker_rows = [row for row in rows if clean(row.get("Speaker")) == speaker]
        if not speaker_rows:
            continue
        dialogue_turns: list[dict[str, Any]] = []
        usr_buffer: list[str] = []
        usr_meta: list[dict[str, Any]] = []

        def flush_usr() -> None:
            nonlocal usr_buffer, usr_meta
            if not usr_buffer:
                return
            text = "\n".join(usr_buffer)
            if dialogue_turns and dialogue_turns[-1]["role"] == "usr":
                dialogue_turns[-1]["text"] = dialogue_turns[-1]["text"] + "\n" + text
                dialogue_turns[-1]["metadata"]["utterances"].extend(usr_meta)
            else:
                dialogue_turns.append(turn("usr", text, {"utterances": list(usr_meta)}))
            usr_buffer = []
            usr_meta = []

        for turn_index, row in enumerate(rows):
            row_speaker = clean(row.get("Speaker"))
            row_text = cped_turn_text(row, hide_speaker_names)
            row_metadata = {
                "turn_id": turn_index,
                "utterance_id": row.get("Utterance_ID"),
                "speaker": row.get("Speaker"),
                "scene": row.get("Scene"),
                "sentiment": row.get("Sentiment"),
                "emotion": row.get("Emotion"),
                "dialogue_act": row.get("DA"),
            }
            if row_speaker == speaker:
                flush_usr()
                if dialogue_turns and dialogue_turns[-1]["role"] == "agent":
                    dialogue_turns[-1]["text"] = dialogue_turns[-1]["text"] + "\n" + row_text
                    dialogue_turns[-1]["metadata"]["utterances"].append(row_metadata)
                else:
                    dialogue_turns.append(turn("agent", row_text, {"utterances": [row_metadata]}))
            else:
                usr_buffer.append(row_text)
                usr_meta.append(row_metadata)
        flush_usr()

        dialogue_turns = valid_turns(dialogue_turns)
        if len(dialogue_turns) < 2 or not any(item["role"] == "agent" for item in dialogue_turns):
            continue
        if dialogue_turns[0]["role"] == "agent":
            dialogue_turns = dialogue_turns[1:]
        last_agent_index = next(
            (index for index in range(len(dialogue_turns) - 1, -1, -1) if dialogue_turns[index]["role"] == "agent"),
            -1,
        )
        if last_agent_index >= 0:
            dialogue_turns = dialogue_turns[: last_agent_index + 1]
        if len(dialogue_turns) < 2 or not any(item["role"] == "agent" for item in dialogue_turns):
            continue

        samples.append(
            Sample(
                id=f"cped-{split}-{dialogue_id}-{speaker}",
                source="cped",
                split=split,
                system=cped_system(speaker_rows[0], speaker, hide_speaker_names),
                turns=dialogue_turns,
                metadata={
                    "tv_id": speaker_rows[0].get("TV_ID"),
                    "dialogue_id": dialogue_id,
                    "target_speaker": speaker,
                },
            )
        )
    return samples


def collect_cped() -> tuple[list[Sample], dict[str, Any]]:
    raw_dir = RAW_DIR / "cped"
    paths: dict[str, Path] = {}
    for name, file_name in CPED_FILES.items():
        path = raw_dir / file_name
        if download(CPED_RAW_PREFIX + file_name, path):
            paths[name] = path

    samples: list[Sample] = []
    for split in ("train", "valid", "test"):
        path = paths.get(split)
        if path is None:
            continue
        by_dialogue: dict[str, list[dict[str, str]]] = {}
        with path.open(newline="", encoding="utf-8-sig", errors="ignore") as file:
            for row in csv.DictReader(file):
                dialogue_id = clean(row.get("Dialogue_ID"))
                utterance = clean(row.get("Utterance"))
                if not dialogue_id or not utterance:
                    continue
                by_dialogue.setdefault(dialogue_id, []).append(row)

        for dialogue_id, rows in by_dialogue.items():
            rows.sort(key=lambda row: clean(row.get("Utterance_ID")))
            samples.extend(cped_samples_for_dialogue(split, dialogue_id, rows))

    return samples, {
        "samples": len(samples),
        "files": [str(path.relative_to(DATA_DIR)) for path in paths.values()],
        "url": "https://github.com/scutcyr/CPED",
        "paper": "https://arxiv.org/abs/2205.14727",
    }


def dedupe(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    unique: list[Sample] = []
    for sample in samples:
        key = (
            sample.source,
            sample.system,
            tuple((turn["role"], turn["text"]) for turn in sample.turns),
        )
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
        ("dailydialog", collect_dailydialog),
        ("personachat", collect_personachat),
        ("livechat", collect_livechat),
        ("naturalconv", collect_naturalconv),
        ("cped", collect_cped),
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
    sys.exit(main())
