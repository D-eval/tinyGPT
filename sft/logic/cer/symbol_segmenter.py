from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent
PARAMS_DIR = BASE_DIR / "params"
STANZA_DIR = PARAMS_DIR / "stanza"
DEFAULT_CONFIG_PATH = PARAMS_DIR / "symbol_segmenter.json"

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
PREDICATE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+")
USR_RE = re.compile(r"<usr>:\s*(.*?)(?:\n<agent>:|\Z)", re.S)
FOL_RE = re.compile(r"(?:premises-FOL|conclusion-FOL):\s*(.*?)(?:\n\n(?:Answer|Reasoning):|\Z)", re.S)
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+")

VARIABLES = {"x", "y", "z", "u", "v", "w"}
LOGIC_WORDS = {"forall", "exists", "true", "false"}
SECTION_WORDS = {"context", "question", "conclusion", "options", "answer"}
QUANTIFIER_WORDS = {"all", "some", "no", "every", "each", "any", "none"}
CONNECTIVE_WORDS = {"and", "or", "not", "nor", "neither", "either", "both", "then", "if"}
PRONOUN_WORDS = {"someone", "somebody", "anyone", "they", "them", "their", "he", "she", "it"}
GRAMMAR_WORDS = {
    "a",
    "an",
    "are",
    "as",
    "at",
    "be",
    "being",
    "by",
    "can",
    "do",
    "does",
    "follow",
    "following",
    "from",
    "have",
    "in",
    "is",
    "of",
    "on",
    "one",
    "premise",
    "statement",
    "than",
    "that",
    "the",
    "this",
    "to",
    "who",
    "whom",
    "whose",
    "with",
}
STOP_LEMMAS = SECTION_WORDS | QUANTIFIER_WORDS | CONNECTIVE_WORDS | PRONOUN_WORDS | GRAMMAR_WORDS
ENTITY_DENY = STOP_LEMMAS | {"people", "person", "thing"}
PREDICATE_DENY = STOP_LEMMAS | {"people", "person", "someone", "true", "false", "uncertain"}
RELATION_OBJECT_DEPS = {"obj", "iobj", "obl", "nmod", "compound", "amod"}


@dataclass(frozen=True)
class SymbolSet:
    entities: list[str]
    predicates: list[str]

    @property
    def all_symbols(self) -> set[str]:
        return set(self.entities) | set(self.predicates)

    def to_dict(self) -> dict[str, list[str]]:
        return {"entities": self.entities, "predicates": self.predicates}


def user_text(dialogue: str) -> str:
    match = USR_RE.search(dialogue)
    return match.group(1).strip() if match else dialogue.strip()


def logic_statement_text(dialogue: str) -> str:
    text = user_text(dialogue)
    text = re.split(r"\n\s*Options\s*:", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\n\s*Answer\s*:\s*$", "", text, flags=re.I)
    text = re.sub(r"\bQuestion\s*:\s*.*?\bConclusion\s*:", "Conclusion:", text, flags=re.I | re.S)
    text = re.sub(r"\b(Context|Conclusion)\s*:", " ", text, flags=re.I)
    return text


def formal_logic_text(dialogue: str) -> str:
    blocks = [match.group(1).strip() for match in FOL_RE.finditer(dialogue)]
    if blocks:
        return "\n".join(blocks)
    marker = "Formal logic:"
    if marker in dialogue:
        return dialogue.split(marker, 1)[1]
    return ""


def split_camel(identifier: str) -> list[str]:
    return [part.lower() for part in CAMEL_RE.findall(identifier) if part]


def normalize_lemma(text: str) -> str:
    text = text.lower().replace("'", "").replace("-", " ")
    return " ".join(WORD_RE.findall(text))


def camel_word(word: str) -> str:
    if word.isdigit():
        return word
    if word.startswith("year") and word[4:].isdigit():
        return word
    return word[:1].upper() + word[1:]


def camel_variants(words: Iterable[str]) -> set[str]:
    base_parts = [part for word in words for part in normalize_lemma(word).split() if part and part not in PREDICATE_DENY]
    if not base_parts:
        return set()
    return {camelize(base_parts)}


def camelize(words: Iterable[str]) -> str:
    clean = [normalize_lemma(word) for word in words]
    parts = [part for word in clean for part in word.split() if part and part not in PREDICATE_DENY]
    return "".join(camel_word(part) for part in parts)


def entity_name(words: Iterable[str]) -> str:
    parts = [part for word in words for part in normalize_lemma(word).split() if part]
    return "".join(parts)


def surface_token(token: str) -> str:
    if token.isdigit():
        return token
    if token.isupper() and len(token) > 1:
        return token
    return token[:1].upper() + token[1:].lower()


def surface_forms(tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    pascal = "".join(surface_token(token) for token in tokens)
    lower_first = pascal[:1].lower() + pascal[1:] if pascal else ""
    compact = "".join(token.lower() for token in tokens)
    forms = {pascal, lower_first, compact}
    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) == 4:
        forms.add("year" + tokens[0])
        forms.add("Year" + tokens[0])
    return {form for form in forms if form and form.lower() not in PREDICATE_DENY}


def parse_fol_symbols(text: str) -> SymbolSet:
    fol = formal_logic_text(text)
    if not fol:
        if "<usr>" in text or "<agent>" in text:
            return SymbolSet([], [])
        fol = text

    predicates = {
        name
        for name in PREDICATE_RE.findall(fol)
        if name not in VARIABLES and name.lower() not in LOGIC_WORDS
    }
    identifiers = set(IDENTIFIER_RE.findall(fol))
    entities = {
        name
        for name in identifiers - predicates
        if name not in VARIABLES
        and name.lower() not in LOGIC_WORDS
        and not (name[:1].isupper() and name not in {"True", "False"})
    }
    return SymbolSet(sorted(entities, key=str.lower), sorted(predicates, key=str.lower))


def ensure_project_root_on_path() -> None:
    root = BASE_DIR.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_dataset_texts() -> list[str]:
    ensure_project_root_on_path()
    from sft.logic.read import Dataset

    return list(Dataset())


def stanza_processors() -> dict[str, str]:
    return {
        "tokenize": "combined",
        "mwt": "combined",
        "pos": "combined_nocharlm",
        "lemma": "combined_nocharlm",
    }


def write_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "stanza_pos_lemma_openie_segmenter",
        "version": 2,
        "params_dir": str(STANZA_DIR.relative_to(BASE_DIR)),
        "processors": stanza_processors(),
        "sources": [
            "https://github.com/stanfordnlp/stanza",
            "https://github.com/CogComp/SRL-English",
            "https://arxiv.org/abs/2010.03147",
        ],
        "note": "Extraction uses only <usr> text with Stanza POS/lemma models and OpenIE/SRL-style chunking. FOL labels are used only for evaluation.",
    }
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def download_params() -> None:
    import stanza

    STANZA_DIR.mkdir(parents=True, exist_ok=True)
    stanza.download(
        "en",
        model_dir=str(STANZA_DIR),
        package=None,
        processors=stanza_processors(),
        verbose=True,
    )
    write_default_config()


@lru_cache(maxsize=1)
def load_stanza_pipeline():
    import stanza
    from stanza.pipeline.core import DownloadMethod

    return stanza.Pipeline(
        "en",
        dir=str(STANZA_DIR),
        package=None,
        processors=stanza_processors(),
        download_method=DownloadMethod.REUSE_RESOURCES,
        use_gpu=False,
        verbose=False,
    )


class SymbolSegmenter:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            write_default_config(self.config_path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "SymbolSegmenter":
        return cls(path)

    @classmethod
    def train(cls, _texts: Iterable[str] | None = None) -> "SymbolSegmenter":
        download_params()
        return cls()

    def save(self, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        write_default_config(path)

    def extract(self, text: str) -> SymbolSet:
        prompt = logic_statement_text(text)
        doc = load_stanza_pipeline()(prompt)
        entities: set[str] = set()
        predicates: set[str] = set()
        compound_parts: set[str] = set()

        for match in re.finditer(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", prompt):
            parts = [normalize_lemma(part) for part in match.group(0).split("-")]
            predicates.update(camel_variants(parts))
            compound_parts.update(parts)

        for year in re.findall(r"\b(?:18|19|20)\d{2}\b", prompt):
            entities.add("year" + year)
            predicates.add("year" + year)

        for sentence in doc.sentences:
            words = sentence.words
            consumed: set[int] = set()
            self._add_nominal_chunks(words, predicates, consumed)
            for index, word in enumerate(words):
                if index in consumed:
                    continue
                lemma = normalize_lemma(word.lemma or word.text)
                text_norm = normalize_lemma(word.text)
                if not lemma or lemma in STOP_LEMMAS:
                    continue

                if word.upos == "PROPN":
                    name_words = self._proper_name_span(words, index)
                    name = entity_name(w.text for w in name_words)
                    if name and name not in ENTITY_DENY:
                        entities.add(name)
                    if word.text.isupper() and len(word.text) > 1:
                        entities.add(word.text)

                if word.upos in {"NOUN", "ADJ"} and lemma not in PREDICATE_DENY:
                    if lemma not in compound_parts:
                        predicates.update(camel_variants([lemma]))

                if word.upos == "VERB" and lemma not in PREDICATE_DENY:
                    obj = self._right_object_phrase(words, index, consumed)
                    if obj:
                        predicates.update(camel_variants([lemma, *obj]))
                        consumed.add(index)
                    else:
                        predicates.update(camel_variants([lemma]))

        return SymbolSet(sorted(entities, key=str.lower), sorted(predicates, key=str.lower))

    def _proper_name_span(self, words: list[Any], index: int) -> list[Any]:
        start = index
        while start > 0 and words[start - 1].upos == "PROPN":
            start -= 1
        end = index + 1
        while end < len(words) and words[end].upos == "PROPN":
            end += 1
        return words[start:end]

    def _right_object_phrase(self, words: list[Any], index: int, consumed: set[int]) -> list[str]:
        phrase: list[str] = []
        for offset, word in enumerate(words[index + 1 : index + 5], start=index + 1):
            lemma = normalize_lemma(word.lemma or word.text)
            if not lemma or lemma in STOP_LEMMAS:
                continue
            if word.upos in {"NOUN", "PROPN", "ADJ", "NUM"}:
                phrase.append(lemma)
                consumed.add(offset)
                continue
            if phrase:
                break
        return phrase[:3]

    def _add_nominal_chunks(self, words: list[Any], predicates: set[str], consumed: set[int]) -> None:
        chunk: list[tuple[int, str]] = []
        for index, word in [*enumerate(words), (None, None)]:
            if word is not None:
                lemma = normalize_lemma(word.lemma or word.text)
                is_chunk_word = word.upos in {"ADJ", "NOUN", "PROPN", "NUM"} and lemma not in STOP_LEMMAS
            else:
                is_chunk_word = False
            if is_chunk_word:
                chunk.append((index, word.text if word.upos == "PROPN" else lemma))
                continue
            if len(chunk) >= 2:
                predicates.update(camel_variants(part for _, part in chunk))
                consumed.update(i for i, _ in chunk)
            chunk = []

    def evaluate(self, texts: Iterable[str]) -> dict[str, Any]:
        totals = Counter()
        misses: list[dict[str, Any]] = []
        samples_with_fol = 0
        for index, text in enumerate(texts):
            target = parse_fol_symbols(text)
            if not target.all_symbols:
                continue
            samples_with_fol += 1
            predicted = self.extract(text)
            hit = predicted.all_symbols & target.all_symbols
            missed = target.all_symbols - predicted.all_symbols
            totals["target"] += len(target.all_symbols)
            totals["hit"] += len(hit)
            totals["predicted"] += len(predicted.all_symbols)
            if missed and len(misses) < 20:
                misses.append(
                    {
                        "index": index,
                        "missed": sorted(missed, key=str.lower),
                        "target": sorted(target.all_symbols, key=str.lower),
                        "predicted": sorted(predicted.all_symbols, key=str.lower),
                    }
                )
        recall = totals["hit"] / totals["target"] if totals["target"] else 0.0
        precision = totals["hit"] / totals["predicted"] if totals["predicted"] else 0.0
        return {
            "samples_with_fol": samples_with_fol,
            "target_symbols": totals["target"],
            "predicted_symbols": totals["predicted"],
            "hits": totals["hit"],
            "recall": recall,
            "precision": precision,
            "misses": misses,
        }


def format_symbols(symbols: SymbolSet) -> str:
    lines = ["entities:"]
    lines += [f"- {entity}" for entity in symbols.entities]
    lines.append("")
    lines.append("predicates:")
    lines += [f"- {predicate}" for predicate in symbols.predicates]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CER FOL symbol segmenter using Stanza POS/lemma params.")
    parser.add_argument("--download", action="store_true", help="download Stanza model params into ./params")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    if args.download:
        download_params()

    model = SymbolSegmenter.load()
    texts = load_dataset_texts() if args.eval or args.sample is not None else []
    if args.sample is not None:
        print(format_symbols(model.extract(texts[args.sample])))
    if args.eval:
        metrics = model.evaluate(texts)
        print(json.dumps({key: value for key, value in metrics.items() if key != "misses"}, indent=2))
        if metrics["misses"]:
            print("first_misses:")
            print(json.dumps(metrics["misses"][:5], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
