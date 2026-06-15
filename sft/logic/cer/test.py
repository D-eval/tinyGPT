from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sft.logic.cer.symbol_segmenter import SymbolSegmenter, format_symbols, parse_fol_symbols, user_text
from sft.logic.read import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CER symbol extraction on one dataset sample.")
    parser.add_argument("index", nargs="?", type=int, default=1000, help="dataset sample index")
    parser.add_argument("--show-text", action="store_true", help="print the <usr> input text")
    parser.add_argument("--show-target", action="store_true", help="print FOL symbols parsed from <agent>")
    args = parser.parse_args()

    dataset = Dataset()
    dialogue = dataset[args.index]
    usr_only = "<usr>: " + user_text(dialogue)
    model = SymbolSegmenter.load()
    symbols = model.extract(usr_only)
    target = parse_fol_symbols(dialogue)

    if args.show_text:
        print("model_input:")
        print(usr_only)
        print()

    print(f"sample_index: {args.index}")
    print("predicted_from_usr:")
    print(format_symbols(symbols))
    print()
    predicted_count = len(symbols.entities) + len(symbols.predicates)
    target_count = len(target.entities) + len(target.predicates)
    overlap = symbols.all_symbols & target.all_symbols
    print(f"predicted_count: {predicted_count}")
    print(f"target_count: {target_count}")
    print(f"exact_overlap_count: {len(overlap)}")

    if args.show_target:
        print()
        print("target_from_formal_logic:")
        print(format_symbols(target))
        if overlap:
            print()
            print("exact_overlap:")
            for symbol in sorted(overlap, key=str.lower):
                print(f"- {symbol}")


if __name__ == "__main__":
    main()
