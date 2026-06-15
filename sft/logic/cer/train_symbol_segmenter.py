from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sft.logic.cer.symbol_segmenter import DEFAULT_CONFIG_PATH, SymbolSegmenter, download_params, load_dataset_texts


def main() -> None:
    download_params()
    texts = load_dataset_texts()
    model = SymbolSegmenter.load()
    metrics = model.evaluate(texts)
    print(f"saved model config: {DEFAULT_CONFIG_PATH}")
    print(f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f}")


if __name__ == "__main__":
    main()
