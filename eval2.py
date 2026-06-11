from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval import evaluate
from model import TinyGPT, checkpoint_model_config, checkpoint_model_state
from read import BPETokenizer, corpus_stats
from read_union import read_dataset2_part, read_dataset_part, read_union_texts
from train import choose_device, estimate_training_memory_mb
from train2 import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DATASET2_LOSS_WEIGHT,
    DATASET_LOSS_WEIGHT,
    MODEL_PATH,
    TOKENIZER_PATH,
    encode_all,
)


def build_result(
    token_acc: float,
    ppl: float,
    dataset_acc: float,
    dataset_ppl: float,
    dataset2_acc: float,
    dataset2_ppl: float,
    param_count: int,
    param_mb: float,
    stats: dict,
    dataset_stats: dict,
    dataset2_stats: dict,
    memory_mb: int,
) -> dict:
    return {
        "token_acc": round(token_acc, 4),
        "ppl": round(ppl, 4),
        "dataset_token_acc": round(dataset_acc, 4),
        "token_acc_dataset": round(dataset_acc, 4),
        "dataset_ppl": round(dataset_ppl, 4),
        "dataset2_token_acc": round(dataset2_acc, 4),
        "token_acc_dataset2": round(dataset2_acc, 4),
        "dataset2_ppl": round(dataset2_ppl, 4),
        "loss_weights": {
            "dataset": DATASET_LOSS_WEIGHT,
            "dataset2": DATASET2_LOSS_WEIGHT,
        },
        "params": int(param_count),
        "param_mb": round(param_mb, 2),
        "stats": stats,
        "dataset_stats": dataset_stats,
        "dataset2_stats": dataset2_stats,
        "linear_attention": False,
        "estimated_training_memory_mb": memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the stage3 tiny GPT on dataset and dataset2.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--result", type=Path, default=Path("result2.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Run python3 train2.py first.")
    if not args.tokenizer.exists():
        raise SystemExit(f"{args.tokenizer} not found. Run python3 train2.py first.")

    tokenizer = BPETokenizer.load(args.tokenizer)
    dataset_texts = read_dataset_part("dataset")
    dataset2_texts = read_dataset2_part("dataset2/data")
    texts = read_union_texts("dataset", "dataset2/data")
    data = encode_all(texts, tokenizer)
    dataset_data = encode_all(dataset_texts, tokenizer)
    dataset2_data = encode_all(dataset2_texts, tokenizer)

    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint_model_config(checkpoint)
    model = TinyGPT(**config)
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)

    device = choose_device()
    model.to(device)
    token_acc, ppl = evaluate(model, data, device, batch_size=args.batch_size)
    dataset_acc, dataset_ppl = evaluate(model, dataset_data, device, batch_size=args.batch_size)
    dataset2_acc, dataset2_ppl = evaluate(model, dataset2_data, device, batch_size=args.batch_size)

    param_count = sum(parameter.numel() for parameter in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    memory_mb = estimate_training_memory_mb(
        param_count,
        BATCH_SIZE,
        BLOCK_SIZE,
        int(config["n_layer"]),
        int(config["n_head"]),
        int(config["n_embd"]),
    )
    stats = corpus_stats(tokenizer, texts)
    dataset_stats = corpus_stats(tokenizer, dataset_texts)
    dataset2_stats = corpus_stats(tokenizer, dataset2_texts)

    result = build_result(
        token_acc,
        ppl,
        dataset_acc,
        dataset_ppl,
        dataset2_acc,
        dataset2_ppl,
        param_count,
        param_mb,
        stats,
        dataset_stats,
        dataset2_stats,
        memory_mb,
    )
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
