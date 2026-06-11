from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model import TinyGPT, checkpoint_model_config, checkpoint_model_state
from read import BPETokenizer, corpus_stats, read_texts


def estimate_training_memory_mb(
    param_count: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
) -> int:
    param_and_adam = param_count * 12
    attention = batch_size * n_layer * n_head * block_size * block_size * 4
    activations = batch_size * block_size * n_embd * n_layer * 8
    return int((param_and_adam + attention + activations) / (1024 * 1024))


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses: list[float] = []

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for start in range(0, len(data) - model.block_size - 1, model.block_size):
        chunk = data[start : start + model.block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
        if len(xs) < batch_size:
            continue

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        logits, loss = model(x, y)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        if loss is not None:
            losses.append(float(loss.item()))
        xs.clear()
        ys.clear()

    if xs:
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        logits, loss = model(x, y)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        if loss is not None:
            losses.append(float(loss.item()))

    token_acc = correct / max(total, 1)
    ppl = math.exp(sum(losses) / max(len(losses), 1))
    model.train()
    return token_acc, ppl


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_dataset(tokenizer: BPETokenizer, dataset_dir: str | Path) -> torch.Tensor:
    ids: list[int] = []
    for _, text in read_texts(dataset_dir):
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained tiny GPT on the dataset.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model", default="params/model.pt")
    parser.add_argument("--tokenizer", default="params/tokenizer.json")
    parser.add_argument("--result", default="result.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    tokenizer = BPETokenizer.load(args.tokenizer)
    data = encode_dataset(tokenizer, args.dataset)
    device = choose_device()
    model = TinyGPT(**checkpoint_model_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)
    token_acc, ppl = evaluate(model, data, device, batch_size=args.batch_size)
    params = sum(parameter.numel() for parameter in model.parameters())
    param_mb = params * 4 / (1024 * 1024)
    config = checkpoint["config"]
    memory_mb = estimate_training_memory_mb(
        params,
        args.batch_size,
        int(config["block_size"]),
        int(config["n_layer"]),
        int(config["n_head"]),
        int(config["n_embd"]),
    )
    stats = corpus_stats(tokenizer, read_texts(args.dataset))
    result = {
        "token_acc": round(token_acc, 4),
        "ppl": round(ppl, 4),
        "params": int(params),
        "param_mb": round(param_mb, 2),
        "stats": stats,
        "linear_attention": False,
        "estimated_training_memory_mb": memory_mb,
    }
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
