from __future__ import annotations

import math

import torch

from model import TinyGPT


@torch.no_grad()
def evaluate(model: TinyGPT, data: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses: list[float] = []
    for start in range(0, len(data) - 1, model.block_size):
        chunk = data[start : start + model.block_size + 1]
        if len(chunk) < 2:
            continue
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
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
