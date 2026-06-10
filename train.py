from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from eval import evaluate
from model import TinyGPT
from read import corpus_stats, load_or_train_tokenizer, read_texts


RESULT_PATH = Path("result.json")
PARAM_DIR = Path("params")
MODEL_PATH = PARAM_DIR / "model.pt"
TOKENIZER_PATH = PARAM_DIR / "tokenizer.json"
TARGET_ACC = 0.70
VOCAB_SIZE = 16000
BLOCK_SIZE = 256
BATCH_SIZE = 32
MAX_STEPS = 3000
EVAL_EVERY = 150


def already_done() -> bool:
    if not RESULT_PATH.exists():
        RESULT_PATH.write_text("{}", encoding="utf-8")
        return False
    try:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return False
    return float(payload.get("token_acc", 0.0)) >= TARGET_ACC


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts]).to(device)
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts]).to(device)
    return x, y


def estimate_training_memory_mb(param_count: int, batch_size: int, block_size: int, n_layer: int, n_head: int) -> int:
    param_and_adam = param_count * 12
    attention = batch_size * n_layer * n_head * block_size * block_size * 4
    activations = batch_size * block_size * 256 * n_layer * 8
    return int((param_and_adam + attention + activations) / (1024 * 1024))


def main() -> None:
    if already_done():
        print("result.json already has token_acc >= 0.7; skipping training.")
        return

    PARAM_DIR.mkdir(exist_ok=True)
    texts = read_texts("dataset")
    if not texts:
        raise RuntimeError("No non-empty txt files found under dataset.")

    tokenizer = load_or_train_tokenizer(texts, TOKENIZER_PATH, vocab_size=VOCAB_SIZE)
    stats = corpus_stats(tokenizer, texts)
    print("corpus stats:", stats)

    ids: list[int] = []
    for _, text in texts:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    if len(ids) <= BLOCK_SIZE + 1:
        raise RuntimeError("Dataset is too small for the configured block size.")

    data = torch.tensor(ids, dtype=torch.long)
    device = choose_device()
    model = TinyGPT(vocab_size=tokenizer.vocab_size).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    memory_mb = estimate_training_memory_mb(param_count, BATCH_SIZE, BLOCK_SIZE, 4, 4)
    print(f"device={device} params={param_count} ({param_mb:.1f}MB) estimated_training_memory={memory_mb}MB")

    if param_mb >= 50:
        raise RuntimeError(f"Model parameters are {param_mb:.1f}MB, over the 50MB limit.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_acc = 0.0
    best_ppl = float("inf")
    started = time.time()

    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % EVAL_EVERY == 0:
            token_acc, ppl = evaluate(model, data, device)
            if token_acc > best_acc:
                best_acc = token_acc
                best_ppl = ppl
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": model.config,
                        "tokenizer_path": str(TOKENIZER_PATH),
                        "token_acc": token_acc,
                        "ppl": ppl,
                    },
                    MODEL_PATH,
                )
            print(f"step={step} loss={loss.item():.4f} token_acc={token_acc:.4f} ppl={ppl:.2f}")
            if token_acc >= TARGET_ACC:
                break

    result = {
        "token_acc": round(best_acc, 4),
        "ppl": round(best_ppl, 4),
        "params": int(param_count),
        "param_mb": round(param_mb, 2),
        "stats": stats,
        "linear_attention": False,
        "estimated_training_memory_mb": memory_mb,
        "seconds": round(time.time() - started, 1),
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", RESULT_PATH, result)


if __name__ == "__main__":
    main()
