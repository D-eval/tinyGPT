from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from eval import evaluate
from model import TinyGPT, checkpoint_model_state
from read import corpus_stats, load_or_train_tokenizer, read_texts


RESULT_PATH = Path("result.json")
PARAM_DIR = Path("params")
MODEL_PATH = PARAM_DIR / "model.pt"
TOKENIZER_PATH = PARAM_DIR / "tokenizer.json"
TARGET_ACC = 0.70
VOCAB_SIZE = 16000
BLOCK_SIZE = 256
BATCH_SIZE = 64
MAX_EPOCHS = 80
STEPS_PER_EPOCH = 120
LEARNING_RATE = 8e-4
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6


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


def encode_all(texts, tokenizer) -> torch.Tensor:
    ids: list[int] = []
    for _, text in texts:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def write_result(token_acc: float, ppl: float, param_count: int, param_mb: float, stats: dict, memory_mb: int) -> None:
    result = {
        "token_acc": round(token_acc, 4),
        "ppl": round(ppl, 4),
        "params": int(param_count),
        "param_mb": round(param_mb, 2),
        "stats": stats,
        "linear_attention": False,
        "estimated_training_memory_mb": memory_mb,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(model: TinyGPT, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float, float]:
    if not MODEL_PATH.exists():
        return 0, 0.0, float("inf")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = int(checkpoint.get("epoch", 0))
    token_acc = float(checkpoint.get("token_acc", 0.0))
    ppl = float(checkpoint.get("ppl", float("inf")))
    print(f"loaded {MODEL_PATH}: epoch={epoch} token_acc={token_acc:.4f} ppl={ppl:.2f}")
    return epoch, token_acc, ppl


def save_checkpoint(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    token_acc: float,
    ppl: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": model.config,
            "tokenizer_path": str(TOKENIZER_PATH),
            "token_acc": token_acc,
            "ppl": ppl,
            "epoch": epoch,
        },
        MODEL_PATH,
    )


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

    data = encode_all(texts, tokenizer)
    if len(data) <= BLOCK_SIZE + 1:
        raise RuntimeError("Dataset is too small for the configured block size.")

    device = choose_device()
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    ).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    param_mb = param_count * 4 / (1024 * 1024)
    memory_mb = estimate_training_memory_mb(param_count, BATCH_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD)
    inference_mb = int(param_count * 2 / (1024 * 1024) + BATCH_SIZE * BLOCK_SIZE * N_EMBD * 2 / (1024 * 1024))
    print(
        f"device={device} params={param_count} ({param_mb:.1f}MB) "
        f"estimated_training_memory={memory_mb}MB estimated_fp16_inference_memory={inference_mb}MB"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    start_epoch, best_acc, best_ppl = load_checkpoint(model, optimizer, device)
    if MODEL_PATH.exists():
        token_acc, ppl = evaluate(model, data, device)
        write_result(token_acc, ppl, param_count, param_mb, stats, memory_mb)
        print(f"pretrain-eval token_acc={token_acc:.4f} ppl={ppl:.2f}")
        if token_acc >= TARGET_ACC:
            return
        best_acc = max(best_acc, token_acc)
        best_ppl = ppl if token_acc >= best_acc else best_ppl

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    started = time.time()

    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                _, loss = model(x, y)
            assert loss is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        token_acc, ppl = evaluate(model, data, device)
        if token_acc >= best_acc:
            best_acc = token_acc
            best_ppl = ppl
        save_checkpoint(model, optimizer, epoch, token_acc, ppl)
        write_result(token_acc, ppl, param_count, param_mb, stats, memory_mb)
        elapsed = round(time.time() - started, 1)
        print(
            f"epoch={epoch} loss={running_loss / STEPS_PER_EPOCH:.4f} "
            f"token_acc={token_acc:.4f} ppl={ppl:.2f} best_acc={best_acc:.4f} seconds={elapsed}",
            flush=True,
        )
        if token_acc >= TARGET_ACC:
            break

    print(f"wrote {RESULT_PATH} best_acc={best_acc:.4f} best_ppl={best_ppl:.2f}")


if __name__ == "__main__":
    main()
