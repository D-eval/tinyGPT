from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
import zlib
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from eval import evaluate
from model import Stage1Config, TinyGPT, checkpoint_model_state
from read import BPETokenizer, ids_corpus_stats
from read_union import (
    IGNORE_INDEX,
    TokenSample,
    iter_dataset_emotion_sft_token_samples,
    load_dataset2,
    read_dataset_part_ids,
    verify_union,
)
from train import choose_device, estimate_training_memory_mb, get_batch


RESULT_PATH = Path("result2.json")
PARAM_DIR = Path("params")
MODEL_PATH = PARAM_DIR / "model2.pt"
TOKENIZER_PATH = PARAM_DIR / "tokenizer2.json"
LOSS_PATH = Path("loss.png")

CFG = Stage1Config()
BATCH_SIZE = CFG.batch_size
BLOCK_SIZE = CFG.block_size
MICRO_BATCH_SIZE = CFG.batch_size
VOCAB_SIZE = 16_000
MAX_EPOCHS = 40
LEARNING_RATE = 1e-4
TARGET_DATASET_ACC = 0.98
TARGET_DATASET2_ACC = 0.80
DATASET_LOSS_WEIGHT = 0.5
DATASET2_LOSS_WEIGHT = 0.1
DATASET_EMOTION_SFT_LOSS_WEIGHT = 0.4
LOSS_AVG_EVERY_STEPS = 100


def ids_to_tensor(samples: list[tuple[Path, list[int]]]) -> torch.Tensor:
    ids: list[int] = []
    for _, sample_ids in samples:
        ids.extend(sample_ids)
    return torch.tensor(ids, dtype=torch.long)


def encode_all(texts: list[tuple[Path, str]], tokenizer: BPETokenizer) -> torch.Tensor:
    ids: list[int] = []
    for _, text in texts:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def train_val_split(data: torch.Tensor, block_size: int, val_ratio: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 2:
        raise RuntimeError("Dataset is too small for train/validation split with the configured block size.")

    val_len = max(int(len(data) * val_ratio), block_size + 2)
    val_len = min(val_len, len(data) - block_size - 2)
    if val_len <= block_size + 1:
        raise RuntimeError("Validation split is too small for evaluation with the configured block size.")
    return data[:-val_len], data[-val_len:]


def dataset2_split_index(total_samples: int, val_ratio: float = 0.01) -> int:
    if total_samples <= 1:
        raise RuntimeError("Dataset2 is too small for train/validation split.")
    valid_count = max(int(total_samples * val_ratio), 1)
    valid_count = min(valid_count, total_samples - 1)
    return total_samples - valid_count


def sample_split_index(total_samples: int, val_ratio: float = 0.01, name: str = "dataset") -> int:
    if total_samples <= 1:
        raise RuntimeError(f"{name} is too small for train/validation split.")
    valid_count = max(int(total_samples * val_ratio), 1)
    valid_count = min(valid_count, total_samples - 1)
    return total_samples - valid_count


def infer_steps_per_epoch(dataset2_train_samples: int, batch_size: int = BATCH_SIZE) -> int:
    if dataset2_train_samples <= 0:
        raise RuntimeError("Dataset2 train split is empty.")
    return math.ceil(dataset2_train_samples / batch_size)


def get_token_sample_batch(
    samples,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = random.choices(indices, k=batch_size)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    min_len: int | None = None
    loaded: list[list[int]] = []
    for index in chosen:
        _, tokens = samples[index]
        if len(tokens) < 2:
            raise RuntimeError(f"Token sample {index} is too short.")
        loaded.append(tokens)
        sample_len = len(tokens) - 1
        min_len = sample_len if min_len is None else min(min_len, sample_len)
    assert min_len is not None
    for tokens in loaded:
        xs.append(torch.tensor(tokens[:min_len], dtype=torch.long))
        ys.append(torch.tensor(tokens[1 : min_len + 1], dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def get_token_sample_batch_by_position(
    samples,
    ordered_indices: list[int],
    start: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = ordered_indices[start : start + batch_size]
    if not chosen:
        raise RuntimeError("No dataset2 indices selected for batch.")
    while len(chosen) < batch_size:
        chosen.append(random.choice(ordered_indices))

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    min_len: int | None = None
    loaded: list[list[int]] = []
    for index in chosen:
        _, tokens = samples[index]
        if len(tokens) < 2:
            raise RuntimeError(f"Token sample {index} is too short.")
        loaded.append(tokens)
        sample_len = len(tokens) - 1
        min_len = sample_len if min_len is None else min(min_len, sample_len)
    assert min_len is not None
    for tokens in loaded:
        xs.append(torch.tensor(tokens[:min_len], dtype=torch.long))
        ys.append(torch.tensor(tokens[1 : min_len + 1], dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def get_masked_token_sample_batch_by_position(
    samples: list[TokenSample],
    ordered_indices: list[int],
    start: int,
    batch_size: int,
    device: torch.device,
    pad_id: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = ordered_indices[start : start + batch_size]
    if not chosen:
        raise RuntimeError("No masked token sample indices selected for batch.")
    while len(chosen) < batch_size:
        chosen.append(random.choice(ordered_indices))

    loaded: list[tuple[list[int], list[int]]] = []
    max_len = 0
    max_sequence_tokens = block_size + 1
    for index in chosen:
        sample = samples[index]
        if sample.labels is None:
            raise RuntimeError(f"Masked token sample {index} has no labels.")
        tokens = sample.tokens
        labels = sample.labels
        if len(tokens) != len(labels):
            raise RuntimeError(f"Masked token sample {index} has mismatched token/label lengths.")
        if len(tokens) > max_sequence_tokens:
            tokens = tokens[-max_sequence_tokens:]
            labels = labels[-max_sequence_tokens:]
        if len(tokens) < 2:
            raise RuntimeError(f"Masked token sample {index} is too short.")
        loaded.append((tokens, labels))
        max_len = max(max_len, len(tokens) - 1)

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for tokens, labels in loaded:
        x_ids = tokens[:-1]
        y_ids = labels[1:]
        pad_len = max_len - len(x_ids)
        xs.append(torch.tensor(x_ids + [pad_id] * pad_len, dtype=torch.long))
        ys.append(torch.tensor(y_ids + [IGNORE_INDEX] * pad_len, dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate_token_samples(
    model: TinyGPT,
    samples,
    indices: list[int],
    device: torch.device,
    max_samples: int,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    eval_indices = indices[:max_samples]
    if not eval_indices:
        raise RuntimeError("No token samples available for evaluation.")
    for index in eval_indices:
        _, tokens = samples[index]
        if len(tokens) < 2:
            continue
        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
        logits, loss = model(x, y)
        assert loss is not None
        preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())
    if total == 0:
        raise RuntimeError("No tokens evaluated.")
    avg_loss = loss_sum / total
    return correct / total, math.exp(min(avg_loss, 20.0))


@torch.no_grad()
def evaluate_masked_token_samples(
    model: TinyGPT,
    samples: list[TokenSample],
    indices: list[int],
    device: torch.device,
    max_samples: int,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    eval_indices = indices[:max_samples]
    if not eval_indices:
        raise RuntimeError("No masked token samples available for evaluation.")
    for index in eval_indices:
        sample = samples[index]
        if sample.labels is None or len(sample.tokens) < 2:
            continue
        tokens = sample.tokens[-(model.block_size + 1) :]
        labels = sample.labels[-(model.block_size + 1) :]
        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        mask = y != IGNORE_INDEX
        if not bool(mask.any().item()):
            continue
        logits, loss = model(x, y)
        assert loss is not None
        preds = logits.argmax(dim=-1)
        correct += int((preds[mask] == y[mask]).sum().item())
        labeled_tokens = int(mask.sum().item())
        total += labeled_tokens
        loss_sum += float(loss.item()) * labeled_tokens
    if total == 0:
        raise RuntimeError("No masked tokens evaluated.")
    avg_loss = loss_sum / total
    return correct / total, math.exp(min(avg_loss, 20.0))


def make_model(vocab_size: int) -> TinyGPT:
    return TinyGPT(
        vocab_size=vocab_size,
        block_size=CFG.block_size,
        n_embd=CFG.n_embd,
        n_head=CFG.n_head,
        n_layer=CFG.n_layer,
    )


def result_meets_targets(path: Path = RESULT_PATH) -> bool:
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return False
    return (
        float(payload.get("dataset_token_acc", payload.get("token_acc_dataset", 0.0))) >= TARGET_DATASET_ACC
        and float(payload.get("dataset2_token_acc", payload.get("token_acc_dataset2", 0.0))) >= TARGET_DATASET2_ACC
        and "dataset_emotion_sft_token_acc" in payload
    )


def write_loss_png(losses: list[float], path: Path = LOSS_PATH) -> None:
    width, height = 640, 360
    rgb = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            rgb[offset : offset + 3] = bytes(color)

    for x in range(48, width - 24):
        put(x, height - 42, (40, 40, 40))
    for y in range(24, height - 41):
        put(48, y, (40, 40, 40))

    values = losses or [0.0]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        x = 56 + int(index * (width - 96) / max(len(values) - 1, 1))
        y = height - 50 - int((value - lo) * (height - 90) / (hi - lo))
        points.append((x, y))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = x0 + (x1 - x0) * step // steps
            y = y0 + (y1 - y0) * step // steps
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    put(x + dx, y + dy, (24, 101, 192))

    raw = b"".join(b"\x00" + rgb[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def save_checkpoint(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: list[float],
    result: dict | None,
    global_step: int,
    epoch_step: int,
    steps_per_epoch: int,
    last_loss: float | None = None,
) -> None:
    PARAM_DIR.mkdir(exist_ok=True)
    result = result or {}
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": model.config,
            "tokenizer_path": str(TOKENIZER_PATH),
            "epoch": epoch,
            "epoch_step": epoch_step,
            "global_step": global_step,
            "steps_per_epoch": steps_per_epoch,
            "last_loss": last_loss,
            "losses": losses,
            "result": result,
            "dataset_token_acc": result.get("dataset_token_acc", 0.0),
            "dataset2_token_acc": result.get("dataset2_token_acc", 0.0),
            "loss_weights": {
                "dataset": DATASET_LOSS_WEIGHT,
                "dataset2": DATASET2_LOSS_WEIGHT,
                "dataset_emotion_sft": DATASET_EMOTION_SFT_LOSS_WEIGHT,
            },
        },
        MODEL_PATH,
    )


def load_checkpoint(model: TinyGPT, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, int, int, list[float]]:
    if not MODEL_PATH.exists():
        return 0, 0, 0, []
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    expected = model.state_dict()
    skipped: list[str] = []
    loadable = {}
    for key, value in checkpoint_model_state(checkpoint, model).items():
        if key not in expected or expected[key].shape != value.shape:
            skipped.append(key)
            continue
        loadable[key] = value
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    checkpoint_config = checkpoint.get("config", {})
    can_resume_optimizer = (
        checkpoint_config.get("vocab_size") == model.config["vocab_size"]
        and not skipped
        and "optimizer_state" in checkpoint
    )
    if can_resume_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = int(checkpoint.get("epoch", 0))
        epoch_step = int(checkpoint.get("epoch_step", 0))
        global_step = int(checkpoint.get("global_step", 0))
    else:
        epoch = 0
        epoch_step = 0
        global_step = 0
    print(
        f"loaded {MODEL_PATH} non-strict: epoch={epoch} epoch_step={epoch_step} global_step={global_step} "
        f"loaded_tensors={len(loadable)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if skipped:
        print(f"skipped checkpoint tensors: {', '.join(skipped[:12])}", flush=True)
    if not can_resume_optimizer:
        print("optimizer state was not loaded because tokenizer/vocab changed; starting optimizer fresh.", flush=True)
    return epoch, epoch_step, global_step, [float(value) for value in checkpoint.get("losses", [])]


def build_result(
    token_acc: float,
    ppl: float,
    dataset_acc: float,
    dataset_ppl: float,
    dataset2_acc: float,
    dataset2_ppl: float,
    emotion_sft_acc: float,
    emotion_sft_ppl: float,
    model: TinyGPT,
    stats: dict,
    dataset_stats: dict,
    dataset2_stats: dict,
    emotion_sft_stats: dict,
) -> dict:
    params = sum(parameter.numel() for parameter in model.parameters())
    memory_mb = estimate_training_memory_mb(params, BATCH_SIZE, BLOCK_SIZE, CFG.n_layer, CFG.n_head, CFG.n_embd)
    return {
        "token_acc": round(token_acc, 4),
        "ppl": round(ppl, 4),
        "dataset_token_acc": round(dataset_acc, 4),
        "token_acc_dataset": round(dataset_acc, 4),
        "dataset_ppl": round(dataset_ppl, 4),
        "dataset2_token_acc": round(dataset2_acc, 4),
        "token_acc_dataset2": round(dataset2_acc, 4),
        "dataset2_ppl": round(dataset2_ppl, 4),
        "dataset_emotion_sft_token_acc": round(emotion_sft_acc, 4),
        "token_acc_dataset_emotion_sft": round(emotion_sft_acc, 4),
        "dataset_emotion_sft_ppl": round(emotion_sft_ppl, 4),
        "params": int(params),
        "param_mb": round(params * 4 / (1024 * 1024), 2),
        "stats": stats,
        "dataset_stats": dataset_stats,
        "dataset2_stats": dataset2_stats,
        "dataset_emotion_sft_stats": emotion_sft_stats,
        "loss_weights": {
            "dataset": DATASET_LOSS_WEIGHT,
            "dataset2": DATASET2_LOSS_WEIGHT,
            "dataset_emotion_sft": DATASET_EMOTION_SFT_LOSS_WEIGHT,
        },
        "linear_attention": False,
        "estimated_training_memory_mb": memory_mb,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
    }


def assert_cuda_memory_pressure() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available; cannot verify the 50% VRAM runtime requirement in this environment.", flush=True)
        return
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_allocated(device)
    pct = peak / total
    print(f"cuda peak_allocated={peak / 1024**3:.2f}GiB ({pct * 100:.1f}%)", flush=True)
    if pct < 0.50:
        raise RuntimeError("10k-context batch_size=8 training used less than 50% VRAM; check the training path.")


def train_epoch(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    dataset_data: torch.Tensor,
    dataset2_samples,
    dataset2_train_indices: list[int],
    emotion_sft_samples: list[TokenSample],
    emotion_sft_train_indices: list[int],
    device: torch.device,
    pad_id: int,
    epoch: int,
    steps_per_epoch: int,
    start_step: int,
    global_step: int,
    save_every_steps: int,
    loss_avg_every_steps: int,
    losses: list[float],
) -> tuple[float, int]:
    model.train()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    accumulation = max(BATCH_SIZE // MICRO_BATCH_SIZE, 1)
    running = 0.0
    running_steps = 0
    loss_window_sum = 0.0
    loss_window_steps = 0
    optimizer.zero_grad(set_to_none=True)
    dataset2_epoch_indices = dataset2_train_indices[:]
    random.shuffle(dataset2_epoch_indices)
    emotion_sft_epoch_indices = emotion_sft_train_indices[:]
    random.shuffle(emotion_sft_epoch_indices)
    micro_start = start_step * accumulation
    for step in range(micro_start, steps_per_epoch * accumulation):
        x, y = get_batch(dataset_data, MICRO_BATCH_SIZE, BLOCK_SIZE, device)
        with autocast("cuda", enabled=device.type == "cuda"):
            _, dataset_loss = model(x, y)
            assert dataset_loss is not None
            scaled_dataset_loss = dataset_loss * DATASET_LOSS_WEIGHT / accumulation
        scaler.scale(scaled_dataset_loss).backward()

        dataset2_start = (step // accumulation) * BATCH_SIZE + (step % accumulation) * MICRO_BATCH_SIZE
        x2, y2 = get_token_sample_batch_by_position(
            dataset2_samples,
            dataset2_epoch_indices,
            dataset2_start,
            MICRO_BATCH_SIZE,
            device,
        )
        with autocast("cuda", enabled=device.type == "cuda"):
            _, dataset2_loss = model(x2, y2)
            assert dataset2_loss is not None
            scaled_dataset2_loss = dataset2_loss * DATASET2_LOSS_WEIGHT / accumulation
        scaler.scale(scaled_dataset2_loss).backward()

        emotion_sft_start = (step // accumulation) * BATCH_SIZE + (step % accumulation) * MICRO_BATCH_SIZE
        x3, y3 = get_masked_token_sample_batch_by_position(
            emotion_sft_samples,
            emotion_sft_epoch_indices,
            emotion_sft_start % len(emotion_sft_epoch_indices),
            MICRO_BATCH_SIZE,
            device,
            pad_id,
            BLOCK_SIZE,
        )
        with autocast("cuda", enabled=device.type == "cuda"):
            _, emotion_sft_loss = model(x3, y3)
            assert emotion_sft_loss is not None
            scaled_emotion_sft_loss = emotion_sft_loss * DATASET_EMOTION_SFT_LOSS_WEIGHT / accumulation
        scaler.scale(scaled_emotion_sft_loss).backward()

        weighted_loss = (
            dataset_loss * DATASET_LOSS_WEIGHT
            + dataset2_loss * DATASET2_LOSS_WEIGHT
            + emotion_sft_loss * DATASET_EMOTION_SFT_LOSS_WEIGHT
        )
        if (step + 1) % accumulation == 0:
            completed_step = (step + 1) // accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            current_loss = float(weighted_loss.detach().cpu())
            loss_window_sum += current_loss
            loss_window_steps += 1
            print(
                f"epoch={epoch} step={completed_step}/{steps_per_epoch} global_step={global_step} "
                f"weighted_loss={current_loss:.4f} "
                f"dataset_loss={float(dataset_loss.detach().cpu()):.4f} "
                f"dataset2_loss={float(dataset2_loss.detach().cpu()):.4f} "
                f"dataset_emotion_sft_loss={float(emotion_sft_loss.detach().cpu()):.4f}",
                flush=True,
            )
            if loss_avg_every_steps > 0 and global_step % loss_avg_every_steps == 0 and loss_window_steps > 0:
                avg_loss = loss_window_sum / loss_window_steps
                losses.append(avg_loss)
                write_loss_png(losses)
                print(
                    f"recorded avg_loss={avg_loss:.4f} over {loss_window_steps} optimizer steps "
                    f"at global_step={global_step}",
                    flush=True,
                )
                loss_window_sum = 0.0
                loss_window_steps = 0
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    losses=losses,
                    result=None,
                    global_step=global_step,
                    epoch_step=completed_step,
                    steps_per_epoch=steps_per_epoch,
                    last_loss=current_loss,
                )
                print(
                    f"saved checkpoint {MODEL_PATH} at epoch={epoch} "
                    f"step={completed_step}/{steps_per_epoch} global_step={global_step}",
                    flush=True,
                )
        running += float(weighted_loss.detach().cpu())
        running_steps += 1
    if running_steps == 0:
        return 0.0, global_step
    return running / running_steps, global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TASK2 stage3 with a pure transformer decoder.")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="defaults to ceil(dataset2_train_samples / batch_size)")
    parser.add_argument("--save-every-steps", type=int, default=500, help="save model2.pt every N optimizer steps; <=0 disables mid-epoch saves")
    parser.add_argument("--loss-avg-every-steps", type=int, default=LOSS_AVG_EVERY_STEPS, help="append avg loss to loss history every N optimizer steps; <=0 disables step loss records")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-dataset2-samples", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="continue training even if result2 already meets targets")
    args = parser.parse_args()

    PARAM_DIR.mkdir(exist_ok=True)
    print(
        "stage3 pure TinyGPT training start: "
        f"batch_size={BATCH_SIZE} micro_batch_size={MICRO_BATCH_SIZE} block_size={BLOCK_SIZE} "
        f"loss_weights=dataset:{DATASET_LOSS_WEIGHT},dataset2:{DATASET2_LOSS_WEIGHT},"
        f"dataset_emotion_sft:{DATASET_EMOTION_SFT_LOSS_WEIGHT}",
        flush=True,
    )
    print("loading read_union.py datasets ...", flush=True)
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"{TOKENIZER_PATH} is required because dataset2 is already pre-tokenized.")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    dataset_id_samples = read_dataset_part_ids(tokenizer, "dataset")
    emotion_sft_samples = list(iter_dataset_emotion_sft_token_samples(tokenizer, split="train"))
    dataset2 = load_dataset2(context_length=BLOCK_SIZE)
    if not dataset_id_samples or len(dataset2) == 0 or not emotion_sft_samples:
        raise RuntimeError("No training data found from read_union.py ids.")

    print("building dataset1 tensor, dataset2 sample index, and dataset_emotion_sft sample index ...", flush=True)
    dataset_data = ids_to_tensor(dataset_id_samples)
    dataset_train_data, dataset_val_data = train_val_split(dataset_data, BLOCK_SIZE)
    dataset2_valid_start = dataset2_split_index(len(dataset2))
    dataset2_train_indices = list(range(dataset2_valid_start))
    dataset2_val_indices = list(range(dataset2_valid_start, len(dataset2)))
    emotion_sft_valid_start = sample_split_index(len(emotion_sft_samples), name="dataset_emotion_sft")
    emotion_sft_train_indices = list(range(emotion_sft_valid_start))
    emotion_sft_val_indices = list(range(emotion_sft_valid_start, len(emotion_sft_samples)))
    steps_per_epoch = args.steps_per_epoch or infer_steps_per_epoch(len(dataset2_train_indices), BATCH_SIZE)
    if steps_per_epoch <= 0:
        raise RuntimeError("--steps-per-epoch must be positive.")
    print(
        "train/validation split: "
        f"dataset={len(dataset_train_data)}:{len(dataset_val_data)} "
        f"dataset2_samples={len(dataset2_train_indices)}:{len(dataset2_val_indices)} "
        f"dataset_emotion_sft_samples={len(emotion_sft_train_indices)}:{len(emotion_sft_val_indices)} "
        f"steps_per_epoch={steps_per_epoch}",
        flush=True,
    )

    dataset_stats = ids_corpus_stats(dataset_id_samples)
    emotion_sft_stats = ids_corpus_stats([(sample.path, sample.tokens) for sample in emotion_sft_samples])
    union_verify = verify_union(context_length=BLOCK_SIZE)
    dataset2_stats = {
        "files": union_verify["dataset2_samples"],
        "tokens": union_verify["dataset2_tokens"],
        "longest_context": BLOCK_SIZE,
        "token_memory_mb": round(int(union_verify["dataset2_tokens"]) * 8 / (1024 * 1024), 3),
    }
    stats = {
        "files": int(dataset_stats["files"]) + int(dataset2_stats["files"]) + int(emotion_sft_stats["files"]),
        "tokens": int(dataset_stats["tokens"]) + int(dataset2_stats["tokens"]) + int(emotion_sft_stats["tokens"]),
        "longest_context": max(
            int(dataset_stats["longest_context"]),
            int(dataset2_stats["longest_context"]),
            int(emotion_sft_stats["longest_context"]),
        ),
        "token_memory_mb": round(
            float(dataset_stats["token_memory_mb"])
            + float(dataset2_stats["token_memory_mb"])
            + float(emotion_sft_stats["token_memory_mb"]),
            3,
        ),
    }
    print("ids corpus stats:", stats, flush=True)

    device = choose_device()
    if device.type != "cuda":
        raise RuntimeError("Stage3 requires CUDA for 10k-context TinyGPT training.")
    model = make_model(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    checkpoint_epoch, checkpoint_epoch_step, global_step, losses = load_checkpoint(model, optimizer, device)

    if result_meets_targets() and not args.force:
        print("result2.json already meets targets for pure TinyGPT; skipping training.", flush=True)
    else:
        started = time.time()
        first_epoch = checkpoint_epoch if checkpoint_epoch_step > 0 else checkpoint_epoch + 1
        for epoch in range(first_epoch, args.max_epochs + 1):
            start_step = checkpoint_epoch_step if epoch == checkpoint_epoch and checkpoint_epoch_step > 0 else 0
            if start_step >= steps_per_epoch:
                start_step = 0
            if start_step:
                print(
                    f"resuming epoch={epoch} from step={start_step}/{steps_per_epoch} "
                    f"global_step={global_step}",
                    flush=True,
                )
            loss, global_step = train_epoch(
                model,
                optimizer,
                dataset_train_data,
                dataset2,
                dataset2_train_indices,
                emotion_sft_samples,
                emotion_sft_train_indices,
                device,
                tokenizer.pad_id,
                epoch,
                steps_per_epoch,
                start_step,
                global_step,
                args.save_every_steps,
                args.loss_avg_every_steps,
                losses,
            )
            losses.append(loss)
            checkpoint_epoch_step = 0
            assert_cuda_memory_pressure()

            dataset_acc, dataset_ppl = evaluate(model, dataset_val_data, device, batch_size=args.eval_batch_size)
            dataset2_acc, dataset2_ppl = evaluate_token_samples(
                model,
                dataset2,
                dataset2_val_indices,
                device,
                max_samples=args.eval_dataset2_samples,
            )
            emotion_sft_acc, emotion_sft_ppl = evaluate_masked_token_samples(
                model,
                emotion_sft_samples,
                emotion_sft_val_indices,
                device,
                max_samples=args.eval_dataset2_samples,
            )
            total_weight = DATASET_LOSS_WEIGHT + DATASET2_LOSS_WEIGHT + DATASET_EMOTION_SFT_LOSS_WEIGHT
            token_acc = (
                dataset_acc * DATASET_LOSS_WEIGHT
                + dataset2_acc * DATASET2_LOSS_WEIGHT
                + emotion_sft_acc * DATASET_EMOTION_SFT_LOSS_WEIGHT
            ) / total_weight
            ppl = (
                dataset_ppl * DATASET_LOSS_WEIGHT
                + dataset2_ppl * DATASET2_LOSS_WEIGHT
                + emotion_sft_ppl * DATASET_EMOTION_SFT_LOSS_WEIGHT
            ) / total_weight
            result = build_result(
                token_acc,
                ppl,
                dataset_acc,
                dataset_ppl,
                dataset2_acc,
                dataset2_ppl,
                emotion_sft_acc,
                emotion_sft_ppl,
                model,
                stats,
                dataset_stats,
                dataset2_stats,
                emotion_sft_stats,
            )
            RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                losses=losses,
                result=result,
                global_step=global_step,
                epoch_step=0,
                steps_per_epoch=steps_per_epoch,
                last_loss=loss,
            )
            write_loss_png(losses)
            print(
                f"epoch={epoch} loss={loss:.4f} "
                f"dataset_acc={dataset_acc:.4f} dataset2_acc={dataset2_acc:.4f} "
                f"dataset_emotion_sft_acc={emotion_sft_acc:.4f} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )
            if dataset_acc >= TARGET_DATASET_ACC and dataset2_acc >= TARGET_DATASET2_ACC:
                break

    print("Run with realtime logging: nohup python3 -u train2.py > train2.log 2>&1 &", flush=True)


if __name__ == "__main__":
    main()
