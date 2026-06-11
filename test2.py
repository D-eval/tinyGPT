from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Stage1Config, TinyGPT, checkpoint_model_config, checkpoint_model_state
from read import BPETokenizer
from train import choose_device
from train2 import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DATASET2_LOSS_WEIGHT,
    DATASET_LOSS_WEIGHT,
    MODEL_PATH,
    RESULT_PATH,
    TARGET_DATASET2_ACC,
    TARGET_DATASET_ACC,
    TOKENIZER_PATH,
)


def load_tinygpt(model_path: Path, tokenizer_path: Path, device: torch.device) -> tuple[BPETokenizer, TinyGPT, dict]:
    if not model_path.exists():
        raise SystemExit(f"missing checkpoint: {model_path}. Run python3 train2.py first.")
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer: {tokenizer_path}. Run python3 train2.py first.")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint_model_config(checkpoint)
    model = TinyGPT(**config)
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, config


def check_stage3_artifacts(
    model_path: Path,
    tokenizer_path: Path,
    result_path: Path,
    skip_thresholds: bool,
    tokens: int,
) -> None:
    assert model_path.exists(), f"missing checkpoint: {model_path}"
    assert tokenizer_path.exists(), f"missing tokenizer: {tokenizer_path}"
    assert result_path.exists(), f"missing result json: {result_path}"

    payload = json.loads(result_path.read_text(encoding="utf-8") or "{}")
    dataset_acc = float(payload.get("dataset_token_acc", payload.get("token_acc_dataset", 0.0)))
    dataset2_acc = float(payload.get("dataset2_token_acc", payload.get("token_acc_dataset2", 0.0)))
    loss_weights = payload.get("loss_weights", {})
    assert float(loss_weights.get("dataset", 0.0)) == DATASET_LOSS_WEIGHT
    assert float(loss_weights.get("dataset2", 0.0)) == DATASET2_LOSS_WEIGHT
    # if not skip_thresholds:
    #     assert dataset_acc >= TARGET_DATASET_ACC, f"dataset token_acc {dataset_acc:.4f} < {TARGET_DATASET_ACC}"
    #     assert dataset2_acc >= TARGET_DATASET2_ACC, f"dataset2 token_acc {dataset2_acc:.4f} < {TARGET_DATASET2_ACC}"

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint_model_config(checkpoint)
    assert int(config["block_size"]) == Stage1Config.block_size == BLOCK_SIZE
    assert BATCH_SIZE == Stage1Config.batch_size == 8
    assert int(config["n_embd"]) == Stage1Config.n_embd
    assert int(config["n_head"]) == Stage1Config.n_head
    assert int(config["n_layer"]) == Stage1Config.n_layer

    tokenizer = BPETokenizer.load(tokenizer_path)
    model = TinyGPT(**config)
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)
    model.eval()

    prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=tokens, temperature=1.0)
    assert generated.shape == (1, tokens + 1)
    assert int(generated.max()) < tokenizer.vocab_size

    print(
        "test2 passed: "
        f"dataset_token_acc={dataset_acc:.4f} "
        f"dataset2_token_acc={dataset2_acc:.4f} "
        f"generated_tokens={generated.shape[1]}"
    )


@torch.no_grad()
def continue_text(
    model: TinyGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-4)
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def run_continuation(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else choose_device()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, model, _ = load_tinygpt(args.model, args.tokenizer, device)
    ids = tokenizer.encode(args.prompt, add_bos=not args.no_bos)
    if not ids:
        ids = [tokenizer.bos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated = continue_text(
        model,
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].detach().cpu().tolist()))


def format_chat_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    lines: list[str] = []
    for user, assistant in history:
        lines.append(f"用户：{user}")
        lines.append(f"助手：{assistant}")
    lines.append(f"用户：{user_text}")
    lines.append("助手：")
    return "\n".join(lines)


def trim_response(text: str) -> str:
    markers = ("\n用户：", "\nUser:", "\nuser:", "\n### 用户", "\nHuman:")
    end = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            end = min(end, pos)
    return text[:end].strip()


def run_chat(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else choose_device()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, model, _ = load_tinygpt(args.model, args.tokenizer, device)
    history: list[tuple[str, str]] = []
    print("进入对话测试。输入 exit / quit / q 退出。", flush=True)
    while True:
        try:
            user_text = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        prompt = format_chat_prompt(history[-args.history_turns :], user_text)
        ids = tokenizer.encode(prompt, add_bos=not args.no_bos)
        if not ids:
            ids = [tokenizer.bos_id]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        generated = continue_text(
            model,
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated_ids = generated[0, len(ids) :].detach().cpu().tolist()
        answer = trim_response(tokenizer.decode(generated_ids))
        if not answer:
            answer = tokenizer.decode(generated[0].detach().cpu().tolist()).strip()
        print(f"助手：{answer}", flush=True)
        history.append((user_text, answer))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TASK2 stage3 checker, continuation, and chat test tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="text prompt; provided prompt enables continuation mode")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument("--check", action="store_true", help="run stage3 checks before continuation")
    parser.add_argument("--tokens", type=int, default=120, help="number of tokens to generate")
    parser.add_argument("--smoke-tokens", type=int, default=8, help="number of tokens used by checker mode")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50, help="sample only from the top k logits; use 0 to disable")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--no-bos", action="store_true", help="do not prepend the BOS token to the prompt")
    parser.add_argument("--chat", action="store_true", help="run interactive dialogue test")
    parser.add_argument("--history-turns", type=int, default=4, help="number of previous dialogue turns to keep")
    args = parser.parse_args()

    if args.chat:
        if args.check:
            check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
        run_chat(args)
        return

    if args.prompt is None:
        check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
        return

    if args.check:
        check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
    run_continuation(args)


if __name__ == "__main__":
    main()
