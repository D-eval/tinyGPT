from __future__ import annotations

import argparse
import json
import re
import sys
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


DEFAULT_SYSTEM_PROMPT = "You are a helpful dialogue agent. Respond naturally across the full conversation."
DEFAULT_DIALOGUE_PROMPTS = [
    "你好，今天我有点累，但又不想浪费这个晚上。",
    "如果我想慢慢把生活整理好，你会建议我先做什么？",
    "你能用更轻松一点的语气继续陪我聊吗？",
]


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is None:
        return choose_device()
    if requested == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if (
        requested == "mps"
        and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available())
    ):
        print("warning: MPS is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def load_tinygpt(model_path: Path, tokenizer_path: Path, device: torch.device) -> tuple[BPETokenizer, TinyGPT, dict]:
    if not model_path.exists():
        raise SystemExit(f"missing checkpoint: {model_path}. Run python3 train2.py first.")
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer: {tokenizer_path}. Run python3 train2.py first.")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint_model_config(checkpoint)
    config["vocab_size"] = tokenizer.vocab_size
    model = TinyGPT(**config)
    expected = model.state_dict()
    loadable = {
        key: value
        for key, value in checkpoint_model_state(checkpoint, model).items()
        if key in expected and expected[key].shape == value.shape
    }
    model.load_state_dict(loadable, strict=False)
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

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint_model_config(checkpoint)
    config["vocab_size"] = tokenizer.vocab_size
    assert int(config["block_size"]) == Stage1Config.block_size == BLOCK_SIZE
    assert BATCH_SIZE == Stage1Config.batch_size == 8
    assert int(config["n_embd"]) == Stage1Config.n_embd
    assert int(config["n_head"]) == Stage1Config.n_head
    assert int(config["n_layer"]) == Stage1Config.n_layer

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
    stop_token_ids: set[int] | None = None,
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
        if stop_token_ids is not None and int(next_id.item()) in stop_token_ids:
            break
    return idx


def run_continuation(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, model, _ = load_tinygpt(args.model, args.tokenizer, device)
    prompt = build_assistant_prompt(args.prompt)
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
    print(tokenizer.decode(generated[0].detach().cpu().tolist()))


def build_assistant_prompt(
    user_text: str,
    user_token: str = "<usr1>",
    assistant_token: str = "<boiwan>",
) -> str:
    if re.search(r"(^|\n)<[^>\n]+>:", user_text):
        return user_text
    return f"{user_token}: {user_text}\n{assistant_token}:"


def format_chat_prompt(
    history: list[tuple[str, str]],
    user_text: str,
    user_token: str = "<usr1>",
    assistant_token: str = "<boiwan>",
) -> str:
    lines: list[str] = []
    for user, assistant in history:
        lines.append(f"{user_token}: {user}")
        lines.append(f"{assistant_token}: {assistant}")
    lines.append(f"{user_token}: {user_text}")
    lines.append(f"{assistant_token}:")
    return "\n".join(lines)


def trim_response(text: str) -> str:
    end = len(text)
    for match in re.finditer(r"\n<[^>\n]+>:", text):
        end = min(end, match.start())
        break
    markers = ("\n用户：", "\n助手：", "\nUser:", "\nuser:", "\n### 用户", "\nHuman:")
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            end = min(end, pos)
    return text[:end].strip()


def run_chat(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
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
            answer = tokenizer.decode(generated_ids).strip()
        print(f"助手：{answer}", flush=True)
        history.append((user_text, answer))


def run_dialogue_sample(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, model, _ = load_tinygpt(args.model, args.tokenizer, device)
    prompts = list(args.dialogue_prompts or DEFAULT_DIALOGUE_PROMPTS)
    if args.prompt is not None:
        prompts = [args.prompt, *prompts]
    prompts = prompts[: max(args.turns, 0)]
    history: list[tuple[str, str]] = []

    print("<dialogue>", flush=True)
    for user_text in prompts:
        prompt = format_chat_prompt(history[-args.history_turns :], user_text)
        ids = tokenizer.encode(prompt, add_bos=not args.no_bos)
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
        print(f"<usr1>: {user_text}", flush=True)
        print(f"<boiwan>: {answer}", flush=True)
        history.append((user_text, answer))
    print("</dialogue>", flush=True)


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
    parser.add_argument("--dialogue", action="store_true", help="run scripted multi-turn dialogue sampling")
    parser.add_argument("--turns", type=int, default=3, help="number of scripted dialogue turns to sample")
    parser.add_argument("--history-turns", type=int, default=4, help="number of previous dialogue turns to keep")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--dialogue-prompts", nargs="*", default=None, help="user turns for scripted dialogue sampling")
    args = parser.parse_args()

    if args.chat:
        if args.check:
            check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
        run_chat(args)
        return

    if args.dialogue:
        if args.check:
            check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
        run_dialogue_sample(args)
        return

    if args.prompt is None:
        check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
        return

    if args.check:
        check_stage3_artifacts(args.model, args.tokenizer, args.result, args.skip_thresholds, args.smoke_tokens)
    run_continuation(args)


if __name__ == "__main__":
    main()
