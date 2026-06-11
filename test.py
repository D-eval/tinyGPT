from __future__ import annotations

import argparse
from pathlib import Path

import torch

from read import BPETokenizer
from model import TinyGPT, checkpoint_model_config, checkpoint_model_state
from train import MODEL_PATH, TOKENIZER_PATH, choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Use the trained tiny GPT to continue text.")
    parser.add_argument("prompt", nargs="?", default="我走进房间，看见", help="text prompt")
    parser.add_argument("--tokens", type=int, default=120, help="number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    args = parser.parse_args()

    if not Path(MODEL_PATH).exists():
        raise SystemExit("params/model.pt not found. Run python3 train.py first.")
    if not Path(TOKENIZER_PATH).exists():
        raise SystemExit("params/tokenizer.json not found. Run python3 train.py first.")

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = TinyGPT(**checkpoint_model_config(checkpoint))
    model.load_state_dict(checkpoint_model_state(checkpoint, model), strict=False)
    device = choose_device()
    model.to(device)

    ids = tokenizer.encode(args.prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature)
    text = tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
