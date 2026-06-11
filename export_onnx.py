from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import torch
from torch.nn import functional as F

from model import CausalSelfAttention
from test2 import load_tinygpt
from train2 import MODEL_PATH, TOKENIZER_PATH


class TinyGPTOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def patch_attention_for_onnx(model: torch.nn.Module) -> None:
    def forward(self: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        q = qkv[:, :, :channels]
        k = qkv[:, :, channels : 2 * channels]
        v = qkv[:, :, 2 * channels :]
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones((steps, steps), dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        y = F.softmax(scores, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(y)

    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.forward = types.MethodType(forward, module)


def export_onnx(
    model_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    metadata_path: Path | None,
    dummy_seq_len: int,
    opset: int,
) -> None:
    device = torch.device("cpu")
    tokenizer, model, config = load_tinygpt(model_path, tokenizer_path, device)
    patch_attention_for_onnx(model)

    wrapper = TinyGPTOnnxWrapper(model).eval()
    dummy_seq_len = max(1, min(dummy_seq_len, int(config["block_size"])))
    dummy = torch.full((1, dummy_seq_len), tokenizer.bos_id, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "logits": {1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    metadata = {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "onnx_path": str(output_path),
        "input_name": "input_ids",
        "output_name": "logits",
        "vocab_size": int(config["vocab_size"]),
        "block_size": int(config["block_size"]),
        "bos_id": int(tokenizer.bos_id),
        "eos_id": int(tokenizer.eos_id),
        "pad_id": int(tokenizer.pad_id),
        "unk_id": int(tokenizer.unk_id),
        "opset": opset,
    }
    if metadata_path is not None:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {output_path}")
    if metadata_path is not None:
        print(f"wrote {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the TinyGPT checkpoint used by test2.py to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=Path("tinygpt.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("tinygpt.onnx.json"))
    parser.add_argument("--dummy-seq-len", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        metadata_path=args.metadata,
        dummy_seq_len=args.dummy_seq_len,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
