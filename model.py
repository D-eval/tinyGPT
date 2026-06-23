from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


GPT_CONFIG_KEYS = ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")


def checkpoint_model_config(checkpoint: dict) -> dict:
    config = checkpoint["config"]
    return {key: config[key] for key in GPT_CONFIG_KEYS}


def checkpoint_model_state(checkpoint: dict, model: "TinyGPT") -> dict:
    expected = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for key, value in checkpoint["model_state"].items():
        if key not in expected:
            continue
        target = expected[key]
        if value.shape == target.shape:
            loaded[key] = value
            continue
        if (
            key in {"token_emb.weight", "head.weight"}
            and value.ndim == 2
            and target.ndim == 2
            and value.shape[1] == target.shape[1]
        ):
            resized = target.clone()
            keep = min(value.shape[0], target.shape[0])
            resized[:keep] = value[:keep]
            loaded[key] = resized
    return loaded


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        if steps > self.block_size:
            raise ValueError(f"Cannot attend sequence of length {steps}; block_size is {self.block_size}")
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16000,
        block_size: int = 256,
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 4,
    ) -> None:
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
        }
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, steps = idx.shape
        if steps > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {steps}; block_size is {self.block_size}")
        pos = torch.arange(steps, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 120, temperature: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


@dataclass(frozen=True)
class Stage1Config:
    vocab_size: int = 1024
    block_size: int = 10_000
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    batch_size: int = 8
    micro_batch_size: int = 1


def _assert_decoder_dense_architecture(model: TinyGPT) -> None:
    assert isinstance(model.token_emb, nn.Embedding)
    assert isinstance(model.pos_emb, nn.Embedding)
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == model.config["n_layer"]
    assert isinstance(model.ln_f, nn.LayerNorm)
    assert isinstance(model.head, nn.Linear)
    assert model.head.weight is model.token_emb.weight

    for block in model.blocks:
        assert isinstance(block, Block)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.attn.qkv, nn.Linear)
        assert isinstance(block.attn.proj, nn.Linear)
        assert block.attn.block_size == model.block_size
        assert isinstance(block.mlp[0], nn.Linear)
        assert isinstance(block.mlp[2], nn.Linear)


def _test_forward_shapes() -> None:
    model = TinyGPT(vocab_size=97, block_size=16, n_embd=32, n_head=4, n_layer=2)
    idx = torch.randint(0, 97, (3, 16))
    targets = torch.randint(0, 97, (3, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, 97)
    assert loss is not None and loss.ndim == 0
    try:
        model(torch.randint(0, 97, (1, 17)))
    except ValueError:
        pass
    else:
        raise AssertionError("TinyGPT must reject sequences longer than block_size")


def _test_causal_mask() -> None:
    torch.manual_seed(7)
    model = TinyGPT(vocab_size=64, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    changed_future = idx.clone()
    changed_future[:, 4:] = torch.randint(0, 64, (1, 4))

    with torch.no_grad():
        logits, _ = model(idx)
        changed_logits, _ = model(changed_future)

    if not torch.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-5, rtol=1e-5):
        raise AssertionError("Changing future tokens changed earlier logits; causal mask is broken")


def _stage1_dummy_train() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available; skipped 10k-context VRAM assertion.")
        return

    cfg = Stage1Config()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = TinyGPT(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
    ).to(device)
    _assert_decoder_dense_architecture(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), device=device)
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), device=device)

    optimizer.zero_grad(set_to_none=True)
    used_micro_batches = False
    loss_value = 0.0
    try:
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        loss_value = float(loss.detach().cpu())
    except torch.cuda.OutOfMemoryError:
        used_micro_batches = True
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        losses = []
        for start in range(0, cfg.batch_size, cfg.micro_batch_size):
            end = start + cfg.micro_batch_size
            _, loss = model(idx[start:end], targets[start:end])
            assert loss is not None
            (loss / (cfg.batch_size / cfg.micro_batch_size)).backward()
            losses.append(float(loss.detach().cpu()))
        loss_value = sum(losses) / len(losses)
    optimizer.step()
    torch.cuda.synchronize(device)

    total = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    allocated_pct = peak_allocated / total * 100
    reserved_pct = peak_reserved / total * 100
    print(
        "stage1 dummy train: "
        f"block_size={cfg.block_size} batch_size={cfg.batch_size} "
        f"micro_batch_size={cfg.micro_batch_size if used_micro_batches else cfg.batch_size} "
        f"n_embd={cfg.n_embd} n_head={cfg.n_head} n_layer={cfg.n_layer} "
        f"loss={loss_value:.4f}"
    )
    print(
        "stage1 cuda memory: "
        f"peak_allocated={peak_allocated / 1024**3:.2f}GiB ({allocated_pct:.1f}%) "
        f"peak_reserved={peak_reserved / 1024**3:.2f}GiB ({reserved_pct:.1f}%)"
    )
    assert peak_allocated >= total * 0.50, "10k-context dummy training did not exceed 50% GPU memory"


def _run_model_tests() -> None:
    _test_forward_shapes()
    _test_causal_mask()
    stage1_model = TinyGPT(
        vocab_size=Stage1Config.vocab_size,
        block_size=Stage1Config.block_size,
        n_embd=Stage1Config.n_embd,
        n_head=Stage1Config.n_head,
        n_layer=Stage1Config.n_layer,
    )
    _assert_decoder_dense_architecture(stage1_model)
    _stage1_dummy_train()
    print("model.py tests passed.")


if __name__ == "__main__":
    _run_model_tests()
