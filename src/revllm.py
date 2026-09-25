"""
revllm.py — a minimal, readable implementation of the reversible LLM architectures from

    "Reversing Large Language Models for Efficient Training and Fine-Tuning"
    Gal, Eliasof, Turek, Ascher, Treister, Haber — arXiv:2512.02056

Four ways to stack transformer blocks are implemented. All share the *same* block
f_theta(p) = Attn(LN1(p)) + MLP(LN2(p + Attn(LN1(p))))   (paper eq. 2.5); they differ
only in how the residual stream is updated from layer to layer:

  mode         update rule (paper eq.)                             reversible?
  ---------    ---------------------------------------------------- -----------
  baseline     p_{l+1} = p_l + f_l(p_l)                        (2.2)   no
  midpoint     p_{l+1} = p_{l-1} + 2h f_l(p_l)                 (2.4)   yes
  leapfrog     p_{l+1} = 2 p_l - p_{l-1} + h^2 f_l(p_l)        (2.6)   yes
  hamiltonian  q_l = q_{l-1} + Attn_l(LN1(p_{l-1}))            (2.8)   yes
               p_l = p_{l-1} + MLP_l(LN2(q_l))                 (2.9)   ("symplectic Euler")

For the reversible modes, `ReversibleSequence` implements memory-free backpropagation:
the forward pass stores only the final pair of hidden states; during backward each
layer's *input* is reconstructed from its *output* with the inverse update rule, the
layer is re-run once with autograd on, and gradients are propagated. Activation memory
therefore does not grow with depth (paper §2.2, Fig. 3).

Set `rev_backprop=False` on a reversible model to run the identical architecture with
ordinary autograd (stores everything). Useful to verify that the memory-free backward
gives the same gradients, and to separate "architecture" effects from "memory" effects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    vocab_size: int = 50257        # GPT-2 tokenizer
    block_size: int = 256          # context length T
    n_layer: int = 10
    n_head: int = 4
    n_embd: int = 256
    mode: str = "baseline"         # baseline | midpoint | leapfrog | hamiltonian
    h: float = 1.0                 # step size. midpoint uses 2h, leapfrog uses h^2
    rev_backprop: bool = True      # memory-free backward for reversible modes
    tie_embeddings: bool = True
    head_chunk: int = 4096         # tokens per chunk in chunked (checkpointed) LM head; 0 = off
    bias: bool = False

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- block pieces
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.IS_RESIDUAL_PROJ = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.IS_RESIDUAL_PROJ = True

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """The shared transformer block. `f(p)` is paper eq. (2.5)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def f(self, p):
        a = self.attn(self.ln_1(p))
        return a + self.mlp(self.ln_2(p + a))


# ----------------------------------------------------------------------------- layer update rules
# Every reversible layer maps a pair of states (x1, x2) -> (y1, y2) and can invert it exactly.

class BaselineLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.block = Block(cfg)

    def forward(self, p):
        return p + self.block.f(p)


class MidpointLayer(nn.Module):
    """x1 = p_{l-1}, x2 = p_l  ->  y1 = p_l, y2 = p_{l+1} = p_{l-1} + 2h f(p_l)."""

    def __init__(self, cfg):
        super().__init__(); self.block = Block(cfg); self.h = cfg.h

    def forward(self, x1, x2):
        return x2, x1 + 2 * self.h * self.block.f(x2)

    def inverse(self, y1, y2):
        return y2 - 2 * self.h * self.block.f(y1), y1


class LeapfrogLayer(nn.Module):
    """x1 = p_{l-1}, x2 = p_l  ->  y1 = p_l, y2 = 2 p_l - p_{l-1} + h^2 f(p_l)."""

    def __init__(self, cfg):
        super().__init__(); self.block = Block(cfg); self.h = cfg.h

    def forward(self, x1, x2):
        return x2, 2 * x2 - x1 + (self.h ** 2) * self.block.f(x2)

    def inverse(self, y1, y2):
        return 2 * y1 - y2 + (self.h ** 2) * self.block.f(y1), y1


class HamiltonianLayer(nn.Module):
    """x1 = p_{l-1} (position), x2 = q_{l-1} (momentum). Symplectic-Euler staggered update:
         q_l = q_{l-1} + Attn(LN1(p_{l-1}))
         p_l = p_{l-1} + MLP(LN2(q_l))
    """

    def __init__(self, cfg):
        super().__init__(); self.block = Block(cfg)

    def forward(self, x1, x2):
        b = self.block
        q = x2 + b.attn(b.ln_1(x1))
        p = x1 + b.mlp(b.ln_2(q))
        return p, q

    def inverse(self, y1, y2):
        b = self.block
        p_new, q = y1, y2
        p = p_new - b.mlp(b.ln_2(q))
        q_prev = q - b.attn(b.ln_1(p))
        return p, q_prev


LAYER_TYPES = {"midpoint": MidpointLayer, "leapfrog": LeapfrogLayer, "hamiltonian": HamiltonianLayer}


# ----------------------------------------------------------------------------- memory-free backprop
class _RevBackprop(torch.autograd.Function):
    """Runs a list of reversible layers without storing intermediate activations.

    forward : (x1, x2) -> (y1, y2), executed under no_grad; only (y1, y2) are saved.
    backward: walk the layers in reverse. For layer l:
        1. (x1, x2) = layer.inverse(y1, y2)              # reconstruct input  (no_grad)
        2. re-run  (y1', y2') = layer(x1, x2)             # with grad enabled
        3. torch.autograd.backward((y1', y2'), (dy1, dy2)) # -> dx1, dx2, and param.grad
        4. (y1, y2, dy1, dy2) <- (x1, x2, dx1, dx2)
    """

    @staticmethod
    def forward(ctx, x1, x2, layers: List[nn.Module]):
        ctx.layers = layers
        ctx.autocast = (torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())
        with torch.no_grad():
            for layer in layers:
                x1, x2 = layer(x1, x2)
        ctx.save_for_backward(x1.detach(), x2.detach())
        return x1, x2

    @staticmethod
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensors
        enabled, dtype = ctx.autocast
        dev = "cuda" if y1.is_cuda else "cpu"
        if dy1 is None: dy1 = torch.zeros_like(y1)
        if dy2 is None: dy2 = torch.zeros_like(y2)
        for layer in reversed(ctx.layers):
            with torch.no_grad(), torch.autocast(dev, dtype=dtype, enabled=enabled):
                x1, x2 = layer.inverse(y1, y2)
            x1 = x1.detach().requires_grad_(True)
            x2 = x2.detach().requires_grad_(True)
            with torch.enable_grad(), torch.autocast(dev, dtype=dtype, enabled=enabled):
                o1, o2 = layer(x1, x2)
                torch.autograd.backward((o1, o2), (dy1, dy2))
            dy1, dy2 = x1.grad, x2.grad
            y1, y2 = x1.detach(), x2.detach()
            del o1, o2
        return dy1, dy2, None


class ReversibleSequence(nn.Module):
    def __init__(self, layers: List[nn.Module], rev_backprop: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.rev_backprop = rev_backprop

    def forward(self, x1, x2):
        if self.rev_backprop and torch.is_grad_enabled():
            return _RevBackprop.apply(x1, x2, list(self.layers))
        for layer in self.layers:            # ordinary autograd (stores activations)
            x1, x2 = layer(x1, x2)
        return x1, x2

    @torch.no_grad()
    def inverse(self, y1, y2):
        for layer in reversed(self.layers):
            y1, y2 = layer.inverse(y1, y2)
        return y1, y2


# ----------------------------------------------------------------------------- the model
class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        if cfg.mode == "baseline":
            self.layers = nn.ModuleList([BaselineLayer(cfg) for _ in range(cfg.n_layer)])
        else:
            L = LAYER_TYPES[cfg.mode]
            self.layers = ReversibleSequence([L(cfg) for _ in range(cfg.n_layer)], cfg.rev_backprop)
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for n, p in self.named_parameters():          # GPT-2 style scaled init of residual projections
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel() + (0 if self.cfg.tie_embeddings else self.lm_head.weight.numel()) + self.wte.weight.numel()
        return n

    # -- trunk: embeddings -> layers -> final hidden state
    def trunk(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        p0 = self.wte(idx) + self.wpe(pos)
        if self.cfg.mode == "baseline":
            p = p0
            for layer in self.layers:
                p = layer(p)
            return p
        # reversible modes need two initial states. midpoint/leapfrog: p_{-1} = p_0 (zero initial velocity)
        # hamiltonian: q_0 = p_0 (as in Reversible ViT, the input is duplicated into both streams)
        y1, y2 = self.layers(p0, p0)
        # midpoint/leapfrog carry (p_{l-1}, p_l): the newest state is y2 = p_L.
        # hamiltonian carries (p, q): the position stream is y1 = p_L.
        return y1 if self.cfg.mode == "hamiltonian" else y2

    # -- chunked LM head: computes LN_f -> logits -> CE per chunk of tokens under checkpoint,
    #    so the (B*T, vocab) logits tensor is never fully materialized. Applied to ALL modes.
    def _head_loss_sum(self, hflat, tflat):
        logits = self.lm_head(self.ln_f(hflat))
        return F.cross_entropy(logits.float(), tflat, reduction="sum", ignore_index=-1)

    def forward(self, idx, targets=None):
        h = self.trunk(idx)
        if targets is None:
            return self.lm_head(self.ln_f(h))
        hflat, tflat = h.reshape(-1, h.size(-1)), targets.reshape(-1)
        n_valid = (tflat != -1).sum().clamp(min=1)
        C = self.cfg.head_chunk
        if C <= 0 or hflat.size(0) <= C or not torch.is_grad_enabled():
            return self._head_loss_sum(hflat, tflat) / n_valid
        loss = 0.0
        for i in range(0, hflat.size(0), C):
            loss = loss + checkpoint(self._head_loss_sum, hflat[i:i + C], tflat[i:i + C], use_reentrant=False)
        return loss / n_valid

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------------------------------------------------------- measurement helpers
class SavedTensorMeter:
    """Counts bytes autograd saves for backward during a forward pass (platform independent)."""

    def __init__(self):
        self.bytes = 0

    def _pack(self, t):
        self.bytes += t.numel() * t.element_size(); return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__(); return self

    def __exit__(self, *a):
        self._ctx.__exit__(*a)


def make_model(**kw) -> Tuple[GPT, Config]:
    cfg = Config(**kw)
    return GPT(cfg), cfg
