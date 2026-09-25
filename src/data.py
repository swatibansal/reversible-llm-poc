"""
data.py — build a token stream for training.

prepare_tinystories(): streams roneneldan/TinyStories from the HF hub, tokenizes with the
GPT-2 BPE (tiktoken), writes train.bin / val.bin as uint16 memmaps (same format as nanoGPT).
prepare_synthetic():   CPU-only fallback (no downloads) — a random Markov-chain "language" so
                       smoke tests have a loss that actually goes down.
get_batch():           random contiguous windows from a memmap.
"""
from __future__ import annotations

import os
import numpy as np
import torch

EOT = 50256  # GPT-2 <|endoftext|>


def prepare_tinystories(out_dir: str, n_train_tokens: int = 55_000_000, n_val_tokens: int = 2_000_000,
                        batch_docs: int = 2000, verbose: bool = True):
    """Tokenize just enough of TinyStories to cover the requested budgets. Skips if files exist."""
    os.makedirs(out_dir, exist_ok=True)
    tr, va = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
    if os.path.exists(tr) and os.path.exists(va):
        n = os.path.getsize(tr) // 2
        if n >= n_train_tokens:
            if verbose: print(f"[data] found {tr} with {n/1e6:.1f}M tokens — reusing")
            return tr, va
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")

    def fill(split, target, path):
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(target,))
        i, buf = 0, []
        for ex in ds:
            buf.append(ex["text"])
            if len(buf) == batch_docs:
                for ids in enc.encode_ordinary_batch(buf):
                    ids.append(EOT)
                    k = min(len(ids), target - i)
                    arr[i:i + k] = np.asarray(ids[:k], dtype=np.uint16); i += k
                    if i >= target: break
                buf = []
                if verbose: print(f"\r[data] {split}: {i/1e6:.1f}M / {target/1e6:.1f}M tokens", end="")
                if i >= target: break
        arr.flush(); print()
        return path

    fill("train", n_train_tokens, tr)
    fill("validation", n_val_tokens, va)
    return tr, va


def prepare_synthetic(out_dir: str, n_train_tokens: int = 2_000_000, n_val_tokens: int = 200_000,
                      vocab: int = 512, seed: int = 0):
    """A toy 'language': the next token is one of 4 fixed successors of the previous token
    with probability 0.8, otherwise uniformly random (a noisy bigram grammar). A model that
    learns the rule drops from ln(vocab)=6.2 to ~2.3 nats — a visible curve, no download."""
    os.makedirs(out_dir, exist_ok=True)
    tr, va = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
    rng = np.random.default_rng(seed)
    table = rng.integers(0, vocab, size=(vocab, 4))

    def gen(n, s):
        r = np.random.default_rng(s)
        out = np.empty(n, dtype=np.uint16); out[0], out[1] = r.integers(0, vocab, 2)
        u = r.random(n); pick = r.integers(0, 4, n); noise = r.integers(0, vocab, n)
        for i in range(2, n):
            out[i] = table[out[i - 1], pick[i]] if u[i] < 0.8 else noise[i]
        return out

    gen(n_train_tokens, seed).tofile(tr); gen(n_val_tokens, seed + 1).tofile(va)
    return tr, va


class TokenStream:
    def __init__(self, path: str, block_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.T, self.device = block_size, device

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        ix = torch.randint(len(self.data) - self.T - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i:i + self.T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.T].astype(np.int64)) for i in ix])
        if self.device == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
