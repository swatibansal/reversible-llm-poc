"""
train.py — one function, `train(...)`, that trains a model for a fixed token budget and returns
a results dict (final loss, val loss, tokens/s, peak memory, loss curve, config). Plus
`find_max_batch(...)` which probes the largest batch that fits in GPU memory.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext

import torch

from revllm import GPT, Config
from data import TokenStream


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _amp(device):
    # T4 has no bf16 tensor cores -> fp16 + GradScaler. Ampere+ -> bf16 (no scaler needed).
    if device != "cuda":
        return nullcontext(), None, torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), None, torch.bfloat16
    return torch.autocast("cuda", dtype=torch.float16), torch.cuda.amp.GradScaler(), torch.float16


def lr_at(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (lr - min_lr)


@torch.no_grad()
def evaluate(model, stream, batch_size, iters, ctx, generator=None):
    model.eval(); losses = []
    for _ in range(iters):
        x, y = stream.get_batch(batch_size, generator)
        with ctx:
            losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: Config, train_bin: str, val_bin: str, *, batch_size: int, tokens_budget: int,
          lr: float = 6e-4, min_lr_ratio: float = 0.1, warmup_frac: float = 0.03, weight_decay: float = 0.1,
          grad_clip: float = 1.0, eval_every: int = 200, eval_iters: int = 20, log_every: int = 50,
          out_json: str | None = None, seed: int = 1337, run_name: str | None = None, compile: bool = False):
    device = _device()
    torch.manual_seed(seed)
    ctx, scaler, dtype = _amp(device)
    model = GPT(cfg).to(device)
    if compile and device == "cuda" and cfg.mode == "baseline":
        model = torch.compile(model)
    n_params = model.num_params()
    tokens_per_step = batch_size * cfg.block_size
    total_steps = max(1, tokens_budget // tokens_per_step)
    warmup = max(1, int(warmup_frac * total_steps))
    run_name = run_name or f"{cfg.mode}{'' if cfg.rev_backprop or cfg.mode=='baseline' else '-norev'}_B{batch_size}"
    print(f"[{run_name}] device={device} dtype={dtype} params={n_params/1e6:.2f}M  steps={total_steps}  "
          f"tokens/step={tokens_per_step}  budget={tokens_budget/1e6:.2f}M tokens")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    tr, va = TokenStream(train_bin, cfg.block_size, device), TokenStream(val_bin, cfg.block_size, device)
    g_eval = torch.Generator().manual_seed(0)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    curve, val_curve, step_times = [], [], []
    t_start = time.time(); tokens_seen = 0
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, warmup, total_steps, lr, lr * min_lr_ratio)
        x, y = tr.get_batch(batch_size)
        t0 = time.time()
        with ctx:
            loss = model(x, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda": torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        tokens_seen += tokens_per_step
        li = loss.item()
        curve.append((step, tokens_seen, li))
        if not math.isfinite(li):
            print(f"[{run_name}] step {step}: loss is {li} — diverged, stopping"); break
        if step % log_every == 0 or step == total_steps - 1:
            el = time.time() - t_start
            print(f"[{run_name}] step {step:5d}/{total_steps}  loss {li:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{tokens_seen/el:,.0f} tok/s  {el/60:.1f} min")
        if (step % eval_every == 0 and step > 0) or step == total_steps - 1:
            vl = evaluate(model, va, min(batch_size, 32), eval_iters, ctx, g_eval)
            val_curve.append((step, tokens_seen, vl))
            print(f"[{run_name}]   val loss {vl:.4f}")

    wall = time.time() - t_start
    # steady-state step time: drop first 10 steps (warm-up / allocator)
    ss = step_times[10:] if len(step_times) > 20 else step_times
    res = {
        "run_name": run_name, "config": cfg.to_dict(), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dtype": str(dtype),
        "n_params": n_params, "batch_size": batch_size, "tokens_per_step": tokens_per_step,
        "steps": len(curve), "tokens_seen": tokens_seen, "wall_time_s": wall,
        "tokens_per_s_overall": tokens_seen / wall,
        "tokens_per_s_steady": tokens_per_step / (sum(ss) / len(ss)),
        "step_time_s": sum(ss) / len(ss),
        "final_train_loss": sum(c[2] for c in curve[-20:]) / len(curve[-20:]),
        "final_val_loss": val_curve[-1][2] if val_curve else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else None,
        "lr": lr, "warmup_steps": warmup, "curve": curve, "val_curve": val_curve,
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f: json.dump(res, f)
        print(f"[{run_name}] saved -> {out_json}")
    print(f"[{run_name}] DONE  train {res['final_train_loss']:.4f}  val {res['final_val_loss']}  "
          f"{res['tokens_per_s_steady']:,.0f} tok/s  peak {res['peak_mem_gb']} GB  {wall/60:.1f} min")
    return res, model


def _try_batch(cfg: Config, b: int, device: str):
    """One full fwd+bwd+step at batch b. Returns peak GB or None on OOM."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ctx, scaler, _ = _amp(device)
    try:
        model = GPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        x = torch.randint(0, cfg.vocab_size, (b, cfg.block_size), device=device)
        for _ in range(2):
            with ctx:
                loss = model(x, x)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, opt, x, loss
        torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:  # cuBLAS/cuDNN sometimes report OOM as a generic RuntimeError
        if "out of memory" in str(e).lower() or "ALLOC_FAILED" in str(e):
            torch.cuda.empty_cache()
            return None
        raise


def find_max_batch(cfg: Config, start: int = 8, cap: int = 8192, verbose=True):
    """Doubling then binary search for the largest batch that survives fwd+bwd+step."""
    device = _device()
    assert device == "cuda", "max-batch probe needs a GPU"
    lo, lo_peak = 0, None
    b = start
    while b <= cap:
        peak = _try_batch(cfg, b, device)
        if verbose: print(f"  [{cfg.mode}{'' if cfg.rev_backprop or cfg.mode=='baseline' else '-norev'}] B={b:5d}  {'OOM' if peak is None else f'{peak:.2f} GB'}")
        if peak is None: break
        lo, lo_peak = b, peak
        b *= 2
    hi = b
    while hi - lo > max(1, lo // 16):          # ~6% resolution is plenty
        mid = (lo + hi) // 2
        peak = _try_batch(cfg, mid, device)
        if verbose: print(f"  [{cfg.mode}] B={mid:5d}  {'OOM' if peak is None else f'{peak:.2f} GB'}")
        if peak is None: hi = mid
        else: lo, lo_peak = mid, peak
    return lo, lo_peak
