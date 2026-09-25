"""Generates the notebooks in ../notebooks from plain-text cells (keeps them diff-friendly).
Run:  python tools/make_notebooks.py
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "notebooks")
os.makedirs(OUT, exist_ok=True)

SETUP = r'''# @title Setup — clone repo (if needed), install deps, detect GPU
import os, sys, subprocess, json, time, math
REPO_URL = "https://github.com/swatibansal/reversible-llm-poc.git"

if not os.path.exists("src/revllm.py"):
    if os.path.exists("../src/revllm.py"):
        os.chdir("..")
    else:
        subprocess.run(["git", "clone", "-q", REPO_URL, "reversible-llm-poc"], check=True)
        os.chdir("reversible-llm-poc")
sys.path.insert(0, os.path.abspath("src"))
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "tiktoken", "datasets", "matplotlib"], check=False)

import torch
from revllm import Config, GPT, SavedTensorMeter
from data import prepare_tinystories, prepare_synthetic, TokenStream
import train as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SMOKE mode = tiny synthetic run that finishes in ~1 min on CPU. Auto-enabled when there is no GPU.
SMOKE = os.environ.get("SMOKE", "0") == "1" or DEVICE == "cpu"
print("device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "| SMOKE mode:", SMOKE)
os.makedirs("results", exist_ok=True)
'''

DRIVE = r'''# @title Persist results/ and data/ to Google Drive (Colab only)
# Surviving session resets matters twice here: results/*.json are the deliverable,
# and the tokenized TinyStories memmaps take ~4 min to rebuild from scratch.
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    drive.mount("/content/drive")
    for name in ("results", "data"):
        target = f"/content/drive/MyDrive/revllm/{name}"
        os.makedirs(target, exist_ok=True)
        os.system(f"rm -rf {name} && ln -s {target} {name}")
    print("results/ and data/ now live on Drive (MyDrive/revllm/)")
else:
    print("not running on Colab - keeping local results/ and data/")
'''

CONFIG = r'''# @title Experiment configuration (shared by all notebooks)
if SMOKE:
    MODEL  = dict(vocab_size=512, block_size=64, n_layer=4, n_embd=128, n_head=4)
    TOKENS = 200_000          # token budget
    BATCH  = 16               # the fixed batch size for notebooks 01/02
    LR     = 2e-3
    LOG    = dict(eval_every=100, eval_iters=5, log_every=50)
else:
    # ~20.8M parameters (12.9M in the tied GPT-2 embedding, 7.9M in 10 transformer blocks of width 256)
    MODEL  = dict(vocab_size=50257, block_size=256, n_layer=10, n_embd=256, n_head=4)
    TOKENS = 50_000_000
    BATCH  = 32               # 32 x 256 = 8,192 tokens / step  -> ~6,100 steps for 50M tokens
    LR     = 6e-4
    LOG    = dict(eval_every=250, eval_iters=20, log_every=50)

def get_data():
    if SMOKE:
        return prepare_synthetic("data/synthetic", 2_000_000, 200_000, vocab=MODEL["vocab_size"])
    return prepare_tinystories("data/tinystories", n_train_tokens=55_000_000, n_val_tokens=2_000_000)

def gpu_table(rows, headers):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    print(line(headers)); print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows: print(line(r))

import matplotlib.pyplot as plt
def plot_runs(results, key="curve", title="training loss", smooth=25):
    plt.figure(figsize=(8, 4.5))
    for r in results:
        c = r[key]
        if not c: continue
        xs = [p[1] / 1e6 for p in c]; ys = [p[2] for p in c]
        if key == "curve" and smooth > 1 and len(ys) > smooth:
            ys = [sum(ys[max(0, i - smooth):i + 1]) / len(ys[max(0, i - smooth):i + 1]) for i in range(len(ys))]
        plt.plot(xs, ys, label=f"{r['run_name']}  (final {ys[-1]:.3f})")
    plt.xlabel("tokens seen (M)"); plt.ylabel("cross-entropy (nats)"); plt.title(title); plt.legend(); plt.grid(alpha=.3)
    plt.show()
'''

# ---------------------------------------------------------------- 00 sanity checks
nb0 = [
("md", r'''# 00 — Sanity checks: is the reversible backward *really* exact and memory-free?

Before training anything, this notebook verifies the three claims that everything else rests on:

1. **Reversibility is exact.** Running the layers forward then `inverse()` returns the input (error ≈ float round-off).
2. **The memory-free backward gives the same gradients** as ordinary autograd on the same architecture.
3. **Activation memory is constant in depth** for reversible models and linear for the baseline (paper Fig. 3).

We also measure the price: how much slower is one training step when activations are recomputed instead of stored?'''),
("code", SETUP), ("code", DRIVE), ("code", CONFIG),
("md", "## 1 & 2 — exact inverse and identical gradients (all three reversible variants)"),
("code", r'''import copy
torch.manual_seed(0)
small = dict(MODEL, n_layer=4)
x = torch.randint(0, small["vocab_size"], (2, small["block_size"]), device=DEVICE)
y = torch.randint(0, small["vocab_size"], (2, small["block_size"]), device=DEVICE)
rows = []
for mode in ["midpoint", "leapfrog", "hamiltonian"]:
    m_rev = GPT(Config(mode=mode, rev_backprop=True, **small)).to(DEVICE)
    m_std = copy.deepcopy(m_rev); m_std.layers.rev_backprop = False
    l_rev = m_rev(x, y); l_rev.backward()
    l_std = m_std(x, y); l_std.backward()
    gdiff = max((a.grad - b.grad).abs().max().item() for a, b in zip(m_rev.parameters(), m_std.parameters()) if a.grad is not None and b.grad is not None)
    with torch.no_grad():
        p0 = m_rev.wte(x) + m_rev.wpe(torch.arange(x.shape[1], device=DEVICE))
        y1, y2 = m_rev.layers(p0, p0)
        r1, r2 = m_rev.layers.inverse(y1, y2)
        recon = max((r1 - p0).abs().max().item(), (r2 - p0).abs().max().item())
    rows.append([mode, f"{l_rev.item():.6f}", f"{l_std.item():.6f}", f"{gdiff:.1e}", f"{recon:.1e}"])
gpu_table(rows, ["variant", "loss (rev backprop)", "loss (std autograd)", "max |grad diff|", "reconstruction err"])
print("\n(gradient differences of ~1e-8 are float32 round-off; anything > 1e-4 would indicate a bug)")'''),
("md", r'''## 3 — activation memory vs depth

`SavedTensorMeter` counts the bytes autograd stashes for the backward pass — a platform-independent measure of activation memory. On a GPU we also read `torch.cuda.max_memory_allocated()` for a full forward+backward. Depth grows, batch stays fixed.'''),
("code", r'''depths = [2, 4, 8, 16, 32] if not SMOKE else [2, 4, 8]
B, Tn = (8, MODEL["block_size"])
rows, saved, peak = [], {}, {}
for L in depths:
    for mode, rev in [("baseline", True), ("midpoint", False), ("midpoint", True)]:
        name = mode + ("" if mode == "baseline" else ("/rev" if rev else "/no-rev"))
        cfg = Config(mode=mode, rev_backprop=rev, head_chunk=0, **dict(MODEL, n_layer=L))
        m = GPT(cfg).to(DEVICE)
        xb = torch.randint(0, cfg.vocab_size, (B, Tn), device=DEVICE)
        if DEVICE == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with SavedTensorMeter() as sm:
            loss = m(xb, xb)
        loss.backward()
        if DEVICE == "cuda": torch.cuda.synchronize(); pk = torch.cuda.max_memory_allocated() / 1e9
        else: pk = float("nan")
        saved.setdefault(name, []).append(sm.bytes / 1e6); peak.setdefault(name, []).append(pk)
        rows.append([L, name, f"{sm.bytes/1e6:.1f}", f"{pk:.2f}" if pk == pk else "n/a"])
        del m, loss, xb
        if DEVICE == "cuda": torch.cuda.empty_cache()
gpu_table(rows, ["layers", "model", "saved-for-backward (MB)", "GPU peak fwd+bwd (GB)"])

fig, ax = plt.subplots(1, 2 if DEVICE == "cuda" else 1, figsize=(11, 4)); ax = ax if isinstance(ax, (list, tuple)) or hasattr(ax, "__len__") else [ax]
for name in saved:
    ax[0].plot(depths, saved[name], marker="o", label=name)
ax[0].set_xlabel("layers"); ax[0].set_ylabel("MB saved for backward"); ax[0].set_title("activation memory vs depth"); ax[0].legend(); ax[0].grid(alpha=.3)
if DEVICE == "cuda":
    for name in peak: ax[1].plot(depths, peak[name], marker="o", label=name)
    ax[1].set_xlabel("layers"); ax[1].set_ylabel("GB (peak, fwd+bwd, B=%d)" % B); ax[1].set_title("GPU peak memory vs depth"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/memory_vs_depth.png", dpi=120); plt.show()
json.dump({"depths": depths, "saved_MB": saved, "peak_GB": peak, "batch": B, "block": Tn, "device": DEVICE},
          open("results/memory_vs_depth.json", "w"), indent=1)'''),
("md", r'''## 4 — the price of not storing activations: step-time overhead

Same midpoint architecture, one full training step, with and without memory-free backward. The paper says recomputation "typically adds only 30–50% overhead" on GPU (§2.2).'''),
("code", r'''def time_step(cfg, B, n=10):
    ctx, scaler, _ = T._amp(DEVICE)
    m = GPT(cfg).to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xb = torch.randint(0, cfg.vocab_size, (B, cfg.block_size), device=DEVICE)
    ts = []
    for i in range(n + 3):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with ctx: loss = m(xb, xb)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if DEVICE == "cuda": torch.cuda.synchronize()
        if i >= 3: ts.append(time.time() - t0)
    return sum(ts) / len(ts)

Bt = 8 if SMOKE else 32
rows = []
base = time_step(Config(mode="baseline", **MODEL), Bt)
for mode in ["midpoint", "leapfrog", "hamiltonian"]:
    t_std = time_step(Config(mode=mode, rev_backprop=False, **MODEL), Bt)
    t_rev = time_step(Config(mode=mode, rev_backprop=True, **MODEL), Bt)
    rows.append([mode, f"{t_std*1000:.0f}", f"{t_rev*1000:.0f}", f"+{(t_rev/t_std-1)*100:.0f}%", f"{t_rev/base:.2f}x"])
gpu_table(rows, ["variant", "step ms (std autograd)", "step ms (rev backprop)", "overhead", "vs baseline step"])
print(f"\nbaseline step: {base*1000:.0f} ms at batch {Bt}")'''),
]

# ---------------------------------------------------------------- 01 baseline
nb1 = [
("md", r'''# 01 — Baseline: ordinary GPT, fixed batch size

Trains the standard (non-reversible) 20M-parameter GPT for 50M tokens of TinyStories at a **fixed batch size of 32 × 256 tokens**.
This is the reference point every other run is compared against. Expected time on a Colab T4: ~25–40 min.

Results are written to `results/baseline_B32.json` (loss curve, tokens/s, peak memory, config).'''),
("code", SETUP), ("code", DRIVE), ("code", CONFIG),
("code", r'''train_bin, val_bin = get_data()
cfg = Config(mode="baseline", **MODEL)
res, model = T.train(cfg, train_bin, val_bin, batch_size=BATCH, tokens_budget=TOKENS, lr=LR,
                     out_json=f"results/baseline_B{BATCH}.json", **LOG)'''),
("code", r'''plot_runs([res], title="baseline — training loss")
plot_runs([res], key="val_curve", title="baseline — validation loss")
print(json.dumps({k: v for k, v in res.items() if k not in ("curve", "val_curve", "config")}, indent=1))'''),
("md", "## Sample generation (sanity check that it learned *something*)"),
("code", r'''if not SMOKE:
    import tiktoken; enc = tiktoken.get_encoding("gpt2")
    prompt = torch.tensor([enc.encode("Once upon a time")], device=DEVICE)
    model.eval()
    with torch.no_grad(), T._amp(DEVICE)[0]:
        out = model.generate(prompt, 80, temperature=0.8, top_k=50)
    print(enc.decode(out[0].tolist()))'''),
]

# ---------------------------------------------------------------- 02 reversible variants
nb2 = [
("md", r'''# 02 — Reversible variants at the *same* fixed batch size

Same model size, same data, same 50M-token budget, same batch (32 × 256) as the baseline — only the layer-to-layer update rule changes,
and the backward pass reconstructs activations instead of storing them.

| variant | update rule | paper eq. |
|---|---|---|
| `midpoint` | p_{l+1} = p_{l-1} + 2h·f(p_l) | 2.4 |
| `leapfrog` | p_{l+1} = 2p_l − p_{l-1} + h²·f(p_l) | 2.6 |
| `hamiltonian` | q_l = q_{l-1} + Attn(p_{l-1}); p_l = p_{l-1} + MLP(q_l)  (symplectic Euler) | 2.8–2.9 |

Each run takes roughly 1.3–1.6× the baseline time (recompute overhead). Results are saved after **each** variant, so if Colab
disconnects you can restart with a shorter `VARIANTS` list. Expected time on T4: ~40–55 min per variant.'''),
("code", SETUP), ("code", DRIVE), ("code", CONFIG),
("code", r'''VARIANTS = ["midpoint", "leapfrog", "hamiltonian"]     # remove entries here to resume a partial run
H = {"midpoint": 0.5, "leapfrog": 1.0, "hamiltonian": 1.0}  # midpoint uses 2h -> 2h=1 matches the baseline's residual scale

train_bin, val_bin = get_data()
results = []
for mode in VARIANTS:
    cfg = Config(mode=mode, h=H[mode], rev_backprop=True, **MODEL)
    res, _ = T.train(cfg, train_bin, val_bin, batch_size=BATCH, tokens_budget=TOKENS, lr=LR,
                     out_json=f"results/{mode}_B{BATCH}.json", **LOG)
    results.append(res)
    if DEVICE == "cuda": torch.cuda.empty_cache()'''),
("md", r'''## Overhead isolation: same architecture, activations *stored* instead of recomputed

A short run (400 steps) of `midpoint` with `rev_backprop=False`. Its loss curve must match the reversible one step-for-step
(same seed, same data order) — proving the memory-free backward changes *only* memory and time, never the math.'''),
("code", r'''steps_short = 400 if not SMOKE else 40
cfg = Config(mode="midpoint", h=H["midpoint"], rev_backprop=False, **MODEL)
res_norev, _ = T.train(cfg, train_bin, val_bin, batch_size=BATCH, tokens_budget=steps_short * BATCH * MODEL["block_size"], lr=LR,
                       out_json=f"results/midpoint-norev_B{BATCH}_short.json", run_name="midpoint-norev_short", eval_every=10**9, eval_iters=1, log_every=100)
cfg = Config(mode="midpoint", h=H["midpoint"], rev_backprop=True, **MODEL)
res_rev_short, _ = T.train(cfg, train_bin, val_bin, batch_size=BATCH, tokens_budget=steps_short * BATCH * MODEL["block_size"], lr=LR,
                       out_json=f"results/midpoint-rev_B{BATCH}_short.json", run_name="midpoint-rev_short", eval_every=10**9, eval_iters=1, log_every=100)
d = max(abs(a[2] - b[2]) for a, b in zip(res_norev["curve"], res_rev_short["curve"]))
print(f"\nmax |loss difference| over {steps_short} steps: {d:.2e}   (fp16 autocast -> expect ~1e-3 or less)")
print(f"tokens/s  stored activations: {res_norev['tokens_per_s_steady']:,.0f}   recomputed: {res_rev_short['tokens_per_s_steady']:,.0f}"
      f"   -> recompute overhead {(res_norev['tokens_per_s_steady']/res_rev_short['tokens_per_s_steady']-1)*100:.0f}%")
if DEVICE == "cuda":
    print(f"peak GB   stored activations: {res_norev['peak_mem_gb']:.2f}   recomputed: {res_rev_short['peak_mem_gb']:.2f}")'''),
("md", "## Compare with the baseline (run notebook 01 first)"),
("code", r'''allres = []
for f in sorted(os.listdir("results")):
    if f.endswith(f"_B{BATCH}.json"):
        allres.append(json.load(open("results/" + f)))
rows = [[r["run_name"], f"{r['final_train_loss']:.4f}", f"{r['final_val_loss']:.4f}", f"{r['tokens_per_s_steady']:,.0f}",
         f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] else "n/a", f"{r['wall_time_s']/60:.1f}"] for r in allres]
gpu_table(rows, ["run", "train loss", "val loss", "tokens/s", "peak GB", "minutes"])
plot_runs(allres, title=f"training loss @ batch {BATCH}")
plot_runs(allres, key="val_curve", title=f"validation loss @ batch {BATCH}")
plt.savefig("results/loss_curves_fixed_batch.png", dpi=120)'''),
]

# ---------------------------------------------------------------- 03 max batch
nb3 = [
("md", r'''# 03 — Push the reversible model to the maximum batch size

The paper's headline (Table 3): with the same GPU, a reversible model fits ~10× the batch of the baseline, and throughput goes up because
GPUs are more efficient on bigger batches (Table 4). Here we:

1. **Probe** the largest batch that survives a full forward+backward+optimizer step, for the baseline and for the reversible model
   (doubling, then binary search).
2. **Train** the reversible model for the same 50M tokens at (≈90% of) its max batch, with the learning rate scaled by √(B/32).
3. Compare loss / tokens-per-second / peak memory against the fixed-batch runs.

> One thing to know before reading the numbers: with only 20M parameters and a 50k-word vocabulary, the *output layer's logits*
> (`batch × 256 × 50257` numbers) are a large share of memory — and reversibility does nothing about them. We use a chunked,
> checkpointed LM head for **all** models (baseline included) so that this doesn't hide the effect of reversibility on the transformer stack.'''),
("code", SETUP), ("code", DRIVE), ("code", CONFIG),
("md", "## 1 — maximum batch that fits"),
("code", r'''BEST = "midpoint"        # pick the variant that did best in notebook 02
H = {"midpoint": 0.5, "leapfrog": 1.0, "hamiltonian": 1.0}
probe = {}
if DEVICE == "cuda":
    for name, cfg in [("baseline", Config(mode="baseline", **MODEL)),
                      (f"{BEST}/no-rev", Config(mode=BEST, h=H[BEST], rev_backprop=False, **MODEL)),
                      (f"{BEST}/rev", Config(mode=BEST, h=H[BEST], rev_backprop=True, **MODEL))]:
        print(name)
        b, pk = T.find_max_batch(cfg, start=16)
        probe[name] = {"max_batch": b, "peak_gb": pk}
    gpu_table([[k, v["max_batch"], f"{v['peak_gb']:.2f}", f"{v['max_batch']/probe['baseline']['max_batch']:.1f}x"] for k, v in probe.items()],
              ["model", "max batch (x%d tokens)" % MODEL["block_size"], "peak GB at max", "vs baseline"])
    json.dump({"gpu": torch.cuda.get_device_name(0), "total_gb": torch.cuda.get_device_properties(0).total_memory/1e9, "probe": probe},
              open("results/max_batch_probe.json", "w"), indent=1)
    MAXB = int(probe[f"{BEST}/rev"]["max_batch"] * 0.9) // 8 * 8      # 10% headroom for fragmentation
else:
    print("no GPU — SMOKE: pretending max batch is 64"); MAXB = 64
print("training batch for the max-batch run:", MAXB)'''),
("md", r'''## 2 — train the reversible model at max batch, same 50M-token budget

Fewer, bigger steps. LR is scaled by √(B/32) (square-root scaling rule) and capped at 3e-3.'''),
("code", r'''train_bin, val_bin = get_data()
lr_big = min(LR * math.sqrt(MAXB / BATCH), 3e-3)
cfg = Config(mode=BEST, h=H[BEST], rev_backprop=True, **MODEL)
res_big, model = T.train(cfg, train_bin, val_bin, batch_size=MAXB, tokens_budget=TOKENS, lr=lr_big,
                         out_json=f"results/{BEST}_maxbatch_B{MAXB}.json",
                         eval_every=max(10, LOG["eval_every"] * BATCH // MAXB), eval_iters=LOG["eval_iters"], log_every=max(1, 50 * BATCH // MAXB))'''),
("md", r'''### Optional: baseline at *its* max batch too (fair "each model at its own limit" comparison)
Set `ALSO_BASELINE_MAX = True` to run it (another ~25–40 min on T4).'''),
("code", r'''ALSO_BASELINE_MAX = False
if ALSO_BASELINE_MAX and DEVICE == "cuda":
    Bb = int(probe["baseline"]["max_batch"] * 0.9) // 8 * 8
    cfgb = Config(mode="baseline", **MODEL)
    T.train(cfgb, train_bin, val_bin, batch_size=Bb, tokens_budget=TOKENS, lr=min(LR * math.sqrt(Bb / BATCH), 3e-3),
            out_json=f"results/baseline_maxbatch_B{Bb}.json",
            eval_every=max(10, LOG["eval_every"] * BATCH // Bb), eval_iters=LOG["eval_iters"], log_every=max(1, 50 * BATCH // Bb))'''),
("md", "## 3 — everything side by side"),
("code", r'''allres = [json.load(open("results/" + f)) for f in sorted(os.listdir("results")) if f.endswith(".json")]
allres = [r for r in allres if "curve" in r and not r["run_name"].endswith("_short")]
rows = [[r["run_name"], r["batch_size"], r["steps"], f"{r['final_train_loss']:.4f}", f"{r['final_val_loss']:.4f}",
         f"{r['tokens_per_s_steady']:,.0f}", f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] else "n/a", f"{r['wall_time_s']/60:.1f}"] for r in allres]
gpu_table(rows, ["run", "batch", "steps", "train loss", "val loss", "tokens/s", "peak GB", "minutes"])
plot_runs(allres, title="training loss — all runs (x axis = tokens, so batch sizes are comparable)")
plot_runs(allres, key="val_curve", title="validation loss — all runs")
plt.savefig("results/loss_curves_all.png", dpi=120)'''),
]

# ---------------------------------------------------------------- 04 report
nb4 = [
("md", r'''# 04 — Collect results into a Markdown report

Reads every `results/*.json`, prints the summary table, saves `results/summary.md` and the comparison plots.
Paste the table into the README's "Results" section.'''),
("code", SETUP), ("code", DRIVE), ("code", CONFIG),
("code", r'''runs = []
for f in sorted(os.listdir("results")):
    if not f.endswith(".json"): continue
    r = json.load(open("results/" + f))
    if "curve" in r and not r["run_name"].endswith("_short"): runs.append(r)
lines = ["| run | mode | rev backprop | batch | steps | final train loss | final val loss | tokens/s | peak GB | minutes |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in runs:
    c = r["config"]
    lines.append(f"| {r['run_name']} | {c['mode']} | {'yes' if c['mode']!='baseline' and c['rev_backprop'] else 'no'} | {r['batch_size']} | {r['steps']} | "
                 f"{r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | {r['tokens_per_s_steady']:,.0f} | "
                 f"{r['peak_mem_gb']:.2f} | {r['wall_time_s']/60:.1f} |" if r["peak_mem_gb"] else
                 f"| {r['run_name']} | {c['mode']} | {'yes' if c['mode']!='baseline' and c['rev_backprop'] else 'no'} | {r['batch_size']} | {r['steps']} | "
                 f"{r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | {r['tokens_per_s_steady']:,.0f} | n/a | {r['wall_time_s']/60:.1f} |")
hdr = [f"**GPU:** {runs[0]['gpu']}  **dtype:** {runs[0]['dtype']}  **params:** {runs[0]['n_params']/1e6:.2f}M  **tokens/run:** {runs[0]['tokens_seen']/1e6:.0f}M", ""] if runs else []
if os.path.exists("results/max_batch_probe.json"):
    p = json.load(open("results/max_batch_probe.json"))
    hdr += ["**Max batch probe** (%s, %.0f GB):" % (p["gpu"], p["total_gb"]), "", "| model | max batch | peak GB |", "|---|---|---|"]
    hdr += [f"| {k} | {v['max_batch']} | {v['peak_gb']:.2f} |" for k, v in p["probe"].items()] + [""]
md = "\n".join(hdr + lines)
open("results/summary.md", "w").write(md); print(md)
plot_runs(runs, title="training loss — all runs"); plt.savefig("results/loss_curves_all.png", dpi=120)
plot_runs(runs, key="val_curve", title="validation loss — all runs"); plt.savefig("results/val_curves_all.png", dpi=120)'''),
("code", r'''# On Colab: download the results folder so you can commit it to the repo
try:
    from google.colab import files
    import shutil; shutil.make_archive("results", "zip", "results"); files.download("results.zip")
except Exception as e:
    print("not on Colab (or download skipped):", e)'''),
]


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    path = os.path.join(OUT, name)
    nbf.write(nb, path); print("wrote", path)


build("00_sanity_checks.ipynb", nb0)
build("01_baseline_fixed_batch.ipynb", nb1)
build("02_reversible_variants_fixed_batch.ipynb", nb2)
build("03_reversible_max_batch.ipynb", nb3)
build("04_report.ipynb", nb4)
