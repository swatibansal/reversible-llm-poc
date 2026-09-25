# CLAUDE.md — working instructions for Claude Code in this repo

## What this repo is

A proof-of-concept re-implementation of arXiv:2512.02056 ("Reversing Large Language Models for Efficient Training and
Fine-Tuning"). A ~20M-parameter GPT is trained four ways — `baseline`, `midpoint`, `leapfrog`, `hamiltonian` — for 50M
TinyStories tokens, and we measure loss, tokens/s and peak GPU memory. The heavy runs happen on **Google Colab (T4)**;
this machine is used for editing, CPU smoke tests, and turning `results/*.json` into the README tables.

Read `README.md` §1–3 first if you need the background. Everything below is operational.

## Layout

```
src/revllm.py      model + 4 layer-update rules + _RevBackprop (memory-free autograd) + chunked LM head + SavedTensorMeter
src/train.py       train(cfg, train_bin, val_bin, batch_size, tokens_budget, ...) -> (results dict, model); find_max_batch(cfg)
src/data.py        prepare_tinystories(), prepare_synthetic(), TokenStream
tools/make_notebooks.py   SOURCE OF TRUTH for notebooks/*.ipynb — edit this, then regenerate
notebooks/00..04   generated; 00 sanity, 01 baseline B=32, 02 reversible variants B=32, 03 max-batch, 04 report
results/           run outputs (*.json with loss curves, *.png, summary.md); results/cpu_smoke/ = verified CPU dry run
```

## Rules

- **Never edit `notebooks/*.ipynb` by hand.** Change `tools/make_notebooks.py`, then `python tools/make_notebooks.py`.
- Any change to `src/revllm.py` must keep these invariants (run the smoke test below to check):
  1. reversible modes with `rev_backprop=True` and `False` give identical loss and gradients (diff < 1e-6 fp32);
  2. `ReversibleSequence.inverse(forward(x))` reconstructs x to ~1e-7;
  3. `SavedTensorMeter` bytes for a reversible model do **not** grow with `n_layer`.
- Don't remove the chunked LM head or apply it to only some modes — it must be identical across variants or the memory
  comparison is meaningless (README §7, finding 2).
- Do not commit `data/` (token memmaps, ~110 MB) — it is gitignored. Do commit `results/*.json` and `*.png`.
- No dropout anywhere (reversibility needs deterministic `f`).
- Keep `SMOKE` mode working: every notebook must run end-to-end on CPU in ~1 min with `SMOKE=1`.
- Free Colab has no `git push` credentials; results come back as `results.zip` from notebook 04 and get committed here.

## Environment (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch CPU build is fine locally
```

## Commands

```bash
# 1. fast correctness check (~20 s, CPU) — run after ANY change to src/
python - <<'EOF'
import sys, copy, torch; sys.path.insert(0, "src")
from revllm import Config, GPT, SavedTensorMeter
torch.manual_seed(0)
for mode in ["midpoint", "leapfrog", "hamiltonian"]:
    kw = dict(mode=mode, n_layer=4, n_embd=64, n_head=2, vocab_size=1000, block_size=32)
    a = GPT(Config(rev_backprop=True, **kw)); b = copy.deepcopy(a); b.layers.rev_backprop = False
    x = torch.randint(0, 1000, (2, 32)); y = torch.randint(0, 1000, (2, 32))
    la = a(x, y); la.backward(); lb = b(x, y); lb.backward()
    gd = max((p.grad - q.grad).abs().max().item() for p, q in zip(a.parameters(), b.parameters()) if p.grad is not None and q.grad is not None)
    assert abs(la.item() - lb.item()) < 1e-6 and gd < 1e-5, (mode, la.item(), lb.item(), gd)
    with torch.no_grad():
        p0 = a.wte(x) + a.wpe(torch.arange(32)); y1, y2 = a.layers(p0, p0); r1, r2 = a.layers.inverse(y1, y2)
        assert (r1 - p0).abs().max() < 1e-5, mode
sizes = []
for L in (2, 8):
    m = GPT(Config(mode="midpoint", n_layer=L, vocab_size=1000, block_size=64, head_chunk=0)); x = torch.randint(0, 1000, (2, 64))
    with SavedTensorMeter() as sm: m(x, x)
    sizes.append(sm.bytes)
assert sizes[0] == sizes[1], sizes
print("OK: exact gradients, exact inverse, depth-constant memory")
EOF

# 2. regenerate notebooks after editing tools/make_notebooks.py
python tools/make_notebooks.py

# 3. execute every notebook in SMOKE mode on CPU (~5 min total). Fails loudly on any cell error.
SMOKE=1 MPLBACKEND=Agg bash -c 'for nb in notebooks/0*.ipynb; do
  python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output /tmp/exec_$(basename $nb) $nb || exit 1; done'

# 4. after GPU runs: unzip results.zip from Colab into results/, then rebuild the summary table
SMOKE=0 python - <<'EOF'
import json, os
runs = [json.load(open(f"results/{f}")) for f in sorted(os.listdir("results")) if f.endswith(".json")]
runs = [r for r in runs if "curve" in r and not r["run_name"].endswith("_short")]
print("| run | batch | steps | train loss | val loss | tokens/s | peak GB | min |\n|---|---|---|---|---|---|---|---|")
for r in runs:
    print(f"| {r['run_name']} | {r['batch_size']} | {r['steps']} | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
          f"{r['tokens_per_s_steady']:,.0f} | {r['peak_mem_gb']:.2f} | {r['wall_time_s']/60:.1f} |")
if os.path.exists("results/max_batch_probe.json"):
    p = json.load(open("results/max_batch_probe.json")); print("\nmax batch:", {k: v["max_batch"] for k, v in p["probe"].items()})
EOF
```

## Typical tasks and how to do them

**"Fill in the README results from the Colab runs."** Run command 4, paste the table into README §6.1–6.3 replacing the
`⏳` rows, state which variant had the lowest val loss under "*Which variant worked best*", copy `results/loss_curves_all.png`
reference into §6, and update §7 findings 3, 4 and 6 with the measured overhead %, max-batch ratio, and ranking. Don't
change the CPU numbers in §5 — they are from a different run.

**"Add a new reversible variant."** Add a `XxxLayer(nn.Module)` in `src/revllm.py` with `forward(x1,x2)->(y1,y2)` and an
exact `inverse(y1,y2)->(x1,x2)`, register it in `LAYER_TYPES`, decide which stream `GPT.trunk` should output, add the
variant's `h` to the `H` dict in `tools/make_notebooks.py` (nb2 and nb3), regenerate notebooks, run command 1 and 3.

**"Change model size / token budget / batch."** Only in the `CONFIG` string in `tools/make_notebooks.py` (the non-SMOKE
branch). Update README §3 "The 20M model" and the notebook time estimates in §4 accordingly. Regenerate notebooks.

**"Make it run on an A100 / bigger GPU."** Nothing to change in code (`_amp` auto-selects bf16). Raise `cap` in
`find_max_batch(cfg, start=16, cap=8192)` if the probe hits the cap; consider `n_layer=24` to make the memory effect larger.

**"The user's Colab run diverged (loss nan)."** `train()` stops on non-finite loss and still writes JSON. Lower `LR`, or for
`leapfrog` lower `h` (0.7), or for `midpoint` set `h=0.25`. Note the change in README §7 finding 5.

## Things that look wrong but are intentional

- `MidpointLayer.forward` returns `(x2, ...)` — the pair is `(p_{l-1}, p_l)` shifting by one; `trunk()` outputs `y2` for
  midpoint/leapfrog and `y1` for hamiltonian. Returning the wrong stream leaves the last layer with zero gradient.
- `_RevBackprop.forward` runs under `torch.no_grad()` on purpose; parameter gradients are accumulated by the inner
  `torch.autograd.backward` call inside `backward()`. Parameters are *not* passed as Function inputs (RevViT pattern).
- `head_chunk=0` in the sanity notebook's memory-vs-depth cell: there we *want* the logits to be stored so the baseline
  curve is the "vanilla PyTorch" number.
- `_early_manual_runs/` in `results/cpu_smoke/` are from a hand-driven first pass with a different tiny config; the
  notebook-generated files next to them are the ones the README quotes.

## Style

Python 3.10+, type hints where cheap, no new dependencies without a reason, comments explain *why* (the paper equation
number is the best comment). Keep files short — this is a teaching repo.
