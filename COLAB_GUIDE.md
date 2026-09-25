# Step-by-step: running the experiments on Google Colab

Total hands-on time ≈ 20 minutes; total GPU time ≈ 3–3.5 hours on a free T4 (can be split across days).
You need: a GitHub account, a Google account, and Claude Code installed locally (`npm i -g @anthropic-ai/claude-code`).

## Part A — publish the repo (5 min, once)

1. Unzip / copy the `reversible-llm-poc` folder somewhere on your machine and open a terminal there.
2. Start Claude Code in the folder: `claude`. It reads `CLAUDE.md` automatically.
   Ask it: *"Run the fast correctness check from CLAUDE.md, then help me push this to GitHub."*
   (Or do it by hand:)
   ```bash
   git init -b main && git add -A && git commit -m "Reversible LLM POC"
   ```
3. On github.com → **New repository** → name `reversible-llm-poc`, public, **no** README/.gitignore (repo must be empty). Create.
4. Back in the terminal:
   ```bash
   git remote add origin https://github.com/<YOUR_USER>/reversible-llm-poc.git
   git push -u origin main
   ```
5. Point the notebooks at your repo. In Claude Code: *"Set REPO_URL in tools/make_notebooks.py to my repo
   https://github.com/<YOUR_USER>/reversible-llm-poc.git, regenerate the notebooks, and commit + push."*
   By hand: edit line 13 of `tools/make_notebooks.py`, then `python tools/make_notebooks.py && git commit -am "repo url" && git push`.

## Part B — open a notebook in Colab (2 min each)

6. Go to <https://colab.research.google.com> → **File → Open notebook → GitHub** tab → paste
   `https://github.com/<YOUR_USER>/reversible-llm-poc` → pick a notebook.
   Shortcut URL: `https://colab.research.google.com/github/<YOUR_USER>/reversible-llm-poc/blob/main/notebooks/01_baseline_fixed_batch.ipynb`
7. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.** (If you skip this the notebook runs in
   SMOKE mode on CPU — a tiny toy run — and prints `SMOKE mode: True` in the first cell.)
8. **Runtime → Run all** (Ctrl/Cmd+F9). The first cell clones your repo and installs `tiktoken`, `datasets`, `matplotlib`.
   Confirm the first cell prints `device: Tesla T4 | SMOKE mode: False`.

## Part C — run the five notebooks in order

| # | notebook | what you'll see | T4 time | can you leave? |
|---|---|---|---|---|
| 1 | `00_sanity_checks` | tables: exact gradients, memory-vs-depth plot, step-time overhead | ~3 min | yes |
| 2 | `01_baseline_fixed_batch` | TinyStories tokenizing (~4 min, once per session), then ~6,100 steps of loss logging, a loss plot, a generated story | ~35 min | yes |
| 3 | `02_reversible_variants_fixed_batch` | same for midpoint → leapfrog → hamiltonian, then a 400-step stored-vs-recomputed check, then a comparison table | ~2.5 h | yes, but see step 10 |
| 4 | `03_reversible_max_batch` | batch probe table (baseline vs reversible max batch), then one training run at the big batch | ~50 min | yes |
| 5 | `04_report` | summary table, two plots, and a `results.zip` download | 1 min | — |

9. **Keep the browser tab open.** Free Colab kills idle sessions after ~90 min and everything after ~12 h. Training keeps the
   session "busy", but closing the laptop lid does not count as busy.
10. **If notebook 02 disconnects mid-way:** the finished variants are already saved in `results/`, but `results/` lives in the
    Colab VM and is lost with the session. So after **each** variant finishes, download it: click the folder icon (left
    sidebar) → `reversible-llm-poc/results/` → right-click the `.json` → Download. On reconnect, edit
    `VARIANTS = ["leapfrog", "hamiltonian"]` (whatever is left) and Run all again. Same for the baseline file from notebook 01
    — download `results/baseline_B32.json` before you close the tab.
11. **Optional but strongly recommended — persist to Google Drive so nothing is lost:** in *any* notebook add a cell right after
    Setup:
    ```python
    from google.colab import drive; drive.mount('/content/drive')
    import os; os.makedirs('/content/drive/MyDrive/revllm/results', exist_ok=True)
    os.system('rm -rf results && ln -s /content/drive/MyDrive/revllm/results results')
    os.makedirs('/content/drive/MyDrive/revllm/data', exist_ok=True)
    os.system('rm -rf data && ln -s /content/drive/MyDrive/revllm/data data')
    ```
    Now results **and** the tokenized dataset survive across sessions (skips the 4-minute tokenization next time).
12. **Notebook 03 needs the results of 01/02 in the same `results/` folder** for its final comparison table (it trains fine
    without them). If you used Drive (step 11) they're already there; otherwise upload the downloaded JSONs via the folder
    icon → Upload.
13. Run `04_report` last; it downloads `results.zip`.

## Part D — bring the numbers home with Claude Code (5 min)

14. Unzip `results.zip` into the repo's `results/` folder on your machine (overwrite is fine).
15. In Claude Code: *"Fill in the README results from the Colab runs."* — CLAUDE.md tells it exactly what to do: build the
    table from `results/*.json`, replace the ⏳ rows in README §6, name the best variant, update §7 with the measured overhead
    and max-batch ratio.
16. Review the diff, then: *"Commit and push."*

## Troubleshooting

| symptom | fix |
|---|---|
| First cell says `SMOKE mode: True` | You didn't select the T4 runtime (step 7). Change it and **Runtime → Restart and run all**. |
| `git clone` fails in first cell | `REPO_URL` still says `YOUR_GITHUB_USER` (step 5), or the repo is private. Alternatively upload the `src/` folder via the file sidebar and re-run. |
| "Cannot connect to GPU backend" | Free-tier GPU quota exhausted for the day. Wait, or Runtime → Change runtime type → CPU to at least run SMOKE mode. |
| `CUDA out of memory` in notebook 01/02 | Shouldn't happen at batch 32 on 15 GB; if it does, another process holds the GPU — Runtime → Disconnect and delete runtime, retry. |
| Loss becomes `nan` | `train()` stops and still saves the JSON. Lower `LR` (e.g. 3e-4) or `H["leapfrog"] = 0.7`, re-run that variant. Note it in README §7 finding 5. |
| Notebook 03 max-batch probe takes forever | It's binary-searching; ~10 probes × ~30 s. Leave it. |
| `datasets` download stalls | HF Hub hiccup. Re-run the cell; tokenization resumes from scratch but is only ~4 min. |
