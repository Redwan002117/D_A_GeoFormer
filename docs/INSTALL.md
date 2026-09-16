# Installation &amp; Setup

How to get Dual-Axis GeoFormer running on your own machine, from a clean
checkout to a verified working install. Everything here was actually run
during development (Windows 11 + Python 3.14, CPU-only) — commands aren't
guessed from documentation, they're what got this repo's own checkpoints
and figures produced.

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.14. No GPU-specific build needed. |
| pip | any recent | Ships with Python. |
| Disk space | ~2GB free | PyTorch alone is ~800MB–2GB depending on platform/CUDA build; add ~1.5GB if you pull real SpaceNet-8 data (§5). |
| RAM | 8GB+ recommended | See §6 — training at full 256px resolution with a large batch size genuinely used enough memory to get this process OOM-killed on a 16GB machine with other things open. Defaults in this repo are chosen to be safe on modest hardware. |
| GPU | optional | Every script auto-detects CUDA (`torch.cuda.is_available()`) and falls back to CPU. Nothing here requires a GPU; a GPU just makes the real-data training in §5 much faster. |
| Git | optional | Only needed to clone the repo / use the Colab notebook's `git clone` step. |
| Docker | optional | Only needed for `docs/DEPLOYMENT.md`'s deployment path. |

## 2. Get the code

```bash
git clone https://github.com/Redwan002117/D_A_GeoFormer.git
cd D_A_GeoFormer
```

(Or download the ZIP from GitHub and extract it, if you don't have `git`.)

## 3. Create a virtual environment (recommended, not required)

Keeps this project's dependencies separate from anything else on your
machine.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script (`running scripts is
disabled`), run PowerShell as Administrator once and execute
`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry.

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked because your shell prompt gets a `(.venv)` prefix.
Every command below assumes this environment is active.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: PyTorch (CPU build by default — see the box below for GPU),
NumPy, SciPy, scikit-image, Matplotlib, Pillow, FastAPI + Uvicorn (for
`serve.py`), boto3 (for `prepare_real_data.py`), and pytest.

> **Want a CUDA GPU build of PyTorch instead of CPU?** `requirements.txt`
> pins `torch>=2.0` with no platform-specific index, so `pip install` gets
> whatever your platform's default wheel is (CPU-only on many setups). To
> get a CUDA build, install PyTorch first, from
> [pytorch.org's own install-selector command](https://pytorch.org/get-started/locally/)
> for your CUDA version, **then** run `pip install -r requirements.txt` — pip
> will see torch is already satisfied and skip reinstalling it.

## 5. Verify the install

Run these in order. Each one should complete with no errors and print the
line shown.

```bash
python model.py
# -> logits: (1, 4, 256, 256)
# -> grid_saliency: (1, 16, 16)
# -> parameters: 10,935,044

python losses.py
# -> Tversky loss on random tensors: <some float>

python postprocess.py
# -> Endpoints found, gaps bridged: 1

python -m pytest tests/ -v
# -> all tests PASSED
```

If all four pass, the install is good — every core piece (model, loss,
post-processing, tests) runs correctly on your machine.

## 6. First real run

```bash
python demo.py
```

This builds the model, generates a synthetic tile pair, runs a full forward
pass, and saves `demo_output.png`. If a trained `checkpoints/best.pt`
doesn't exist yet, it automatically falls back to random-init weights and
tells you so — that's expected on a first run, not an error. To get a
trained checkpoint, see the training commands in `README.md`.

## 7. A real memory limit we actually hit — plan around it

Training at `--image-size 256` with `--batch-size 8` was observed to get
the training process killed by the OS for running out of memory on a
16GB-RAM machine that had other applications (browser, editor) open at the
same time — **not** a bug in this code, but a real resource limit worth
planning around rather than discovering the hard way:

| Your free RAM (with other apps already running) | Recommended `--batch-size` at `--image-size 256` |
|---|---|
| &lt; 6GB | `2` (this repo's real-data default) |
| 6–10GB | `4` |
| 10GB+, or a dedicated/CI machine | `8` or higher |

If a training run disappears with no Python traceback at all (just stops),
that's almost always this — the OS killed the process outright before it
could print anything. Lower `--batch-size` and/or `--image-size` and retry;
there's no error message to "fix" because the process never got the chance
to raise one.

## 8. Optional: real SpaceNet-8 data

No AWS account needed — the bucket is public. See `docs/MANUAL.md` §6 and
§12, or just run:

```bash
python prepare_real_data.py --n-tiles 24 --out-dir real_sn8_dataset
```

## 9. Optional: the deployable API

```bash
uvicorn serve:app --port 8000
```

Then in another terminal: `curl -F "pre=@your_pre.jpg" -F "post=@your_post.jpg" http://localhost:8000/predict`.
Full deployment instructions (Docker, cloud platforms) are in
`docs/DEPLOYMENT.md`.

## Troubleshooting the install itself

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or any other package) | You're not in the virtual environment (§3), or `pip install -r requirements.txt` (§4) didn't complete — re-run it and read the output for errors. |
| `pip install` fails on `scikit-image` or another compiled package | Usually a missing platform build tool. On Windows, install the ["Desktop development with C++" workload](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if a wheel isn't available for your Python version; on Linux, `sudo apt install build-essential python3-dev` (Debian/Ubuntu) first. |
| `python: command not found` (macOS/Linux) | Use `python3` instead of `python` — some systems don't alias `python` to Python 3. |
| PowerShell: `cannot be loaded because running scripts is disabled` | See the venv-activation note in §3. |
| Everything installs but `python -m pytest tests/` fails | Re-read the actual failure — it names the exact assertion. If it's unrelated to your changes, check `docs/MANUAL.md` §9 for known, already-diagnosed issues before assuming it's new. |
| A training run just silently stops, no traceback | See §7 — almost certainly the OS killing the process for memory, not a code bug. |

Once you're past this page, `docs/MANUAL.md` is the full usage reference
(CLI flags, data format, architecture, the honest real-data findings) and
`docs/DEPLOYMENT.md` covers running this anywhere/in the cloud.
