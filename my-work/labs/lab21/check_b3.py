"""check_b3.py - verifies the B3 mini-project. Put it in lab21 and run it."""
import json, math, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "_shared"))
FAILED = []


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not passed:
        FAILED.append(name)


def die(msg):
    print(f"FAIL  {msg}")
    sys.exit(1)


rep_file = HERE / "train_report.json"
if not rep_file.exists():
    die("train_report.json not found. Train, then run report.py.")
try:
    rep = json.loads(rep_file.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    die(f"train_report.json is not valid JSON ({e})")
missing = [k for k in ("corpus_file", "checkpoint", "final_val_loss", "history",
                       "samples") if k not in rep]
if missing:
    die(f"train_report.json is missing keys: {missing}")

corpus, ckpt = HERE / rep["corpus_file"], HERE / rep["checkpoint"]
if not corpus.exists():
    die(f"corpus {rep['corpus_file']} not found next to this checker")
if not ckpt.exists():
    die(f"checkpoint {rep['checkpoint']} not found next to this checker")
text = corpus.read_text(encoding="utf-8", errors="replace")
check("corpus is 20,000+ characters", len(text) >= 20_000, f"{len(text):,} chars")

blob = None
try:
    import torch
    from tinygpt import TinyGPT
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = TinyGPT(len(blob["stoi"]), blob["config"])
    model.load_state_dict(blob["state"])
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1, 8), dtype=torch.long))
    check("checkpoint loads and runs", True, f"{model.n_params():,} parameters")
except Exception as e:
    check("checkpoint loads and runs", False, f"{type(e).__name__}: {e}"[:80])

if blob:
    vocab = sorted(blob["stoi"])
    check("vocabulary matches the corpus", vocab == sorted(set(text)),
          f"{len(vocab)} distinct characters")
    base, final = math.log(len(vocab)), float(rep["final_val_loss"])
    check("report matches the checkpoint", abs(final - blob["val_loss"]) < 0.02,
          f"report {final:.3f}, checkpoint {blob['val_loss']:.3f}")
    check("beats the random-guess baseline by 1.0+",
          final <= base - 1.0, f"{final:.3f} vs baseline {base:.3f}")
    stray = sorted(set("".join(rep["samples"])) - set(vocab))
    check("samples use only your vocabulary", bool(rep["samples"]) and not stray,
          f"{len(rep['samples'])} samples" if not stray else f"stray {stray[:5]}")

try:
    h = [float(p["val_loss"]) for p in rep["history"]]
except (TypeError, KeyError, ValueError):
    h = []
mid = len(h) // 2
check("history has 3+ measured points", len(h) >= 3, f"{len(h)} points")
check("validation loss went down overall",
      len(h) >= 3 and h[-1] < h[0]
      and sum(h[mid:]) / len(h[mid:]) < sum(h[:mid]) / len(h[:mid]),
      f"{h[0]:.3f} -> {h[-1]:.3f}" if h else "no history")

print()
if FAILED:
    print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
    sys.exit(1)
print("All checks passed. You trained a language model from random numbers.")
