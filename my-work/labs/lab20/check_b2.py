"""Checker for Build-track B2. Put it next to your test_model.py and run it."""
import importlib.util
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
FAILED = []


def report(name, passed, detail=""):
    print(("PASS  " if passed else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not passed:
        FAILED.append(name)


test_file = HERE / "test_model.py"
if not test_file.exists():
    report("test_model.py exists", False, "expected it at " + str(test_file))
    sys.exit(1)
report("test_model.py exists", True)

src = test_file.read_text(encoding="utf-8", errors="replace")
report("your test asks for the parameter count", "n_params" in src)
report("your test asks for a shape", "shape" in src)
report("your test mentions the mask", "mask" in src.lower() or "causal" in src.lower())

run = subprocess.run([sys.executable, str(test_file)], capture_output=True,
                     text=True, cwd=str(HERE))
tail = (run.stdout + run.stderr).strip().splitlines()
report("test_model.py runs and exits 0", run.returncode == 0,
       tail[-1][:110] if tail else "")

model_path = HERE / "my_gpt.py"
if not model_path.exists():
    model_path = HERE.parent / "_shared" / "tinygpt.py"
if not model_path.exists():
    report("model file found", False, "no my_gpt.py and no _shared/tinygpt.py")
    sys.exit(1)

try:
    import torch
    spec = importlib.util.spec_from_file_location("b2model", model_path)
    gpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpt)
    model = gpt.TinyGPT(vocab_size=65, cfg=gpt.Config()).eval()
except Exception as exc:
    report("model imports and builds", False, type(exc).__name__ + ": " + str(exc)[:90])
    sys.exit(1)
report("model imports and builds", True, model_path.name)

n = model.n_params()
report("parameter count is 824,897", n == 824897, "got {:,}".format(n))

torch.manual_seed(0)
x = torch.randint(0, 65, (2, 16))
with torch.no_grad():
    logits, loss = model(x, x)
report("logits shape is (2, 16, 65)", tuple(logits.shape) == (2, 16, 65),
       "got " + str(tuple(logits.shape)))
report("loss is one number", loss is not None and loss.dim() == 0)

y = x.clone()
y[:, -1] = (y[:, -1] + 1) % 65
with torch.no_grad():
    a, _ = model(x)
    b, _ = model(y)
same = torch.equal(a[:, :-1], b[:, :-1])
drift = float((a[:, :-1] - b[:, :-1]).abs().max())
report("changing the last token leaves every earlier output identical", same,
       "largest difference {:.2e}".format(drift))
report("changing the last token does change the last output",
       not torch.equal(a[:, -1], b[:, -1]),
       "largest difference {:.4f}".format(float((a[:, -1] - b[:, -1]).abs().max())))

print()
if FAILED:
    print("{} check(s) failed: {}".format(len(FAILED), "; ".join(FAILED)))
    sys.exit(1)
print("All checks passed. Your model is correct without a single training step.")
sys.exit(0)
