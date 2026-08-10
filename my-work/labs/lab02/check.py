# check.py - checks the Module 2 mini-project. Run: python check.py
# Standard library only. No API key, no internet, no model call.
import json, math, os, sys

ART = "rings_result.json"
fails = []

def report(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def finish():
    print("Not checked automatically: whether rings_boundary.png shows a closed ring. Open it and look.")
    print("ALL CHECKS PASSED" if not fails else "%d check(s) failed" % len(fails))
    sys.exit(0 if not fails else 1)

if not os.path.exists(ART):
    sys.exit("FAIL %s not found. Run your mini-project script in this folder first." % ART)
try:
    with open(ART, encoding="utf-8") as f:
        d = json.load(f)
except json.JSONDecodeError as e:
    sys.exit("FAIL %s is not valid JSON: %s" % (ART, e))

need = ["hidden", "train_X", "train_y", "test_X", "test_y",
        "train_accuracy", "test_accuracy", "weights", "small_net"]
missing = [k for k in need if k not in d]
report(not missing, "all required keys present" + ("; missing " + ", ".join(missing) if missing else ""))
if missing:
    finish()

wk = d["weights"] if isinstance(d["weights"], dict) else {}
if not all(k in wk for k in ("W1", "b1", "W2", "b2")):
    sys.exit("FAIL weights must be an object holding W1, b1, W2 and b2 as nested lists")
rows = lambda m: m if m and isinstance(m[0], list) else [m]   # accept [1,2] or [[1,2]]
H = int(d["hidden"])
W1, b1, W2, b2 = rows(wk["W1"]), rows(wk["b1"]), rows(wk["W2"]), rows(wk["b2"])
report(len(W1) == 2 and all(len(r) == H for r in W1) and len(b1[0]) == H
       and len(W2) == H and len(W2[0]) == 1 and len(b2[0]) == 1,
       "weight shapes match hidden=%d (W1 is 2x%d, W2 is %dx1)" % (H, H, H))

# the hold-out set is real and separate
ntr, nte = len(d["train_X"]), len(d["test_X"])
report(nte >= 0.15 * (ntr + nte), "test set holds %d of %d points (at least 15%%)" % (nte, ntr + nte))
seen = {(round(a, 6), round(b, 6)) for a, b in d["train_X"]}
leak = sum(1 for a, b in d["test_X"] if (round(a, 6), round(b, 6)) in seen)
report(leak == 0, "no test point also appears in the training set (found %d)" % leak)

# the data really is two rings, not something one straight line could split
radii = lambda lab: sorted(math.hypot(x, y) for (x, y), t in zip(d["train_X"], d["train_y"]) if t[0] == lab)
inner, outer = radii(0), radii(1)
report(bool(inner) and bool(outer) and inner[int(0.9 * len(inner)) - 1] < outer[int(0.1 * len(outer))],
       "label 0 sits inside label 1 (concentric rings, both labels present)")

# recompute the accuracy from the saved weights instead of trusting the reported number
def predict(x):
    h = [math.tanh(sum(x[i] * W1[i][j] for i in range(2)) + b1[0][j]) for j in range(H)]
    z = sum(h[j] * W2[j][0] for j in range(H)) + b2[0][0]
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

hits = sum(1 for x, t in zip(d["test_X"], d["test_y"]) if (predict(x) > 0.5) == (t[0] > 0.5))
recomputed = 100.0 * hits / nte
report(abs(recomputed - float(d["test_accuracy"])) <= 0.5,
       "reported test accuracy %.1f%% matches the %.1f%% these weights actually score"
       % (float(d["test_accuracy"]), recomputed))
report(recomputed >= 95.0, "held-out accuracy is %.1f%% (needs 95%% or better)" % recomputed)
gap = float(d["train_accuracy"]) - recomputed
report(gap <= 5.0, "train minus test gap is %.1f points (needs 5 or less, or it memorised)" % gap)

# the 2-neuron comparison
sn = d["small_net"] if isinstance(d["small_net"], dict) else {}
report(int(sn.get("hidden", 0)) == 2 and "test_accuracy" in sn, "small_net records a run with hidden=2")
report(float(sn.get("test_accuracy", 100.0)) < recomputed,
       "2 neurons scored %.1f%%, below the %d-neuron run" % (float(sn.get("test_accuracy", -1)), H))

report(os.path.exists("rings_boundary.png") and os.path.getsize("rings_boundary.png") > 0,
       "rings_boundary.png exists and is not empty")
finish()
