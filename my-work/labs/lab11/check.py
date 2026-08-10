"""Checks the Module 11 mini-project. Run:  python check.py"""
import importlib.util, json, os, sys

RUN, RULE = "voice_run.json", "style_check.py"
fails, crashes = [], []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def die(msg):                       # missing or broken artefact: stop, no traceback
    print("FAIL " + msg + "\n1 check failed.")
    sys.exit(1)

# --- load the two artefacts -------------------------------------------------
if not os.path.isfile(RUN):
    die(RUN + " not found in " + os.getcwd())
if not os.path.isfile(RULE):
    die(RULE + " not found in " + os.getcwd() + " (it must define passes(text))")
try:
    run = json.load(open(RUN, encoding="utf-8"))
except json.JSONDecodeError as e:
    die(RUN + " is not valid JSON: " + str(e))
spec = importlib.util.spec_from_file_location("style_check", RULE)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    die(RULE + " would not import: " + repr(e))
if not callable(getattr(mod, "passes", None)):
    die(RULE + " has no passes(text) function")

def rule(text):                     # your rule, with crashes turned into a check
    try:
        return bool(mod.passes(text))
    except Exception as e:
        crashes.append(repr(e))
        return False

train, held = run.get("train", []), run.get("heldout", [])
before, after = run.get("before", {}), run.get("after", {})

# --- dataset shape ----------------------------------------------------------
check(len(train) >= 15, "train has 15+ rows (got %d)" % len(train))
check(len(held) == 5, "heldout has exactly 5 rows (got %d)" % len(held))
check(bool(train) and all(r.get("prompt") and r.get("completion") for r in train),
      "every train row has a non-empty prompt and completion")

# --- no leakage: you must not test on something you trained on --------------
trained = {r.get("prompt", "").strip() for r in train}
leak = [h.get("prompt", "") for h in held if h.get("prompt", "").strip() in trained]
check(not leak, "no held-out prompt appears in train (leaked: %s)" % leak[:1])

# --- your rule must fit your own writing, and reject writing that is not ----
own = sum(rule(r.get("completion", "")) for r in train)
check(own == len(train), "passes() accepts all your own writing (%d/%d)" % (own, len(train)))
DECOYS = ["",
          "Certainly! Here is a breakdown of the topic. Let me know if you want more detail!",
          "Great question. There are several factors to consider, and I am happy to help."]
taken = [(d[:28] + "...") if d else "(empty string)" for d in DECOYS if rule(d)]
check(not taken, "passes() rejects empty text and generic assistant prose (took: %s)" % taken)
check(not crashes, "passes() never crashes (%s)" % crashes[:1])

# --- before and after must cover the same five prompts ----------------------
hp = [h.get("prompt", "") for h in held]
check(set(before) == set(hp) == set(after),
      "before and after each hold answers for exactly the 5 held-out prompts")

# --- the measurement --------------------------------------------------------
b = sum(rule(before[p]) for p in hp if p in before)
a = sum(rule(after[p]) for p in hp if p in after)
print("   scored: before %d/5, after %d/5" % (b, a))
check(b <= 1, "base model mostly fails your rule (before %d/5, want 0 or 1)" % b)
check(a >= 3, "tuned model mostly passes your rule (after %d/5, want 3+)" % a)
moved = sum(1 for p in hp if before.get(p) != after.get(p))
check(moved >= 4, "at least 4 of 5 answers actually changed (got %d)" % moved)

# --- you kept an adapter, not a copy of the whole model ---------------------
size = run.get("adapter_bytes")
check(isinstance(size, int) and 10**6 <= size <= 2 * 10**8,
      "adapter_bytes is between 1 MB and 200 MB (got %r)" % (size,))

print("\nNot checked automatically: whether the after-answers sound like you.")
print("Read them yourself. A rule only confirms the habits you wrote down.")
if fails:
    print("%d check(s) failed." % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
