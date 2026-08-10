"""Checks the B4 mini-project. Run:  python check_compare.py"""
import json, os, sys

F = "compare.json"
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def die(msg):                      # missing or broken artefact: stop, no traceback
    print("FAIL " + msg + "\n1 check failed.")
    sys.exit(1)

if not os.path.isfile(F):
    die(F + " not found in " + os.getcwd())
try:
    data = json.load(open(F, encoding="utf-8"))
except json.JSONDecodeError as e:
    die(F + " is not valid JSON: " + str(e))
if not isinstance(data, dict):
    die(F + " must be a JSON object with keys prompts, scratch, tuned, judgement")

prompts = data.get("prompts")
scratch = data.get("scratch")
tuned = data.get("tuned")
judge = data.get("judgement")
if not isinstance(prompts, list) or not prompts:
    die("prompts must be a non-empty list of the instructions you sent")
if not isinstance(scratch, dict) or not isinstance(tuned, dict):
    die("scratch and tuned must each be an object mapping prompt to answer")
if not isinstance(judge, dict):
    die("judgement must be an object with keys better and why")

# --- the same three prompts, answered twice --------------------------------
check(len(prompts) == 3, "exactly 3 prompts (got %d)" % len(prompts))
check(all(isinstance(p, str) and p.strip() for p in prompts), "every prompt is non-empty text")
check(len({str(p).strip() for p in prompts}) == len(prompts), "the 3 prompts differ from each other")
check(set(scratch) == set(prompts), "scratch answers cover exactly those prompts")
check(set(tuned) == set(prompts), "tuned answers cover exactly those prompts")

both = [p for p in prompts if p in scratch and p in tuned]
check(len(both) == len(prompts), "both models answered all 3 (got %d)" % len(both))
check(all(str(scratch[p]).strip() for p in both), "no empty answer from your from-scratch model")
check(all(str(tuned[p]).strip() for p in both), "no empty answer from the tuned model")
differ = sum(1 for p in both if str(scratch[p]).strip() != str(tuned[p]).strip())
check(differ == len(prompts), "all 3 answers differ between the models (%d of 3)" % differ)

# --- the judgement, in your own words ---------------------------------------
better = str(judge.get("better", "")).strip().lower()
check(better in ("scratch", "tuned"), "judgement.better is 'scratch' or 'tuned' (got %r)" % better)
why = str(judge.get("why", "")).strip()
check(len(why.split()) >= 25, "judgement.why is 25 words or more (got %d)" % len(why.split()))
STOCK = ("todo", "n/a", "tbd", "it is better", "the tuned model is better")
check(why.lower() not in STOCK, "judgement.why is not a placeholder")
answers = {str(v).strip().lower() for v in list(scratch.values()) + list(tuned.values())}
check(why.lower() not in answers, "judgement.why is your writing, not a pasted answer")
check(any(w in why.lower() for w in ("because", "since", "which", "whereas", "so that")),
      "judgement.why gives a reason, not only a verdict")

print("\nNot checked: whether your reason is a good one. Read the answers yourself.")
if fails:
    print("%d check(s) failed." % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
