"""check.py - checks the Module 12 mini-project. Run: python check.py"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_CAP = 6
FIELDS = ("design", "model_calls", "input_tokens", "seconds", "totals", "total_correct")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)
def read(name):
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        print("FAIL missing file: " + path)
        print("The mini-project steps say to create it. Do that, then run check.py again.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return f.read()
def totals_of(r):
    return {str(k).lower(): round(float(v), 2) for k, v in (r.get("totals") or {}).items()}

# 1. Rebuild the right answer here, from your own expenses.txt.
rows, skipped = [], 0
for raw in read("expenses.txt").splitlines():
    if not raw.strip():
        continue
    parts = [p.strip() for p in raw.split(",")]
    try:
        amount = float(parts[-1]) if len(parts) >= 4 else None
    except ValueError:
        amount = None
    if amount is None:
        skipped += 1          # a line with no readable amount
        continue
    rows.append((parts[1], parts[2].lower(), amount))
truth = {}
for _, cat, amt in rows:
    truth[cat] = round(truth.get(cat, 0.0) + amt, 2)
flagged = sorted(desc for desc, _, amt in rows if amt > 200)
check(len(rows) >= 15, "expenses.txt has %d usable lines (need 15 or more)" % len(rows))
check(any(a < 0 for _, _, a in rows), "expenses.txt has a negative amount (the refund)")
check(skipped >= 1, "expenses.txt has one line with no readable amount")

# 2. Load runs.json and check every run record.
try:
    data = json.loads(read("runs.json"))
except json.JSONDecodeError as e:
    print("FAIL runs.json is not valid JSON: %s" % e)
    sys.exit(1)
runs = data.get("runs") if isinstance(data.get("runs"), list) else []
check(len(runs) == 6, "runs.json holds 6 run records, found %d" % len(runs))
for design in ("workflow", "agent"):
    n = sum(1 for r in runs if isinstance(r, dict) and r.get("design") == design)
    check(n == 3, "3 %s runs recorded, found %d" % (design, n))
for i, r in enumerate(runs, 1):
    tag = "run %d" % i
    if not isinstance(r, dict) or any(k not in r for k in FIELDS):
        check(False, tag + " is missing one of: " + ", ".join(FIELDS))
        continue
    check(all(isinstance(r[k], (int, float)) and r[k] > 0 for k in FIELDS[1:4]),
          tag + " has positive model_calls, input_tokens and seconds")
    if r["design"] == "agent":
        check(r["model_calls"] <= STEP_CAP,
              tag + " stayed inside the %d step cap (recorded %s)" % (STEP_CAP, r["model_calls"]))
    # Honesty check: the flag you wrote must agree with the totals you wrote.
    redone = totals_of(r) == truth
    check(r["total_correct"] is redone, tag + " total_correct=%s agrees with its own totals"
          " (recomputed %s)" % (r["total_correct"], redone))
check(any(isinstance(r, dict) and r.get("design") == "workflow" and totals_of(r) == truth for r in runs),
      "one workflow run reproduces the totals exactly: " + json.dumps(truth, sort_keys=True))
check(sorted(data.get("flagged_over_200") or []) == flagged,
      "flagged_over_200 lists exactly " + json.dumps(flagged))

# 3. The written decision, mechanical parts only.
d = data.get("decision") if isinstance(data.get("decision"), dict) else {}
check(d.get("ship") in ("workflow", "agent"), "decision.ship is 'workflow' or 'agent'")
check(d.get("metric") in ("model_calls", "input_tokens", "seconds", "total_correct"),
      "decision.metric names one of the numbers you recorded")
check(all(isinstance(d.get(k), (int, float)) for k in ("workflow_value", "agent_value")),
      "decision has workflow_value and agent_value as numbers")
check(len(str(d.get("changes_my_mind", ""))) >= 40, "decision.changes_my_mind is a real sentence")

print("not checked automatically: whether your five sentences argue the case well.")
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
