"""Checks drops.json from the Module 8 mini-project.
Run inside my-work/labs/lab08:  python check.py
Standard library only. No API key, no internet.
"""
import json, os, re, sys

PATH = "drops.json"
EXPECTED = {1200: (1, 814), 700: (3, 697), 400: (12, 396)}  # budget: (drops, tokens used)
CODES = ("source_cap_reached", "no_room_left", "bigger_than_source_cap")
FIELDS = ("source", "reason", "budget", "needed", "cap", "source_used", "used", "why", "fix")
NUMS = ("budget", "needed", "cap", "source_used", "used")
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


# the artefact has to exist and parse before anything else can be said about it
if not os.path.exists(PATH):
    print("FAIL " + PATH + " not found. Run 'python explain.py' inside my-work/labs/lab08 first.")
    sys.exit(1)
try:
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print("FAIL " + PATH + " is not valid JSON: " + str(e))
    sys.exit(1)

runs = data.get("budgets") if isinstance(data, dict) else None
check(isinstance(runs, list) and len(runs) == 3, "top level is {'budgets': [...]} holding 3 runs")
runs = [r for r in (runs or []) if isinstance(r, dict)]
by_budget = {r.get("budget"): r for r in runs}
check(sorted(k for k in by_budget if isinstance(k, int)) == [400, 700, 1200],
      "the three runs are budgets 400, 700 and 1200")

# drop count and tokens used must match what the lab packer really produces
for b in (1200, 700, 400):
    n_want, used = EXPECTED[b]
    run = by_budget.get(b, {})
    got = run.get("drops")
    n = len(got) if isinstance(got, list) else -1
    check(n == n_want, "budget %d has %d dropped items (got %d)" % (b, n_want, n))
    check(run.get("used") == used, "budget %d used %d tokens (got %r)" % (b, used, run.get("used")))

drops = [d for r in runs if isinstance(r.get("drops"), list) for d in r["drops"] if isinstance(d, dict)]
missing = sorted({k for d in drops for k in FIELDS if k not in d})
check(not missing, "every drop record carries all 9 fields (missing: %s)" % (", ".join(missing) or "none"))
full = [d for d in drops if all(k in d for k in FIELDS)]
numeric = [d for d in full if all(isinstance(d[k], int) for k in NUMS)]
check(len(numeric) == len(full), "the five drop numbers are integers, not strings")

# the sentences must be English, and must name the numbers that decided the drop
leaks = [d for d in full if any(c in str(d["why"]) for c in CODES)]
check(not leaks, "no 'why' sentence leaks a raw reason code (%d do)" % len(leaks))
thin = [d for d in numeric
        if len(set(re.findall(r"\d+", str(d["why"]))) & {str(d[k]) for k in NUMS}) < 2]
check(not thin, "every 'why' names at least 2 of its own numbers (%d do not)" % len(thin))
check(all(re.search(r"\d", str(d["fix"])) for d in full), "every 'fix' names a number")

# the numbers must support the reason given, so records cannot be invented
bad = 0
for d in numeric:
    if d["reason"] == "source_cap_reached" and d["source_used"] + d["needed"] <= d["cap"]:
        bad += 1
    if d["reason"] == "no_room_left" and d["used"] + d["needed"] <= d["budget"]:
        bad += 1
check(not bad, "the numbers behind each reason add up (%d records do not)" % bad)

# the planted trap: the passage holding the answer dies at 700 and survives at 1200
trap = [d for d in by_budget.get(700, {}).get("drops", []) if isinstance(d, dict)
        and d.get("source") == "docs" and d.get("needed") == 89
        and d.get("reason") == "source_cap_reached"]
check(len(trap) == 1, "budget 700 drops the 89-token 'Refund window and approval' doc on the docs cap")
check(not [d for d in by_budget.get(1200, {}).get("drops", []) if isinstance(d, dict)
           and d.get("source") == "docs"], "budget 1200 drops no documents at all")

print("\nNot checked automatically: whether your sentences read clearly to someone who has")
print("never seen packer.py. Read the budget 700 run out loud.")
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
