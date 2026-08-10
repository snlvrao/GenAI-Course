"""check.py - verifies your Module 16 regression report. Standard library only."""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
REPORT = HERE / "regression_report.json"
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

# 1. the artefact must exist and parse before anything else can be checked
if not REPORT.exists():
    print("FAIL regression_report.json not found next to check.py. Write it (step 4) and rerun.")
    sys.exit(1)
try:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    print(f"FAIL regression_report.json is not valid JSON: {e}")
    sys.exit(1)

base, variants = rep.get("baseline", {}), rep.get("variants", [])
bpr = base.get("pass_rate")

check(isinstance(bpr, (int, float)) and 0 <= bpr <= 1, "baseline.pass_rate is a number from 0 to 1")
check(isinstance(base.get("kappa"), (int, float)) and base.get("kappa", -9) >= 0.4,
      "baseline.kappa is recorded and at least 0.4 (below that the rubric is too vague to use)")
check(rep.get("threshold") == 0.10, "threshold is 0.10, the same ten points gate.py uses")
check(len(variants) == 3, f"exactly 3 variants recorded, found {len(variants)}")
check(len({v.get("name") for v in variants}) == len(variants), "variant names are unique")

# 2. the baseline counts must match the files they came from, not be typed by hand
cases, labels = HERE / "cases.jsonl", HERE / "labels.jsonl"
if cases.exists():
    n = sum(1 for l in cases.read_text(encoding="utf-8").splitlines() if l.strip())
    check(n == base.get("n_cases"), f"baseline.n_cases matches cases.jsonl, which has {n} lines")
    check(n >= 50, f"cases.jsonl holds at least 50 frozen cases, found {n}")
else:
    check(False, "cases.jsonl not found; the case set has to be frozen on disk")
if labels.exists():
    ids = {json.loads(l)["trace_id"]
           for l in labels.read_text(encoding="utf-8").splitlines() if l.strip()}
    check(len(ids) == base.get("n_labels"), f"baseline.n_labels matches labels.jsonl, {len(ids)} unique")
    check(len(ids) >= 40, f"at least 40 blind labels behind kappa, found {len(ids)}")
else:
    check(False, "labels.jsonl not found; kappa has to come from labels you made yourself")

# 3. per variant: the verdict must follow from the numbers, and the numbers from the saved run
missed = 0
for v in variants:
    name = str(v.get("name", "?"))
    pr, code, det = v.get("pass_rate"), v.get("gate_exit_code"), v.get("detected")
    ok_pr = isinstance(pr, (int, float)) and 0 <= pr <= 1
    check(ok_pr, f"{name}: pass_rate is a number from 0 to 1")
    check(len(str(v.get("defect", "")).strip()) > 10, f"{name}: defect is described in one line")
    caught = ok_pr and isinstance(bpr, (int, float)) and pr < bpr - 0.10
    check(det is caught,
          f"{name}: detected={det} should be {caught}, from {pr} against baseline {bpr} minus 0.10")
    check(code == (1 if caught else 0),
          f"{name}: gate_exit_code {code} should be {1 if caught else 0} for a defect the gate "
          f"{'caught' if caught else 'missed'}")
    log = HERE / "runs" / f"{name}.txt"
    if log.exists():
        m = re.search(r"pass rate\s+([0-9.]+)", log.read_text(encoding="utf-8"))
        check(bool(m) and ok_pr and abs(float(m.group(1)) - pr) < 0.011,
              f"{name}: pass_rate matches the 'pass rate' line in runs/{name}.txt")
    else:
        check(False, f"{name}: runs/{name}.txt missing; save each variant's gate output there")
    if not caught:
        missed += 1

# 4. every defect the gate missed needs a written blind spot
spots = [s for s in rep.get("blind_spots", []) if isinstance(s, str) and len(s.strip()) > 20]
check(len(spots) >= missed,
      f"blind_spots has a sentence for each of the {missed} defect(s) the gate did not catch")

print("\nNot checked automatically: whether your blind_spots sentences are true, and whether "
      "your 50 cases resemble what users actually ask.")
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(0 if not fails else 1)
