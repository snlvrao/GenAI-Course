# check.py - verifies the module 15 mini-project. Run: python check.py
# Reads failure_report.json and cross-checks its run ids against shared_memory.db.
import json, os, re, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "failure_report.json")
DB = os.path.join(HERE, "shared_memory.db")
MODES = {"information_withheld", "task_derailment", "no_verification"}
HEX8 = re.compile(r"^[0-9a-f]{8}$")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def need(ok, msg):                     # if this one fails, nothing later can run
    check(ok, msg)
    if not ok:
        print("%d CHECK(S) FAILED" % len(fails))
        sys.exit(1)

# 1. the artefact exists and is one JSON object
need(os.path.isfile(REPORT), "failure_report.json is present in " + HERE)
try:
    with open(REPORT, encoding="utf-8") as fh:
        report = json.load(fh)
except json.JSONDecodeError as e:
    need(False, "failure_report.json is valid JSON (line %d: %s)" % (e.lineno, e.msg))
need(isinstance(report, dict), "failure_report.json holds one JSON object")

# 2. required keys, the mode name, and three written answers
gone = [k for k in ("mast_mode", "break_description", "fix_description",
                    "still_missed", "before", "after") if k not in report]
need(not gone, "six top-level keys present" + ("; missing: " + ", ".join(gone) if gone else ""))
check(report["mast_mode"] in MODES, "mast_mode is one of %s" % ", ".join(sorted(MODES)))
for k in ("break_description", "fix_description", "still_missed"):
    check(len(str(report[k]).strip()) >= 20, "%s is a real sentence, 20+ characters" % k)

# 3. ten run records a side, right shape, no id reused
def runs(name):
    v = report[name]
    return [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []
for name in ("before", "after"):
    rs = runs(name)
    check(len(rs) == 10, "%s holds 10 run records (got %d)" % (name, len(rs)))
    bad = [r for r in rs if not (HEX8.match(str(r.get("run_id", "")))
           and isinstance(r.get("failed"), bool) and str(r.get("topic", "")).strip())]
    check(not bad, "%s records all carry run_id (8 hex), topic, failed true/false"
                   " (%d malformed)" % (name, len(bad)))
ids = [str(r.get("run_id", "")) for n in ("before", "after") for r in runs(n)]
check(len(set(ids)) == len(ids), "all 20 run ids are distinct (%d unique)" % len(set(ids)))

# 4. those ids must be on disk, so the counts cannot be written from memory
if not os.path.isfile(DB):
    check(False, "shared_memory.db not found next to check.py; run the lab first")
else:
    con = sqlite3.connect(DB)
    absent, nodraft = [], []
    for rid in ids:
        c = dict(con.execute("SELECT kind, COUNT(*) FROM memory WHERE run_id=? "
                             "GROUP BY kind", (rid,)).fetchall())
        if not c:
            absent.append(rid)
        elif c.get("draft", 0) < 1:
            nodraft.append(rid)
    con.close()
    check(not absent, "every run id is in shared_memory.db (%d missing: %s)"
          % (len(absent), ", ".join(absent[:3]) or "-"))
    check(not nodraft, "every run stored a writer draft (%d without: %s)"
          % (len(nodraft), ", ".join(nodraft[:3]) or "-"))

# 5. the two counts, with room for an honest null result
b = sum(1 for r in runs("before") if r.get("failed") is True)
a = sum(1 for r in runs("after") if r.get("failed") is True)
check(b >= 7, "the break is strong: %d/10 before-runs failed (7 is the floor, below that "
              "a fix cannot be shown to work)" % b)
if a < b:
    check(True, "the fix moved the number: %d/10 failed before, %d/10 after" % (b, a))
else:
    ok = len(str(report.get("honest_null_result", "")).strip()) >= 20
    check(ok, "no movement (%d/10 -> %d/10)%s" % (b, a, ", and honest_null_result says why"
          if ok else "; add an honest_null_result note of 20+ characters"))
print("not checked automatically: whether the fix is structural rather than a firmer "
      "instruction, and whether each failed judgement was called correctly.")
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
