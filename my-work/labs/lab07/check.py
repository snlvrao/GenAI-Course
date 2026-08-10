# check.py - verifies the mini-project artefacts next to this file. Stdlib only.
import json, math, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
fails = []

def check(ok, msg):                    # one PASS/FAIL line per assertion
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def bail(msg):                         # stop cleanly, never a traceback
    print("FAIL " + msg)
    sys.exit(1)

def load(name):
    path = HERE / name
    if not path.exists():
        bail("missing file: %s\n      Create it first, the module gives the shape." % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        bail("%s is not valid JSON: %s" % (name, err))

cases, report = load("cases.json"), load("report.json")

# shape guards, so every check below can assume lists of dicts
if not (isinstance(cases, list) and all(isinstance(c, dict) for c in cases)):
    bail("cases.json must be a JSON list of case objects")
if not isinstance(report, dict):
    bail("report.json must be a JSON object")
for key, kind in (("versions", list), ("heldout", list), ("notes", dict)):
    if not isinstance(report.get(key), kind):
        bail("report.json needs a %r key of type %s" % (key, kind.__name__))
if not all(isinstance(v, dict) for v in report["versions"] + report["heldout"]):
    bail("every entry in versions and heldout must be a JSON object")

# the test set
work, held = [[c for c in cases if c.get("split") == s] for s in ("working", "heldout")]
check(len(work) == 15, "15 working cases (found %d)" % len(work))
check(len(held) == 5, "5 held-out cases (found %d)" % len(held))
check(all(c.get("text") and c.get("expected") for c in cases),
      "every case has a non-empty text and expected")
check(len({c.get("text") for c in cases}) == len(cases),
      "no two cases share the same text")

# the version ladder
vs = report["versions"]
check(len(vs) >= 3, "at least 3 versions recorded (found %d)" % len(vs))
check([v.get("id") for v in vs] == ["v%d" % i for i in range(len(vs))],
      "version ids run v0, v1, v2 with no gaps")
check(all(v.get("total") == 15 for v in vs), "every version scored on all 15 cases")
check(all(isinstance(v.get("passed"), int) and 0 <= v["passed"] <= 15 for v in vs),
      "every passed count is a whole number from 0 to 15")
check(all(len((v.get("change") or "").split()) >= 3 for v in vs[1:]),
      "every version after v0 says what changed")

# the 30 percent bar, then the held-out run on v0 and the winner only
base = vs[0].get("passed", 0) if vs else 0
need = math.ceil(base * 1.30)
best_id = report.get("best")
best = next((v for v in vs if v.get("id") == best_id), None)
check(base <= 12, "v0 passes 12 or fewer, so the set is not trivial (v0=%s)" % base)
check(best is not None, "best (%r) names one of the versions" % best_id)
if best:
    check(best.get("passed", 0) >= need, "best beats v0 by 30 percent relative: "
          "need %d/15, got %d/15" % (need, best.get("passed", 0)))
ho = {h.get("id"): h for h in report["heldout"]}
check(set(ho) == {"v0", best_id}, "held-out run covers v0 and the best version only")
check(all(h.get("total") == 5 for h in ho.values()), "every held-out total is 5")

# one saved prompt file per version, then the three written notes
for v in vs:
    path = HERE / "prompts" / ("%s.txt" % v.get("id"))
    body = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    check(body != "", "prompts/%s.txt exists and is not empty" % v.get("id"))
v0 = HERE / "prompts" / "v0.txt"
check(v0.exists() and len(v0.read_text(encoding="utf-8").strip()) <= 200,
      "v0.txt is one short sentence, 200 characters or fewer")
for key in ("biggest_win", "surprise_that_failed", "heldout_vs_working"):
    check(len((report["notes"].get(key) or "").strip()) >= 40,
          "notes.%s is filled in" % key)

print("\nNot checked automatically: whether your pass rules are fair, whether each")
print("version changed only one thing, and the wording of the prompts themselves.")
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
