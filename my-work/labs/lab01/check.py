"""Checks runs.json, the artefact from the Module 1 mini-project.
Run: python my-work/labs/lab01/check.py   (no API key, no internet, stdlib only)
"""
import hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "runs.json"          # what compare.py wrote
SRC = HERE / "compare.py"         # the script that wrote it
fails = 0


def check(ok, msg):
    """Print one PASS/FAIL line and remember the failures."""
    global fails
    ok = bool(ok)
    print(("PASS " if ok else "FAIL ") + msg)
    fails += 0 if ok else 1
    return ok


def stop(msg):
    """Artefact missing or unreadable: say so plainly, no traceback."""
    print("FAIL " + msg)
    print("1 check failed")
    sys.exit(1)


if not ART.exists():
    stop("runs.json not found at %s. Run compare.py once per provider first." % ART)
try:
    data = json.loads(ART.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    stop("runs.json is not valid JSON (line %d): %s" % (e.lineno, e.msg))
if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
    stop("runs.json must be an object with a 'runs' list. Check the shape in the steps.")

runs = data["runs"]
check(isinstance(data.get("question"), str) and data.get("question", "").strip(),
      "'question' is a non-empty string")
if not check(len(runs) == 2, "'runs' holds exactly 2 entries (found %d)" % len(runs)):
    print("%d check(s) failed" % fails)
    sys.exit(1)

# every run carries the same five fields, with usable values
need = {"provider": str, "model": str, "seconds": (int, float),
        "answer": str, "script_sha256": str}
for i, r in enumerate(runs):
    if not isinstance(r, dict):
        check(False, "runs[%d] is not a JSON object" % i)
        continue
    bad = [k for k, t in need.items() if not isinstance(r.get(k), t)]
    check(not bad, "runs[%d] has provider, model, seconds, answer, script_sha256%s"
          % (i, " (missing or wrong type: %s)" % ", ".join(bad) if bad else ""))
    check(isinstance(r.get("seconds"), (int, float)) and r.get("seconds", 0) > 0,
          "runs[%d].seconds is above zero (got %r)" % (i, r.get("seconds")))
    check(len(str(r.get("answer", ""))) >= 10,
          "runs[%d].answer is at least 10 characters" % i)
    for k in ("input_tokens", "output_tokens"):
        v = r.get(k)
        check(v is None or (isinstance(v, int) and not isinstance(v, bool) and v > 0),
              "runs[%d].%s is a positive whole number, or null if the "
              "provider reported none (got %r)" % (i, k, v))

a, b = runs[0], runs[1]
check(a.get("model") != b.get("model"),
      "the two runs name different models (%r vs %r)" % (a.get("model"), b.get("model")))
check(a.get("answer") != b.get("answer"),
      "the two answers differ (identical text means one run was copied, not called)")

# the point of the exercise: same code, different provider
same = a.get("script_sha256") == b.get("script_sha256")
check(same, "both runs recorded the same script_sha256, so compare.py did not change between them")
if SRC.exists():
    now = hashlib.sha256(SRC.read_bytes()).hexdigest()
    check(same and a.get("script_sha256") == now,
          "that hash matches compare.py as it stands now (%s...)" % now[:12])
else:
    check(False, "compare.py not found beside check.py, so the hash cannot be verified")

print()
print("Not checked automatically: whether either answer is factually correct. Read them yourself.")
print("ALL CHECKS PASSED" if fails == 0 else "%d check(s) failed" % fails)
sys.exit(0 if fails == 0 else 1)
