# check.py - checks decision.json for the Module 14 mini-project.
# Standard library only. No API key, no internet, no model call.
import json, os, re, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "decision.json"
CAPS = ["state_between_steps", "checkpoints", "retries", "human_approval", "tracing"]
RATINGS = ("needed", "nice", "no")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

# Load the artefact. Stop with a sentence, never a traceback.
if not os.path.exists(PATH):
    print("FAIL " + PATH + " not found. Run this from my-work/labs/lab14, where the file lives.")
    sys.exit(1)
try:
    with open(PATH, encoding="utf-8") as f:
        d = json.load(f)
except json.JSONDecodeError as e:
    print("FAIL %s is not valid JSON: %s at line %d" % (PATH, e.msg, e.lineno))
    sys.exit(1)
if not isinstance(d, dict):
    print("FAIL %s must hold one JSON object, got a %s" % (PATH, type(d).__name__))
    sys.exit(1)
print("PASS " + PATH + " parsed as a JSON object")

sc = d.get("scenario")
check(sc in ("A", "B", "C"), "scenario is A, B or C (got %r)" % (sc,))

caps = d.get("capabilities") if isinstance(d.get("capabilities"), dict) else {}
check(sorted(caps) == sorted(CAPS), "capabilities holds exactly: " + ", ".join(CAPS))
bad = sorted(k for k, v in caps.items() if v not in RATINGS)
check(not bad, "every rating is needed/nice/no (wrong: %s)" % (", ".join(bad) or "none"))
needed = [k for k, v in caps.items() if v == "needed"]

# These rules come from the scenario wording, not from taste.
if sc == "A":
    check(caps.get("human_approval") != "needed",
          "scenario A runs with no human, so human_approval is not 'needed'")
if sc == "B":
    check(caps.get("human_approval") == "needed" and caps.get("checkpoints") == "needed",
          "scenario B waits for a person and survives a restart, so human_approval "
          "and checkpoints are both 'needed'")
if sc == "C":
    check(caps.get("tracing") == "needed",
          "scenario C answers to compliance, so tracing is 'needed'")
check(d.get("deciding_capability") in needed,
      "deciding_capability %r is one you marked 'needed'" % (d.get("deciding_capability"),))

n = d.get("diy_count")
check(isinstance(n, int) and 0 <= n <= 5, "diy_count is a whole number 0 to 5 (got %r)" % (n,))
choice = str(d.get("choice", "")).strip().lower()
if isinstance(n, int) and n <= 1:
    check(choice == "none", "diy_count is %d, so choice must be 'none' (got %r)" % (n, choice))
else:
    check(choice not in ("", "none"), "diy_count is above 1, so choice names a framework")
sl = [str(s).strip().lower() for s in (d.get("shortlist") or [])]
if choice == "none":
    print("PASS choice is 'none', so no shortlist is required")
else:
    check(len(sl) == 2 and len(set(sl)) == 2, "shortlist has two different names (got %r)" % (sl,))
    check(choice in sl, "choice %r is one of the two you shortlisted" % (choice,))

pv = str(d.get("python_version", ""))
check(re.match(r"^3\.\d+", pv) is not None, "python_version looks like 3.x (got %r)" % (pv,))
if choice == "crewai" and re.match(r"^3\.\d+", pv):
    check(int(pv.split(".")[1]) < 14, "CrewAI needs Python older than 3.14, you run " + pv)

# Mechanical checks on the writing. Decimals are joined first, so 3.14 is not a sentence end.
txt = str(d.get("defence", "")).strip()
flat = re.sub(r"(\d)\.(\d)", r"\1\2", txt)
sents = [s for s in re.split(r"[.!?]+", flat) if s.strip()]
check(len(sents) == 3, "defence is exactly 3 sentences (found %d)" % len(sents))
check(0 < len(txt.split()) < 80, "defence is under 80 words (found %d)" % len(txt.split()))
check(len(str(d.get("accepted_cost", "")).split()) >= 3,
      "accepted_cost is a phrase naming what you give up, not one word")
cm = str(d.get("change_my_mind", "")).strip()
check(cm != "" and "\n" not in cm, "change_my_mind is one non-empty line")

print("NOT CHECKED AUTOMATICALLY: whether your reasoning is sound. Read the defence "
      "back in a week and see if you still agree with it.")
if fails:
    print("%d CHECKS FAILED" % len(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
