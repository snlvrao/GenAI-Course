"""check.py - checks rag_report.json from the module 10 mini-project. Run it from
my-work/labs/lab10 with: python check.py   It reads what your run recorded: no key, no internet."""
import json, os, re, sqlite3, sys

REPORT = "rag_report.json"
REFUSAL = "I don't have that in these documents."  # must match REFUSAL in ask.py
fails = []

def check(ok, msg, detail=""):
    print("PASS " + msg if ok else f"FAIL {msg} -> {detail}")
    if not ok: fails.append(msg)
num = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
norm = lambda s: " ".join(str(s or "").split())  # compare text, ignore whitespace
ids = lambda rs: [r.get("id") for r in rs]

if not os.path.exists(REPORT):
    sys.exit(f"FAIL {REPORT} is not in this folder. Run your ten questions and write it here.")
try:
    with open(REPORT, encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    sys.exit(f"FAIL {REPORT} is not valid JSON: {e}")

rows = data.get("questions") or []
thr = num(data.get("threshold"))
yes = [r for r in rows if r.get("answerable") is True]
no = [r for r in rows if r.get("answerable") is False]
ans = [r for r in rows if not r.get("refused")]
marks = lambda r: set(re.findall(r"\[(\d+)\]", r.get("answer") or ""))
cited = lambda r: {str(c.get("n")) for c in (r.get("citations") or [])}

# the test set itself
check(thr is not None, "threshold is a number", f"got {data.get('threshold')!r}")
check(len(rows) == 10, "ten questions recorded", f"got {len(rows)}")
check(len(yes) == 5 and len(no) == 5, "five answerable, five not", f"got {len(yes)} and {len(no)}")
check(all(num(r.get("top_score")) is not None for r in rows), "every question records a top_score",
      "at least one row has no numeric top_score")

# the refusal gate is one rule, applied the same way to every question
off = ids([r for r in rows if thr is not None and num(r.get("top_score")) is not None
           and bool(r.get("refused")) != (r["top_score"] < thr)])
check(not off, "refused always equals top_score < threshold", f"ids {off} disagree with the rule")
leaked = ids([r for r in no if not r.get("refused")])
check(not leaked, "all five unanswerable questions refused", f"ids {leaked} answered instead")
loud = ids([r for r in rows if r.get("refused") and (r.get("answer") != REFUSAL or r.get("citations"))])
check(not loud, "refusals use the exact refusal line and cite nothing", f"ids {loud}")

# receipts: markers resolve, and each citation carries readable text
nocite = ids([r for r in ans if not r.get("citations")])
check(not nocite, "every answered question lists its citations", f"ids {nocite} cite nothing")
bad = ids([r for r in ans if not marks(r) or marks(r) - cited(r)])
check(not bad, "answers carry [n] markers and each matches a citation", f"ids {bad}")
thin = ids([r for r in ans for c in (r.get("citations") or []) if len(norm(c.get("chunk_text"))) < 50])
check(not thin, "every citation carries its chunk text, 50 characters or more", f"ids {thin}")

# the judge pass, kept binary
verdicts = [j for r in ans for j in (r.get("judged") or [])]
check(len(verdicts) >= len(ans), "every answered question has a judged sentence",
      f"{len(verdicts)} verdicts for {len(ans)} answers")
check(bool(verdicts) and all(isinstance(j.get("supported"), bool) for j in verdicts),
      "judge verdicts are true or false, not a 1 to 5 score", "no verdicts, or one is not a boolean")
unlinked = ids([r for r in ans for j in (r.get("judged") or []) if str(j.get("cite")) not in cited(r)])
check(not unlinked, "each judged sentence names a citation that exists", f"ids {unlinked}")
absent = ids([r for r in ans for j in (r.get("judged") or [])
              if j.get("supported") and norm(j.get("sentence")) not in norm(r.get("answer"))])
check(not absent, "supported sentences appear word for word in the answer", f"ids {absent}")

# the two rates, reported separately
rates = data.get("rates") or {}
for key, group, hits in [("answered_when_answerable", yes, sum(1 for r in yes if not r.get("refused"))),
                         ("refused_when_unanswerable", no, sum(1 for r in no if r.get("refused")))]:
    want, got = (hits / len(group) if group else -1), num(rates.get(key))
    check(got is not None and abs(got - want) < 0.005, f"rates.{key} matches the rows",
          f"rows give {want:.2f}, report says {rates.get(key)!r}")

# trace the quoted text back to the index, when the index is here
stored = None
if os.path.exists("rag.db"):
    try: stored = {norm(t[0]) for t in sqlite3.connect("rag.db").execute("SELECT text FROM chunks")}
    except sqlite3.Error: pass
if stored is None:
    print("SKIP no readable rag.db here, so citation text was not traced back to your index")
else:
    ghosts = ids([r for r in ans for c in (r.get("citations") or []) if norm(c.get("chunk_text")) not in stored])
    check(not ghosts, "every cited chunk appears word for word in rag.db", f"ids {ghosts} quote text not in the index")

print("NOT CHECKED: whether each answer is factually right, and whether your threshold sits in the right place.")
print(f"{len(fails)} check(s) failed" if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
