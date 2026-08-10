"""check.py - verify the lab 06 router mini-project. Run it from my-work/labs/lab06."""
import importlib.util
import json
import pathlib
import sys

REPORT = pathlib.Path("router_report.json")
ROUTER = pathlib.Path("router.py")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def die(msg):
    print("FAIL " + msg)
    sys.exit(1)

def cost(calls):
    return sum((c["in"] * prices[c["model"]]["in"]
                + c["out"] * prices[c["model"]]["out"]) / 1e6 for c in calls)

def ok_call(c):
    return (c.get("model") in prices and isinstance(c.get("in"), int) and c["in"] > 0
            and isinstance(c.get("out"), int) and c["out"] >= 0)

# Load both artefacts. Plain messages, never a traceback.
if not REPORT.exists():
    die("router_report.json not found. Run this from my-work/labs/lab06.")
if not ROUTER.exists():
    die("router.py not found. It must define route(question) -> 'cheap' or 'strong'.")
try:
    r = json.loads(REPORT.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    die(f"router_report.json is not valid JSON: {e}")
prices = r.get("prices_used", {})
if not {"cheap", "strong"} <= set(prices):
    die("prices_used needs a 'cheap' and a 'strong' entry, each with 'in' and 'out'.")
spec = importlib.util.spec_from_file_location("router", ROUTER)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    die(f"router.py would not import: {type(e).__name__}: {e}")
if not callable(getattr(mod, "route", None)):
    die("router.py has no route() function.")

qs = r.get("questions", [])
check(len(qs) == 20, f"20 questions recorded (found {len(qs)})")
easy = sum(1 for q in qs if q.get("label") == "easy")
hard = sum(1 for q in qs if q.get("label") == "hard")
check(easy == 10 and hard == 10, f"10 easy + 10 hard hand labels (found {easy} + {hard})")
cheap_n = sum(1 for q in qs if q.get("routed_to") == "cheap")
check(0 < cheap_n < len(qs), f"router splits traffic ({cheap_n} cheap, {len(qs)-cheap_n} strong)")
# Re-run the learner's own router. The report must be reproducible from it.
try:
    off = [q.get("id") for q in qs if mod.route(q.get("text", "")) != q.get("routed_to")]
except Exception as e:
    die(f"route() raised on one of your questions: {type(e).__name__}: {e}")
check(not off, f"route() reproduces every recorded decision (mismatched ids: {off})")
bad = sorted({q.get("id") for q in qs for c in
              list(q.get("router_calls", [])) + [q.get("strong_only_call", {})] if not ok_call(c)})
check(not bad, f"every call names a model and whole token counts (bad ids: {bad})")
if bad:
    die("Fix the calls above. The cost checks cannot run without them.")

routed = sum(cost(q["router_calls"]) for q in qs)
alone = sum(cost([q["strong_only_call"]]) for q in qs)
t = r.get("totals", {})
check(abs(routed - t.get("cost_routed", -1)) < 1e-6,
      f"cost_routed matches the calls (yours {t.get('cost_routed')}, recomputed {routed:.6f})")
check(abs(alone - t.get("cost_strong_only", -1)) < 1e-6,
      f"cost_strong_only matches (yours {t.get('cost_strong_only')}, recomputed {alone:.6f})")
saved = 100 * (1 - routed / alone) if alone else 0.0
check(abs(saved - t.get("percent_saved", -999)) < 0.1,
      f"percent_saved matches (yours {t.get('percent_saved')}, recomputed {saved:.1f})")
wrong = sum(1 for q in qs if q.get("routed_to") == "cheap" and q.get("cheap_correct") is False)
check(wrong == t.get("misrouted_count"),
      f"misrouted_count matches your flags (yours {t.get('misrouted_count')}, counted {wrong})")
esc = [q.get("id") for q in qs if q.get("escalated")
       and (len(q.get("router_calls", [])) < 2 or q["router_calls"][-1]["model"] != "strong")]
check(not esc, f"escalated questions are billed for both calls (bad ids: {esc})")
print("\nNot checked automatically: whether your easy/hard labels are fair, and whether your\n"
      "cheap_correct flags are right. Those are your reading of the answers.")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED")
    sys.exit(1)
print("\nALL CHECKS PASSED")
