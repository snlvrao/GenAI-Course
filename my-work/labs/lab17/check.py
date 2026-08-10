# check.py - checks budget.py and runs.jsonl. Standard library only, no API key, no network.
import importlib.util, json, sys
from pathlib import Path

HERE, fails = Path(__file__).parent, []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok: fails.append(msg)

def t(fn, msg):  # run one check; print the error rather than a traceback
    try: check(bool(fn()), msg)
    except Exception as e: check(False, "%s  (%s: %s)" % (msg, type(e).__name__, e))

def need(name):
    f = HERE / name
    if not f.exists():
        print("FAIL " + name + " is not next to check.py. Create it, then run again.")
        sys.exit(1)
    return f

spec = importlib.util.spec_from_file_location("budget", need("budget.py"))
b = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(b)
except Exception as e:
    print("FAIL budget.py did not import (%s: %s)" % (type(e).__name__, e)); sys.exit(1)

P = getattr(b, "PRICES", {})
t(lambda: {"input_per_m", "output_per_m"} <= set(P), "PRICES has input_per_m and output_per_m")
t(lambda: P["input_per_m"] > 0 and P["output_per_m"] > 0, "both prices are above zero")
t(lambda: abs(b.cost(1_000_000, 0) - P["input_per_m"]) < 1e-9, "cost(1M input, 0) equals the input price")
t(lambda: abs(b.cost(0, 1_000_000) - P["output_per_m"]) < 1e-9, "cost(0, 1M output) equals the output price")
t(lambda: b.cost(0, 0) == 0, "cost(0, 0) is zero")
t(lambda: issubclass(b.BudgetExceeded, Exception), "BudgetExceeded is an Exception subclass")

missing = [n for n in ("PRICES", "cost", "Meter", "BudgetExceeded") if not hasattr(b, n)]
if missing:
    print("\nbudget.py is missing " + ", ".join(missing) + ". Add them, then run again."); sys.exit(1)

per_turn = b.cost(60_000, 300)   # one typical turn, priced with your own numbers
try:
    m = b.Meter(limit_usd=per_turn * 20, max_turns=50)
    m.add(60_000, 300); m.add(60_000, 300)
except Exception as e:
    m = None; check(False, "Meter(limit_usd=..., max_turns=...) plus two adds failed (%s)" % e)
if m is None:
    print("\nFix the Meter class first, then run this again."); sys.exit(1)
t(lambda: abs(m.spent - 2 * per_turn) < 1e-9, "two adds under the ceiling total exactly 2 turns of spend")
t(lambda: m.turn == 2, "Meter.turn is 2 after two adds")

def run_until_raise(meter, tokens):   # returns (turn it stopped on, message)
    for _ in range(10):
        try: meter.add(*tokens)
        except b.BudgetExceeded as e: return getattr(meter, "turn", None), str(e)
        except Exception as e: return None, "wrong exception type: " + type(e).__name__
    return None, ""

turn, msg = run_until_raise(b.Meter(limit_usd=per_turn * 2.5, max_turns=50), (60_000, 300))
t(lambda: turn is not None, "Meter raises BudgetExceeded once spend crosses the ceiling: " + msg)
t(lambda: turn == 3, "it raises on turn 3, mid-run, rather than after the loop finished")
t(lambda: "$" in msg and str(turn) in msg, "the message carries a dollar amount and the turn number")
turn2, _ = run_until_raise(b.Meter(limit_usd=10 ** 9, max_turns=4), (100, 10))
t(lambda: turn2 is not None and turn2 <= 5, "max_turns=4 stops the loop by the 5th call")

rows = []
for i, line in enumerate(need("runs.jsonl").read_text(encoding="utf-8").splitlines(), 1):
    if line.strip():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as e: check(False, "runs.jsonl line %d is not JSON (%s)" % (i, e))

KEYS = {"task", "limit_usd", "spent_usd", "turns", "stopped"}
done = [r for r in rows if not r.get("stopped")]
over = [r for r in rows if r.get("stopped")]
t(lambda: len(rows) >= 2, "runs.jsonl holds at least 2 runs")
t(lambda: all(KEYS <= set(r) for r in rows), "every run has task, limit_usd, spent_usd, turns, stopped")
t(lambda: done and over, "one run finished and one run hit its ceiling")
t(lambda: all(r["spent_usd"] <= r["limit_usd"] for r in done), "no finished run spent more than its ceiling")
t(lambda: all(r["spent_usd"] > 0 and r["turns"] >= 1 for r in rows), "every run logged real spend and at least 1 turn")
t(lambda: min(r["limit_usd"] for r in over) < max(r["limit_usd"] for r in done), "the tiny ceiling really is smaller than the normal one")

print("\nNot checked automatically: that Meter.add is wired into your agent loop,")
print("and that the token counts came from the provider's usage block.")
if fails:
    print("\n%d check(s) failed." % len(fails)); sys.exit(1)
print("\nALL CHECKS PASSED")
