# check.py - checks the Module 9 mini-project. Run from my-work/labs/lab09: python check.py
# Reads what your run recorded. It never calls a model, so no API key, no internet.
import json, sys, pathlib, importlib.util
HERE, fails, NOTHING = pathlib.Path(__file__).resolve().parent, [], object()

def check(ok, msg, got=NOTHING):      # got is printed only when the check fails
    print(("PASS " if ok else "FAIL ") + msg + ("" if ok or got is NOTHING else ", got %.100r" % (got,)))
    if not ok: fails.append(msg)

def need(name):                       # stop with a sentence, not a traceback
    p = HERE / name
    if not p.exists():
        print("FAIL missing file: %s" % p)
        print("Write it first. The mini-project steps give the exact shape."); sys.exit(1)
    return p

res_path, tools_path = need("results.json"), need("tools.py")
try:
    data = json.loads(res_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    print("FAIL results.json is not valid JSON: %s" % e); sys.exit(1)
spec = importlib.util.spec_from_file_location("learner_tools", tools_path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)      # tools.py must not call a model at import time
except Exception as e:
    print("FAIL tools.py failed to import: %r" % e); sys.exit(1)

# --- the three tool definitions --------------------------------------------
tools = getattr(mod, "TOOLS", None)
ok_tools = isinstance(tools, list) and len(tools) == 3
check(ok_tools, "TOOLS is a list of exactly 3 definitions")
names, props = [], {}
if ok_tools:
    for t in tools:
        fn = (t or {}).get("function") or {}
        par = fn.get("parameters") or {}
        names.append(fn.get("name"))
        props[fn.get("name")] = par.get("properties") if isinstance(par.get("properties"), dict) else {}
        check(bool(fn.get("description")) and bool(props[fn.get("name")]),
              "tool %r has a description and named parameters" % fn.get("name"))
    check(len(set(names)) == 3 and all(names), "three different tool names: %s" % names)

# --- dispatch returns data, never an exception ------------------------------
run_tool = getattr(mod, "run_tool", None)
check(callable(run_tool), "run_tool(name, raw_args) exists")
if callable(run_tool):
    def call(n, raw):                 # a raise here is itself a failure, not a crash
        try: return run_tool(n, raw)
        except Exception as e: return {"raised instead of returning a dict": repr(e)}
    for n, raw, msg in [("no_such_tool", "{}", "unknown tool name returns an error dict"),
                        ("get_price", '{"item":"bread","quantity":-5}', "quantity -5 is refused"),
                        ("get_price", "not json", "unparseable arguments return an error dict")]:
        r = call(n, raw)
        check(isinstance(r, dict) and "error" in r, msg, r)
    for n, raw, want, msg in [
            ("convert_mass", '{"value":5,"from_unit":"kg","to_unit":"lb"}', 11.02, "5 kg is about 11.02 lb"),
            ("convert_currency", '{"amount":10,"from_currency":"USD","to_currency":"EUR"}', 9.20,
             "10 USD is 9.20 EUR at the fixed 0.92 rate")]:
        r = call(n, raw)
        v = r.get("converted") if isinstance(r, dict) else r
        check(isinstance(v, (int, float)) and abs(v - want) < 0.02, msg, r)

# --- the recorded run -------------------------------------------------------
runs, keys = data.get("runs"), {"question", "expected_tool", "chosen_tool", "arguments"}
ok_runs = isinstance(runs, list) and len(runs) == 5 and all(
    isinstance(r, dict) and keys <= set(r) for r in runs)
check(ok_runs, "results.json holds 5 runs, each with %s" % sorted(keys))
if ok_runs:
    check(sum(r["expected_tool"] is None for r in runs) == 1, "exactly one question expects no tool")
    check(all(r["chosen_tool"] in names + [None] for r in runs),
          "every chosen_tool is one of your three tools, or null")
    wrong = [r["question"] for r in runs if r["chosen_tool"] != r["expected_tool"]]
    check(not wrong, "5 of 5 picks match your prediction. Wrong: %s" % wrong)
    bad = [r["question"] for r in runs if r["chosen_tool"]
           and not set(r["arguments"] or {}) <= set(props.get(r["chosen_tool"], {}))]
    check(not bad, "no invented argument names. Offenders: %s" % bad)

tok = data.get("tokens") or {}
n0, n3 = tok.get("no_tools"), tok.get("three_tools")
check(isinstance(n0, int) and isinstance(n3, int) and n3 > n0,
      "prompt_tokens recorded, 3 tools cost more input than none (%s vs %s)" % (n0, n3))
print("\nNot checked automatically: whether your description wording is the clearest fix.")
if fails: print("%d CHECK(S) FAILED" % len(fails)); sys.exit(1)
print("ALL CHECKS PASSED")
