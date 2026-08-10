"""check.py - checks the Module 23 mini-project. Run: python check.py"""
import json, os, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER, REPORT = os.path.join(HERE, "serve_mine.py"), os.path.join(HERE, "serve_report.json")
fails = []

def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

def die(*lines):     # print a plain explanation, then stop
    print("\n".join(lines)); sys.exit(1)

if not os.path.isfile(SERVER):
    die("FAIL missing file: " + SERVER,
        "This check starts your server itself, so it cannot run without that file.")
if not os.path.isfile(REPORT):
    die("FAIL missing file: " + REPORT,
        "The mini-project steps say to create it. Do that, then run check.py again.")

s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
BASE = "http://127.0.0.1:%d/v1" % PORT      # a free port, so a server you left running is safe
proc = subprocess.Popen([sys.executable, SERVER], cwd=HERE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, env=dict(os.environ, MINE_PORT=str(PORT)))
models, stop = None, time.time() + 120
while models is None and time.time() < stop and proc.poll() is None:
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=3) as r:
            models = json.load(r)
    except Exception:
        time.sleep(0.5)
try:
    if models is None:
        die("FAIL your server never answered on 127.0.0.1:%d within 120 seconds." % PORT,
            "Run 'python serve_mine.py' in a terminal and read the error it prints.")
    check(isinstance(models.get("data"), list) and len(models["data"]) > 0,
          "GET /v1/models answered with a non-empty data list")
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(
        {"model": "mine", "messages": [{"role": "user", "content": "ROMEO:"}],
         "max_tokens": 24}).encode("utf-8"), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.load(r)
    except Exception as e:
        die("FAIL POST /v1/chat/completions failed: %s: %s" % (type(e).__name__, e),
            "Your server started but could not answer one real request.")
    ch = body.get("choices")
    check(isinstance(ch, list) and len(ch) >= 1, "the reply carries a choices list")
    m = (ch[0].get("message") or {}) if isinstance(ch, list) and ch else {}
    check(m.get("role") == "assistant", "choices[0].message.role is 'assistant'")
    check(isinstance(m.get("content"), str) and m.get("content") != "",
          "choices[0].message.content is a non-empty string")
    check(body.get("object") == "chat.completion", "object is 'chat.completion'")
    u = body.get("usage") or {}
    ok = all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    check(ok, "usage carries three whole-number token counts")
    check(ok and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
          "usage.total_tokens equals prompt_tokens plus completion_tokens")
finally:
    proc.terminate()

try:
    with open(REPORT, encoding="utf-8") as f:
        rep = json.load(f)
except json.JSONDecodeError as e:
    die("FAIL serve_report.json is not valid JSON: %s" % e)
c, l, a, v = (rep.get(k) or {} for k in ("chat", "via_llm_py", "agent", "verdict"))
check(isinstance(c.get("content"), str) and len(c.get("content") or "") >= 20,
      "report chat.content keeps at least 20 characters your model wrote")
check(all(isinstance(c.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")),
      "report chat records prompt_tokens, completion_tokens and total_tokens")
check(l.get("provider") == "mine", "report via_llm_py.provider is 'mine'")
check(isinstance(l.get("reply"), str) and (l.get("reply") or "").strip() != "",
      "report via_llm_py.reply holds the text llm.py got back")
check(a.get("outcome") in ("pass", "partial", "fail"), "report agent.outcome is pass, partial or fail")
check(isinstance(a.get("steps_used"), int) and (a.get("steps_used") or 0) > 0,
      "report agent.steps_used is a positive whole number")
check(isinstance(a.get("tool_calls_parsed"), int) and a.get("tool_calls_parsed", -1) >= 0,
      "report agent.tool_calls_parsed is a whole number, zero allowed")
check(len(str(a.get("what_happened", ""))) >= 40, "report agent.what_happened is a real sentence")
check(v.get("works_as_endpoint") is True, "report verdict.works_as_endpoint is true")
check(isinstance(v.get("passes_agent_lab"), bool), "report verdict.passes_agent_lab is true or false")
check(len(str(v.get("honest_use", ""))) >= 60, "report verdict.honest_use says where this model belongs")
print("not checked automatically: whether your honest_use sentence is honest.")
if fails:
    die("%d CHECK(S) FAILED" % len(fails))
print("ALL CHECKS PASSED")
