# Lab 16: Trace and judge your agent

**Module 16: Knowing whether your agent works**

You are going to give your Module 12 agent a flight recorder, read what it records, then build a pass/fail checker and measure how often it agrees with you. A span is one timed record of one thing the agent did, and a trace is every span from a single user question tied together, which is what lets you see where a run went wrong instead of only that it did. Everything here runs on the Python standard library plus the shared llm.py helper you already have, so there is nothing new to install and no dashboard to sign up for. Before you start, make sure python llm.py works, see setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Write the span logger

This step builds the recorder that every later step reads from. A span is one timed record of one thing your agent did, such as one model call or one tool call, with its inputs, how long it took, and whether it raised an error. A trace is all the spans from one user question, tied together by a shared trace_id. The tricky part is knowing which span sits inside which, and this file solves it with contextvars, a standard library module that holds a value which any nested code can read without you passing it in as an argument. Without that you would have to thread a parent id through every function in your agent by hand, and you would stop bothering within a week. Each span writes its own JSON line the moment its with block exits, so if the agent crashes halfway you still have every step up to the crash sitting on disk. Running this file right now prints nothing useful, it is a library, and the next step is what proves it works.

```python
"""spans.py - tiny span logger, standard library only."""
import json, os, time, uuid, contextvars
from contextlib import contextmanager
from pathlib import Path

LOG_PATH = Path(os.getenv("SPAN_LOG", "traces.jsonl"))
_current = contextvars.ContextVar("current_span", default=None)
_trace = contextvars.ContextVar("trace_id", default=None)


def _write(rec):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


@contextmanager
def span(name, **attrs):
    new_trace = _trace.get() is None
    tid_token = _trace.set(uuid.uuid4().hex[:12]) if new_trace else None
    rec = {
        "trace_id": _trace.get(),
        "span_id": uuid.uuid4().hex[:8],
        "parent_id": _current.get(),
        "name": name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attrs": attrs,
    }
    cur_token = _current.set(rec["span_id"])
    t0 = time.perf_counter()
    try:
        yield rec["attrs"]           # add fields as the step runs
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _current.reset(cur_token)
        if tid_token is not None:
            _trace.reset(tid_token)
        _write(rec)


def read(path=None):
    path = Path(path or LOG_PATH)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = {}
    for r in rows:
        traces.setdefault(r["trace_id"], []).append(r)
    return traces


def show(rows):
    kids = {}
    for r in rows:
        kids.setdefault(r["parent_id"], []).append(r)

    def walk(pid, depth):
        for r in kids.get(pid, []):
            flag = "  OK " if r["status"] == "ok" else " FAIL"
            print(f'{flag} {"  " * depth}{r["name"]}  {r["ms"]}ms  {r["attrs"]}')
            if r.get("error"):
                print(f'      {"  " * depth}{r["error"]}')
            walk(r["span_id"], depth + 1)

    walk(None, 0)


if __name__ == "__main__":
    for tid, rows in read().items():
        print(f"--- trace {tid} ---")
        show(rows)
```

- `LOG_PATH = Path(os.getenv("SPAN_LOG", "traces.jsonl"))`: This reads the environment variable SPAN_LOG once, at the moment the module is imported, and falls back to traces.jsonl if it is not set. Because it runs at import time, setting SPAN_LOG after the line 'import spans' has no effect at all, which is the rule that steps 2 and 8 both depend on.
- `_current = contextvars.ContextVar("current_span", default=None)`: A ContextVar is a slot that holds a value for the current flow of execution, so nested code can read it without being handed it as an argument. This one holds the id of the span you are currently inside, which becomes the parent_id of the next span you open, and it starts as None so the outermost span records no parent.
- `tid_token = _trace.set(uuid.uuid4().hex[:12]) if new_trace else None`: Only the first span in a run creates a trace id, and every span opened inside it reads that same id back out. The set() call returns a token that remembers the previous value, which is what lets the finally block put everything back exactly as it was.
- `yield rec["attrs"]`: This hands the attributes dictionary back to your 'with ... as s' variable, so you can add facts while the step is running, such as s["out_tokens"] = 812. It is the same dictionary object that gets written to disk afterwards, so anything you put into it lands in the log.
- `the finally block`: Everything that must happen no matter what lives here: the duration is measured, the two context variables are put back, and the line is written. That is why a span that raised an exception still appears in the log marked as an error, instead of vanishing.
- `walk(None, 0)`: The tree is rebuilt by starting at parent_id None, which is the root span, then recursing into each span's children. This is why the order of lines in the file does not matter, and it needs to be, because inner spans finish first and are written before their own parents.

**The maths, spelled out**

```
Two numbers in this file are worth understanding properly.

1) The duration field.
Formula: ms = round((t1 - t0) * 1000, 1)
t0 is the reading from time.perf_counter() taken just before the step runs, t1 is the reading taken in the finally block, and both are seconds as floating point numbers. perf_counter is used instead of time.time() because it only ever counts forwards, so a clock adjustment mid run cannot produce a negative duration.
Worked example: t0 = 12.340000 and t1 = 13.827412. The difference is 1.487412 seconds. Times 1000 that is 1487.412 milliseconds. Rounded to one decimal place the log stores 1487.4.
Intuition: you record milliseconds because most model and tool calls land somewhere between 100 and 5000 of them, and whole seconds would flatten nearly all of them to 1 or 2.

2) How safe the shortened ids are.
Formula: chance of at least one clash among n ids is roughly (n * n) / (2 * N), where N is the number of possible ids.
uuid4().hex is 32 hexadecimal characters. Each hex character carries 4 bits, so [:8] keeps 8 * 4 = 32 bits, giving N = 2 to the power 32 = 4,294,967,296 possible span ids. The trace id keeps 12 characters, so 48 bits, giving 281,474,976,710,656 possible trace ids.
Worked example for span ids with n = 1000 spans in one file: n * n = 1,000,000, and 2 * N = 8,589,934,592. Divide: 1,000,000 / 8,589,934,592 = 0.000116, which is about 1 chance in 8,600.
Intuition: cutting the id to 8 characters keeps the log readable by eye and still almost never repeats at lab scale, but it would not be a safe choice for millions of spans.
```

> **Watch out:** Running python spans.py before any run has happened raises FileNotFoundError, because read() tries to open traces.jsonl and that file does not exist yet.

### 2. Prove it records a tree, including failures

Test the recorder on a fake run before you point it at your real agent. If you skip this and a trace later looks wrong, you will not know whether the agent misbehaved or the logger did, and you will spend an afternoon reading the wrong file. The fake run opens one nested model span and one tool span, and the second call deliberately raises an error inside the tool span. Notice that the exception is caught outside the with block and the span is still written with status: error, because the writing happens in a finally block that runs on the way out either way. On screen you should see two blocks headed --- trace ... ---, each with an indented tree, and the second one carries a FAIL line for tool_call with ValueError: index missing printed underneath it. The file t.jsonl will hold six lines, three per run, with children written before their parents because inner blocks always exit first.

```python
# test_spans.py
import os
os.environ["SPAN_LOG"] = "t.jsonl"      # must be set BEFORE importing spans
from pathlib import Path
Path("t.jsonl").unlink(missing_ok=True)
import spans

def fake_run(q):
    with spans.span("agent_run", question=q) as s:
        with spans.span("llm_call", model="fake") as l:
            l["out_tokens"] = 12
        try:
            with spans.span("tool_call", tool="search", args={"q": q}):
                if "boom" in q:
                    raise ValueError("index missing")
        except ValueError:
            pass
        s["steps"] = 2

fake_run("hello")
fake_run("boom now")
for tid, rows in spans.read().items():
    print(f"--- trace {tid} ---")
    spans.show(rows)
```

- `os.environ["SPAN_LOG"] = "t.jsonl"`: This has to be the first thing the file does, above 'import spans', because spans.py reads SPAN_LOG at import time and never checks it again. It keeps the test output in its own file so your fake runs do not get mixed into the real traces.jsonl you read in step 4.
- `Path("t.jsonl").unlink(missing_ok=True)`: The logger only ever appends, so without this delete you would be reading today's fake run stacked on top of every previous one. The missing_ok=True part means the very first run does not fail just because the file is not there yet.
- `with spans.span("llm_call", model="fake") as l:  /  l["out_tokens"] = 12`: These two lines show the two ways to attach facts to a span: keyword arguments for what you know when the step starts, and the yielded dictionary for what you only learn once it has finished. Token counts, reply lengths and result sizes all belong in the second group.
- `the try / except ValueError around the tool span`: The exception is raised inside the tool span and caught outside it, which is exactly what a real agent does when it recovers from a failed tool call. The tool span is still written with status error while the agent_run span around it stays ok, so the log shows you both the failure and the recovery.
- `fake_run("boom now")`: The word boom is the trigger for the raise, so this second call is the one that produces the FAIL line. One clean case and one broken case is the smallest test that proves both the success path and the error path actually write to disk.

> **Watch out:** If the line setting SPAN_LOG ends up anywhere below 'import spans', it is silently ignored and your fake runs get appended to traces.jsonl instead, quietly polluting the real log you read in step 4.

### 3. Wire the logger into your Module 12 agent

Now put the recorder into the agent loop you built in Module 12. You wrap three things and no more: the whole run, each model call, and each tool call. That gives you one root span per question with its children underneath, which is exactly the tree you need to find the step where things went wrong. SYSTEM, parse_tool_call and your tools dictionary all come from your Module 12 file, so paste this structure around your existing code rather than replacing it. Store result_chars plus only the first 200 characters of each tool result, because a full retrieval result is often several thousand characters and your log would otherwise be mostly retrieved text. Be honest with yourself about what that cap costs: result_head is the only evidence your judge sees in step 5, so a fact sitting at character 900 of a tool result will look invented to the judge even though the agent read it. Raise 200 to 1000 if your tool results are long, and accept a bigger log in exchange.

```python
# agent.py  (your Module 12 loop, now instrumented)
import sys
sys.path.insert(0, "../_shared")
from llm import chat
from spans import span

def run_agent(question, tools, max_steps=6):
    with span("agent_run", question=question) as run:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question}]
        for step in range(max_steps):
            with span("llm_call", step=step) as s:
                reply = chat(messages)
                s["reply_chars"] = len(reply)
            call = parse_tool_call(reply)          # your Module 12 parser
            if call is None:
                run["final"] = reply
                run["steps"] = step + 1
                return reply
            with span("tool_call", tool=call["name"], args=call["args"]) as t:
                result = tools[call["name"]](**call["args"])
                t["result_chars"] = len(str(result))
                t["result_head"] = str(result)[:200]
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"TOOL RESULT: {result}"}]
        run["final"] = "gave up"
        run["steps"] = max_steps
        return "gave up"
```

- `sys.path.insert(0, "../_shared")`: This tells Python to look in the shared folder first when importing, which is how every lab finds llm.py without installing anything. If you run agent.py from a different folder the relative path will not resolve, so run it from inside the lab16 directory.
- `with span("agent_run", question=question) as run:`: This is the outermost span, so it is the one that creates the trace id, and every span opened inside the loop attaches itself to it automatically. It also holds the two facts you look up most often later, the question at the start and run["final"] at the end.
- `reply = chat(messages)`: This is the one line you will probably have to change. The shared llm.py chat() takes a plain prompt string with an optional system= argument, not a message list, so either flatten your messages into a single string or switch to chat_raw(messages).choices[0].message.content, which does accept a list.
- `t["result_head"] = str(result)[:200]`: Only the first 200 characters of the tool result are stored, sitting next to the full length in result_chars, so you can still see that a result was 4000 characters long without keeping all 4000. Remember that this exact text becomes the evidence your judge reads in step 5.
- `messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"TOOL RESULT: {result}"}]`: The full result goes back to the model, not the truncated head, so the agent is never limited by your logging cap. The tool output is sent back as a user turn because this loop parses tool calls out of plain text rather than using the provider's tool calling API, so there is no tool role available to put it in.
- `run["final"] = "gave up"`: If the loop reaches max_steps without the model producing a final answer, the run is recorded as a give up rather than silently returning nothing. In step 7 that becomes an answer the judge will fail, which is the correct outcome, because a run that never answered is a failed run.

**The maths, spelled out**

```
The 200 character cap is the only real number in this step, so here is what it buys and what it costs.

Formula: characters kept = runs * tool_calls_per_run * cap
Formula: rough token count = characters / 4, because English averages about 4 characters per token for most tokenisers. This is an approximation and it is off for code, numbers and non English text.

Worked example. Say 20 questions, 3 tool calls each, and an average tool result of 2000 characters.
Without the cap: 20 * 3 * 2000 = 120,000 characters, so about 120 KB of log, and 120,000 / 4 = 30,000 tokens if you ever feed it to a model.
With the cap: 20 * 3 * 200 = 12,000 characters, so about 12 KB, and 12,000 / 4 = 3,000 tokens.
That is a factor of 10 saved on both file size and token cost.

The same cap sets the size of your judge prompt in step 5, because result_head is the evidence. Per run the evidence is 3 * 200 = 600 characters, about 150 tokens, instead of 3 * 2000 = 6000 characters, about 1500 tokens. Across the 40 calibration runs in step 7 that is 6,000 evidence tokens instead of 60,000.

One more count worth knowing: with max_steps = 6, a single run writes at most 1 agent_run span plus 6 llm_call spans plus 6 tool_call spans, which is 13 lines in the log.

Intuition: you are trading evidence for size, and that trade is fine right up until the fact you care about sits past character 200.
```

> **Watch out:** If you are using the shared llm.py, chat(messages) fails because that chat() expects a prompt string, so fix that one line before you run anything else in this lab.

### 4. Read twenty traces before you write any evaluator

This is the step that decides what your evaluator will check, so do not jump past it. Error analysis means reading real runs end to end and writing one plain sentence for each thing that went wrong, before you invent any metric at all. If you write the judge first you will end up measuring something generic like helpfulness, which gives you a number that moves up and down without ever telling you what to fix. Write twenty questions you actually care about into questions.txt, run them, dump the printed trees into review.txt, and read that file in your editor rather than skimming the terminal. For every run that is wrong or annoying, put one sentence and its trace id into failures.md, then group the similar sentences and give each group a name. Expect two or three groups to cover most of the damage, and expect review.txt to be a few hundred lines, which is a twenty minute read and not more. Note that agent.py as written has no command line handling yet, so add a small if __name__ == "__main__": block that reads the file and calls run_agent once per line.

```python
python agent.py --questions questions.txt     # writes traces.jsonl
python spans.py > review.txt                  # one printed tree per run

# then, by hand, in failures.md:
# 3f9a1c  answered from memory, never called search
# 8b02de  searched the user's typo, returned nothing, still answered
# c41f77  quoted a price that is not in any tool result
```

- `python agent.py --questions questions.txt`: This runs every question through the instrumented agent and appends one full trace per question into traces.jsonl. The agent.py from step 3 defines a function and never calls it, so add a small block at the bottom that reads the file line by line and calls run_agent for each line.
- `python spans.py > review.txt`: The 'if __name__ == "__main__":' block at the bottom of spans.py prints one tree per trace, and the > sends that printout into a file instead of the screen. Read the file in your editor, because scrolling a terminal makes you skim, and skimming is exactly how you miss the middle steps where agents break.
- `# 3f9a1c  answered from memory, never called search`: Each line is a trace id and one plain sentence, nothing more, because the point is to be able to sort them into groups afterwards. Keeping the trace id means you can always go back to the exact run and check whether you remembered it correctly.

**The maths, spelled out**

```
Twenty runs is a real sample, so it has real limits. Two calculations tell you what twenty can and cannot show you.

1) Coverage of a failure group.
Formula: share = failures_in_group / total_failures
Worked example: 20 runs produce 7 bad ones. Grouping the sentences gives 4 in "answered from memory", 2 in "searched the typo", and 1 in "quoted a price nobody returned". The shares are 4/7 = 0.571, 2/7 = 0.286 and 1/7 = 0.143. So the top group alone is 57 percent of your damage and the top two together are 86 percent.
Intuition: one checker aimed at the top group buys you more than a vague checker that tries to cover everything at once.

2) The chance twenty runs miss a failure mode completely.
Formula: P(never seen) = (1 - p) to the power n, where p is how often the failure really happens and n is how many runs you looked at.
Worked example with p = 0.10 and n = 20, so 0.9 to the power 20. Step it up: 0.9 squared = 0.81, to the 4th = 0.6561, to the 8th = 0.4305, to the 16th = 0.1853, and 0.1853 * 0.6561 = 0.1216.
So there is about a 12 percent chance you see zero examples of a bug that hits one run in ten.
Intuition: twenty runs is enough to find your big problems and nowhere near enough to prove a rare one does not exist, so treat an empty group as unmeasured rather than as fixed.
```

> **Watch out:** The first command does nothing until you add an 'if __name__ == "__main__":' block to agent.py, because the code in step 3 only defines run_agent and never calls it.

### 5. Write one binary judge for your biggest failure group

Now write one checker for whichever group came top of your error analysis. The rubric here checks made up facts, which is usually the biggest group, but replace it with your own if yours differs. Three defences are built into this prompt on purpose: it checks exactly one rule so the model cannot trade one quality off against another, it is told to ignore length because judges reliably score longer answers higher, and it must reply with a tiny JSON object so you can parse a verdict instead of reading prose. Keep the verdict binary, pass or fail, because a 1 to 5 scale produces a pile of threes and fours that nobody can reproduce twice. Set LLM_PROVIDER in your .env to a different model family than your agent uses, since a model marking its own family's work is measurably more generous. The function returns a tuple, a boolean and a short reason string, and that reason is the thing you will read when you and the judge disagree in step 7. One line differs from the module page: the shared chat() takes a plain prompt string, not a message list, so the judge passes the formatted template straight in.

```python
# judge.py
import json, re, sys
sys.path.insert(0, "../_shared")
from llm import chat

RULE = ("PASS if every number, name and date in the answer also appears in at least one "
        "tool result below. FAIL if the answer states any fact that is not in the tool "
        "results, even if that fact is true in the real world.")

TEMPLATE = """You check one rule and nothing else.

RULE:
{rule}

QUESTION:
{question}

TOOL RESULTS THE AGENT SAW:
{evidence}

ANSWER TO CHECK:
{answer}

Ignore length, tone and style. A short answer is not worse than a long one.
Reply with only this JSON: {{"verdict": "pass" or "fail", "why": "12 words max"}}"""


def judge(question, evidence, answer):
    raw = chat(TEMPLATE.format(
        rule=RULE, question=question, evidence=evidence, answer=answer))
    m = re.search(r"\{.*\}", raw, re.S)          # survives ```json fences
    if not m:
        return False, "no JSON in judge reply"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False, "bad JSON in judge reply"
    return str(d.get("verdict", "")).strip().lower() == "pass", d.get("why", "")
```

- `even if that fact is true in the real world.`: That last clause is the part that makes the rule checkable. Without it the judge starts using its own knowledge and passes any answer that happens to be correct, which is not what you are measuring here; you are measuring whether the agent actually used the evidence it retrieved.
- `Reply with only this JSON: {{"verdict": "pass" or "fail", "why": "12 words max"}}`: The braces are doubled because this string goes through .format(), and .format turns a doubled brace back into a single one. The 12 word limit on 'why' keeps each reason short enough that you can scan forty of them in step 7 without losing patience.
- `raw = chat(TEMPLATE.format(rule=RULE, question=question, evidence=evidence, answer=answer))`: One formatted string goes in and one string comes back. This differs from the module page, which passes a message list, because the shared llm.py chat() takes a prompt string and a list would fail at the API call.
- `m = re.search(r"\{.*\}", raw, re.S)`: Models often wrap JSON in a markdown code fence or add a sentence before it, so this pulls out everything from the first opening brace to the last closing one. The re.S flag makes the dot match newlines as well, which it does not by default, and the pattern is greedy on purpose so a JSON object spread over several lines is captured whole.
- `return False, "no JSON in judge reply"`: A reply you cannot parse is treated as a fail rather than crashing the run or being skipped. That is the safe direction, because an unreadable verdict should never quietly count as a pass and inflate your score.
- `str(d.get("verdict", "")).strip().lower() == "pass"`: Anything that is not exactly the word pass, after trimming spaces and lowering the case, comes out as False. That single line covers PASS, ' pass ', a missing key entirely, and a model that decided to answer with true instead.

**The maths, spelled out**

```
The judge calls the shared chat(), which does not pass a temperature, so it uses the helper's default of 0.2. That number is doing real work here, so it is worth knowing what it changes.

A language model does not pick words directly. For every possible next token it produces a raw score called a logit, and those scores are turned into probabilities by softmax with temperature.

Formula: P(token i) = exp(z_i / T) divided by the sum over all j of exp(z_j / T)
z_i is the logit for token i, T is the temperature, and exp is the exponential function. Every P comes out between 0 and 1 and they all add up to 1.

Worked example with three candidate tokens and logits z = 2.0, 1.0 and 0.5.
At T = 1.0: exp(2.0) = 7.389, exp(1.0) = 2.718, exp(0.5) = 1.649. The sum is 11.756. The probabilities are 7.389 / 11.756 = 0.629, then 2.718 / 11.756 = 0.231, then 1.649 / 11.756 = 0.140.
At T = 0.2: first divide each logit by 0.2, giving 10, 5 and 2.5. Then exp(10) = 22026.5, exp(5) = 148.4, exp(2.5) = 12.2. The sum is 22187.1. The probabilities are 0.993, 0.0067 and 0.0005.
Intuition: dividing by a small T spreads the scores further apart before the exponential, so the leading token goes from about a 63 percent chance to about a 99 percent chance. For a judge that is exactly what you want, because the same input should give you the same verdict tomorrow.

One more number, from the module notes. A judge tends to favour answers written by its own model family by something like 10 to 25 percentage points. On 40 labelled runs, 15 points is 6 runs flipping from fail to pass, which on its own is enough to push kappa across the 0.6 line. Those percentages come from measurements on particular tasks and they move around, so read them as an effect big enough to change your conclusion rather than as a fixed constant.
```

> **Watch out:** If LLM_PROVIDER points at the same model family that wrote the answers, the pass rate comes out flattering for no good reason, and the fix belongs in .env and never in this file.

### 6. Label forty runs yourself, before you see the judge's verdicts

This is the step people skip, and it is the step that turns your judge's output into a number that means something. Label blind, which means you decide pass or fail before you have seen any verdict from the judge, because once you have seen its answer you will drift towards agreeing with it and your measurement becomes circular. The script prints the question, then each tool call with the first 200 characters of its result, then the final answer, and waits for one keypress. Press y for pass, n for fail, and s to skip anything you genuinely cannot decide. Aim for forty labelled runs and make sure at least ten of them are ones you call failures, because the true negative rate is computed only from the failures and ten is the smallest number that gives it any resolution at all. Each label is written and flushed straight away, so you can stop with Ctrl+C after twelve runs, come back tomorrow, and carry on without losing work.

```python
# label.py
import json
from spans import read

with open("labels.jsonl", "a", encoding="utf-8") as out:
    for tid, rows in read("traces.jsonl").items():
        run = next(r for r in rows if r["name"] == "agent_run")
        print("\nQ:", run["attrs"]["question"])
        for r in rows:
            if r["name"] == "tool_call":
                print("   tool", r["attrs"]["tool"], "->", r["attrs"].get("result_head"))
        print("A:", run["attrs"].get("final"))
        v = input("pass? [y/n/s] ").strip().lower()
        if v in ("y", "n"):
            out.write(json.dumps({"trace_id": tid, "human_pass": v == "y"}) + "\n")
            out.flush()
```

- `with open("labels.jsonl", "a", encoding="utf-8") as out:`: The file is opened in append mode so you can label across several sittings without losing earlier work. The cost is that running label.py twice over the same traces writes a second label for the same trace id, and calibrate.py will then count that run twice.
- `run = next(r for r in rows if r["name"] == "agent_run")`: This picks the single root span out of the trace, which is where the question and the final answer live. It raises StopIteration if a trace has no agent_run span, which happens when spans from something other than run_agent got logged into the same file.
- `print("   tool", r["attrs"]["tool"], "->", r["attrs"].get("result_head"))`: You are shown the same 200 character heads that the judge will see, not the full tool output. That is deliberate, because if you label using evidence the judge never receives, you will disagree with it for reasons that are your fault and not its.
- `v = input("pass? [y/n/s] ").strip().lower()`: One keypress per run keeps the pace fast enough that forty runs is twenty minutes rather than a whole evening. Only y and n are recorded, so s writes nothing and that run simply stays unlabelled.
- `out.flush()`: Python buffers writes, so without this a Ctrl+C could throw away the last several labels while they are still sitting in memory. Flushing after every line makes the file safe to interrupt at any moment.

**The maths, spelled out**

```
How many failures you label decides how precise your true negative rate can possibly be, so here is the arithmetic behind the "at least ten" rule.

TNR is computed only from the runs you called fail. If you label k failures, TNR can only take the values 0/k, 1/k, and so on up to k/k.
Formula for the uncertainty: SE = square root of ( p * (1 - p) / k ), and a rough 95 percent range is p plus or minus 1.96 * SE.
p is the TNR you observed, k is how many failures you labelled, and SE stands for standard error, which is how much that fraction would wobble if you labelled a different set of the same size.

Worked example with k = 3 and 2 of them caught, so p = 0.667.
p * (1 - p) = 0.667 * 0.333 = 0.222. Divide by 3: 0.0740. Square root: 0.272. Times 1.96: 0.533.
The range is 0.667 plus or minus 0.533, so 0.13 to 1.00. That range tells you nothing at all.

Worked example with k = 13 and 10 of them caught, so p = 0.769.
p * (1 - p) = 0.769 * 0.231 = 0.178. Divide by 13: 0.01366. Square root: 0.117. Times 1.96: 0.229.
The range is 0.769 plus or minus 0.229, so 0.54 to 1.00. Still wide, but it now rules out a judge that only catches half your failures.

Intuition: three failures give you a number with four possible values and an error bar wider than the scale itself, so ten is the floor and not the target.
```

> **Watch out:** Running label.py a second time over the same traces.jsonl appends duplicate labels, so delete labels.jsonl and start again rather than trying to spot the duplicates afterwards.

### 7. Calibrate, then fix the rubric and repeat

Now compare the judge against your own labels and read four numbers plus the list of disagreements. TPR (true positive rate) is the share of the runs you called good that the judge also called good, and TNR (true negative rate) is the share of the runs you called bad that it also called bad. Cohen's kappa is agreement after subtracting the agreement two random guessers would reach at those same base rates, which is why you report it instead of a raw percentage. A low TNR is the normal result the first time round, and it means your judge is lenient and is passing runs you called bad. Go and read those specific disagreements, because roughly half the time the judge is right and your label was sloppy, and the other half your rule was ambiguous and needs one more sentence. Change the rubric wording in judge.py, rerun this file, and stop when kappa is above about 0.6 and you can defend every remaining disagreement out loud. Be aware that this makes one model call per label, so forty labels means forty calls and a minute or two of waiting.

```python
# calibrate.py
import json
from spans import read
from judge import judge


def scores(pairs):
    """pairs = list of (human_pass, judge_pass). Positive class = PASS."""
    tp = sum(1 for h, j in pairs if h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fp = sum(1 for h, j in pairs if not h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    n = len(pairs) or 1
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    po = (tp + tn) / n
    ph, pj = (tp + fn) / n, (tp + fp) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": round(tpr, 3), "tnr": round(tnr, 3),
            "raw_agreement": round(po, 3), "kappa": round(kappa, 3)}


if __name__ == "__main__":
    traces = read("traces.jsonl")
    labels = [json.loads(l) for l in open("labels.jsonl", encoding="utf-8")]
    pairs, disagree = [], []
    for lab in labels:
        rows = traces.get(lab["trace_id"])
        if not rows:
            continue
        run = next(r for r in rows if r["name"] == "agent_run")
        evidence = "\n".join(str(r["attrs"].get("result_head", ""))
                             for r in rows if r["name"] == "tool_call")
        v, why = judge(run["attrs"]["question"], evidence, run["attrs"].get("final", ""))
        pairs.append((lab["human_pass"], v))
        if lab["human_pass"] != v:
            disagree.append((lab["trace_id"], "you said pass" if lab["human_pass"]
                             else "you said fail", why))
    print(json.dumps(scores(pairs), indent=2))
    for d in disagree:
        print("DISAGREE", *d)
```

- `tp = sum(1 for h, j in pairs if h and j)`: Each of the four counters is one pass over the same list of (your label, judge label) pairs. Writing them out as four separate sums is longer than calling a confusion matrix library, and it means you can read exactly what each number counts without looking anything up.
- `tpr = tp / (tp + fn) if tp + fn else float("nan")`: The guard stops a divide by zero when you labelled nothing as pass, and returns nan (not a number) instead. Seeing nan in the output is useful information in itself, because it tells you that rate was computed from no examples at all.
- `pe = ph * pj + (1 - ph) * (1 - pj)`: This is the chance you and the judge agree purely by luck, given how often each of you says pass. Both of you landing on pass has probability ph * pj, both landing on fail has probability (1 - ph) * (1 - pj), and kappa subtracts that total from the agreement you actually got.
- `evidence = "\n".join(str(r["attrs"].get("result_head", "")) for r in rows if r["name"] == "tool_call")`: The evidence handed to the judge is rebuilt from the trace itself, so the judge sees exactly what the agent saw, capped at the 200 characters you stored in step 3. If a run made no tool calls this comes out as an empty string, and the judge should then fail any answer containing specific facts.
- `disagree.append((lab["trace_id"], "you said pass" if lab["human_pass"] else "you said fail", why))`: Every disagreement is printed with its trace id and the judge's own one line reason. That reason is the thing you read to decide whether the judge misunderstood your rule or your rule was genuinely vague.

**The maths, spelled out**

```
This step prints five derived numbers. Here is every one of them, with the same worked example carried through.

The four counts, with PASS as the positive class:
tp = you said pass and the judge said pass
tn = you said fail and the judge said fail
fp = you said fail but the judge said pass
fn = you said pass but the judge said fail

Formulas:
TPR = tp / (tp + fn)
TNR = tn / (tn + fp)
raw agreement, written po = (tp + tn) / n
ph = (tp + fn) / n, which is how often you said pass
pj = (tp + fp) / n, which is how often the judge said pass
pe = ph * pj + (1 - ph) * (1 - pj), the agreement two random guessers would reach at those rates
kappa = (po - pe) / (1 - pe)
n is the number of labelled runs.

Worked example A, a realistic first calibration on 40 runs: tp = 24, tn = 10, fp = 3, fn = 3.
TPR = 24 / (24 + 3) = 24 / 27 = 0.889
TNR = 10 / (10 + 3) = 10 / 13 = 0.769
po = (24 + 10) / 40 = 34 / 40 = 0.850
ph = (24 + 3) / 40 = 27 / 40 = 0.675
pj = (24 + 3) / 40 = 27 / 40 = 0.675
pe = 0.675 * 0.675 + 0.325 * 0.325 = 0.4556 + 0.1056 = 0.5613
kappa = (0.850 - 0.5613) / (1 - 0.5613) = 0.2887 / 0.4387 = 0.658
That clears 0.6, so this judge is usable.

Worked example B, the trap. Same 40 runs, but the judge says pass to everything: tp = 27, fn = 0, fp = 13, tn = 0.
TPR = 27 / 27 = 1.000, which looks perfect
TNR = 0 / 13 = 0.000, which is the truth
po = 27 / 40 = 0.675
ph = 27 / 40 = 0.675, and pj = 40 / 40 = 1.000
pe = 0.675 * 1.000 + 0.325 * 0.000 = 0.675
kappa = (0.675 - 0.675) / (1 - 0.675) = 0 / 0.325 = 0.000
So raw agreement of 67.5 percent sounds acceptable while kappa says the judge carries no information whatsoever.

Intuition: kappa asks how much better than lucky guessing you did, so it collapses to zero exactly when a judge is simply betting on whichever class is more common. The usual reading is that above 0.6 is usable and below 0.4 means the rubric is ambiguous, but those cut-offs are a 1977 convention with no statistical backing, so treat them as a nudge and not a law.
```

> **Watch out:** If you regenerated traces.jsonl after labelling, every trace id in labels.jsonl now points at nothing, traces.get returns None for all of them, and you get n of 1 with zeros and nan everywhere.

### 8. Turn the calibrated judge into a deploy gate

The last step turns your calibrated judge into something a build server can run on its own. Copy your questions into cases.jsonl, one {"question": "..."} per line, then freeze that file, because a case set that keeps changing cannot tell you whether today is worse than last week. Run the gate once on today's code, then write that pass rate into baseline.json yourself as {"pass_rate": 0.88}, since this script reads that file but never writes it. From then on the script exits with code 1 whenever the pass rate falls more than ten points below the baseline, and a CI job treats a non-zero exit as a failed build. The os.environ line has to come before the spans and agent imports, because spans.py reads SPAN_LOG once at import time and never looks again. Runs that crash are caught, printed and still counted, which is what you want, because a crashed run is a failed run and its trace has no final answer for the judge to pass. On screen you get one line: the pass rate, the baseline, and how many cases were counted.

```python
# gate.py
import json, os, sys
os.environ["SPAN_LOG"] = "gate_run.jsonl"     # before importing spans / agent
if os.path.exists("gate_run.jsonl"):
    os.remove("gate_run.jsonl")

from agent import run_agent, TOOLS
from judge import judge
from spans import read

cases = [json.loads(l) for l in open("cases.jsonl", encoding="utf-8")]
for c in cases:
    try:
        run_agent(c["question"], TOOLS)
    except Exception as e:
        print("crashed:", c["question"], e)

traces = read("gate_run.jsonl")
ok = 0
for tid, rows in traces.items():
    run = next(r for r in rows if r["name"] == "agent_run")
    ev = "\n".join(str(r["attrs"].get("result_head", ""))
                   for r in rows if r["name"] == "tool_call")
    v, _ = judge(run["attrs"]["question"], ev, run["attrs"].get("final", ""))
    ok += bool(v)

rate = ok / max(len(traces), 1)
base = json.load(open("baseline.json"))["pass_rate"] if os.path.exists("baseline.json") else rate
print(f"pass rate {rate:.2f}   baseline {base:.2f}   cases {len(traces)}")
sys.exit(1 if rate < base - 0.10 else 0)
```

- `os.environ["SPAN_LOG"] = "gate_run.jsonl"`: This must sit above the 'from agent import ...' line, because importing agent imports spans, and spans.py reads SPAN_LOG once at import. Put it lower and the gate would append into traces.jsonl, mixing your gate runs into the labelled runs you calibrated against.
- `if os.path.exists("gate_run.jsonl"): os.remove("gate_run.jsonl")`: The logger only appends, so the file is deleted first to make sure the pass rate is computed from this run alone. Without it your gate would slowly average today together with every previous day and stop reacting to anything.
- `the try / except around run_agent`: A crashed case is printed and the loop carries on, so one broken question does not abandon the whole gate. The crashed run still leaves a trace behind with no final answer, so it goes on to be judged and counted as a failure, which is the right result.
- `rate = ok / max(len(traces), 1)`: The max(..., 1) is a divide by zero guard for the case where nothing ran at all. Note that the denominator is the number of traces found in the log, not the number of cases you fed in, so a case that never produced a span quietly disappears from the score instead of counting against it.
- `base = json.load(open("baseline.json"))["pass_rate"] if os.path.exists("baseline.json") else rate`: If baseline.json is missing, the baseline is set to today's rate, so the comparison is against itself and the gate always passes. That is deliberate first run behaviour, and it is also the trap, because nothing in this script ever writes baseline.json for you.
- `sys.exit(1 if rate < base - 0.10 else 0)`: Exit code 1 means failure to any shell or CI system and exit code 0 means success, which is the entire interface a build server needs. The 0.10 is ten percentage points and not ten percent of the baseline, so a baseline of 0.88 fails only below 0.78.

**The maths, spelled out**

```
Two numbers here: the ten point threshold and the token cost.

1) Why ten points and not one.
Model output varies between runs, so a small drop can be nothing but noise. The wobble on a pass rate is:
SE = square root of ( p * (1 - p) / n )
p is your baseline pass rate, n is the number of frozen cases, and a rough 95 percent range is 1.96 * SE either side.
Worked example with p = 0.88 and n = 50: 0.88 * 0.12 = 0.1056. Divide by 50: 0.002112. Square root: 0.0460. Times 1.96: 0.090.
So with 50 cases a perfectly healthy agent can swing about 9 points either way without anything actually being wrong, which is where the 10 point threshold comes from.
Worked example of the gate itself: baseline 44/50 = 0.88. A run scoring 41/50 = 0.82 has dropped 0.06, which is less than 0.10, so sys.exit(0) and the build passes. A run scoring 36/50 = 0.72 has dropped 0.16, so sys.exit(1) and the build fails.
Intuition: the threshold is set just outside normal noise, so the gate fires on a real regression and not on a bad afternoon.

2) What one gate run costs in tokens.
Formula: input tokens = cases * (fixed prefix tokens + variable tokens), and cached input costs roughly a tenth of fresh input.
Worked example. Say each judge prompt is about 1600 input tokens, of which about 1300 are the fixed instructions and RULE at the top and 300 are the question, evidence and answer underneath.
With no caching: 50 * 1600 = 80,000 input tokens per gate run.
With prefix caching: 50 * (1300 * 0.1 + 300) = 50 * 430 = 21,500 effective tokens, so about 3.7 times cheaper.
Caching matches on prefixes, which is exactly why the fixed instructions and the RULE sit at the top of TEMPLATE in step 5 and the changing parts sit below them. Be honest about the limits here: whether caching happens at all depends on your provider, and the shared llm.py does not ask for it, so treat this as the reason for the prompt layout rather than a bill you will definitely see.
Intuition: keep the unchanging text first and the case specific text last, and a 50 case gate stops being something you avoid running.
```

> **Watch out:** Forgetting to create baseline.json means the gate compares today's rate against today's rate and exits 0 forever, so it will never once tell you anything.

## You are done when

Running python spans.py prints a readable step by step tree for every agent run, with one indented line per model call and tool call and a FAIL line wherever a step raised. Running python calibrate.py prints n, tp, tn, fp, fn, TPR, TNR, raw agreement and Cohen's kappa against at least forty of your own blind labels, with kappa above 0.6. Running python gate.py prints a pass rate next to a baseline you wrote yourself and exits 0. If kappa is below 0.6, you have a written explanation for each remaining disagreement, which counts as finished too.

---

## Mini-project: Catch a real regression

Break your agent three ways on purpose and record whether your gate notices. The result is regression_report.json, which check.py reads and cross-checks against the gate output you saved.

- Freeze the case set: 50 lines of {"question": "..."} in cases.jsonl, then stop editing it. Run gate.py once against today's agent, write that rate into baseline.json as {"pass_rate": 0.88}, and note the kappa calibrate.py printed.
- Make three copies of agent.py with exactly one defect each, named truncate_100 (tool results cut to the first 100 characters), no_retrieval (the search tool removed from the tools dict), and cheap_model (the cheapest model tier you have access to).
- Run gate.py against each copy without touching judge.py or cases.jsonl. Save each run's stdout to runs/<name>.txt, so runs/truncate_100.txt holds a line like "pass rate 0.62   baseline 0.88   cases 50", and note each exit code.
- Write regression_report.json with this shape: {"baseline": {"pass_rate": 0.88, "kappa": 0.66, "n_cases": 50, "n_labels": 41}, "threshold": 0.10, "variants": [{"name": "truncate_100", "defect": "tool results cut to 100 chars", "pass_rate": 0.62, "gate_exit_code": 1, "detected": true}, ...], "blind_spots": ["..."]}. Set detected true only when that variant's rate falls more than 0.10 below the baseline.
- For every variant where detected is false, read that copy's traces until you find the failure by hand. Then either sharpen the rubric in judge.py, recalibrate and rerun, or write one sentence in blind_spots saying what your eval cannot see.
- Run python check.py from inside my-work/labs/lab16.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab16 && python check.py
```


**You are done when** check.py prints one PASS line per check and ends with ALL CHECKS PASSED and exit code 0. It fails when a detected flag disagrees with the numbers, when a pass_rate does not match runs/<name>.txt, when n_cases or n_labels do not match the files on disk, or when a missed defect has no blind_spots sentence. It prints a line saying that whether your blind_spots sentences are true is not checked automatically.

**If you want more:** Add a second binary check that grades the trajectory instead of the answer, such as "did the agent call search_docs at least once before answering?". Build a fourth defect that only the trajectory check catches while the answer check still says pass, and add it to the report. check.py expects exactly 3 variants, so change that number to 4 when you do.
