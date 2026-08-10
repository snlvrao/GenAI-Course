# Lab 15: Two agents, one job

**Module 15: Many agents, memory, and A2A**

You will build two agents, a researcher and a writer, that never pass their conversations to each other. They share one SQLite file instead: the researcher writes its findings into it, and the writer reads them back out and writes the final text, so the only thing travelling between the two is an eight character run id. This is the smallest honest version of shared memory, and unlike a Python dictionary it is still there after you close the terminal. Before you start, make sure python llm.py works, see setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Set up the shared memory store

Create memory.py in your my-work/labs/lab15 folder. It opens a SQLite file, which is a complete database kept inside one ordinary file on your disk, and gives you two functions: remember() writes one memory, recall() reads memories back for one run. Every row stores a run_id, so two different runs never see each other's data, and a created_at timestamp, so you can later ask how old a stored fact is. Nothing in this file mentions the researcher or the writer, and that is deliberate: memory is a place both agents visit, not a feature bolted onto one of them. Without it the two agents would have to hand each other their whole conversation, which is the expensive habit this lab exists to break. Saving this file prints nothing, because it only defines functions. You will see shared_memory.db appear next to memory.py the first time an agent actually writes something.

```python
# memory.py
import sqlite3, json, time, os

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_memory.db")

def connect():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        agent  TEXT NOT NULL,
        kind   TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL)""")
    con.commit()
    return con

def remember(run_id, agent, kind, content):
    con = connect()
    con.execute(
        "INSERT INTO memory (run_id, agent, kind, content, created_at) VALUES (?,?,?,?,?)",
        (run_id, agent, kind, json.dumps(content), time.time()))
    con.commit(); con.close()

def recall(run_id, kind=None):
    con = connect()
    if kind:
        rows = con.execute("SELECT agent, kind, content, created_at FROM memory "
                           "WHERE run_id=? AND kind=? ORDER BY id", (run_id, kind)).fetchall()
    else:
        rows = con.execute("SELECT agent, kind, content, created_at FROM memory "
                           "WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    con.close()
    return [{"agent": a, "kind": k, "content": json.loads(c), "created_at": t}
            for a, k, c, t in rows]
```

- `DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_memory.db")`: This pins the database file next to memory.py, not to whichever folder you happen to be standing in when you type a command. That is what lets run.py and inspect_memory.py, started from different places and at different times, open the same file.
- `CREATE TABLE IF NOT EXISTS memory (`: IF NOT EXISTS makes connect() safe to call on every single read and write. The first call builds the table and every later call quietly does nothing, so you never need a separate setup script.
- `run_id TEXT NOT NULL`: This column is what keeps runs apart. Every query filters on it, so ten runs can share one file without their findings bleeding into each other.
- `json.dumps(content) ... json.loads(c)`: A SQLite text column holds text, not Python dictionaries, so you turn the finding into a JSON string on the way in and parse it back on the way out. This pair is the reason a finding survives the process exiting.
- `VALUES (?,?,?,?,?)`: The question marks are placeholders and the values arrive separately as a tuple, so they are never glued into the SQL text. A claim containing a quote mark therefore cannot break the query or rewrite it.
- `ORDER BY id`: id is AUTOINCREMENT, so it only ever goes up. Ordering by it hands the findings back in exactly the order they were written, without you needing to sort on the timestamp.

**The maths, spelled out**

```
Formula:
  age_seconds = time.time() - created_at
  age_days    = age_seconds / 86400

What the symbols mean:
  time.time() is the current Unix timestamp, meaning the number of seconds that have passed since 1 January 1970 at 00:00 UTC. It is a plain float, for example 1772582400.0.
  created_at is that same clock, captured at the moment the row was written.
  86400 is the number of seconds in a day, because 60 seconds * 60 minutes * 24 hours = 86400.

Worked example:
  created_at = 1772323200.0
  now        = 1772582400.0
  age_seconds = 1772582400.0 - 1772323200.0 = 259200.0
  age_days    = 259200 / 86400 = 3.0
  So that stored finding is exactly 3 days old.

What it means:
  The timestamp is the only thing in the row that lets you say "this was true three days ago", which is how you catch a stored fact that still reads as relevant but is no longer correct.
```

> **Watch out:** If you copy memory.py into a second folder, you get a second shared_memory.db, and recall() will return an empty list with no error at all.

### 2. Write the researcher agent

researcher.py asks the model for four findings about your topic and stores each one as its own row. Look at the shape it demands back: a claim, a confidence word, and an unknowns field for whatever the model could not verify. That last field matters, because doubt that stays inside the model's reasoning is doubt the writer will never see. The model replies as ordinary text, so you slice out the part between the first { and the last } before handing it to json.loads. Each finding is stored as a separate row rather than one blob, so recall() can hand them back as a clean list later. Note that the call passes your prompt first and the system message as system=, because that is the real signature of chat() in my-work/labs/_shared/llm.py. When you run this you should see exactly one line: researcher: stored 4 findings for run &lt;id&gt;.

```python
# researcher.py
import json, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
from llm import chat
from memory import remember

SYS = ("You are a research agent. Given a topic, produce 4 short findings.\n"
       "Return ONLY JSON: {\"findings\":[{\"claim\":\"...\",\"confidence\":\"high|medium|low\","
       "\"unknowns\":\"what you could not verify, or empty string\"}]}\n"
       "Never guess a number you are unsure of. Put the doubt in unknowns.")

def research(run_id, topic):
    raw = chat(f"Topic: {topic}", system=SYS)
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1])
    for f in data["findings"]:
        remember(run_id, "researcher", "finding", f)
    print(f"researcher: stored {len(data['findings'])} findings for run {run_id}")
    return run_id
```

- `sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))`: This adds my-work/labs/_shared to the list of folders Python searches when you import something. Without it, from llm import chat fails, because Python only looks in the current folder and the standard library.
- `the SYS string`: This is the system message, the standing instruction the model reads before your question. It fixes the reply shape so your code can parse it, and the last line pushes uncertainty into the unknowns field instead of into a confident sentence.
- `raw = chat(f"Topic: {topic}", system=SYS)`: chat() in my-work/labs/_shared/llm.py is defined as chat(prompt, *, system=None, ...), so the star makes system keyword-only. Passing the system text as a second positional argument, as some earlier drafts of this lab did, raises TypeError before the model is ever called.
- `start, end = raw.find("{"), raw.rfind("}")`: find() returns the index of the first opening brace and rfind() the index of the last closing brace. Together they locate the JSON even when the model wrapped it in a code fence or added a friendly sentence around it.
- `json.loads(raw[start:end + 1])`: The +1 is there because a Python slice stops just before the end index, so without it you would cut off the final closing brace and the parse would fail.
- `for f in data["findings"]: remember(run_id, "researcher", "finding", f)`: One row per finding, each tagged with kind "finding". That tag is what lets the writer later ask for only the findings and never accidentally pick up a draft.

**The maths, spelled out**

```
The model does not pick words directly. It produces a score for every possible next token and turns those scores into probabilities with the softmax function, adjusted by temperature.

Formula:
  p(i) = exp(z_i / T) / sum over all j of exp(z_j / T)

What the symbols mean:
  z_i is the logit for token i, meaning the raw score the model gives that token before it becomes a probability.
  T is the temperature. chat() in my-work/labs/_shared/llm.py defaults to T = 0.2.
  exp is the exponential function, exp(1) = 2.718.
  p(i) is the chance that token i gets chosen.

Worked example with three candidate tokens whose logits are 2.0, 1.0 and 0.5.

At T = 1.0:
  exp(2.0) = 7.389, exp(1.0) = 2.718, exp(0.5) = 1.649
  total = 7.389 + 2.718 + 1.649 = 11.756
  p = 7.389/11.756 = 0.629, 2.718/11.756 = 0.231, 1.649/11.756 = 0.140

At T = 0.2, divide each logit by 0.2 first:
  2.0/0.2 = 10, 1.0/0.2 = 5, 0.5/0.2 = 2.5
  exp(10) = 22026, exp(5) = 148.4, exp(2.5) = 12.2
  total = 22186.6
  p = 22026/22186.6 = 0.993, 148.4/22186.6 = 0.0067, 12.2/22186.6 = 0.00055

What it means:
  A low temperature squashes the runners-up, so the model almost always takes its top choice. That is what you want when the reply has to parse as JSON, and it is also why running the same topic twice gives you nearly the same four findings.
```

> **Watch out:** chat() caps the reply at 800 tokens, so a chatty model can get cut off mid-JSON and json.loads raises JSONDecodeError; print raw before parsing and you will see the truncated text immediately.

### 3. Write the writer agent

writer.py never sees the researcher's conversation. It calls recall(run_id, kind="finding"), pulls the four findings out of SQLite, flattens them into four bullet lines, and asks the model for a 150 word brief. The system message tells it to use only the claims given and to hedge any sentence whose finding had a non-empty unknowns, which is how the researcher's doubt survives the handoff. Look at how thin the connection between the two agents is: the writer receives a run id of eight characters and fetches everything else itself. The draft is written back into the same table with kind "draft", so the finished text sits next to the evidence it came from. If you call write() with a run id that has nothing stored, it raises RuntimeError rather than quietly carrying on. That guard matters, because an empty findings list would otherwise give you a confident, entirely invented brief.

```python
# writer.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
from llm import chat
from memory import recall, remember

SYS = ("You are a writing agent. You are given findings from a research agent.\n"
       "Write a 150-word brief. Rules:\n"
       "- Use ONLY the claims given. Add nothing.\n"
       "- If a finding has a non-empty 'unknowns', you MUST hedge that sentence "
       "and say what is unverified.\n"
       "- If confidence is 'low', say so in the text.")

def write(run_id):
    findings = [m["content"] for m in recall(run_id, kind="finding")]
    if not findings:
        raise RuntimeError(f"no findings in memory for run {run_id}")
    lines = [f"- {f['claim']} (confidence: {f['confidence']}; unknowns: {f['unknowns'] or 'none'})"
             for f in findings]
    brief = chat("Findings:\n" + "\n".join(lines), system=SYS)
    remember(run_id, "writer", "draft", brief)
    return brief
```

- `findings = [m["content"] for m in recall(run_id, kind="finding")]`: The kind filter keeps drafts out, so the writer only ever reads the researcher's rows. The list comprehension then unwraps each row down to the finding dictionary itself, dropping the agent and timestamp the writer does not need.
- `if not findings:
        raise RuntimeError(f"no findings in memory for run {run_id}")`: An empty list is falsy in Python, so this catches a wrong or stale run id. Failing loudly here is the difference between an obvious error and a fluent brief built on nothing.
- `f['unknowns'] or 'none'`: An empty string is falsy in Python, so this prints the word "none" when the researcher had no doubts. The model then sees an explicit word rather than a label with nothing after it, which it might otherwise try to fill in.
- `brief = chat("Findings:\n" + "\n".join(lines), system=SYS)`: Same keyword-only system argument as the researcher, for the same reason. join() glues the bullets with newlines so the model reads a list of four separate items rather than one run-on paragraph.
- `remember(run_id, "writer", "draft", brief)`: Storing the output with kind "draft" keeps it separate from the inputs. Without that separation a second call to write() would recall its own previous draft as if it were a finding.

**The maths, spelled out**

```
This step is where you can see why a pointer beats a transcript. Suppose each step forwards the entire conversation so far.

Formula:
  total input tokens = N * B + C * N * (N - 1) / 2

What the symbols mean:
  N is the number of steps or agents in the chain.
  B is the base tokens in every prompt, meaning the system message and instructions that are always there.
  C is the tokens each step adds to the growing transcript.
  The N * (N - 1) / 2 part is just 0 + 1 + 2 + ... + (N - 1), the number of times earlier text gets re-sent.

Worked example with N = 10, B = 200, C = 500:
  first term  = 10 * 200 = 2000
  second term = 500 * (10 * 9 / 2) = 500 * 45 = 22500
  total       = 24500 input tokens

Now the shared-state version, where each step reads only what it needs, roughly B + C = 700 tokens:
  total = 10 * 700 = 7000 input tokens
  ratio = 24500 / 7000 = 3.5 times cheaper

What it means:
  Passing the transcript makes your cost grow with the square of the chain length, while passing a run id keeps it a straight line. At two agents nobody notices. At ten, it is the whole bill.
```

> **Watch out:** A KeyError on 'unknowns' means the model returned a finding without that field; fix it by rejecting bad findings at write time in step 2, not by hoping the next run behaves.

### 4. Add a supervisor that runs both

run.py is the supervisor, the piece that decides who runs and in what order. It makes a fresh run id, calls research(), then calls write(), then prints the brief and the id. Keep it deliberately dumb, because a supervisor written in plain Python is a workflow, meaning a fixed path you control and can read top to bottom when something goes wrong. The alternative, letting a model choose the order, is something you should reach for only after you hit a real case where the fixed order fails. Notice that the supervisor holds no findings of its own; it passes the run id along and nothing else, so its memory stays tiny no matter how much the agents produce. Run it with python run.py "your topic" and you should see the researcher's stored-count line, a divider, a brief of roughly 150 words, then the run id.

```python
# run.py
import uuid, sys
from researcher import research
from writer import write

topic = sys.argv[1] if len(sys.argv) > 1 else "the cost of running LLM agents in production"
run_id = uuid.uuid4().hex[:8]

research(run_id, topic)
print("\n--- BRIEF ---\n")
print(write(run_id))
print(f"\nrun_id={run_id}  (inspect with: python inspect_memory.py {run_id})")
```

- `topic = sys.argv[1] if len(sys.argv) > 1 else "the cost of running LLM agents in production"`: sys.argv[1] is the first thing you typed after the filename. The length check avoids an IndexError when you run the file with no argument, and hands you a sensible default topic instead.
- `run_id = uuid.uuid4().hex[:8]`: uuid4() makes a random 128 bit identifier, .hex turns it into 32 hexadecimal characters, and [:8] keeps the first eight so it is short enough to retype by hand into the inspector.
- `research(run_id, topic)
print(...)
print(write(run_id))`: This is the whole handoff. The only value crossing between the two agents is run_id, because everything else travels through the database.
- `print(f"\nrun_id={run_id}  (inspect with: python inspect_memory.py {run_id})")`: Printing the id is not decoration. Lose it and the findings are still on disk but you have no way to ask for them, since every query filters on run_id.

**The maths, spelled out**

```
You threw away 24 of the 32 hex characters, so it is fair to ask when two runs collide.

Formula:
  number of possible ids  N = 16 ^ 8
  approximate collision chance for n runs, p = n^2 / (2 * N)

What the symbols mean:
  16 because each hex character is one of 0 to 9 or a to f, which is 16 options.
  8 because you kept 8 characters.
  n is how many runs you do.
  N is how many distinct ids exist.
  This is the birthday approximation, and it holds while p stays well below 1.

Worked example:
  N = 16 ^ 8 = 4,294,967,296 (about 4.29 billion)
  With n = 1,000 runs:
    p = 1,000^2 / (2 * 4,294,967,296) = 1,000,000 / 8,589,934,592 = 0.000116
    That is about 1 chance in 8,600.
  With n = 100,000 runs:
    p = 10,000,000,000 / 8,589,934,592 = 1.16
    A result above 1 just tells you the approximation has broken down and a collision is near certain.

What it means:
  The chance grows with n squared, not with n, so it stays negligible for a long time and then arrives fast. Eight characters is fine for a lab notebook and not fine for a production log.
```

> **Watch out:** from researcher import research only works when you run python run.py from inside my-work/labs/lab15, because Python adds the script's own folder to the import path; launching it from the parent folder gives ModuleNotFoundError.

### 5. Prove the memory is real

Memory that only exists inside one running process is not memory, it is a variable. inspect_memory.py proves the difference: it is five lines, it imports nothing from either agent, and it reads the database directly. Open a brand new terminal, run python inspect_memory.py &lt;run_id&gt; with the id run.py printed, and you should see five lines, four tagged researcher/finding and one tagged writer/draft. Then do the stronger test: start a fresh Python session, from writer import write, and call write(run_id) with that same id. You will get a new brief with the researcher never running at all, because everything it knew is sitting on disk. If that second test fails, you do not have shared memory, you have a variable that happened to still be in scope.

```python
# inspect_memory.py
import sys
from memory import recall

run_id = sys.argv[1]
for m in recall(run_id):
    print(f"[{m['agent']}/{m['kind']}] {str(m['content'])[:160]}")
```

- `run_id = sys.argv[1]`: There is no default here on purpose. The inspector is meaningless without an id, and an immediate IndexError is a clearer signal than silently printing nothing.
- `for m in recall(run_id):`: Calling recall without a kind returns every row for that run, findings and draft together, in the order they were written. That mixed view is exactly what you want when checking a handoff.
- `str(m['content'])[:160]`: content can be a dictionary (a finding) or a long string (the draft), so str() handles both without you branching. The slice caps each line at 160 characters so one memory stays on one terminal row.

**The maths, spelled out**

```
A quick sanity check on how big this file gets, so you know whether to clean it up.

Formula:
  rows_per_run = findings + drafts
  total_rows   = rows_per_run * runs
  file_size    is roughly total_rows * bytes_per_row

What the symbols mean:
  findings is 4 per run in this lab, drafts is 1 per run.
  bytes_per_row is an estimate, not a measurement, because it depends entirely on how long your text is.

Worked example:
  rows_per_run = 4 + 1 = 5
  200 runs     -> 5 * 200 = 1,000 rows
  a finding is a short JSON dict and a draft is about 150 words, so call it 400 bytes per row on average
  file_size    = 1,000 * 400 = 400,000 bytes, about 0.4 MB

What it means:
  This table will not get large by accident during the lab, so leave old runs in place and treat shared_memory.db as a log you can go back through. Be clear that the 400 bytes is a guess; run the numbers again on your own file if it ever matters.
```

> **Watch out:** Empty output almost always means a mistyped or half-copied run id, not lost data, because recall() returns an empty list rather than raising when nothing matches.

### 6. Count what the split cost you

Now measure what splitting the work actually bought you. Add a len() print of the prompt each agent sends, so the researcher reports the length of its topic prompt plus SYS, and the writer reports the length of its bullet list plus SYS. The writer's number should be far smaller than the researcher's request and reply combined, because the writer only ever received four short lines. That gap is the entire reason shared state exists: agent runs are heavy on input tokens and light on output tokens, so the size of what you send on each step is your main lever on cost. Characters are not tokens, so treat these as a ratio you are watching rather than as a bill. Write both numbers into your notes now, because the mini-project asks you to take the same measurement again and compare.

**The maths, spelled out**

```
Two formulas turn your character counts into money.

Formula:
  tokens is roughly characters / 4
  cost = (input_tokens / 1,000,000) * price_per_million_input
       + (output_tokens / 1,000,000) * price_per_million_output

What the symbols mean:
  The 4 is a rough average of characters per token for ordinary English. Real tokenisers land somewhere between about 3 and 5, and JSON or code sits at the low end.
  price_per_million is whatever your provider charges per million tokens, and input is normally much cheaper than output.

Worked example:
  researcher prompt = 1,200 characters -> 1,200 / 4 = 300 tokens
  writer prompt     =   600 characters ->   600 / 4 = 150 tokens
  if you had instead pasted the researcher's whole exchange into the writer:
                      4,800 characters -> 4,800 / 4 = 1,200 tokens
  saving per run    = 1,200 - 150 = 1,050 input tokens
  over 10,000 runs  = 1,050 * 10,000 = 10,500,000 tokens
  at $0.50 per million input tokens:
    10,500,000 / 1,000,000 = 10.5
    10.5 * 0.50 = $5.25 saved

What it means:
  One run's saving looks trivial, and that is exactly why this gets ignored. The number only becomes visible once you multiply by how often the thing runs.
```

> **Watch out:** Do not quote character counts as if they were tokens; JSON, code and non-English text can run closer to 2 or 3 characters per token, which makes the real number bigger than your estimate.

## You are done when

Running python run.py "your topic" prints the researcher's stored-count line and then a brief of roughly 150 words. Running python inspect_memory.py <run_id> in a brand new terminal prints five lines for that run, four tagged researcher/finding and one tagged writer/draft, which proves shared_memory.db outlived the process that wrote it. At least one sentence in the brief hedges, because the finding behind it had a non-empty unknowns field. And you have the two prompt lengths from step 6 written down, ready to compare in the mini-project.

---

## Mini-project: Make it fail on purpose

Break your two-agent system on purpose, name the MAST failure mode, fix it structurally, then prove the fix with counts from twenty real runs. The artefact is my-work/labs/lab15/failure_report.json, and check.py verifies it against the run ids sitting in shared_memory.db.

- Pick one mode and write the rule you will judge by. Use exactly one of these strings: information_withheld, task_derailment, no_verification. Then write one binary test you can apply to any brief, for example "the brief states a figure whose stored finding had a non-empty unknowns, with no hedge".
- Break it in code, not by luck. For information_withheld, drop the unknowns key before calling remember(). For task_derailment, remove the topic from the writer's input. For no_verification, store one finding you know is false.
- Run run.py ten times on the same topic. Keep every run id it prints, apply your rule to each brief, and record pass or fail. If fewer than seven of the ten fail, the break is too weak to prove a fix, so make it harder before you carry on.
- Fix it structurally, not with a firmer instruction. Make remember() reject a finding that is missing a required key, pass the topic explicitly into the writer prompt, or add a third checker agent that answers supported or unsupported for each sentence. Run ten fresh runs and judge them with the same rule.
- Write my-work/labs/lab15/failure_report.json with these keys: mast_mode (one of the three strings), break_description, fix_description, still_missed (each a sentence of 20 characters or more), and before and after, each a list of ten objects shaped {"run_id": "9f2c1ab4", "topic": "solid-state batteries", "failed": true}. Use the real run ids run.py printed. If the after count did not drop, add honest_null_result saying why.
- Save check.py next to it and run python check.py from inside my-work/labs/lab15.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab15 && python check.py
```


**You are done when** python check.py prints one PASS or FAIL line per check and ends with ALL CHECKS PASSED and exit code 0. Passing means all twenty run ids really exist in shared_memory.db with a stored draft, your break failed at least seven of ten runs, and the after count dropped or honest_null_result explains why it did not. The checker also prints a line naming what it cannot judge: whether your fix is structural rather than a firmer prompt, and whether you called each pass or fail correctly.

**If you want more:** Add a fourth agent and spread eight more tools across the team, then run the same ten topics again as a third block and count the failures. If the rate rises with nothing else changed, you have measured coordination overhead in your own system instead of trusting the roughly-a-dozen-tools rule of thumb.
