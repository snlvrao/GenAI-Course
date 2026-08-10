# Lab 14: Rebuild your agent in a framework

**Module 14: Agent frameworks, and how to choose**

You already wrote an agent loop by hand in Module 12, so you know exactly how many lines that loop takes and where every bug can hide. In this lab you build the same agent twice, once as your own for loop and once as a LangGraph graph, and both versions share the same tools, the same system prompt and the same model call so the comparison is fair. Then you run both on the same two tasks and watch what actually changes. The goal is not to prove LangGraph is better, it is to see in your own code which three things a framework hands you (state saved after every step, a pause a human can resume, and one place to hang tracing) and what it charges you in extra lines and extra places for a bug to hide.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Install LangGraph and check it

Make a folder my-work/labs/lab14 next to your other labs, then install the exact version this lab was written against. Pinning means asking pip for one specific version instead of the newest one, and you do it here because LangGraph moved and renamed several things during its 1.x line, so an unpinned install can hand you a library that no longer matches this page. The second command does two separate jobs, it proves the package really imports, and it prints the version pip actually recorded on disk, which is not always the one you asked for if another package pulled in a different one. You should see 1.2.10 printed and nothing else. If your .env already has LLM_PROVIDER set from an earlier module you do not need to touch it, because every model call in this lab goes through the shared my-work/labs/_shared/llm.py helper. Nothing here needs a GPU, a vendor desktop app, or a new API key.

```python
pip install langgraph==1.2.10

python -c "import langgraph, importlib.metadata as m; print(m.version('langgraph'))"
```

- `pip install langgraph==1.2.10`: The double equals asks pip for that exact release and refuses anything else. Without it pip takes whatever is newest on the day you run it, and the imports this lab relies on (InMemorySaver, interrupt, Command) are only guaranteed to sit where they do in 1.2.x.
- `import langgraph`: This is the cheap half of the check. An install can print a success message and still fail to import if a dependency did not build on your machine, and importing is the only way to find that out now instead of three steps later.
- `importlib.metadata as m; print(m.version('langgraph'))`: importlib.metadata reads the version pip wrote into the package metadata, rather than a __version__ attribute the package might not define. If this prints anything other than 1.2.10, something else in your environment quietly changed it.

> **Watch out:** If pip prints a dependency resolver warning or the version check prints a number other than 1.2.10, another package in the same environment pinned LangGraph too, and the fix is a fresh virtual environment rather than forcing the install.

### 2. Write the parts both agents share

Both agents have to use identical tools, identical tool descriptions and an identical model call, otherwise any difference you measure later might just be a difference in wording rather than a difference between the two designs. So everything shared lives in one file that both agents import. calc does arithmetic safely by parsing the expression into a tree and walking it, instead of calling eval, which would happily run any Python a model handed it. save_note writes a real file, which makes it the risky tool, meaning one with a side effect you cannot undo by pressing back, and it is the action you will put a human in front of in step 4. The SCHEMAS list describes those tools in JSON Schema form, which is the shape the model reads when deciding which tool to reach for and what arguments to invent. One risky assumption is baked in here, chat_raw is called with keyword arguments, so if your copy of the helper takes positional ones instead, change that single line inside call_model and nothing else in this lab moves.

```python
# my-work/labs/lab14/common.py
import ast, json, operator, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw  # noqa: E402

NOTES = pathlib.Path(__file__).parent / "notes"

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")

def calc(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))

def save_note(name: str, text: str) -> str:
    NOTES.mkdir(exist_ok=True)
    p = NOTES / (name + ".txt")
    p.write_text(text, encoding="utf-8")
    return "saved " + p.name

TOOLS = {"calc": calc, "save_note": save_note}
RISKY = {"save_note"}

SCHEMAS = [
  {"type": "function", "function": {
     "name": "calc",
     "description": "Evaluate one arithmetic expression, for example 0.17*4830.",
     "parameters": {"type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]}}},
  {"type": "function", "function": {
     "name": "save_note",
     "description": "Save a short text note to disk under a given name.",
     "parameters": {"type": "object",
        "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
        "required": ["name", "text"]}}},
]

SYSTEM = ("You are a careful assistant. Use calc for any arithmetic. "
          "Use save_note only when the user asks you to save something. "
          "When you are done, answer in one short sentence.")

def _as_message(reply):
    if hasattr(reply, "choices"):
        reply = reply.choices[0].message
    if hasattr(reply, "model_dump"):
        reply = reply.model_dump()
    if not isinstance(reply, dict):
        raise TypeError("chat_raw gave back a " + type(reply).__name__
                        + ", expected a response or an assistant message")
    return reply

def call_model(messages):
    t0 = time.time()
    msg = _as_message(chat_raw(messages=messages, tools=SCHEMAS))
    return msg, time.time() - t0

def assistant_message(msg):
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out

def tool_calls_of(msg):
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc["function"]
        args = fn["arguments"]
        calls.append({"id": tc["id"], "name": fn["name"],
                      "args": json.loads(args) if isinstance(args, str) else args})
    return calls

def run_tool(call):
    try:
        return TOOLS[call["name"]](**call["args"])
    except Exception as e:
        return "ERROR: " + str(e)
```

- `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: __file__ is this file's own path, .resolve() makes it absolute so it works no matter where you launched Python from, and parents[1] steps up two folders to my-work/labs/. Inserting at position 0 means Python looks in my-work/labs/_shared before anywhere else, so you get the course helper and not some other llm module.
- `the _eval function`: It walks the parsed expression tree one node at a time and returns a value only for a number, a two-sided operator, or a leading minus. Everything else falls through to the raise, and that final raise is the only thing making calc safe, so do not delete it.
- `_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,`: This maps a syntax-tree operator class to the real Python function that performs it. Because it is a lookup table rather than a chain of if statements, anything without an entry (a function call, an attribute access, a percent sign) simply has nowhere to go and gets rejected.
- `RISKY = {"save_note"}`: A plain set of tool names that both agents check before running anything. Keeping it as data instead of an if buried inside a function is what lets the hand-written loop and the graph enforce the same rule without sharing any control flow.
- `def _as_message(reply):`: Different providers hand back slightly different objects, so this narrows all of them to one plain dict before the rest of the lab touches it. It tries .choices[0].message first, then .model_dump(), and if neither works it raises an error that names the type it actually received, which saves you a debugging session.
- `"args": json.loads(args) if isinstance(args, str) else args`: The model returns tool arguments as a JSON string, not as a dict, so it has to be parsed before you can splat it into a Python function. The isinstance guard is there because a few providers already parsed it for you, and calling json.loads on a dict would crash.

**The maths, spelled out**

```
PERCENT, THE THING THE TOOL DESCRIPTION IS TEACHING

Formula: percent of a number = (p / 100) x n
p is the percentage, n is the number you are taking it from.

Worked example, the exact task used later in this lab:
p = 17, n = 4830
17 / 100 = 0.17
0.17 x 4830 = 821.1

That is why the tool description says "for example 0.17*4830". The description is quietly teaching the model to do the percent conversion itself and hand calc a plain multiplication, because calc has no percent operator.

HOW _eval ACTUALLY GETS 821.1

ast.parse("0.17*4830", mode="eval").body produces:
BinOp(left=Constant(0.17), op=Mult, right=Constant(4830))

_eval sees a BinOp, so it runs _OPS[Mult](_eval(left), _eval(right)).
_eval(left) hits the Constant branch and returns 0.17.
_eval(right) hits the same branch and returns 4830.
operator.mul(0.17, 4830) = 821.1

WHERE OPERATOR PRECEDENCE COMES FROM

You never wrote a precedence rule, and you did not need to. "2+3*4" parses as:
BinOp(left=Constant(2), op=Add, right=BinOp(left=Constant(3), op=Mult, right=Constant(4)))

The multiplication is a child node, so _eval reaches it first:
3 x 4 = 12
then 2 + 12 = 14

If precedence were ignored and you went left to right you would get (2+3) x 4 = 20, which is wrong. Python's parser already put the tree in the right shape, and _eval just reads it from the bottom up.

Intuitively: the tree is the sum already arranged into the correct order of operations, and _eval is only doing the adding up.
```

> **Watch out:** If the model hands calc something like "17% of 4,830", ast.parse raises a SyntaxError, run_tool catches it and returns a string starting with "ERROR:", so read the trace rather than assuming the tool ran and gave a wrong number.

### 3. The hand-written agent, one more time

This is the Module 12 loop again, trimmed down to the two shared tools, and it is your baseline for everything that follows. The shape is the whole idea of an agent, send the messages, read the reply, run any tools the model asked for, append the results, and go round again until the model replies with plain text and no tool calls. max_steps is the only thing standing between you and a model that calls tools forever, so treat it as a safety belt rather than a tuning knob. Read the function once and count what it does not do, because there is no state saved anywhere outside this function, no way to resume after a crash, no retry when a model call fails, and the human approval is a callback you had to invent yourself. Those four gaps are the shopping list you carry into the framework in the next step. When you run the file you should see two or three trace rows printed, then a line starting ANSWER: with 821.1 in it.

```python
# my-work/labs/lab14/agent_plain.py
from common import (SYSTEM, RISKY, call_model, assistant_message,
                    tool_calls_of, run_tool)

def run(task, approve=lambda call: True, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    trace = []
    for _ in range(max_steps):
        msg, secs = call_model(messages)
        messages.append(assistant_message(msg))
        calls = tool_calls_of(msg)
        trace.append({"step": "model", "seconds": round(secs, 2),
                      "asked_for": [c["name"] for c in calls]})
        if not calls:
            return msg.get("content") or "", trace
        for c in calls:
            if c["name"] in RISKY and not approve(c):
                result = "DENIED by the human reviewer."
            else:
                result = run_tool(c)
            trace.append({"step": "tool", "name": c["name"],
                          "result": str(result)[:80]})
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": str(result)})
    return "gave up: hit the step limit", trace

if __name__ == "__main__":
    answer, trace = run("What is 17 percent of 4830?")
    for row in trace:
        print(row)
    print("ANSWER:", answer)
```

- `for _ in range(max_steps):`: This is the agent loop, and each pass is exactly one model call plus whatever tools that call asked for. The underscore name says the counter is never used, it exists only to put a hard ceiling on how many model calls a single run can make.
- `def run(task, approve=lambda call: True, max_steps=6):`: approve is passed in rather than hard-coded, so the same function can run unattended or behind a human gate. The default lambda always says yes, which keeps the file runnable on its own for the safe task at the bottom.
- `if not calls:
            return msg.get("content") or "", trace`: This is the stopping condition. A reply with no tool calls means the model believes it is finished, so you hand back its text and leave the loop. Without this line the loop would burn all six steps on every single run.
- `if c["name"] in RISKY and not approve(c):`: The approval gate, hand-rolled in one line. Note that a denial does not raise or stop the run, it feeds the string "DENIED by the human reviewer." back as the tool result so the model can explain itself in its final answer.
- `messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": str(result)})`: The tool result must go back to the model tagged with the same id the model used when it asked. Get that id wrong and the provider rejects the next call, because it cannot match your answer to its question.
- `return "gave up: hit the step limit", trace`: If the loop runs out of steps you return a plain string instead of raising. An agent that quits loudly and still hands you its trace is far easier to debug than one that throws from inside a loop and takes the evidence with it.

**The maths, spelled out**

```
WHY max_steps MATTERS MORE THAN IT LOOKS

The message list only ever grows, and you resend the whole list on every step. So the cost is not the cost of one call multiplied by the number of steps, it is the sum of a growing list.

Formula: total input tokens = T1 + T2 + ... + TN
Tk is the size of the whole message list at step k, and N is the number of steps.

If each step adds about the same T tokens (one assistant message plus one tool result), then Tk is roughly k x T, and the sum becomes:
total = T x (1 + 2 + ... + N) = T x N x (N + 1) / 2

T is tokens added per step, N is the number of steps.

Worked example with T = 300 tokens and N = 6:
1 + 2 + 3 + 4 + 5 + 6 = 21
total = 300 x 21 = 6300 input tokens

Same numbers with N = 3:
1 + 2 + 3 = 6
total = 300 x 6 = 1800 input tokens

6300 / 1800 = 3.5

So doubling the step limit from 3 to 6 multiplies your input tokens by about 3.5, not by 2.

Intuitively: agent cost grows with the square of the step count, not in a straight line, which is the real reason a runaway loop gets expensive so fast and why the ceiling is not optional.
```

> **Watch out:** If you see "gave up: hit the step limit", look at the trace rows first, because the usual cause is a tool result beginning with "ERROR:" that the model keeps trying to work around by calling the same tool again.

### 4. The same agent as a graph

LangGraph asks you to break that one loop apart into nodes, which are ordinary functions that take the state and return only the parts of the state they changed, and edges, which are the arrows saying which node runs next. Your for loop becomes two nodes, think and act, plus an edge from act back to think that closes the circle. Annotated[list, operator.add] is a reducer, which is the rule LangGraph uses to combine what a node returned with what was already saved, and here it means append instead of replace. InMemorySaver is the checkpointer, which writes a copy of the state after every node, and it is the only reason the approval pause in step 5 and the follow-up in step 6 work at all. interrupt() stops the run and throws a description of the pending action back out to your code, and later you resume it with a decision. Watch carefully where that call sits, it is above the line that runs the tool, because when you resume LangGraph re-runs the whole node from the top, and any real side effect placed before the interrupt would happen twice.

```python
# my-work/labs/lab14/agent_graph.py
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from common import (SYSTEM, RISKY, call_model, assistant_message,
                    tool_calls_of, run_tool)

class State(TypedDict):
    messages: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    pending: list

def think(state: State):
    msg, secs = call_model(state["messages"])
    calls = tool_calls_of(msg)
    return {"messages": [assistant_message(msg)],
            "trace": [{"step": "model", "seconds": round(secs, 2),
                       "asked_for": [c["name"] for c in calls]}],
            "pending": calls}

def act(state: State):
    msgs, rows = [], []
    for c in state["pending"]:
        if c["name"] in RISKY:
            decision = interrupt({"name": c["name"], "args": c["args"]})
            ok = decision in (True, "y", "yes")
            result = run_tool(c) if ok else "DENIED by the human reviewer."
        else:
            result = run_tool(c)
        rows.append({"step": "tool", "name": c["name"], "result": str(result)[:80]})
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})
    return {"messages": msgs, "trace": rows, "pending": []}

def next_step(state: State):
    return "act" if state["pending"] else END

builder = StateGraph(State)
builder.add_node("think", think)
builder.add_node("act", act)
builder.add_edge(START, "think")
builder.add_conditional_edges("think", next_step, {"act": "act", END: END})
builder.add_edge("act", "think")

app = builder.compile(checkpointer=InMemorySaver())

def fresh_input(task):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": task}],
            "trace": [], "pending": []}
```

- `messages: Annotated[list, operator.add]`: Annotated glues an extra hint onto a type, and LangGraph reads that hint as the reducer for this key. operator.add on two lists is concatenation, so whatever a node returns under "messages" gets appended to what was already saved.
- `pending: list`: No Annotated, so no reducer, so this key is replaced outright by whatever a node returns. That is deliberate, because pending tool calls belong to the current turn only, and appending them would make act keep re-running calls it already handled.
- `def think(state: State):`: This is exactly the top half of the hand-written loop, one model call and a look at what it asked for. It returns a dict with only the three keys it changed, and LangGraph merges each key using that key's own rule rather than overwriting the whole state.
- `decision = interrupt({"name": c["name"], "args": c["args"]})`: This stops the graph, saves the state, and hands that dict out to your caller as state["__interrupt__"]. When you later resume with a value, this same line returns that value instead of stopping, which is why the code after it reads like ordinary Python.
- `builder.add_conditional_edges("think", next_step, {"act": "act", END: END})`: A conditional edge calls next_step, takes the value it returned, and looks that value up in the mapping to decide where to go. This is the "if not calls: return" line from the hand-written loop, lifted out of the function and turned into part of the graph.
- `app = builder.compile(checkpointer=InMemorySaver())`: compile turns the description of the graph into something you can invoke, and passing a checkpointer is what makes every node write a saved copy of state under a thread id. Leave the checkpointer out and interrupt() raises an error, because a pause with nowhere to save state is not a pause, it is a crash.

**The maths, spelled out**

```
WHAT A REDUCER ACTUALLY COMPUTES

A reducer is one small rule LangGraph applies to every state key after every node.

Formula with a reducer:    new_saved = reducer(old_saved, value_the_node_returned)
Formula without a reducer: new_saved = value_the_node_returned

For "messages" the reducer is operator.add, and operator.add(a, b) on lists is a + b, which is concatenation.

WORKED EXAMPLE, ONE FULL TURN, COUNTING MESSAGES

Start: saved = [system, user], length 2
think returns {"messages": [assistant]}  ->  [system, user] + [assistant] = length 3
act returns {"messages": [tool_result]}  ->  length 3 + 1 = length 4
think runs again, returns {"messages": [final_assistant]}  ->  length 5

THE SAME TURN IF messages HAD NO REDUCER

After think, saved = [assistant], length 1.
The system prompt and the user's question are gone, and the next model call has no idea what it was asked. That is the failure the reducer prevents.

WHY pending DELIBERATELY HAS NO REDUCER

think returns {"pending": [call_a]}  ->  saved pending = [call_a]
act returns {"pending": []}          ->  saved pending = []  (the empty list replaces it)

If pending used operator.add instead:
[call_a] + [] = [call_a], forever.
next_step would then always see a non-empty list, always route to act, and the graph would loop until you killed it.

HOW MANY TIMES EACH NODE RUNS

For a run with k rounds of tools:
think runs k + 1 times, act runs k times.
node visits = 2k + 1, model calls = k + 1

With k = 1: 3 node visits and 2 model calls, which matches the 5 messages counted above (2 starting messages plus 3 added).

Intuitively: the reducer is you telling the framework, for each piece of state separately, whether new information adds to the old or wipes it out.
```

> **Watch out:** If you move run_tool(c) above the interrupt() line, the note file gets written before you are ever asked and then written a second time when you approve, which is exactly the double side effect the ordering in this node exists to prevent.

### 5. Run both on the same tasks

Run this from inside my-work/labs/lab14, because both agent files import common by plain name and Python only finds it when that folder is the working directory. Task one needs only calc, so both versions should run straight through without asking you anything. Task two asks for a note, so both stop and wait at the same point, and the single approve function serves both because the hand-written call dict and the interrupt payload carry the same name and args keys. Answer n the first time, then check that notes/ is still missing or empty, because that is the real test that the gate holds rather than just printing a prompt. Run it again and answer y, and notes/maths.txt should appear containing 163. The two printed times will be close to each other, and that is the honest result, because the framework did not make the model any faster and the model is where nearly all the time goes.

```python
# my-work/labs/lab14/run_both.py
import time
from langgraph.types import Command

from agent_plain import run as run_plain
from agent_graph import app as graph_app, fresh_input

TASKS = [
    "What is 17 percent of 4830?",
    "Multiply 12 by 13, add 7, then save the result as a note called maths.",
]

def approve(req):
    print("  APPROVE " + req["name"] + " with " + str(req["args"]) + " ? [y/N] ",
          end="", flush=True)
    return input().strip().lower() == "y"

def run_graph(task, thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    state = graph_app.invoke(fresh_input(task), cfg)
    while state.get("__interrupt__"):
        ok = approve(state["__interrupt__"][0].value)
        state = graph_app.invoke(Command(resume=ok), cfg)
    return state["messages"][-1].get("content", ""), state["trace"]

for i, task in enumerate(TASKS):
    print("\n=== TASK: " + task)
    t0 = time.time()
    answer, trace = run_plain(task, approve=approve)
    print("[hand-written] %.1fs, %d steps -> %s" % (time.time() - t0, len(trace), answer))
    t0 = time.time()
    answer, trace = run_graph(task, "task-%d" % i)
    print("[langgraph]    %.1fs, %d steps -> %s" % (time.time() - t0, len(trace), answer))
```

- `cfg = {"configurable": {"thread_id": thread_id}}`: The thread id is the filing label for every checkpoint of this run. Reuse the same string and you continue the same conversation, use a new string and you start a clean one, which is why each task gets its own "task-0" or "task-1".
- `while state.get("__interrupt__"):`: A run can pause more than once, for example if the model decides to save two notes. Looping until that key disappears handles every pause, whereas a single if would silently skip the second one and leave the graph unfinished.
- `ok = approve(state["__interrupt__"][0].value)`: __interrupt__ is a list because several nodes could pause at once, so you take the first entry, and .value is the dict you passed into interrupt(). That is how the tool name and arguments reach your prompt without the graph knowing anything about terminals.
- `state = graph_app.invoke(Command(resume=ok), cfg)`: Command(resume=...) tells LangGraph to reload the saved state for this thread and hand ok back as the return value of the interrupt() call. You do not resend the task or the message list, because the checkpoint already holds all of it.
- `print("[hand-written] %.1fs, %d steps -> %s" % (time.time() - t0, len(trace), answer))`: len(trace) counts rows, and a row is added for each model call and for each tool call, so it is a proxy for work done rather than a count of turns. If both versions print the same number, they really did take the same path.

**The maths, spelled out**

```
WHAT THE TWO PRINTED NUMBERS ACTUALLY COUNT

The steps number is trace rows, not turns. One model call adds one row, one tool call adds one row.

Formula: rows = (k + 1) model rows + k tool rows = 2k + 1
k is the number of rounds of tool calls.

Worked example, task one, "What is 17 percent of 4830?":
The model calls calc once, so k = 1.
rows = 2 x 1 + 1 = 3
Both agents should print 3, and the answer should contain 821.1 (from 0.17 x 4830).

Worked example, task two, "Multiply 12 by 13, add 7, then save the result as a note called maths.":
12 x 13 = 156, then 156 + 7 = 163.
The model usually sends one calc call with 12*13+7 and one save_note call, so k = 2 and rows = 2 x 2 + 1 = 5.
If it splits the arithmetic into two calc calls, k = 3 and rows = 7.
Both are correct behaviour, so a different row count is not a failure.

THE SECONDS

total seconds = sum of model call latencies + tool time + your own typing time at the prompt

Example for task one with two model calls at 1.2 s and 0.9 s and calc taking about 0.001 s:
1.2 + 0.9 + 0.001 = about 2.1 s

calc is roughly 0.001 s out of 2.1 s, which is about 0.05 percent of the run. So over 99 percent of the time is the model waiting on the network.

Important honesty note: for task two your reading and typing at the [y/N] prompt sits inside the measured seconds, so the task-two timings are not a fair comparison between the two agents. Compare the task-one numbers only.

Intuitively: the framework costs almost nothing at runtime, because the clock is dominated by the model, and both versions make the same model calls.
```

> **Watch out:** If you get "ModuleNotFoundError: No module named 'common'", you started Python from the wrong folder, so change directory into my-work/labs/lab14 and run the file by its bare name instead of giving a long path from elsewhere.

### 6. See what the checkpoint bought you

The hand-written agent forgets everything the instant run() returns, because its message list was just a local variable inside a function. The graph does not, because every node wrote a checkpoint filed under the thread id demo. Here you invoke once, read the saved state back with get_state without running anything, then send a second message on the same thread and watch the model answer "Now halve that" correctly. That is the whole feature in three lines, your second call carries one new message and the framework supplies the other four. Be clear about the limit though, InMemorySaver keeps checkpoints in this process's memory only, so they disappear when the script ends, which is honest and enough for a lab but is not persistence. Swapping in the SQLite checkpointer, which is a separate install, is what makes a paused run survive a restart, and it is the same two lines with a different class name.

```python
# my-work/labs/lab14/followup.py
from agent_graph import app, fresh_input

cfg = {"configurable": {"thread_id": "demo"}}
app.invoke(fresh_input("What is 17 percent of 4830?"), cfg)

snapshot = app.get_state(cfg)
print("messages saved on this thread:", len(snapshot.values["messages"]))

out = app.invoke({"messages": [{"role": "user", "content": "Now halve that."}],
                  "trace": [], "pending": []}, cfg)
print("follow-up:", out["messages"][-1].get("content", ""))
```

- `cfg = {"configurable": {"thread_id": "demo"}}`: One config object is reused by both invokes and by get_state, and that shared thread id is the only thing tying the three calls into a single conversation. Change the string between the two invokes and the follow-up starts from nothing.
- `snapshot = app.get_state(cfg)`: This reads the most recent checkpoint for the thread without running a single node or spending a token. snapshot.values is the state dict, so snapshot.values["messages"] is literally what the next model call will be handed, which makes it the first thing to print when an agent behaves oddly.
- `out = app.invoke({"messages": [{"role": "user", "content": "Now halve that."}],
                  "trace": [], "pending": []}, cfg)`: You pass only the one new user message, and the operator.add reducer appends it to the saved list rather than replacing it. trace and pending are sent as empty lists because the State TypedDict declares all three keys and LangGraph expects the shape to match.

**The maths, spelled out**

```
WHAT NUMBER SHOULD "messages saved on this thread" PRINT?

Count what each part of the first run adds:
1  system message
2  user message
3  assistant message asking for calc
4  tool result from calc
5  assistant's final sentence

So the usual answer is 5.

It is not always 5, and here is the arithmetic for the other cases using the rows = 2k + 1 idea from step 5:

messages = 2 (system + user) + (2k + 1), where k is the number of tool rounds.
k = 1, the normal case:                        2 + 3 = 5
k = 0, the model answers 821.1 with no tool:   2 + 1 = 3
k = 2, it calls calc twice:                    2 + 5 = 7

AFTER THE FOLLOW-UP

The list keeps growing, it is never trimmed. Starting from 5 and assuming the follow-up also uses one tool round:
5 + 1 (your new user message) + 3 (assistant, tool result, final assistant) = 9

THE ANSWER YOU ARE LOOKING FOR

821.1 / 2 = 410.55

The model can only produce 410.55 because 821.1 is still sitting in the saved message list. Run the same follow-up against the hand-written agent and it has nothing to halve.

Intuitively: nothing is ever removed, so the saved list is a running total of everything that happened, and that running total is exactly what "memory" means here.
```

> **Watch out:** If the follow-up replies with something like "halve what?", check that both invokes really used the same cfg, because a different thread id gives the model an empty history and there is nothing for it to halve.

### 7. Score the trade honestly

Now put a number on the trade instead of arguing about it from feeling. Count the lines in both agent files and you should get about 32 for agent_plain.py and about 53 for agent_graph.py, so the framework version is roughly two thirds longer for identical behaviour. Those extra lines bought exactly three things, state saved after every step, a pause a person can resume, and one obvious place to hang tracing. Write down which of those three you would genuinely have used in the last thing you built, and be strict, because "nice to have" is how projects collect dependencies nobody ever needed. There is one more thing worth noticing before you finish, neither file let LangGraph anywhere near your prompt, because call_model stayed inside your own code and the model saw your SYSTEM string and nothing else. That stops being true the moment you swap in a prebuilt agent from any framework, so log the exact request it sends before you trust it, and read what got inserted above your system message.

```python
python -c "import pathlib;[print(p.name, sum(1 for _ in p.open())) for p in pathlib.Path('.').glob('agent_*.py')]"
```

- `pathlib.Path('.').glob('agent_*.py')`: Matches both agent files in the current folder and nothing else. It is the reason both files were named with the same agent_ prefix, so one pattern catches both without you typing either name.
- `sum(1 for _ in p.open())`: Iterating an open text file yields one item per line, so adding 1 for each is a line count that never loads the whole file into memory. The underscore says out loud that you are throwing the line content away and only counting.
- `the list comprehension around print`: A comprehension is used only because python -c cannot easily take a multi-line for statement. It builds and discards a list of None values, which is an ugliness worth recognising as fine in a one-liner and wrong in a real file.

**The maths, spelled out**

```
PART 1, THE LINE COUNT TRADE

agent_plain.py  = 32 lines
agent_graph.py  = 53 lines

extra lines = 53 - 32 = 21
ratio = 53 / 32 = about 1.66, so 66 percent longer
capabilities gained = 3 (saved state, resumable pause, tracing hook point)
cost per capability = 21 / 3 = 7 lines each

Seven lines per capability is cheap if you need the capability and pure waste if you do not. That is the whole decision in one number.

PART 2, WHY AN INSERTED LINE AT THE TOP OF A PROMPT COSTS REAL MONEY

Formula: cost = (fresh tokens x price) + (cached tokens x price / 10)

Cached input is priced at roughly one tenth of fresh input across the major providers. The cache matches on the PREFIX, meaning the longest run of tokens counted from the very start of the prompt that is identical to last time. The moment one token differs, everything from that point onward is charged as fresh.

Symbols: price is the per-token price of fresh input. For this example use 3 dollars per million tokens, which is 0.000003 dollars per token. One tenth of that is 0.0000003 dollars per token.

Worked example. An agent prompt of 50,000 input tokens, where the first 45,000 are byte for byte the same as the previous call.
cached part: 45,000 x 0.0000003 = 0.0135 dollars
fresh part:   5,000 x 0.000003  = 0.015 dollars
total per call = 0.0285 dollars

Now a framework quietly inserts its own system text at the very top. The prefix stops matching at token 1, so nothing is cached.
total per call = 50,000 x 0.000003 = 0.15 dollars

0.15 / 0.0285 = about 5.3 times more, for the same task, the same model and the same answer.
Over 1,000 calls that is 150 dollars instead of 28.50 dollars, a difference of 121.50 dollars.

Honest note: the 3 dollars per million is an example number, not any one vendor's price. What matters and holds widely is the one tenth ratio and the fact that the cache matches from the front.

Intuitively: the cache is a bookmark pinned at the start of your prompt, and anything added above your system message tears the bookmark out.
```

> **Watch out:** Your counts will not be exactly 32 and 53 if your editor added or stripped a trailing newline when you pasted, so treat the ratio as the point rather than chasing the exact figures.

## You are done when

You are done when all four of these are true. First, python run_both.py gives an answer containing 821.1 for task one and 163 for task two, from both the hand-written agent and the LangGraph one. Second, after a run where you typed n, notes/maths.txt does not exist. Third, after a run where you typed y, notes/maths.txt exists and contains 163. Fourth, python followup.py prints a saved message count (5 in the normal case, or 3 or 7 as explained in step 6) followed by a follow-up answer containing 410.55, which the model could only produce because the earlier result was still in the saved state.

---

## Mini-project: Pick and defend

Decide which framework you would use for one scenario, and record the decision in a file a program can check. You write my-work/labs/lab14/decision.json, then run check.py, which enforces that your ratings, your choice and your defence agree with each other and with the scenario.

- Pick one scenario and create my-work/labs/lab14/decision.json with a "scenario" key set to "A", "B" or "C". A: a nightly job that reads 200 support tickets, tags each one and writes a CSV, no human involved. B: a refund assistant that must stop and ask a person before moving money, and must still be waiting after the server restarts. C: a research helper for five analysts that runs three searches at once, joins the results, and has to be traceable because compliance will ask what it did.
- Add "capabilities": an object with exactly these five keys, each set to "needed", "nice" or "no": state_between_steps, checkpoints, retries, human_approval, tracing. Be strict about needed versus nice. The checker enforces what the scenario text already states, so B must mark human_approval and checkpoints needed, C must mark tracing needed, and A must not mark human_approval needed.
- Sketch the no-framework version in bullet points, count how many of your "needed" items you would have to write yourself, and put that count in "diy_count" as a whole number 0 to 5. If it is 0 or 1, set "choice": "none" and "shortlist": [], and the checker treats that as a finished answer.
- Otherwise put two different lowercase names in "shortlist", for example ["langgraph", "crewai"], read only the documentation for your hardest needed capability rather than the quickstart, and set "choice" to one of the two. Add "python_version" from python -V, for example "3.12.4". CrewAI needs Python older than 3.14, and the checker rejects that pairing.
- Add "deciding_capability" (one of the five keys, and one you marked needed), "accepted_cost" (a phrase of three words or more naming what you give up), "defence" (exactly three sentences, under 80 words, naming the choice, the deciding capability and the cost), and "change_my_mind" (one line naming evidence that would flip you within a month).
- Save check.py next to decision.json and run: python check.py

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py    (from my-work/labs/lab14; optionally pass another path, e.g. python check.py decision_c.json)
```


**You are done when** python check.py prints about fifteen PASS lines and ends with ALL CHECKS PASSED and exit code 0. A failure names the field, for example "FAIL scenario B waits for a person and survives a restart, so human_approval and checkpoints are both 'needed'" or "FAIL defence is exactly 3 sentences (found 5)", and the run ends with a failure count and exit code 1. The checker prints one honest disclaimer: it cannot tell whether your reasoning is sound, only whether the file contradicts itself.

**If you want more:** Build your hardest needed capability, not the hello-world, in both shortlisted frameworks against a stub task, and time yourself. If neither took over an hour, that is evidence the choice matters less than you feared, so pick the one your team can read. Then write a second file for a different scenario and run python check.py decision_c.json, since the checker takes an optional path.
