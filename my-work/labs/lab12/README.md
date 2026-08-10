# Lab 12: Write an agent loop from scratch

**Module 12: Workflows, agents, and the loop**

An agent is a loop. Your code sends the conversation to the model, the model asks for one tool, your Python runs that tool for real, and the result goes back into the conversation as a new message. In this lab you write that loop yourself in about a hundred lines with no framework: four tools, a system prompt that teaches the reply format, a small parser, and three brakes that stop it running away. Every block below adds to the same file, my-work/labs/lab12/react_agent.py, so paste them in order and you end up with one script you can run. Before you start, make sure python llm.py works, as described in setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Create the file and connect to the shared helper

This block creates the file and gives you one function that talks to the model. ask is the only place in the whole lab that makes a network call, so if you later need to change providers, add logging, or add a retry, there is exactly one place to change. chat_raw comes from my-work/labs/_shared/llm.py, and which company actually answers is decided by LLM_PROVIDER in your .env file, never by anything you type in this file. chat_raw returns whatever the provider's library handed back, which is usually an object rather than text, so ask unwraps it once and returns a plain string; without that, every later function would have to know about response objects. If your provider accepts it, call chat_raw(messages=messages, temperature=0) so repeated runs give the same reply and you are debugging your parser rather than random sampling. If you are running Ollama on your own machine, pick a model of about 3B parameters or larger, because smaller ones lose the reply format after a few turns. Run the file now and nothing should happen, which is correct, because so far you have only defined a function.

```python
# my-work/labs/lab12/react_agent.py
import re
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw


def ask(messages):
    """Send the whole conversation, get back one chunk of text."""
    resp = chat_raw(messages=messages)
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "choices"):
        return resp.choices[0].message.content
    return resp["choices"][0]["message"]["content"]
```

- `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: __file__ is the path of this script, resolve() turns it into a full absolute path, and parents[1] is the folder two levels up, which is labs. Joining "_shared" onto that and adding it to the import path lets the next line find llm.py without installing anything or copying files around.
- `from llm import chat_raw`: One import gives you the shared model client that every lab in this course uses. chat_raw is the lower level of the two helpers, and you want it here because this lab prints, trims and parses the reply text itself.
- `resp = chat_raw(messages=messages)`: You send the entire conversation every single turn, not just the newest line, because the model keeps no memory between calls. messages is a list of small dictionaries, each with a role ("system", "user" or "assistant") and the text content for that role.
- `the three return branches`: Different providers hand back different shapes: a plain string, an object with a .choices attribute, or a plain dictionary. Handling all three means the same file works whether LLM_PROVIDER points at Groq, Gemini or a local Ollama model, and every other function in the lab only ever sees a string.

**The maths, spelled out**

```
Why temperature=0 makes runs repeatable.

Formula: p(i) = exp(z(i) / T) divided by the sum over all j of exp(z(j) / T)

Symbols: z(i) is the raw score (the "logit") the model gives to candidate next token i, T is the temperature you pass, exp is the exponential function, and p(i) is the chance that token i gets picked.

Worked example with three candidate tokens scoring 3.0, 2.0 and 1.0.
At T = 1: exp(3) = 20.09, exp(2) = 7.39, exp(1) = 2.72. The sum is 30.20. So the chances are 20.09 / 30.20 = 0.67, 7.39 / 30.20 = 0.24, and 2.72 / 30.20 = 0.09.
At T = 0.5 the scores are first divided by 0.5, giving 6, 4 and 2: exp(6) = 403.4, exp(4) = 54.6, exp(2) = 7.4. The sum is 465.4, so the chances become 0.87, 0.12 and 0.02.
As T gets closer to 0, the top token's share climbs towards 1.00 and everything else falls towards 0.00.

Intuitively: temperature 0 means always take the highest scoring token, so the same input gives you the same output and any change you see came from your code and not from luck. Honest simplification: providers implement T = 0 as a greedy pick, and hosted models can still vary slightly run to run because of floating point rounding and how requests are batched on their servers.
```

> **Watch out:** If you get ModuleNotFoundError: No module named 'llm', your file is not saved inside my-work/labs/lab12/, so parents[1] points at the wrong folder and my-work/labs/_shared was never added to the path.

### 2. Write the tools as ordinary functions

A tool here is just a normal Python function, nothing more. Each one takes a single string and returns a single string, and that choice is what keeps the parser in step four down to a few lines. PRICES and STOCK stand in for a database, so the whole lab runs with no network and no account, and the numbers are small enough that you can check the agent in your head. _key cleans up whatever the model typed, because across runs the same model will write "monitors", "Monitor" and "monitor", and all three should find the same row. calc is the dangerous one: eval runs whatever text you hand it as Python, so the character whitelist in front of it is the only thing stopping your agent running arbitrary code, and you must not remove it. save_note writes to a file, which you cannot undo, so it is marked approve: True and step five will stop and ask you before running it. The help text is not a comment for you; it is copied straight into the system prompt in step three, so it is literally the text the model reads when choosing which tool to reach for.

```python
PRICES = {"laptop": 900.0, "monitor": 220.0, "dock": 150.0}
STOCK = {"laptop": 3, "monitor": 7, "dock": 4}


def _key(arg):
    k = arg.strip().strip('"\'').lower()
    if k.endswith("s") and k[:-1] in PRICES:
        k = k[:-1]
    return k


def price_of(arg):
    k = _key(arg)
    return f"{k} costs {PRICES[k]:.2f} dollars each" if k in PRICES else f"no item called {k}"


def stock_of(arg):
    k = _key(arg)
    return f"we hold {STOCK[k]} units of {k}" if k in STOCK else f"no item called {k}"


def calc(arg):
    if not re.fullmatch(r"[0-9+\-*/(). ]+", arg.strip()):
        return "error: only numbers and + - * / ( ) are allowed"
    return str(eval(arg.strip(), {"__builtins__": {}}, {}))


def save_note(arg):
    with open("agent_note.txt", "a", encoding="utf-8") as f:
        f.write(arg.strip() + "\n")
    return "saved to agent_note.txt"


TOOLS = {
    "price_of":  {"fn": price_of,  "approve": False, "help": "price_of(item) gives the price of one unit"},
    "stock_of":  {"fn": stock_of,  "approve": False, "help": "stock_of(item) gives how many units we hold"},
    "calc":      {"fn": calc,      "approve": False, "help": "calc(expression) does arithmetic, for example 7 * 220"},
    "save_note": {"fn": save_note, "approve": True,  "help": "save_note(text) appends one line to agent_note.txt"},
}
```

- `PRICES = {"laptop": 900.0, "monitor": 220.0, "dock": 150.0}`: This is your pretend database. Keeping it as a plain dictionary in the file means the lab runs offline, and it means you know every correct answer in advance, which is the only way to tell a working agent from a convincing one.
- `if k.endswith("s") and k[:-1] in PRICES:`: The model writes "monitors" about as often as "monitor". k[:-1] drops the final s, and the check only accepts the shorter word if it is a real item, so nothing gets mangled by accident. Without this line the tool answers "no item called monitors" and the agent burns a step recovering.
- `if not re.fullmatch(r"[0-9+\-*/(). ]+", arg.strip()):`: fullmatch means the whole string must be built from the 18 allowed characters: the ten digits plus plus, minus, star, slash, open bracket, close bracket, dot and space. No letters can pass, so no function name or variable name can pass, and that is what makes the eval on the next line safe.
- `return str(eval(arg.strip(), {"__builtins__": {}}, {}))`: eval runs the string as Python code. Passing {"__builtins__": {}} strips out built in names like open and __import__, giving you a second layer of protection under the whitelist. str() wraps the number because every tool in this lab must return text.
- `the TOOLS registry`: One dictionary holds three things per tool: the function to call, whether a human must approve it, and the single line of help the model reads. Keeping them together means step three builds the prompt from the same source the loop calls, so the description and the behaviour can never drift apart.

**The maths, spelled out**

```
What calc actually computes, and why the expression matters more than the calculator.

Rule: Python does star and slash before plus and minus, and does anything in brackets first.

Worked example: the string "7 * 220 + 4 * 150" is read as (7 x 220) + (4 x 150) = 1540 + 600 = 2140. It is not read as 7 x (220 + 4) x 150, which would give 7 x 224 = 1568, then 1568 x 150 = 235200. Those two answers differ by a factor of about 109, from the same digits in the same order.

One more number to expect: in Python 3 the slash always gives a decimal, so calc("1540 / 7") returns the string "220.0" and not "220". The model sees "220.0" in its Observation line and usually copies that trailing .0 into its final answer.

Intuitively: calc is a real calculator, so the arithmetic is always right. The agent can still be wrong, because it chose the expression, and a bracket in the wrong place is its mistake arriving inside a correct tool.
```

> **Watch out:** If you loosen or delete the whitelist regex, calc becomes a way to run any Python at all, and nothing on screen will warn you, because the successful attack looks exactly like a successful sum.

### 3. Teach the model the format in the system prompt

This block is all of ReAct, and ReAct is only a prompt shape plus a small parser, not a library you install. You are asking the model for three lines and then silence: one Thought, one Action, one Action Input. The silence matters because your code needs a predictable place to cut the reply, which is exactly what step four does. TOOL_LINES is built by looping over the TOOLS dictionary rather than typed out a second time, so when you add a fifth tool the prompt updates itself and can never describe a tool that no longer exists. The sentence telling the model never to write its own Observation helps a lot, but it is a request and not a control, so step four enforces the same rule in Python where the model cannot argue with it. Print SYSTEM once and you should see four dashed lines sitting under the word Tools, matching your four functions exactly.

```python
TOOL_LINES = "\n".join(f"- {t['help']}" for t in TOOLS.values())

SYSTEM = f"""You answer questions by using tools, one at a time.

Tools:
{TOOL_LINES}

Reply in exactly this shape and then stop:

Thought: one short sentence about what you need next
Action: the tool name on its own
Action Input: the single argument, plain text, no quotes

Stop after the Action Input line. The user will run the tool and send back one
line starting with Observation:.

Never write an Observation line yourself. Never invent a tool result.

When you have enough to answer, reply with one line only:

Final Answer: your answer
"""
```

- `TOOL_LINES = "\n".join(f"- {t['help']}" for t in TOOLS.values())`: This walks every entry in TOOLS, pulls out its help string, puts a dash in front, and joins them with newlines. Generating the list means the prompt is always a true description of the tools that exist, and typing that list by hand is the usual way a prompt and its code quietly stop matching.
- `SYSTEM = f"""You answer questions by using tools, one at a time.`: The f in front of the triple quoted string is what allows {TOOL_LINES} to be swapped in for its value. Everything else inside the quotes is literal text that gets sent to the model on every single turn of the loop.
- `the three line reply shape`: Thought, then Action, then Action Input is the entire ReAct format. Each one is its own line with a fixed label, because your parser finds them with simple line based pattern matching instead of trying to understand free prose.
- `Never write an Observation line yourself. Never invent a tool result.`: This lowers how often the model fakes a result, but it does not stop it, because finishing a pattern is easier for a model than stopping in the middle of one. Treat every instruction of this kind as a preference, which is why the real fix lives in code in step four.
- `Final Answer: your answer`: This gives your loop an exit door it can recognise. Without one fixed phrase to search for, your code has no reliable way to tell "I am finished" apart from "I am still working".

**The maths, spelled out**

```
How much this prompt costs you, since it is resent on every turn.

Rule of thumb: for ordinary English, 1 token is roughly 4 characters, so tokens are about characters divided by 4.

Symbols: a token is the unit providers bill you in, and it is a common chunk of text rather than a whole word.

Worked example: the four tool lines are about 199 characters and the rest of SYSTEM is about 518, so the whole prompt is roughly 717 characters. 717 divided by 4 is about 180 tokens. A six step run sends it six times, so 6 x 180 = 1080 tokens of instructions alone. At a price of 0.50 dollars per million input tokens that is 1080 x 0.50 / 1,000,000 = 0.00054 dollars, so about five hundredths of a cent per run.

Intuitively: the prompt is cheap here, but the cost is multiplied by the number of steps, so a long system prompt in a twenty step agent is twenty copies of that prompt. Honest simplification: the 4 characters per token rule is an average for English, and code, numbers and rare words use noticeably more tokens per character.
```

> **Watch out:** Because SYSTEM is an f-string, any literal curly brace you add later (a JSON example, for instance) is treated as a substitution slot and raises a KeyError or NameError; write them doubled as {{ and }} to get one brace out.

### 4. Cut the reply, then read it

These two small functions are where your agent stops being fiction, so read them twice. Models are trained to complete patterns, so after writing the Action Input line the model very often carries straight on, writes its own Observation line with a made up number, and then gives a confident Final Answer built on top of it. trim walks the reply line by line and stops the moment a line starts with the words action input, throwing away everything after it, so the invented part never reaches your loop, your log or your user. parse then reports one of exactly three outcomes: final means the model is done, act means it wants a tool and hands you the name and the argument, and bad means the reply did not match the shape at all. A bad reply is a normal outcome and not a crash, and the loop handles it by sending back a short correction and asking again. Small local models fall out of the format far more often than hosted ones, which is exactly why the loop has to survive it rather than raise an exception.

```python
def trim(text):
    """Keep the reply only up to its Action Input line, so the model cannot
    write its own Observation and then reason from it."""
    kept = []
    for line in text.splitlines():
        kept.append(line)
        if line.strip().lower().startswith("action input:"):
            break
    return "\n".join(kept).strip()


def parse(text):
    if "Final Answer:" in text:
        return "final", text.split("Final Answer:", 1)[1].strip()
    action = re.search(r"^\s*Action:\s*(.+)$", text, re.M)
    arg = re.search(r"^\s*Action Input:\s*(.+)$", text, re.M)
    if action and arg:
        return "act", (action.group(1).strip(), arg.group(1).strip())
    return "bad", text
```

- `the for loop over text.splitlines()`: It copies lines into the kept list one at a time and breaks out the moment it sees the Action Input line. Everything after that point is dropped and never stored, and that single break is the whole defence against invented tool results.
- `if line.strip().lower().startswith("action input:"):`: strip() removes leading and trailing spaces and lower() makes the comparison ignore capitals, so "Action Input:", "  action input:" and "ACTION INPUT:" all match. Models change their capitalisation between runs, so a strict check would let the guard fail at random.
- `return "final", text.split("Final Answer:", 1)[1].strip()`: The 1 tells split to cut at most once, so an answer that itself contains the words Final Answer still comes back whole. Index [1] takes everything after the label, which is the answer text you return to the caller.
- `action = re.search(r"^\s*Action:\s*(.+)$", text, re.M)`: re.M is multiline mode, which makes ^ and $ mean start and end of a line instead of start and end of the whole reply. \s* allows any leading spaces, and (.+) captures the rest of that line, which is the tool name.
- `return "bad", text`: A reply that fits neither shape is reported as data, not raised as an error. The loop turns this into a short correction message and tries again, so one badly formatted turn costs you a step instead of killing the run.

**The maths, spelled out**

```
What keeping the untrimmed text would actually cost, on top of being wrong.

Formula: extra tokens paid = e x n x (n - 1) / 2

Symbols: e is the number of extra tokens the model tacks on after Action Input each turn, and n is the number of turns in the run. The n x (n - 1) / 2 part is there because junk added on turn 1 is resent on turns 2 through n, junk added on turn 2 is resent on turns 3 through n, and so on.

Worked example: say the model adds a fake Observation and a Final Answer worth about 120 extra tokens each turn, over a 6 turn run. Extra = 120 x 6 x 5 / 2 = 120 x 15 = 1800 tokens you paid for and did not want. The first turn's junk alone is resent 5 times, so 120 becomes 600.

Intuitively: text you keep in an agent transcript is not paid for once, it is paid for again on every remaining turn. The money is the smaller problem here though, because the real cost is a number nobody calculated being treated as a fact for the rest of the run.
```

> **Watch out:** parse looks for "Final Answer:" with those exact capitals, so a model that writes "final answer:" in lower case falls through to bad and your agent loops until the step cap.

### 5. Write the loop with its brakes on

This is the agent, and the point of reading it is to see how ordinary it is: send the messages, cut the reply, run one tool, append the real result, repeat. There is no hidden agent state on the provider's side, so the model remembers nothing between turns except what you resend in messages. Three brakes are built in and each one stops a different failure. max_steps caps how many model calls a confused run can make, so it ends instead of continuing all night. already_tried holds every tool name and argument pair you have already run and blocks an exact repeat, which is the most common way an agent quietly burns money. The approve flag stops everything and waits for you to type y before any tool that writes to disk runs, which is a gate in your code rather than a polite line in a prompt the model can skip. Watch the two append lines at the bottom, because they are why your sixth model call is much bigger and more expensive than your first.

```python
def run(question, max_steps=6, verbose=True):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    already_tried = set()

    for step in range(1, max_steps + 1):
        reply = trim(ask(messages))
        if verbose:
            print(f"\n--- step {step} ---\n{reply}")

        kind, payload = parse(reply)
        if kind == "final":
            return payload
        if kind == "bad":
            obs = "I could not read that. Use the exact Thought / Action / Action Input shape."
        else:
            name, arg = payload
            if name not in TOOLS:
                obs = f"error: there is no tool called {name}"
            elif (name, arg) in already_tried:
                obs = "error: you already ran that exact call. Use the earlier result or give a Final Answer."
            else:
                already_tried.add((name, arg))
                if TOOLS[name]["approve"] and input(f"Allow {name}({arg})? [y/N] ").strip().lower() != "y":
                    obs = "error: the human refused this action"
                else:
                    obs = TOOLS[name]["fn"](arg)

        if verbose:
            print(f"Observation: {obs}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return f"stopped: hit the {max_steps} step limit without a Final Answer"
```

- `the initial messages list`: The conversation starts with exactly two messages: the rules as the system message, then the question as the user message. Everything the agent later learns is appended to this same list, so this list is the agent's entire memory and nothing outside it exists.
- `reply = trim(ask(messages))`: One line does the model call and the safety cut together. Written this way you can never accidentally use the untrimmed text later, because the untrimmed text is never given a variable name.
- `elif (name, arg) in already_tried:`: already_tried is a set of (tool name, argument) pairs. Checking the pair rather than just the name lets the agent call calc many times with different sums while still blocking the identical call twice, which is the usual shape of a stuck loop.
- `if TOOLS[name]["approve"] and input(f"Allow {name}({arg})? [y/N] ").strip().lower() != "y":`: This is the human gate. Python's and stops as soon as the left side is False, so input() only ever runs for tools marked approve: True, and anything other than y (including just pressing Enter) counts as a refusal.
- `the two messages.append lines`: The model's trimmed reply goes back in as the assistant turn, and the real tool result goes back as a user turn labelled Observation. This is how a fact produced by your Python gets into the model's context, and it is the only part of the loop that is not text prediction.
- `return f"stopped: hit the {max_steps} step limit without a Final Answer"`: If the for loop finishes without ever returning, the agent failed to answer. Returning a clear sentence instead of None means the caller sees what went wrong rather than printing ANSWER: None and leaving you guessing.

**The maths, spelled out**

```
Why step six costs far more than step one.

Formula for one turn: input tokens on turn k = S + (k - 1) x d
Formula for the whole run: total input tokens = n x S + d x n x (n - 1) / 2

Symbols: S is the starting size in tokens (system prompt plus the question), d is how many tokens each completed turn adds (the assistant reply plus the observation), k is which turn you are on, and n is how many turns the run takes.

Worked example with S = 200, d = 60.
Turn 1 sends 200 tokens. Turn 6 sends 200 + 5 x 60 = 500 tokens, which is 2.5 times turn 1.
Over 6 turns the total is 6 x 200 = 1200, plus 60 x 6 x 5 / 2 = 60 x 15 = 900, giving 2100 input tokens.
Over 20 turns the total is 20 x 200 = 4000, plus 60 x 20 x 19 / 2 = 60 x 190 = 11400, giving 15400 input tokens.
So 3.3 times as many steps costs 15400 / 2100 = 7.3 times as many tokens.

Intuitively: because every turn resends the whole growing transcript, cost grows with the square of the number of steps, not in a straight line. That is why a step cap saves you real money and not just time.
```

> **Watch out:** input() needs a real console, so running this from an IDE Run button that does not attach a terminal gives you EOFError at the approval prompt; run it from a command window instead.

### 6. Run it and read the trace

Now you run it and read what comes out, which is the actual skill this lab is teaching. The first question needs the price of a monitor, the number of monitors, and one multiplication, so a healthy trace shows three tool calls and then a Final Answer of 1540. The second question ends in save_note, so the run pauses, prints Allow save_note(...)? [y/N] and waits for you. Answer y once and check that agent_note.txt appears, then run it again and answer n, and read what the agent does with the refusal, because handling a no without falling apart is a real product requirement. Read the whole trace and not just the last line, because the interesting failures always sit in the middle, for example a tool called with the wrong argument that still returned something plausible. If the run finishes in one step with the right number, the model did the arithmetic in its head and skipped your tools entirely, which is worth noticing even though this particular answer happened to be right.

```python
if __name__ == "__main__":
    q1 = "What is the total value of the monitors we currently hold in stock?"
    print("\nANSWER:", run(q1))

    q2 = "Work out the total value of the docks we hold, then save that number as a note."
    print("\nANSWER:", run(q2))
```

- `if __name__ == "__main__":`: Everything under this line runs only when you run the file directly. If you later import react_agent from another script, the two demo questions will not fire, which matters for the mini-project where you reuse run() from elsewhere.
- `q1 = "What is the total value of the monitors we currently hold in stock?"`: The question needs three separate facts and never names a tool. That is deliberate: you are testing whether the model can pick tools from their one line descriptions, not whether it can follow a recipe you already wrote out for it.
- `q2 = "Work out the total value of the docks we hold, then save that number as a note."`: This one ends in an action that changes something on your disk, so it is the question that exercises the approval gate. The word "then" also forces at least two steps, so you get to watch the transcript grow.
- `print("\nANSWER:", run(q1))`: run() prints the trace as it goes because verbose defaults to True, then returns the final answer, which this line prints last. If you see a full trace but ANSWER is the step limit message, the loop ran out of steps without ever producing a Final Answer line.

**The maths, spelled out**

```
The two answers you should already know before you press run.

Formula: total value = unit price x units held

Symbols: unit price comes from the PRICES dictionary, units held comes from the STOCK dictionary, both defined in step two.

Worked example: a monitor costs 220.00 dollars and you hold 7, so 220.00 x 7 = 1540.00. A dock costs 150.00 dollars and you hold 4, so 150.00 x 4 = 600.00. Laptops, which neither question asks about, would be 900.00 x 3 = 2700.00.

Intuitively: you can check the agent because you can do the sum yourself. A demo where you cannot check the answer teaches you nothing about agents, because in a trace a wrong number and a right number look exactly the same.
```

> **Watch out:** agent_note.txt is opened with a relative path, so it lands in whatever folder you ran the command from and not next to the script; if you cannot find it, check your current directory.

### 7. Break it on purpose, three ways

These three experiments teach you more than the successful run did, so do all of them rather than reading about them. First, ask run("When is the next monitor shipment arriving?"), where no tool can possibly help, and watch the agent spend its steps guessing until max_steps saves you, because nothing inside the model tells it to give up. Second, comment out the already_tried check and ask the same question again, then count the identical calls, which is what an unguarded loop looks like from the inside. Third, change reply = trim(ask(messages)) to reply = ask(messages) and ask the monitor question again, and you will often see the model write its own Observation line and hand you a confident total it never calculated. That fake answer looks exactly as convincing as a real one on screen, and that is the single most important thing in this module. Put all three guards back afterwards and check the file line by line, because a forgotten experiment here makes every later run untrustworthy without ever showing an error.

**The maths, spelled out**

```
Why the repeat detector helps but does not solve the problem.

How it works: already_tried stores pairs of (tool name, argument string), and only an exact match of both is blocked.

Worked example: with 4 tools, and just 3 spellings the model might use for one item ("monitor", "monitors", "Monitor"), there are 4 x 3 = 12 different pairs it can produce about that one item. All 12 are treated as different calls, so the guard only stops the 13th if it repeats one of them exactly. Argument text is free form, so the real number of possible pairs is effectively unlimited.

What the guard is worth in tokens: using the growth formula from step five with S = 200 and d = 60, a run that gives up after 2 turns costs 2 x 200 + 60 x 2 x 1 / 2 = 460 tokens, while one that runs the full 6 costs 2100 tokens. That is 3 times the steps for about 4.6 times the tokens.

Intuitively: the repeat detector catches an agent stuck in an exact circle, which is common, but it cannot catch an agent wandering through slightly different useless calls. The step cap is the brake that always works, so never rely on the repeat detector alone.
```

> **Watch out:** The easiest mistake here is leaving reply = ask(messages) in the file after experiment three, and you will not get an error, you will just get answers that are sometimes invented.

## You are done when

Your agent answers the monitor question with 1540 in three tool calls or fewer, the printed trace shows every Thought, Action and Observation line in order, the save_note question pauses and waits for you to type y (and refusing with n does not crash it), and the shipment question you cannot answer stops with the "hit the 6 step limit" message instead of looping forever.

---

## Mini-project: Same task, two designs

Build the same expense-totalling job twice, once as a fixed workflow and once as the lab's agent, and record what every run cost. The artefact is my-work/labs/lab12/runs.json, and check.py recomputes the right answer from your own expenses.txt and tests your record against it.

- Create my-work/labs/lab12/expenses.txt. One expense per line, exactly four comma-separated fields: date, description, category, amount. No commas inside a field. At least 15 lines that parse, categories as lowercase words (travel, meals, office, software), plus two awkward ones on purpose: a refund as a negative amount (`15 Mar, refund cancelled hotel, travel, -214.00`) and one line with the amount missing (`17 Mar, stationery, office`).
- Build my-work/labs/lab12/workflow.py. Fixed path: one model call per line to pull out the four fields, then plain Python sums by category and lists every description whose amount is over 200. The model never picks the next step and never does the arithmetic.
- Build my-work/labs/lab12/agent_expenses.py. Import run() from react_agent.py, add a read_expenses tool that returns the file contents, keep calc, keep max_steps=6 and the repeat detector. One instruction: total the expenses by category and flag anything over 200.
- Run each version three times and write my-work/labs/lab12/runs.json in exactly this shape: {"runs": [{"design": "workflow", "run": 1, "model_calls": 17, "input_tokens": 3910, "seconds": 21.4, "totals": {"travel": 434.00, "meals": 254.35}, "total_correct": true}, ... six entries, three per design ...], "flagged_over_200": ["annual IDE licence", "flight to Dublin"], "decision": {...}}. Category keys are lowercase, amounts to the cent.
- Fill the decision block: {"ship": "workflow" or "agent", "metric": one of model_calls / input_tokens / seconds / total_correct, "workflow_value": number, "agent_value": number, "changes_my_mind": one sentence of at least 40 characters}. Those are your five sentences compressed into fields you can be held to.
- Save check.py beside expenses.txt in my-work/labs/lab12 and run `python check.py` from that folder.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab12 && python check.py
```


**You are done when** check.py prints one PASS or FAIL line per check and ends with ALL CHECKS PASSED and exit code 0. It rebuilds the category totals from your own expenses.txt, so passing means at least one workflow run matched them to the cent, every agent run stayed inside the 6 step cap, every total_correct flag agrees with the totals written beside it, and flagged_over_200 is exactly right. It also prints one line saying your written argument is not checked automatically.

**If you want more:** Add `14 Mar, hotel Berlin, travel, 180 EUR`. check.py counts it as a line with no readable amount, and your workflow will drop or mangle it just as quietly, because no step converts currency and nothing is allowed to notice. Decide whether the fix is one more workflow step or the agent, and put the reason in decision.changes_my_mind.
