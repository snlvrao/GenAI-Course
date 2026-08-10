# Lab 09: Make the model call your function

**Module 9: Getting exact data back, and calling your code**

In this lab you build one tool by hand and watch every single message that crosses the wire. You will see the model ask for a function, you will run that function yourself, and you will hand the answer back so the model can turn a raw dict into a sentence. Nothing is hidden inside a framework, so by the end you can point at the exact line in your own file where the decision to act was made, and that line is yours, not the model's. You will also measure what one tool definition costs you in input tokens on every request, which is the number that decides an agent bill later.

Before you start, make sure `python llm.py` works, see `setup.html`.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Check the helper and note the token count

Make a file at my-work/labs/lab09/tool_roundtrip.py and run this before you write anything else, because a broken key or a wrong model will look like a tool bug three steps from now. The shared helper chat_raw hands your messages straight to the provider and gives you back the raw reply object, so you can look at fields that a plain string would hide, such as tool_calls and usage. A token is a chunk of text the model counts in, usually a short word or part of a word, and prompt_tokens is how many of them the provider charged you for on the way in. On screen you should see one line naming your provider, model, endpoint and key state, then three words of greeting, then a small number somewhere between about 10 and 20. If the key line says MISSING, fix your .env now, and remember you change providers by editing .env, never by editing this file. If you are running a local model through Ollama, use one around 3B parameters or larger, because smaller ones call tools unreliably. If your copy of llm.py takes messages positionally rather than by keyword, adjust the calls below to match.

```python
import sys, json, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from llm import chat_raw, whoami

print(whoami())
r = chat_raw(messages=[{"role": "user", "content": "say hi in three words"}])
print("reply       :", r.choices[0].message.content)
print("input tokens:", r.usage.prompt_tokens)
```

- `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: Python can only import from folders it already knows about, and my-work/labs/_shared is not one of them. This works out the path from this file's own location (__file__), goes up one folder (parents[1]), and adds _shared to the import list, so the script runs from any working directory.
- `from llm import chat_raw, whoami`: chat returns a tidy string, which is useless here because the interesting parts of a tool call are not in the text. chat_raw returns the whole reply object, which is the only way to reach tool_calls and usage.
- `print(whoami())`: Prints which provider, model and endpoint you are actually about to use, plus whether the key is set. Half of all confusion in this lab is talking to a different model than you thought.
- `messages=[{"role": "user", "content": "say hi in three words"}]`: A conversation is a list of dicts, each with a role and content. This is the shape every provider in the registry expects, which is why one helper can talk to all of them.
- `r.choices[0].message.content`: choices is a list because the API can return several alternative replies in one call. You asked for one, so you take index 0.
- `r.usage.prompt_tokens`: The count of input tokens the provider billed for this request. You print it now so you have a baseline to compare against in step 8.

**The maths, spelled out**

```
Rough token count, and what it costs.

Formula:
  tokens is approximately characters / 4      (for ordinary English)
  cost = (tokens / 1,000,000) * price_per_million

What the symbols mean:
  characters = how many characters are in your text
  4 = the rough average number of characters per token for English prose
  price_per_million = what your provider charges per million input tokens

Worked example:
  "say hi in three words" is 21 characters.
  21 / 4 = 5.25, so about 5 tokens of your actual words.
  But prompt_tokens will print more like 12 to 15. The extra is the chat wrapping the provider adds around your message (role markers, start and end markers) that you never typed.
  At $0.50 per million input tokens, 15 tokens costs 15 / 1,000,000 * 0.50 = $0.0000075.

Intuitively: a token is about three quarters of an English word, and you pay for everything that is sent, including the packaging you did not write. The divide-by-4 rule is a rough guide only, not the real tokenizer.
```

> **Watch out:** If the whoami line ends with key=MISSING, stop and fix .env now, because every later step will fail with an authentication error that looks nothing like a tool problem.

### 2. Write the tool as an ordinary Python function

Start with plain code that has nothing to do with the model at all, because a tool is just a normal function that you later agree to expose. Call it yourself once from a Python prompt, for example get_price("bread", 3), so you know it works before any model is involved. Two design choices matter here and both are deliberate. First, it returns a dict (a set of named values) rather than a sentence, because in step 5 you will turn that dict into JSON text and hand it back to the model, and named fields are far easier for a model to read correctly than prose. Second, an unknown item comes back as {"error": ...} instead of raising an exception, because an exception kills your program while an error dict travels back to the model as data it can read and correct on the next turn. You should see {'item': 'bread', 'unit_price': 2.4, 'quantity': 3, 'total': 7.2} printed at your prompt.

```python
PRICES = {"apple": 0.50, "bread": 2.40, "milk": 1.15}

def get_price(item: str, quantity: int = 1) -> dict:
    key = item.strip().lower()
    if key not in PRICES:
        return {"error": f"unknown item: {item}", "known_items": sorted(PRICES)}
    return {
        "item": key,
        "unit_price": PRICES[key],
        "quantity": quantity,
        "total": round(PRICES[key] * quantity, 2),
    }
```

- `PRICES = {"apple": 0.50, "bread": 2.40, "milk": 1.15}`: A stand-in for whatever real thing you would query in production, such as a database or an internal API. Keeping it a three line dict means nothing in this lab can fail for a reason that is not about tool calling.
- `key = item.strip().lower()`: The model writes the argument, and it may write "Bread", " bread " or "BREAD". strip() removes spaces at the ends and lower() makes it lowercase, so all of those find the same entry.
- `return {"error": f"unknown item: {item}", "known_items": sorted(PRICES)}`: An unknown item is returned as data, not raised as an exception, so your loop keeps running. Sending known_items back gives the model a concrete list to pick from instead of guessing again.
- `"total": round(PRICES[key] * quantity, 2)`: Money needs rounding to two decimal places, because binary floating point cannot store 2.40 exactly. Rounding once at the end, rather than at every step, keeps the answer correct.
- `def get_price(item: str, quantity: int = 1) -> dict:`: The : str, : int and -> dict parts are type hints. Python does not enforce them at runtime, they are notes for you and for tools, which is exactly why step 6 adds a real check.

**The maths, spelled out**

```
Why 2.40 times 3 is not quite 7.20 inside the computer.

Formula:
  total = round(unit_price * quantity, 2)

What the symbols mean:
  unit_price = the price of one item, taken from PRICES
  quantity = how many the model asked for
  round(x, 2) = the nearest value to x that has two decimal places

Worked example:
  2.40 * 3
  A computer stores numbers in binary, and 2.40 cannot be written exactly in binary. What is actually stored is 2.39999999999999991118...
  Multiply that by 3 and Python prints 7.199999999999999, not 7.2.
  round(7.199999999999999, 2) = 7.2, which is the answer you want.
  Compare 0.50 * 3 = 1.5 exactly, because 0.5 is one half and halves are exact in binary.

Intuitively: binary cannot write 2.40 exactly, in the same way decimal cannot write one third exactly. Rounding at the end hides the tiny error. For real money you would store integer cents or use decimal.Decimal instead, and this lab is a simplification on that point.
```

> **Watch out:** Python prints 7.2 rather than 7.20, so do not think the arithmetic is broken when the trailing zero is missing.

### 3. Describe the tool for the model

This block is the only thing the model ever learns about your function, because your Python code is invisible to it. What you are writing is a JSON Schema, which is a written description of an object: which fields exist, what type each one is, and which ones are required. Read the description out loud and ask whether a new colleague, given only these words, would know when to use it and when not to. Notice the last sentence tells the model when not to call it, which is the single most effective line in most tool descriptions. Nothing prints when you run this step, it is data sitting in a variable, and it does nothing until you attach it in step 4. Every character here is pasted into your prompt and charged as input tokens on every single request, so the job is to be specific and short at the same time.

```python
GET_PRICE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_price",
        "description": (
            "Look up the shop price of one grocery item and multiply it by a "
            "quantity. Use it for any question about what an item costs or what "
            "several of them cost. Do not use it for questions with no item in them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string",
                         "description": "Single item name, for example apple, bread, milk"},
                "quantity": {"type": "integer", "minimum": 1,
                             "description": "How many of the item. Defaults to 1."},
            },
            "required": ["item"],
            "additionalProperties": False,
        },
    },
}
```

- `"name": "get_price"`: This exact string is what the model echoes back when it wants the tool, and it is what your dispatch code compares against in step 6. It happens to match your Python function name here, but nothing forces that, the link is a string you maintain.
- `"Do not use it for questions with no item in them."`: A negative instruction, telling the model when to stay away. Models over-call tools that are attached, and one sentence like this usually fixes more wrong picks than rewriting the positive half.
- `"parameters": {"type": "object", "properties": {...}}`: This is JSON Schema saying the arguments come as an object with named fields. Each field gets a type and its own description, and those per-field descriptions are where you put example values.
- `"description": "Single item name, for example apple, bread, milk"`: Examples inside a parameter description steer the format of the value the model writes. Without "single item name", models happily send "3 loaves of bread" as the item.
- `"required": ["item"]`: Says item must always be present while quantity may be left out. That matches your Python default of quantity=1, so the two descriptions of the same function agree.
- `"additionalProperties": False`: Asks that no extra invented fields appear in the arguments. Treat it as a request rather than a guarantee, because providers differ in how much JSON Schema they really enforce.

**The maths, spelled out**

```
What this block costs you in tokens.

Formula:
  added_tokens is approximately len(json.dumps(tool)) / 4
  cost_per_conversation = added_tokens * number_of_turns

What the symbols mean:
  json.dumps(tool) = the definition turned back into one line of JSON text, which is roughly what gets sent
  len(...) = its length in characters
  4 = the same rough characters-per-token figure from step 1
  number_of_turns = how many times you call the API in one conversation

Worked example:
  json.dumps of this exact definition is 576 characters.
  576 / 4 = 144, so roughly 144 tokens added to every request.
  The description string alone is 202 characters, about 50 tokens. Cutting it in half saves about 25 tokens per request.
  A 12 turn conversation pays 144 * 12 = 1,728 tokens for this one tool, whether it is used or not.

Intuitively: a tool definition is text pasted into your prompt, so its length is money, charged again on every turn. Step 8 measures the real number for your provider instead of estimating it.
```

> **Watch out:** "minimum": 1 and "additionalProperties": False are hints inside the schema and not enforcement, so do not skip the real validation in step 6 just because you wrote them here.

### 4. Ask, and look at what comes back without running anything

Send the question with the tool attached, print the reply, and stop there on purpose. You will usually see that content is None or empty, and that tool_calls holds one entry with an id, the name get_price, and arguments as a string of JSON such as '{"item":"bread","quantity":3}', not a Python dict. Sit with that for a second: your function has not run, no price has been looked up, and nothing in the world has changed. All the model did was write a message saying what it would like you to do, and that message is now ordinary data sitting in a variable. Notice too that the user never typed the word "quantity", the model read "3 loaves" and mapped it onto your schema, which is the actual work it did. The word "usually" in this step is honest, because the model picks its output by probability and not by rule, so an occasional plain text reply is not a bug in your code.

```python
messages = [
    {"role": "system", "content": "You answer shop price questions. Use the tools when a price is involved."},
    {"role": "user", "content": "How much for 3 loaves of bread?"},
]

resp = chat_raw(messages=messages, tools=[GET_PRICE_TOOL], tool_choice="auto")
msg = resp.choices[0].message
print("text reply:", msg.content)
print("tool calls:", msg.tool_calls)
```

- `{"role": "system", "content": "You answer shop price questions. Use the tools when a price is involved."}`: The system message sets the job before the user speaks. It nudges tool use, but it does not force it, the schema descriptions still do most of the steering.
- `"How much for 3 loaves of bread?"`: Deliberately worded in human terms. The model has to map "loaves" onto item: bread and "3" onto quantity: 3, which is the translation step you are here to watch.
- `tools=[GET_PRICE_TOOL], tool_choice="auto"`: tools is a list because you normally attach several. "auto" means the model may answer in plain text or call a tool, and it is the right default for real conversation.
- `msg = resp.choices[0].message`: Pulls out the single assistant message from the reply. You keep the whole object rather than just the text, because you need msg.tool_calls and, in step 5, the message itself.
- `print("tool calls:", msg.tool_calls)`: Prints None if the model chose to answer in text, or a list of call objects if it asked for the tool. This print is your proof that a tool call is data and not an action.

**The maths, spelled out**

```
How the model picks what to write next, and why the lab says "usually".

Formula (softmax):
  P(token i) = exp(z_i) / sum over all j of exp(z_j)

What the symbols mean:
  z_i = the raw score, called a logit, that the model gives to candidate token i
  exp(x) = the number e (about 2.718) raised to the power x, which turns any score into a positive number
  the sum on the bottom = the same thing added up over every token in the vocabulary, so the results add to 1
  P(token i) = the probability of choosing token i next

Worked example with three candidates for the very next token:
  z = 2.0 for the token that starts a tool call, 1.0 for the token that starts plain text, 0.5 for a refusal.
  exp(2.0) = 7.39, exp(1.0) = 2.72, exp(0.5) = 1.65
  total = 7.39 + 2.72 + 1.65 = 11.76
  P(tool call) = 7.39 / 11.76 = 0.63
  P(plain text) = 2.72 / 11.76 = 0.23
  P(refusal)   = 1.65 / 11.76 = 0.14

Intuitively: even a clear favourite is only a favourite. About one run in four in this made up example would answer in text instead of calling the tool, which is exactly why your code must handle <code>tool_calls</code> being <code>None</code>.
```

> **Watch out:** If tool_calls prints None every time, your model chose plain text, so try a larger local model, and if you are on llama.cpp check the server was started with --jinja or tool calling silently does nothing.

### 5. Run the function yourself and send the result back

This is the step where the round trip closes, and it has four moves in a fixed order. First you append the model's own message to the list, so the conversation contains the request before it contains the answer, and most providers reject a tool result that does not follow a message with tool_calls. Then you parse the arguments string into a dict, run your function, and append the result with the role tool and the matching tool_call_id, which is how the model knows which answer belongs to which request when several tools were called at once. The last call uses tool_choice="none" so the model must write a sentence instead of cheerfully calling the same tool a second time. Remember the model has no memory between calls, so if you skip this step it never learns what happened and simply cannot mention the price. On screen you should see the arguments the model asked for, your dict with total: 7.2, and a final sentence containing 7.20.

```python
messages.append(msg.model_dump(exclude_none=True))

for call in msg.tool_calls or []:
    args = json.loads(call.function.arguments)
    print("model asked for:", call.function.name, args)
    result = get_price(**args)          # <-- THIS line is your code choosing to act
    print("your function returned:", result)
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })

final = chat_raw(messages=messages, tools=[GET_PRICE_TOOL], tool_choice="none")
print("final answer:", final.choices[0].message.content)
```

- `messages.append(msg.model_dump(exclude_none=True))`: The reply is an object, not a dict, so model_dump converts it into a plain dict you can put back in the list. exclude_none=True drops empty fields such as content=None, which several providers reject if you send them back.
- `for call in msg.tool_calls or []:`: The model can ask for more than one tool in a single reply, so this is a loop. The or [] turns None into an empty list, so a plain text reply skips the loop instead of crashing.
- `args = json.loads(call.function.arguments)`: arguments arrives as a string of JSON text, not as a dict, because it is streamed back character by character. json.loads parses that string into a real Python dict.
- `result = get_price(**args)`: The one line in the whole file where something actually happens. The ** spreads the dict into keyword arguments, so {"item": "bread", "quantity": 3} becomes get_price(item="bread", quantity=3). Delete this line and the model can ask forever with no effect.
- `{"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}`: The result goes back as a message with the role tool. tool_call_id pairs it with the exact request it answers, and json.dumps turns your dict into text because message content must be a string.
- `final = chat_raw(messages=messages, tools=[GET_PRICE_TOOL], tool_choice="none")`: The second API call, now with the tool result in the history. "none" forbids another tool call, which is what makes the model summarise instead of looping.

**The maths, spelled out**

```
Why a tool call costs more than it looks like it should.

Formula:
  P(k+1) = P(k) + A(k) + T(k)
  total input billed = P(1) + P(2) + ... + P(N)

What the symbols mean:
  P(k) = input tokens sent on API call number k
  A(k) = tokens the model wrote on call k (the tool call itself)
  T(k) = tokens of your tool result, the JSON you appended
  N = how many API calls one user question ended up needing

Worked example for this step, using round numbers:
  P(1) = 150 (system message, user question and the tool definition)
  A(1) = 25 (the tool call the model wrote)
  T(1) = 30 (your JSON result)
  P(2) = 150 + 25 + 30 = 205
  Total input billed for one question = 150 + 205 = 355 tokens, even though the user typed seven words.
  If a task needed 4 rounds each adding about 55 tokens: total = 4 * 150 + 55 * (1 + 2 + 3) = 600 + 330 = 930 tokens.

Intuitively: every round re-sends everything that came before it, so input cost grows roughly with the square of the number of rounds, not in a straight line. That is why the mini-project asks you to cap the loop at four rounds.
```

> **Watch out:** If you forget the messages.append(msg...) line first, most providers return a 400 error saying a tool message must follow a message containing tool_calls, which reads like a nonsense error until you know this.

### 6. Stop trusting the arguments

get_price(**args) is fine until the model invents an argument name, at which point Python raises TypeError: unexpected keyword argument and your program dies mid conversation. Worse, it will happily accept quantity: -5, because nothing in your Python code checks the number, and the schema in step 3 only asked politely. Pydantic is a library where you describe the data you expect as a class, and it checks real data against that description for you. PriceArgs.model_validate_json(raw_args) either hands you a typed object with clean values or raises a ValidationError naming the exact bad field. Replace the old line with result = run_tool(call.function.name, call.function.arguments), and note that run_tool never raises, it always returns a dict, so a bad call becomes a tool result the model can read and correct on its next turn instead of a crash. By default Pydantic ignores extra fields the model made up, and you can add model_config = ConfigDict(extra="forbid") if you would rather reject them loudly. Test it by hand: call run_tool("get_price", '{"item": "bread", "quantity": -5}') and read the error dict that comes back.

```python
from pydantic import BaseModel, Field, ValidationError

class PriceArgs(BaseModel):
    item: str
    quantity: int = Field(default=1, ge=1)

def run_tool(name: str, raw_args: str) -> dict:
    if name != "get_price":
        return {"error": f"no tool called {name}"}
    try:
        args = PriceArgs.model_validate_json(raw_args)
    except ValidationError as e:
        return {"error": "bad arguments", "detail": str(e)}
    return get_price(args.item, args.quantity)
```

- `class PriceArgs(BaseModel):`: Describes the arguments you are willing to accept as a class. It is the same information as the JSON Schema in step 3, but this copy is enforced by your own machine rather than requested from the provider.
- `quantity: int = Field(default=1, ge=1)`: int rejects 2.5 and "three". ge=1 means greater than or equal to 1, so a negative or zero quantity is refused. default=1 matches the Python function default, so an omitted quantity still works.
- `if name != "get_price": return {"error": f"no tool called {name}"}`: A dispatch guard. Models sometimes ask for a tool you never attached, and this returns that fact as data instead of letting a lookup blow up. With three tools this becomes a small lookup table.
- `args = PriceArgs.model_validate_json(raw_args)`: Parses the JSON string and checks it in one move, so you no longer need json.loads here. If the string is not even valid JSON, this raises the same ValidationError you are already catching.
- `except ValidationError as e: return {"error": "bad arguments", "detail": str(e)}`: The error text names the field and the rule that failed, and you send it back to the model as the tool result. Models repair a named error far better than a vague complaint, so this detail is worth the tokens.
- `return get_price(args.item, args.quantity)`: Calls your function with validated values, passed one by one instead of spread with **. Nothing the model invented can reach your function signature any more.

**The maths, spelled out**

```
The constraint you just added, in plain arithmetic.

Rule:
  accept quantity q only if q is a whole number and q >= 1

What the symbols mean:
  q = the quantity the model wrote in its arguments
  >= means greater than or equal to, which is what ge=1 says in Field

Worked example:
  q = 3     accepted. total = 2.40 * 3 = 7.20
  q = -5    rejected by ge=1. Without the check, the old code returned total = 2.40 * -5 = -12.00, a refund the shop never agreed to.
  q = 0     rejected. total would have been 0.00, which looks like a free loaf.
  q = 2.5   rejected by the int type, because half a loaf is not something this shop sells.

Intuitively: the schema in step 3 asks the model nicely, this check enforces the rule on your side. Only one of those two runs on hardware you control, and it is this one.
```

> **Watch out:** Do not forget to actually swap the old line for result = run_tool(call.function.name, call.function.arguments), because defining run_tool without calling it leaves the unsafe path running and everything still looks fine.

### 7. Take control with tool_choice

tool_choice is the one parameter that settles whether using a tool is optional, and this step shows both extremes side by side. Run the two calls and compare them honestly. With "none" the tool is attached but forbidden, so the model answers from memory and usually invents a price like 2.50 or 3.00, which proves the tool was doing real work in step 5 rather than decorating an answer the model already knew. With a named tool forced onto "Tell me a joke", watch it call get_price anyway with an invented item such as "joke" or "bread", because forcing removes the option of saying no, it does not create understanding. That failure mode is the important half of this step, so read the arguments it made up before moving on. Some providers served through an OpenAI compatible endpoint reject a forced choice, so if you get an error here, write it down as a real portability limit rather than a bug in your code.

```python
a = chat_raw(messages=[{"role": "user", "content": "How much for 3 loaves of bread?"}],
             tools=[GET_PRICE_TOOL], tool_choice="none")
print("none   ->", a.choices[0].message.content)

b = chat_raw(messages=[{"role": "user", "content": "Tell me a joke."}],
             tools=[GET_PRICE_TOOL],
             tool_choice={"type": "function", "function": {"name": "get_price"}})
print("forced ->", b.choices[0].message.tool_calls)
```

- `tools=[GET_PRICE_TOOL], tool_choice="none"`: The tool is still attached, so you still pay for its definition, but the model is not allowed to call it. This is the setting you want on a final summarising turn, and here it is a controlled experiment showing what the model knows without help.
- `print("none   ->", a.choices[0].message.content)`: Reads content rather than tool_calls, because with "none" there will not be any tool calls. Compare the number it invents against the real 7.20 from step 5.
- `tool_choice={"type": "function", "function": {"name": "get_price"}}`: Names one specific tool and forces it. This turns tool calling into a dependable data extractor when the tool genuinely fits, and into confident nonsense when it does not.
- `print("forced ->", b.choices[0].message.tool_calls)`: Prints the invented arguments for a question with no item in it. Seeing the model fill in a plausible looking item for a joke request is the whole point of this step.

**The maths, spelled out**

```
What forcing actually does inside the model.

Formula (the same softmax from step 4, with a mask):
  masked score z_i' = z_i if token i is allowed, otherwise negative infinity
  P(token i) = exp(z_i') / sum over all j of exp(z_j')
  and exp(negative infinity) = 0

What the symbols mean:
  z_i = the score the model gave token i before any forcing
  allowed = the set of tokens that are legal given your tool_choice setting
  P(token i) = the probability after the illegal options are removed

Worked example with three candidates:
  z = 2.0 plain text, 1.0 call tool A, 0.5 call tool B
  Free choice: exp values 7.39, 2.72, 1.65, total 11.76, so P = 0.63, 0.23, 0.14
  tool_choice "required": plain text is masked to 0, total = 2.72 + 1.65 = 4.37, so P(A) = 0.62 and P(B) = 0.38
  tool_choice naming tool A: plain text and B are both masked to 0, so P(A) = 2.72 / 2.72 = 1.00

Intuitively: forcing does not persuade the model that a tool fits, it deletes every other option and renormalises what is left. That is exactly why a forced call on "Tell me a joke" still produces confident, invented arguments instead of an admission that the tool does not apply.
```

> **Watch out:** A 400 error on the second call means your endpoint does not support a named tool_choice, so record it as a portability limit and move on rather than rewriting your file.

### 8. See what the definition costs

Same two word question, sent twice, once with the tool attached and once without, so the only difference between the two numbers is the tool definition. The gap in prompt_tokens is what that one definition costs you on every request for the rest of the project, whether the tool is used or not. Expect the plain call to report something like 10 to 15 tokens and the armed call to report roughly 100 to 170, and your exact numbers will differ by provider because each one formats tool definitions its own way. Write the difference down in a comment at the top of your file, then multiply it by fifteen tools and by every turn in a long conversation, and you are looking at the main line item in an agent bill. Prompt caching can cut repeated input to roughly a tenth of the price, but it matches on prefixes, so keep the tool block early and unchanging and push timestamps and session IDs to the end of the prompt. This is also the cheapest habit in the course: measure the token cost of a design choice instead of guessing at it.

```python
plain = chat_raw(messages=[{"role": "user", "content": "say hi"}])
armed = chat_raw(messages=[{"role": "user", "content": "say hi"}],
                 tools=[GET_PRICE_TOOL], tool_choice="auto")
print("without tools:", plain.usage.prompt_tokens)
print("with 1 tool  :", armed.usage.prompt_tokens)
```

- `plain = chat_raw(messages=[{"role": "user", "content": "say hi"}])`: No tools argument at all, which is your baseline. The question is deliberately tiny so the tool definition dominates the difference rather than the words.
- `armed = chat_raw(..., tools=[GET_PRICE_TOOL], tool_choice="auto")`: Identical question, one tool attached. Keeping every other part of the call the same is what makes the subtraction meaningful.
- `plain.usage.prompt_tokens and armed.usage.prompt_tokens`: usage is the provider's own billing count, not an estimate, which is why it is worth reading rather than guessing with the divide-by-4 rule from step 1. Subtract one from the other to get the real cost of your definition.

**The maths, spelled out**

```
Turning the two printed numbers into a monthly bill.

Formula:
  d = armed_prompt_tokens - plain_prompt_tokens
  tokens_per_month = d * number_of_tools * turns_per_conversation * conversations_per_month
  cost = (tokens_per_month / 1,000,000) * price_per_million_input

What the symbols mean:
  d = the extra input tokens one tool definition adds to one request
  turns_per_conversation = how many API calls one user session makes
  price_per_million_input = your provider's input price per million tokens

Worked example (your numbers will differ):
  plain prints 12, armed prints 130, so d = 130 - 12 = 118 tokens.
  15 tools of similar size: 15 * 118 = 1,770 tokens per request. This is slightly high, because some wrapper text is shared across tools rather than repeated.
  12 turns per conversation: 1,770 * 12 = 21,240 input tokens per conversation.
  At $0.50 per million: 21,240 / 1,000,000 * 0.50 = $0.0106 per conversation.
  10,000 conversations a month: about $106 a month, spent before the user said anything useful.
  With prompt caching working well, repeated input costs roughly a tenth, so that becomes about $11, but only while the cached prefix stays byte for byte identical.

Intuitively: tool definitions are a fixed tax charged on every request, and the only ways to lower it are fewer tools, shorter descriptions, or a cache hit.
```

> **Watch out:** If either number prints None, your provider is not reporting usage on that call (this happens with some local servers), which means the delta is unmeasurable there rather than zero.

## You are done when

You are done when you can point at four things on screen. One, the tool call the model asked for, showing name get_price and arguments containing quantity 3. Two, the dict your own function returned, with total 7.2. Three, a final sentence containing 7.20 that the model could only write because you fed the result back, and which it got wrong in step 7 when you forbade the tool. Four, the two prompt_tokens numbers from step 8 and their difference, written in a comment at the top of your file. You can also name the exact line where your code, not the model, chose to act (result = get_price(**args), or result = run_tool(call.function.name, call.function.arguments) after step 6), and say what happens if you delete it.

---

## Mini-project: Three tools, no hints

Give the model three tools and see whether it picks the right one on all five questions. You write two files, tools.py and results.json, and check.py grades both.

- Create my-work/labs/lab09/tools.py with three plain functions and no model involved. get_price(item, quantity=1) returns a dict with a total key, using the PRICES dict from the lab. convert_currency(amount, from_currency, to_currency) returns a dict with a converted key at fixed rates, USD to EUR 0.92 and EUR to USD 1.087. convert_mass(value, from_unit, to_unit) returns a dict with a converted key using 1 kg = 2.20462 lb. Round money and weights to 2 decimal places, and call each function yourself before going on. Nothing in this file may call a model when it is imported.
- In the same file add TOOLS, a list of exactly three definitions in the shape from lab step 3. The name in each definition must match the Python function name and the property names must match the argument names above, or the checker cannot line them up. Make the two converters genuinely similar in purpose, because that is where the model slips and where the descriptions earn their keep.
- Add run_tool(name, raw_args) to tools.py. It validates raw_args with Pydantic and always returns a dict, never raises. An unknown tool name, unparseable JSON, and quantity -5 must all come back as {"error": ...} so the model can read the problem and correct itself.
- Write your five questions in a runner file, my-work/labs/lab09/run.py, with the tool you expect written next to each one before you run anything. One question must need no tool at all (expected null), and one must sit close to the line between the two converters.
- Run the five with tool_choice="auto" and write my-work/labs/lab09/results.json in exactly this shape: {"runs": [{"question": string, "expected_tool": string or null, "chosen_tool": string or null, "arguments": object}, five of them], "tokens": {"no_tools": int, "three_tools": int}}. Use {} for arguments when no tool was called. Take the two token counts from prompt_tokens on "say hi" sent once with the three tools attached and once without.
- Run python check.py in my-work/labs/lab09. When a pick is wrong, edit only the tool description, never the question, then rerun run.py and check.py. Fixing one and breaking another is normal and is the lesson.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab09 && python check.py
```


**You are done when** python check.py prints 17 PASS lines and ends with ALL CHECKS PASSED and exit code 0. A wrong pick prints FAIL with the question that missed, an invented argument name prints FAIL with the question that caused it, and a run_tool that raises instead of returning an error dict prints the exception it threw. The last line reminds you that the wording of your descriptions is not graded, so read that yourself once.

**If you want more:** Add a sixth question that needs two tools in sequence, such as the price of three loaves converted into euros. Feed each result back until the model stops asking, with a hard cap of four rounds so a confused model cannot spin forever. Record it under a new top-level key such as "bonus", because check.py reads only "runs" and "tokens" and ignores anything else.
