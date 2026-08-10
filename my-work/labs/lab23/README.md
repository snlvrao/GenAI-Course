# Lab 23: Serve your model

**Module 23: B5 · Serve it, and plug it in**

my-work/labs/lab23/serve_mine.py is already written and it is 131 lines of standard library Python, no Flask, no FastAPI, no Docker. You are going to read it end to end, start it, point the whole course at it with one line in .env, and then break it on purpose with the Module 12 agent. Keep the file open beside these steps, because every block below is quoted from it.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Read the top: where the model comes from and which port it answers on

Open my-work/labs/lab23/serve_mine.py and read the first thirty lines before you run anything. The imports tell you the whole story: json, os, pathlib, sys, time and http.server, and nothing else, so this file runs on a fresh Python with no installs. The sys.path line adds my-work/labs/_shared so the file can import the TinyGPT class you wrote in B2, which matters because a checkpoint is only numbers and needs the class definition to become a model again. PORT and CKPT are both read from environment variables with defaults, which is what lets a test harness start this server on a different port without editing the file. The default port is 8100 because that is the address llm.py already has for the provider called mine, so the two halves are already wired together and you have nothing to configure. Run nothing yet, just confirm those two defaults with your own eyes.

```python
import json
import os
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))

PORT = int(os.environ.get("MINE_PORT", "8100"))
CKPT = os.environ.get("MINE_CKPT", str(
    pathlib.Path(__file__).resolve().parents[1] / "lab21" / "tinygpt.pt"))
```

- `from http.server import BaseHTTPRequestHandler, HTTPServer`: This is the entire web framework. BaseHTTPRequestHandler parses the request line and the headers for you and then calls a method named after the verb, so a POST arrives at do_POST. Everything a framework would add on top of this is convenience, not capability.
- `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))`: __file__ is this script's path, resolve() makes it absolute, and parents[1] is the labs folder. Adding my-work/labs/_shared to the import path is what lets the next section do `from tinygpt import TinyGPT` with nothing installed and nothing copied.
- `PORT = int(os.environ.get("MINE_PORT", "8100"))`: The port is configuration, not code, so a checker or a second copy can move it without editing the file. 8100 is the default because llm.py already lists http://127.0.0.1:8100/v1 for the provider named mine.
- `CKPT = os.environ.get("MINE_CKPT", str(... "lab21" / "tinygpt.pt"))`: The path is built from this file's own location, not from the folder you happen to be standing in, so the server finds your B3 checkpoint whichever directory you launch it from. Set MINE_CKPT if you trained more than one model and want to serve a different one.

**The maths, spelled out**

```
Why one agreed shape is worth the trouble.

Formula: hand-written glue = P x L. Glue with one agreed shape = P + L.

Symbols: P is how many providers you support, L is how many programs call a model.

Worked example with this course: llm.py holds 12 providers (groq, gemini, huggingface, openrouter, openai, deepseek, together, anthropic, mine, ollama, llamacpp, lmstudio) and there are 23 labs. Per provider glue would be 12 x 23 = 276 pieces of code to write and keep working. One agreed shape means 12 registry entries plus 23 labs that call one function, which is 35. That is 276 / 35 = 7.9 times less code, and adding a thirteenth provider costs one line instead of 23.

Intuitively: your own model gets into the course for the price of one registry entry, because it agreed to speak the same shape as everyone else.
```

> **Watch out:** The note beside the mine entry in llm.py points at an older folder name. The path that matters is the base_url, http://127.0.0.1:8100/v1, so if you change MINE_PORT you must change the registry entry too or llm.py will knock on an empty door.

### 2. Bring your checkpoint back to life

load_model turns a 4.4 MB file of numbers back into something that can predict a character. It checks for the checkpoint first and, if there is none, prints two lines and returns False rather than crashing, which is what lets you prove the HTTP plumbing before you have trained anything. torch is imported inside the function, not at the top of the file, so a learner with no torch installed still gets a working server with the stand-in. The checkpoint carries four things you need: the weights, the character-to-number map, the number-to-character map, and the Config that says how big the model is. Notice that the vocabulary travels with the model, because your model does not know the alphabet, it knows the 65 characters that happened to be in your training text. The last two lines build encode and decode as one-line functions, and the filter inside encode is the quiet detail that will bite you in step seven.

```python
def load_model():
    """Load the checkpoint written by train_gpt.py, if there is one."""
    global _model, _encode, _decode
    if not os.path.exists(CKPT):
        print(f"No checkpoint at {CKPT}. Serving the stand-in model instead.")
        print("Train one with train_gpt.py, then restart this file.")
        return False
    import torch
    from tinygpt import TinyGPT              # the model you wrote in B2

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    stoi, itos = blob["stoi"], blob["itos"]
    _model = TinyGPT(len(stoi), blob["config"])
    _model.load_state_dict(blob["state"])
    _model.eval()
    _encode = lambda s: [stoi[c] for c in s if c in stoi]
    _decode = lambda ids: "".join(itos[i] for i in ids)
    print(f"Loaded {CKPT}: {sum(p.numel() for p in _model.parameters()):,} parameters")
    return True
```

- `if not os.path.exists(CKPT): ... return False`: A missing checkpoint is a normal state for someone starting the Build track, not an error. Returning False leaves _model as None, which the next function reads as a signal to use the stand-in, so the server still starts and still answers correctly shaped JSON.
- `import torch (inside the function)`: Moving the heavy import inside means the file starts in a fraction of a second when there is no checkpoint, and it means someone who has not installed torch yet can still run the server and see the shape. Imports at the top of a file are paid for on every run whether you use them or not.
- `blob = torch.load(CKPT, map_location="cpu", weights_only=False)`: map_location="cpu" loads a file saved anywhere onto your processor. weights_only=False is needed because the checkpoint also contains a Config object, not only numbers, and unpickling an object runs code from the file, so only ever point this at a checkpoint you made.
- `_model = TinyGPT(len(stoi), blob["config"])`: The weights alone are meaningless. You rebuild the exact same architecture from the saved Config, then pour the saved numbers into it with load_state_dict. A mismatch of one layer here raises a size error rather than producing silently wrong output, which is the good outcome.
- `_model.eval()`: This switches layers like dropout out of training behaviour. Your Config has dropout 0.0 so it changes nothing today, and leaving it out would make a dropout-trained model give a different answer to the same prompt on every call.
- `_encode = lambda s: [stoi[c] for c in s if c in stoi]`: The `if c in stoi` silently drops any character your training text never contained. It stops a KeyError from killing the server, and it also means part of the caller's prompt can vanish with no warning anywhere.

**The maths, spelled out**

```
How 824,897 numbers become a 4.4 MB file.

Formula: weight bytes = parameters x 4, because each parameter is a 32 bit float and 32 bits is 4 bytes.

Worked example: 824,897 x 4 = 3,299,588 bytes, which is 3.15 MB. The file on disk is 4,387,939 bytes, about 4.4 MB. The extra 1.1 MB is everything else in the blob: the two character maps, the Config, the loss history, and pickle's own bookkeeping.

Compare that with B4, where the LoRA adapter for a 135M parameter model was about 1.8 MB, because it stored 460,800 trained numbers instead of all 134,975,808.

Intuitively: a model is mostly its weights, and weights are a long list of numbers with a fixed size each. That is why you can predict a checkpoint's size before you save it, and why a surprise in that number means you saved something you did not mean to.
```

> **Watch out:** weights_only=False runs code stored inside the file when it loads. That is fine for a checkpoint your own train_gpt.py wrote, and it is the reason you should never point MINE_CKPT at a .pt file someone sent you.

### 3. The two functions that do the actual work

generate is nine lines and it is the only place your model is used. If there is no checkpoint it returns a fixed sentence, so the plumbing can be tested by anyone. When there is a model, it encodes the prompt, keeps the last block_size characters, calls the generate method you wrote in B2, and slices the new characters off the end. count_tokens is one line, and the docstring in the file explains why: characters are this model's tokens, so len is the honest count for it. Read the slice on the last line of generate carefully and hold the number 128 in your head, because that is where step seven starts. This is also the moment to notice that nothing here looks at temperature, so whatever the caller asks for, your model samples at 1.0.

```python
def generate(prompt: str, max_tokens: int) -> str:
    """Your model's answer to a prompt."""
    if _model is None:
        # The stand-in. Proves the wiring without needing a trained model.
        return ("[stand-in model] I have no weights yet. Train one with "
                "train_gpt.py and restart the server.")
    import torch
    ids = _encode(prompt) or [0]
    out = _model.generate(torch.tensor([ids[-_model.block_size:]]), max_tokens)
    return _decode(out[0].tolist()[len(ids):])


def count_tokens(s: str) -> int:
    """Characters are this model's tokens, so the count is honest for it."""
    return len(s)
```

- `ids = _encode(prompt) or [0]`: An empty prompt encodes to an empty list, and an empty list is falsy, so `or [0]` substitutes a single token to start from. Without it, a client that posts an empty messages list takes your server down with a tensor that has no elements.
- `ids[-_model.block_size:]`: block_size is 128, from the Config in B2. This keeps only the last 128 characters and throws the rest away with no error and no log line. It has to crop, because the position embedding table has exactly 128 rows, but the silence is what makes long prompts fail invisibly.
- `_model.generate(torch.tensor([...]), max_tokens)`: The tensor has an extra pair of brackets because the model expects a batch. generate runs one forward pass per new character and appends each one, so max_tokens of 120 means 120 full passes through all 4 layers, not one.
- `out[0].tolist()[len(ids):]`: generate returns the input tokens followed by the new ones, so this slice is meant to drop the prompt and keep the answer. It uses len(ids), the length before cropping, which is correct while the prompt fits in 128 characters and wrong the moment it does not.
- `return len(s)`: This model has no subword tokenizer. Its vocabulary is 65 single characters, so one character is exactly one token, and any other count you reported would be a number borrowed from a different model.

**The maths, spelled out**

```
Exactly how much answer you get back, measured.

Formula: characters returned = max(0, max_tokens - max(0, encoded_prompt_length - 128))

Symbols: encoded_prompt_length is len(ids) after unknown characters are dropped, 128 is block_size, and max_tokens is what the caller asked for.

Worked example with max_tokens = 120, measured by calling the running server on a laptop processor with no graphics card:
  prompt 100 characters: 120 - 0 = 120 returned.
  prompt 128 characters: 120 - 0 = 120 returned.
  prompt 129 characters: 120 - 1 = 119 returned.
  prompt 200 characters: 120 - 72 = 48 returned.
  prompt 247 characters: 120 - 119 = 1 returned.
  prompt 248 characters: 120 - 120 = 0, an empty string.

Intuitively: the slice on the last line subtracts the uncropped prompt length from a list that only ever holds 128 prompt characters, so every character of prompt past 128 eats one character of your answer. Past 247, there is no answer left.
```

> **Watch out:** That empty string comes back with HTTP 200 and finish_reason "stop", so nothing anywhere reports a problem. If you want long prompts answered with nonsense instead of silence, the fix is one line: change `out[0].tolist()[len(ids):]` to `out[0].tolist()[min(len(ids), _model.block_size):]`. Do not apply it yet. Step seven is worth more if you see the empty strings first.

### 4. The HTTP plumbing, which is smaller than you expect

_send is the only place this file writes a response, which is why the headers are correct everywhere. It turns your dictionary into JSON, encodes it to bytes, sends a status line, sets the content type and the exact byte length, and writes the body. Content-Length must be the number of bytes and not the number of characters, which is why the encode happens before the header is set. do_GET answers one route, /v1/models, and returns a list with a single entry describing the model you trained. Everything else gets a 404 in the same JSON envelope, so a client that pokes at an unknown route gets a message it can parse rather than an HTML error page. Start the server now and open http://127.0.0.1:8100/v1/models in a browser before you go near the interesting route.

```python
class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # The client asks what models exist. Answer with the one you trained.
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [
                {"id": "mine", "object": "model", "owned_by": "you"}]})
        else:
            self._send({"error": {"message": "not found"}}, 404)
```

- `body = json.dumps(obj).encode("utf-8")`: Two steps in one line: the dictionary becomes a JSON string, then the string becomes bytes. HTTP moves bytes, so this conversion has to happen before anything is measured or written.
- `self.send_header("Content-Length", str(len(body)))`: len(body) is the length of the bytes object, so non-ASCII characters are counted correctly. Measuring the string instead would understate the length and the client would sit waiting for bytes that never arrive.
- `self.path.rstrip("/").endswith("/models")`: Clients disagree about trailing slashes, so /v1/models and /v1/models/ both have to work. rstrip removes trailing slashes and endswith accepts any prefix, which is why this one handler answers whether a base_url ends in /v1 or /v1/.
- `self._send({"error": {"message": "not found"}}, 404)`: An unknown route still returns JSON in the shape clients expect an error to take. Sending the default HTML error page here would make the client library raise a parse error instead of showing your message.

**The maths, spelled out**

```
Why Content-Length counts bytes and not characters.

Rule: UTF-8 uses 1 byte for ASCII and 2 to 4 bytes for everything else.

Worked example: the models reply above is 82 bytes and 82 characters, because every character in it is ASCII. Now imagine a model whose training text contained "cafe" written with an accent. That word is 4 characters but 5 bytes, because the accented letter takes 2. Setting Content-Length from the string would promise 4 and send 5, and the client would either truncate the body or wait on a length that no longer matches.

Intuitively: characters are what you read, bytes are what the wire carries, and the header describes the wire. Encoding first and measuring second makes the mistake impossible to make.
```

> **Watch out:** A wrong Content-Length raises nothing in your Python. The failure shows up as a client that hangs, or a JSON parse error in the caller, which sends you hunting for the bug in the wrong file.

### 5. do_POST: the route the whole course goes through

This is the method every lab, every agent and your capstone will hit. It rejects any path that is not /v1/chat/completions, reads exactly Content-Length bytes from the socket, and parses the JSON body. Then comes the honest compromise: it flattens the whole messages list into one string joined with newlines, because your model was trained on plain text and knows nothing about roles or chat templates. A real instruction model would apply its own template here, with special tokens marking where the system message ends and the user message begins, and yours has none of that. Everything after that is the envelope, and it is worth reading field by field, because that dictionary is the entire contract between your model and every other piece of software in this course. Notice that finish_reason is the literal string "stop" whatever happened, and that usage is computed with count_tokens, so your token numbers are character counts and honest for this model only.

```python
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send({"error": {"message": "not found"}}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")

        # Flatten the message list into one prompt. A real chat model would
        # apply a template here; yours was trained on plain text, so this is
        # the honest thing to do.
        prompt = "\n".join(m.get("content", "") for m in req.get("messages", []))
        answer = generate(prompt, int(req.get("max_tokens") or 120))

        self._send({
            "id": "chatcmpl-mine",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mine"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": count_tokens(prompt),
                "completion_tokens": count_tokens(answer),
                "total_tokens": count_tokens(prompt) + count_tokens(answer),
            },
        })
```

- `n = int(self.headers.get("Content-Length", 0)) then self.rfile.read(n)`: You must read exactly the number of bytes the client promised. Reading fewer leaves the rest in the socket and corrupts the next request, and reading without a limit blocks forever waiting for an end that never comes.
- `json.loads(self.rfile.read(n) or b"{}")`: An empty body reads as b"", which is falsy, so `or b"{}"` gives json.loads something valid to parse. Without it, a client that posts nothing kills your server with a JSONDecodeError.
- `prompt = "\n".join(m.get("content", "") for m in req.get("messages", []))`: System, user and assistant turns are all concatenated with newlines and the role labels are thrown away. Your model has no tokens for roles, so inventing a template would only add characters it cannot read, and those characters compete for the 128 it can see.
- `int(req.get("max_tokens") or 120)`: `or 120` catches both a missing key and an explicit null, so a caller that omits max_tokens gets 120 characters back. It is a cap on work, and since your model never emits an end-of-text token, it is also the only thing that stops generation.
- `"finish_reason": "stop"`: This is hardcoded and therefore always wrong. The model always ran until max_tokens, so the true value would be "length" on every call. A caller that branches on finish_reason to detect truncation will believe every reply is complete.
- `"model": req.get("model", "mine")`: The reply echoes whatever model name the caller sent, which is what real providers do. It means llm.py setting LLM_MODEL to anything at all still works, because there is only one model here regardless of the label.

**The maths, spelled out**

```
Reading the usage block on a real call.

Formula: total_tokens = prompt_tokens + completion_tokens, where one token is one character.

Worked example, measured against the running server with the prompt "ROMEO:" and max_tokens 120:
  {"prompt_tokens": 6, "completion_tokens": 120, "total_tokens": 126}
"ROMEO:" is 6 characters, so prompt_tokens is 6. completion_tokens is exactly 120, not 119 or 121, because the model has no way to stop early and always runs to the cap.

The envelope around the text: a reply carrying 86 characters of generated text serialises to 356 bytes of JSON, so 270 bytes are wrapper. On a 24 character reply the wrapper is over 90 percent of the traffic, and on a 2,000 character reply it is under 12 percent.

Intuitively: completion_tokens equal to max_tokens on every single call is your signal that finish_reason is not telling the truth. On a hosted model, a completion that stops short of the cap is the normal case.
```

> **Watch out:** This method reads only messages, max_tokens and model. temperature, top_p, stop, response_format, tools and stream are all accepted and silently ignored, so a lab that sets temperature=0 for repeatable runs still gets sampling at 1.0, and a lab that asks for a stream gets one whole JSON body instead.

### 6. Start it, then point the course at it

The last five lines load the model, print where it is listening, and hand control to serve_forever, which blocks until you press Ctrl+C. You need two terminals from here on, one running the server and one running labs. Put LLM_PROVIDER=mine in your .env at the top of the course folder, then run python llm.py from my-work/labs/_shared, which prints what it is about to use and then makes one real call. If whoami says endpoint=your own machine and key=none needed, the registry entry and your server agree. On a laptop processor with no graphics card the server answered its first request 1.6 seconds after launch, and a 120 character reply came back in under a second.

```python
if __name__ == "__main__":
    load_model()
    print(f"\nYour model is serving on http://127.0.0.1:{PORT}/v1")
    print("Put LLM_PROVIDER=mine in your .env, then run any lab.\n")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

# --- terminal 1 ---
# cd my-work/labs/lab23
# python serve_mine.py
#   Loaded ...tinygpt.pt: 824,897 parameters
#   Your model is serving on http://127.0.0.1:8100/v1

# --- .env, at the top of the course folder ---
# LLM_PROVIDER=mine

# --- terminal 2 ---
# cd my-work/labs/_shared
# python llm.py
#   provider=mine  model=mine  endpoint=your own machine  key=none needed
#   reply: How hear-betime. What all have doth waith more and good
#          she with save you contemply for our wife?
#   Working. You are ready to start.
```

- `HTTPServer(("127.0.0.1", PORT), Handler)`: Binding to 127.0.0.1 means only your own machine can reach it. Binding to 0.0.0.0 would put an unauthenticated model server on your whole network, so leave this alone.
- `.serve_forever()`: This blocks the terminal and handles one request at a time. Two labs calling at once will queue rather than fail, which is fine here and is the first thing you would replace in anything real.
- `load_model() before the server starts`: The 4.4 MB checkpoint is loaded once at startup rather than per request. That is why the first call is fast, and why you must restart this file after you train a new checkpoint.
- `LLM_PROVIDER=mine`: This one line redirects every lab in the course. No lab file mentions your model, because the choice lives in configuration and llm.py reads it when it is imported.
- `python llm.py`: Running the shared helper directly prints whoami and then makes one real call. It is the fastest way to tell a wrong registry entry (wrong port) from a stopped server (connection refused) from a broken response shape (a parse error).

**The maths, spelled out**

```
What that reply costs, and what it would cost from a provider.

Measured on a laptop processor with no graphics card: 1.6 seconds from launch to the first answered request, which is the torch import plus loading 4,387,939 bytes. Then about 0.4 seconds for 120 characters, which is 120 forward passes through 4 layers, so roughly 300 characters per second.

Cost comparison: those 126 tokens are characters. The same text at a hosted provider's 4 bytes per token would be about 32 tokens, and at 0.50 dollars per million input tokens that is 32 x 0.50 / 1,000,000 = 0.000016 dollars. Your model costs 0 dollars and about 0.4 seconds of your own processor.

Intuitively: local models trade money for latency and quality. Free is a real advantage for a test loop that runs hundreds of times, and it buys you nothing if the answers cannot be used.
```

> **Watch out:** If an old window is already holding port 8100 you get an address-in-use error at startup. Set MINE_PORT to something else and update the mine entry in llm.py to match, because the client will not discover the new port on its own.

### 7. Now break it with the Module 12 agent

Leave the server running, keep LLM_PROVIDER=mine, and run your Module 12 agent from my-work/labs/lab12/react_agent.py against it. What you will see is six steps, zero parsed tool calls, and the step limit message, because every reply comes back as an empty string. Before you conclude that the model is simply bad, run the script below, which measures how much of the agent's prompt the model can physically read. The agent's system prompt plus its question is 786 characters, 17 of them are characters your training text never contained (the brackets, the star, the underscore and the digits 0, 2 and 7) so they vanish in silence, and only the last 128 of the remaining 769 reach the model. That is 16.6 percent, and the surviving part does not contain a single tool name or the word Thought. Now apply the one-line fix from step three and run it again: you get 120 characters of Shakespeare-flavoured nonsense instead of an empty string, and the agent still fails, which is the point.

```python
# my-work/labs/lab23/window.py - how much of an agent prompt your model can see
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "lab12"))
from react_agent import SYSTEM

blob = torch.load("../lab21/tinygpt.pt", map_location="cpu", weights_only=False)
stoi, n = blob["stoi"], blob["config"].block_size
q = "What is the total value of the monitors we currently hold in stock?"
prompt = SYSTEM + "\n" + q
kept = [c for c in prompt if c in stoi]

print(f"prompt          {len(prompt)} characters")
print(f"after encoding  {len(kept)} (dropped {len(prompt) - len(kept)}: "
      f"{sorted(set(c for c in prompt if c not in stoi))})")
print(f"the model sees  the last {n}, which is {n / len(kept) * 100:.1f}% of it")
print("---- all it actually reads ----")
print("".join(kept[-n:]))

# Measured output:
# prompt          786 characters
# after encoding  769 (dropped 17: ['(', ')', '*', '0', '2', '7', '_'])
# the model sees  the last 128, which is 16.6% of it
# ---- all it actually reads ----
# nswer, reply with one line only:
#
# Final Answer: your answer
#
# What is the total value of the monitors we currently hold in stock?
```

- `from react_agent import SYSTEM`: You are measuring the real prompt from Module 12, not an imitation of it. If you have not built that file yet, paste any 700 character system prompt in its place and the arithmetic comes out the same.
- `kept = [c for c in prompt if c in stoi]`: This is the same filter as _encode in serve_mine.py. It shows which characters disappear before the model ever runs, and the list is a shock: the training text contains only one digit, the character 3, so the agent's prices 900.0 and 220.0 cannot even be spelled.
- `n / len(kept) * 100`: 128 out of 769 is 16.6 percent. The other 83.4 percent, which is the whole tool list and the whole reply format, is discarded before the first prediction, with no error and no warning.
- `"".join(kept[-n:])`: Printing the surviving window is the moment the lesson lands. The model is being asked to follow a format it was never shown, using tools it was never told exist, and it is being judged on the result.

**The maths, spelled out**

```
The failure, in one calculation.

Formula: fraction of the prompt seen = block_size / encoded_prompt_length

Worked example: 128 / 769 = 0.166, so 16.6 percent. Turning it round, showing this model the whole prompt needs a context of 769 characters, which is 6.0 times what it has. The position embedding table has exactly block_size rows, so that means retraining a bigger model from scratch, not editing a number in a config.

It gets worse per turn, not better. Module 12's growth formula says turn k sends S + (k - 1) x d characters. With S = 786 and d = 180 characters per completed turn, turn 6 sends 786 + 5 x 180 = 1,686 characters, of which the model reads 128, or 7.6 percent.

Intuitively: the agent labs do not fail because the model is stupid, they fail because 83 percent of the instructions never arrive. A model is not only its weights, it is its context window, and yours is 128 characters wide.
```

> **Watch out:** The tempting fix is to raise block_size in Config and restart. It will not load: the checkpoint's position embedding table has exactly 128 rows and load_state_dict refuses a mismatch. A larger window is a new model and a new training run, which is the honest price of the thing you just measured.

## You are done when

Your server prints the parameter count at startup, http://127.0.0.1:8100/v1/models returns a data list in a browser, python llm.py with LLM_PROVIDER=mine prints endpoint=your own machine followed by a line of your model's own nonsense, and you can state from your own measurements what fraction of the Module 12 agent prompt your model actually reads.

---

## Mini-project: Run the course on it

Serve your model, drive it through the shared client the whole course uses, then point the Module 12 agent at it and write down exactly what happened. The result you are recording is a failure, and recording a failure precisely is the skill being taught. The checker starts your server itself on a free port and makes a real HTTP request to it, so nothing in this report can be faked.

- Start the server in one terminal and confirm both routes from another. GET http://127.0.0.1:8100/v1/models should return a data list with one entry. POST to http://127.0.0.1:8100/v1/chat/completions with {"model": "mine", "messages": [{"role": "user", "content": "ROMEO:"}], "max_tokens": 120} and keep the whole reply, including the usage block.
- Put LLM_PROVIDER=mine in your .env, then run python llm.py from my-work/labs/_shared. Copy the whoami line and the reply text exactly as printed, crude parts included. If it fails, read the error before changing anything: connection refused means the server is not running, and a parse error means your response shape is wrong.
- Run your Module 12 agent (my-work/labs/lab12/react_agent.py) against it with the monitor question and count three things: how many steps it used, how many replies parsed as a real Action, and what the final line said. Then run window.py from step seven of the lab and note what fraction of the prompt the model could read.
- Write my-work/labs/lab23/serve_report.json in exactly this shape: {"chat": {"prompt": "ROMEO:", "content": "<at least 20 characters your model wrote>", "prompt_tokens": 6, "completion_tokens": 120, "total_tokens": 126}, "via_llm_py": {"provider": "mine", "whoami": "<the line llm.py printed>", "reply": "<what came back>"}, "agent": {"question": "...", "steps_used": 6, "tool_calls_parsed": 0, "final_answer": "...", "outcome": "fail", "what_happened": "<one sentence of at least 40 characters>"}, "verdict": {"works_as_endpoint": true, "passes_agent_lab": false, "honest_use": "<at least 60 characters saying where a model this size belongs>"}}
- Save check.py beside serve_mine.py in my-work/labs/lab23 and run it from that folder. It starts your server on a free port of its own, so you can leave your copy running on 8100 while it works.

### Check it

`check.py` is in this folder. Run it:

```bash
cd my-work/labs/lab23 then python check.py
```


**You are done when** check.py prints one PASS or FAIL line per check and ends with ALL CHECKS PASSED and exit code 0. Passing means your server really started, really answered a real HTTP request, and returned choices, a message with role and content, an object of chat.completion, and three token counts that add up. It also means your report records the agent failure with real numbers rather than a shrug.

**If you want more:** Change one line in .env to a hosted provider and run the same agent with the same question, then add a second agent block to your report and compare steps_used and tool_calls_parsed side by side. Nothing in react_agent.py changes, which is the whole point of the module: you swapped the brain and kept the body.
