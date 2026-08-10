# Lab 01: Set up your AI workspace

**Module 1: What machine learning you still need**

You are going to open the folder you will live in for the next seventeen modules, then make a language model answer one question in two different ways from a single small script. The first way is the short one you will use most days, and the second way shows you what the short one hides, including which model really answered and how many tokens it cost. Nothing here costs money if you take the local option, because you can run a model on your own machine with no account and no key. Before you start, make sure python llm.py works, which is covered in setup.html.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Activate the environment and open the lab folder

The setup page already built your environment and the course already ships the folders, so this step is about finding them and switching the environment on. A virtual environment (venv) is a private folder of Python packages that belongs only to this project, so anything you install here cannot break Python for other work on your machine. There is one for the whole course, at my-work/.venv, and every lab shares it. Activating it is the step people forget, and you can tell it worked because your terminal prompt gains a (.venv) label at the front. Keep that terminal open, because activation only lasts for the window you ran it in. Your own work lives under my-work, and the helper every lab imports is at my-work/labs/_shared/llm.py. This course was checked on Python 3.14 on Windows, and anything from 3.11 upwards is fine.

```python
cd path\to\GenAI-Course
.\my-work\.venv\Scripts\Activate.ps1
cd my-work\labs\lab01

# macOS or Linux, instead of the two lines above:
#   source my-work/.venv/bin/activate
#   cd my-work/labs/lab01
```

- `.\my-work\.venv\Scripts\Activate.ps1`: Activation edits the PATH for this terminal window only, so that python and pip now mean the copies inside .venv. PowerShell needs the leading .\ because it will not run a script from the current folder without it. Close the window and the effect is gone, which is why you re-run this line at the start of every session.
- `cd my-work\labs\lab01`: This folder already exists and already holds hello_llm.py, so you are moving into it rather than creating it. Everything you write in this course lives under my-work, which is the only folder you ever need to edit. Every command in this lab assumes you are standing here, because the script reaches the shared helper by going one folder up to _shared.
- `source my-work/.venv/bin/activate`: The macOS and Linux equivalent. You need the word source rather than just the path, because the script changes variables in your current shell, and running it the normal way would change them in a child shell that exits immediately.

> **Watch out:** The usual trap is opening a fresh terminal a day later and forgetting the activate line, which shows up as a missing package error for something you know you installed. On Windows the activate line is refused the first time, with "running scripts is disabled on this system". That is the default on a fresh install, not something you broke. Run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned once, which needs no administrator rights, then run the activate line again. If you would rather not change that setting, skip activation entirely and call the environment by its path instead: .\my-work\.venv\Scripts\python.exe hello_llm.py.

### 2. Install the two packages you need

Now you install the only two packages this first lab needs. openai is a client library, meaning code that formats your request, sends it over HTTP, and turns the reply back into a Python object you can read. You use it for hosted models and for models running on your own laptop, because almost every provider now accepts the same request format that OpenAI published. python-dotenv reads settings out of a plain text file called .env, so your keys live outside your code and never get committed by accident. You pin openai to one exact version so a future release cannot quietly change how a call behaves halfway through the course. When pip finishes you should see a line beginning Successfully installed naming openai-2.53.0 and python-dotenv, with no red error text underneath.

```python
# save this as requirements.txt in my-work
openai==2.53.0
python-dotenv

# then, with (.venv) active:
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- `openai==2.53.0`: The double equals pins one exact version. Without it, pip fetches whatever is newest on the day you run it, and a later release could change behaviour that the rest of this course depends on.
- `python-dotenv`: Left unpinned on purpose, because it is small and its job (reading a text file into environment variables) does not change. It provides load_dotenv(), which llm.py calls for you so you never have to.
- `python -m pip install --upgrade pip`: Written as python -m pip rather than plain pip so it definitely uses the Python you just activated, not some other pip earlier on your PATH. Upgrading pip first avoids old-pip errors when installing recently published packages.
- `pip install -r requirements.txt`: -r means read the package list from this file. Keeping the list in a file rather than typing names by hand means you can rebuild this exact environment on another machine, or after you delete .venv and start again.

> **Watch out:** If pip reports success but Python still says the package is missing, the (.venv) label was absent from your prompt when you ran pip, so it installed into your system Python instead.

### 3. Get a model you can actually call

A script that cannot reach a model does nothing, so you need at least one model to talk to. The free path is Ollama, a small program that downloads a model file onto your disk and then serves it from your own machine, with no account and no bill. Pick a model with about 3 billion parameters or more, where parameters are the numbers inside the model that were set during training. Below roughly 2 billion, the tool calling you need in later modules starts failing in ways that look like your code is broken when it is not. The pull command downloads a few gigabytes once, and after that the model answers with your internet switched off. ollama serve starts a small web server on your machine at port 11434, which is exactly the address the course helper already points at. If you already own a paid API key you can set that up as a second option, but do not buy one just for this course.

```python
# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b

# start the server if it is not already running
# (the Windows installer usually starts it for you)
ollama serve

# quick sanity check, in another terminal
ollama run llama3.2:3b "say hello in five words"
```

- `ollama pull llama3.2:3b`: pull downloads the model file once and stores it on your disk. The part after the colon is the size tag, so 3b means the 3 billion parameter build, and a different tag downloads a completely different file.
- `ollama serve`: Starts a small web server listening on port 11434. Nothing can call the model until this is running, and llm.py already points the ollama provider at http://localhost:11434/v1/, so you never type that address yourself.
- `ollama run llama3.2:3b "say hello in five words"`: A one-off check that the download works, done before you write any Python. If this prints words, then any later error is in your code or your .env, which cuts your debugging in half.

**The maths, spelled out**

```
Formula: file size in bytes = number of parameters x bytes used per parameter

number of parameters: how many numbers sit inside the model. llama3.2:3b has about 3 billion, written 3,000,000,000.
bytes used per parameter: how much space each of those numbers takes. Full precision uses 4 bytes each. Ollama ships models quantised, which means each number is squashed into fewer bits, commonly about 4 bits, which is 0.5 bytes.

Worked example:
Full precision: 3,000,000,000 x 4 bytes = 12,000,000,000 bytes, so about 12 GB.
Quantised to 4 bits: 3,000,000,000 x 0.5 bytes = 1,500,000,000 bytes, so about 1.5 GB. Add roughly 0.5 GB of working space and you need about 2 GB of free memory to run it.

What it means: the b in 3b is the parameter count, and it tells you both how much disk the download eats and roughly how much memory the model needs while it answers. Quantisation is what makes a 12 GB model fit in 2 GB, and it does cost a little accuracy, which is part of why very small models get tool calling wrong.
```

> **Watch out:** If ollama serve says the address is already in use, the installer already started it for you, so leave it alone and move on rather than trying to kill it.

### 4. Write your .env and your .gitignore

The .env file is a plain text file of name=value lines holding everything that differs between machines, including secrets. The line that matters most is LLM_PROVIDER, because that single word is the switch you flip for the rest of the course, and it must match one of the names the shared helper knows (ollama, groq, gemini, openai and a few more). The .gitignore file is a list of things git must never upload, and .env belongs on it. Leaked API keys get scraped off public repositories within minutes by automated bots, and the bill lands on you, so write this file now rather than later. Both files go in my-work, beside the environment you made. The helper finds your .env by looking in the folder you are standing in and then working upwards, so anywhere inside my-work can see it. Both names start with a dot, which means Windows Explorer and macOS Finder may hide them from you until you turn on the setting that shows hidden files.

```python
# my-work/.env
LLM_PROVIDER=ollama

# --- local, free, no key needed ---

# --- hosted, optional: leave blank if you have no key ---
OPENAI_API_KEY=
# copy the exact model id from your provider's model list page


# .gitignore
my-work/.venv/
my-work/.env
__pycache__/
*.db
```

- `LLM_PROVIDER=ollama`: This value is looked up against the provider list inside llm.py, so it has to be spelled exactly as one of the known names. Change this one word and every lab in the course talks to a different model, with no code edit anywhere.
- `OPENAI_API_KEY=`: Left empty on purpose. llm.py only reads a key for hosted providers, and when it is missing it raises a clear MissingKey error telling you what to add, instead of failing later with a confusing network message.
- `.env  (the line inside .gitignore)`: This is the line that stops your keys reaching a public repository. Add it before your first commit, because git keeps history, and deleting a key later does not remove it from the commits that already contain it.
- `*.db`: Later modules create SQLite database files for the vector store, and the star matches any filename ending in .db. Those files are rebuildable data rather than source code, so they should never be committed.

> **Watch out:** On Windows, Notepad silently saves the file as .env.txt, and a file with that name is never read, so nothing you put in it takes effect.

### 5. Point Python at the shared helper

Every lab in this course imports one file, my-work/labs/_shared/llm.py, which hides the differences between providers behind four functions: chat() sends text and returns text, chat_raw() returns the provider's full reply object, client_for() hands you a configured client plus a model name, and whoami() reports what you are about to use. Copy that file into my-work/labs/_shared/ before you run anything. Python only imports from folders on its import path, which is the list of directories it searches when you write an import, and my-work/labs/_shared is not on that list by default. The two lines below add it while your script is running, so a file sitting in my-work/labs/lab01/ can find it no matter which directory you launched from. Without them you get ModuleNotFoundError: No module named 'llm', however correct the rest of your code is. Run just this much first, because if whoami() prints your provider, model and endpoint, your setup is already correct and everything after this is easy.

```python
# my-work/labs/lab01/hello_llm.py
import sys, pathlib

# make my-work/labs/_shared importable from anywhere
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))

from llm import whoami, chat, chat_raw

print("Active setup:", whoami())
```

- `import sys, pathlib`: sys gives you sys.path, the list of folders Python searches on every import. pathlib builds file paths in a way that behaves the same on Windows and macOS, so you never hardcode a backslash that then breaks on someone else's machine.
- `pathlib.Path(__file__).resolve()`: __file__ is the path of the script that is currently running, and it can be a relative path depending on how you launched it. .resolve() turns it into a full absolute path, which is what makes the next part reliable from any working directory.
- `.parents[1] / "_shared"`: parents[0] is the folder holding the file (my-work/labs/lab01), and parents[1] is one level above it (labs). The forward slash is pathlib's way of joining path pieces, so the result is my-work/labs/_shared on every operating system.
- `sys.path.append(str(...))`: This adds that folder to Python's search list while the program runs, which is why the very next line can import llm even though llm.py is not sitting beside your script. str() is needed because sys.path stores plain strings, not Path objects.
- `from llm import whoami, chat, chat_raw`: Pulls three of the four helper functions in by name. You never edit llm.py to change provider, you edit .env, and keeping that decision in one file is the entire reason the shared folder exists.
- `print("Active setup:", whoami())`: whoami() reports the provider, model, endpoint and whether a key was found, and it does this without making any network call. Running only this much proves your configuration before you spend time chasing a failing model call.

> **Watch out:** If you still see ModuleNotFoundError: No module named 'llm', check that llm.py is inside my-work/labs/_shared and not saved one folder too high in my-work/labs/.

### 6. Call the model the easy way

chat() is text in, text out, and it is what you will use nine times out of ten in this course. It builds the message list for you, sends it to whichever provider your .env names, waits for the reply, and hands back a plain Python string with surrounding whitespace stripped off. Underneath it also sets two defaults worth knowing: temperature 0.2, which controls how adventurous the word choice is, and max_tokens 800, which caps how long the reply may get. Add these lines to the same file you started in step 5. You should see one sentence of English appear after a short pause, and the exact wording will differ slightly each time you run it, which is normal and is explained in the maths below. If the answer stops mid-word, the model hit the max_tokens cap rather than finishing its thought.

```python
question = "In one sentence, what is a Python virtual environment?"

answer = chat(question)
print("\n--- chat() ---")
print(answer)
```

- `question = "In one sentence, what is a Python virtual environment?"`: Stored in a variable because step 7 sends the exact same question a second way. Using one variable for both is what makes the two outputs fair to compare.
- `answer = chat(question)`: One string in, one string out. Inside, chat() wraps your text as a user message, calls the provider named in .env, then returns resp.choices[0].message.content with the whitespace trimmed, so you never touch the response object.
- `print("\n--- chat() ---")`: The \n prints a blank line before the header, so the two sections of output do not run into each other. It is a small thing, but you will be reading this output many times.

**The maths, spelled out**

```
Formula: probability of token i = exp(score_i / T) divided by the sum over all candidates j of exp(score_j / T)

score_i: the raw number (called a logit) the model produces for one candidate next token. Higher means the model likes it more.
T: the temperature you pass in. chat() uses 0.2 by default.
exp(x): the number e (about 2.718) raised to the power x. It turns any score, including a negative one, into a positive number.
The division at the end makes all the probabilities add up to 1.

Worked example, using three candidate words with scores 2.0, 1.0 and 0.5.

At T = 1.0:
exp(2.0) = 7.39, exp(1.0) = 2.72, exp(0.5) = 1.65
total = 11.76
probabilities = 7.39/11.76 = 0.63, 2.72/11.76 = 0.23, 1.65/11.76 = 0.14

At T = 0.2, the course default:
2.0/0.2 = 10, 1.0/0.2 = 5, 0.5/0.2 = 2.5
exp(10) = 22026, exp(5) = 148, exp(2.5) = 12
total = 22186
probabilities = 22026/22186 = 0.993, 148/22186 = 0.0067, 12/22186 = 0.0005

What it means: dividing by a small temperature stretches the gaps between scores before they are compared, so the model's favourite word jumps from a 63 percent chance to a 99 percent chance. Low temperature therefore gives you answers that repeat well, which is what you want while learning. This is simplified: a real model scores tens of thousands of candidate tokens at each step rather than three, and it redoes the whole calculation for every single word it writes, which is why the wording still wobbles a little between runs.
```

> **Watch out:** An empty line of output does not mean the call failed, because chat() returns an empty string when the model produced only whitespace or ran straight into the max_tokens cap.

### 7. Call it the honest way

chat_raw() hands you the provider's full reply object instead of only the words, so you can see what chat() was quietly throwing away. You want it whenever you need the model id that actually answered, the reason the model stopped, or the token counts, and token counts are how you estimate cost in a later module. It also shows you the real shape of a chat request, which is not a string but an ordered list of messages, each one a small dictionary with a role and some content. The attribute names below (choices, usage) are the OpenAI-compatible shape, which is also what Ollama serves, and that is exactly why one script can work against both. Some providers send no usage block at all, so you check for it rather than assuming it is there. Expect the printed model name to be the full id, something like llama3.2:3b, which is often not the short name you had in your head.

```python
messages = [
    {"role": "system", "content": "Answer in one short sentence."},
    {"role": "user", "content": question},
]

raw = chat_raw(messages)
print("\n--- chat_raw() ---")
print("model that answered:", raw.model)
print("text:", raw.choices[0].message.content)

usage = getattr(raw, "usage", None)
if usage:
    print("input tokens: ", usage.prompt_tokens)
    print("output tokens:", usage.completion_tokens)
else:
    print("this provider reported no token counts")
```

- `the messages list`: This is the true shape of a chat request. The model does not receive a string, it receives an ordered list of messages and reads them top to bottom as the conversation so far, which is the structure every later module builds on.
- `{"role": "system", "content": "Answer in one short sentence."}`: The system message carries instructions about how to behave, kept separate from the user's actual question. Placing it first in its own role is what gives it more weight than the same words pasted into the end of the question.
- `raw = chat_raw(messages)`: Returns the provider's whole response object rather than just the text, so nothing is discarded. This is also the function the later tool-calling labs use, because tools are passed alongside the messages.
- `raw.choices[0].message.content`: choices is a list because the API can be asked to generate several alternative answers in one call. You asked for one, so you take index 0, and .message.content is the actual text sitting inside that choice.
- `usage = getattr(raw, "usage", None)`: getattr with a third argument returns that default instead of raising AttributeError when the field is absent. This is how you support providers that report no token counts without your script crashing on them.
- `if usage:`: Only print the numbers when the provider actually sent them. This one guard is the difference between a script that works on your provider and a script that works on all of them.

**The maths, spelled out**

```
Formula 1, rough token count: tokens = characters divided by 4, for ordinary English text.
Formula 2, cost of one call: cost = (input_tokens / 1,000,000) x input_price + (output_tokens / 1,000,000) x output_price

input_tokens: what usage.prompt_tokens reports, meaning everything you sent (the system message, your question, and the hidden role markers the chat template wraps around them).
output_tokens: what usage.completion_tokens reports, meaning only the words the model wrote back.
input_price and output_price: what a provider charges per million tokens. Output usually costs more than input, because generating text takes real compute while reading it does not.

Worked example.
"In one sentence, what is a Python virtual environment?" is 54 characters, so 54 / 4 = about 13 tokens.
"Answer in one short sentence." is 29 characters, so 29 / 4 = about 7 tokens.
The chat template adds role markers around each message, roughly 8 more tokens.
Total input is about 28 tokens, which is why the printed number is bigger than your question on its own.
Say the reply comes back at 25 tokens, and imagine a provider charging 0.50 dollars per million input tokens and 1.50 dollars per million output tokens:
input cost = (28 / 1,000,000) x 0.50 = 0.000014 dollars
output cost = (25 / 1,000,000) x 1.50 = 0.0000375 dollars
one call = 0.0000515 dollars, so 100,000 calls = about 5.15 dollars.

What it means: a token is roughly three quarters of a word, one call costs a fraction of a penny, and cost only becomes real once you multiply by traffic. Those two prices are invented for the arithmetic, so check your own provider's page for real ones. On Ollama the money is always zero, but the token numbers still matter, because the same counts are what fill up the model's limited input window.
```

> **Watch out:** An AttributeError on raw.choices almost always means you passed the result of chat() by mistake, because that is a plain string and only chat_raw() returns the full object.

### 8. Run it and read what comes back

Run the file from inside my-work/labs/lab01 with the virtual environment still active, so the python you invoke is the one inside .venv. Expect the local model to pause for several seconds on the very first call while the file is read off disk into memory, then to be much quicker on every call afterwards. You should see four things printed: the whoami line, a sentence from chat(), the same question answered through chat_raw() with the real model id, and two token numbers. If you see ModuleNotFoundError: llm, your _shared path is wrong or llm.py is not in it. If you see a connection refused error, Ollama is not running, so start it and try again. If the token counts look larger than the question you typed, that is expected, because the system message and the hidden chat formatting are counted too.

```python
python hello_llm.py

# macOS or Linux:
#   python my-work/labs/lab01/hello_llm.py
```

- `python hello_llm.py`: Run this from inside my-work/labs/lab01. llm.py finds your .env by searching from the folder you are standing in and working upwards, so launching from somewhere outside the project means your settings are never loaded and the built-in defaults take over silently.
- `# macOS or Linux:  python my-work/labs/lab01/hello_llm.py`: The same command with forward slashes. The only difference between the two lines is the path separator your shell needs, because the Python code itself uses pathlib and does not care which one you have.

**The maths, spelled out**

```
Formula 1, first-call delay: seconds = model file size divided by disk read speed
Formula 2, answer time: seconds = tokens generated divided by tokens per second

Worked example.
The quantised llama3.2:3b file is about 2 GB. A normal SSD reads at roughly 0.5 GB per second, so 2 / 0.5 = about 4 seconds before the first word appears. After that the file is already sitting in memory, so the same delay on the next call drops to well under a second.
If your machine generates about 25 tokens per second on CPU and the answer is 30 tokens long, then 30 / 25 = 1.2 seconds of writing.
First call total: about 4 + 1.2 = 5.2 seconds. Second call: about 1.2 seconds.

What it means: that long first pause is disk work, not the model thinking, so never judge a model's speed from your first run. These numbers are rough and your own disk and processor will shift them, but the shape (one slow call, then fast ones) is what you should expect to see.
```

> **Watch out:** The folder name has to match everywhere, so if you created my-work\labs\m01 but run my-work\labs\lab01, Python reports that it cannot open the file, which is a path typo rather than anything wrong with your code.

## You are done when

You are done when one command runs one file and prints four things: your active provider and model from whoami(), a sentence from chat(), the same question answered through chat_raw() showing the real model id, and an input and output token count. Your .env file exists in my-work, LLM_PROVIDER names a provider the helper knows, and .env is listed inside .gitignore. If you use git, run git status and check that .env does not appear anywhere in the output.

---

## Mini-project: Two providers, one script

Prove mechanically that the provider is a setting and not a code change. Your script writes my-work/labs/lab01/runs.json, and check.py verifies that two different models answered while the script's own SHA-256 hash stayed identical.

- Add a second provider you can reach for free to your .env. A second local model is enough (`ollama pull llama3.2:1b`, or any other you already have), or a hosted key you already own. Buy nothing.
- Write my-work/labs/lab01/compare.py. It reads the provider from whoami(), sends one fixed question through chat_raw(), times the call with time.perf_counter(), and hashes its own source with hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest().
- Have it append to my-work/labs/lab01/runs.json, creating the file when missing. Exact shape: {"question": "<the question you asked>", "runs": [ ... ]}, where each run is {"provider": str, "model": str, "seconds": float, "answer": str, "input_tokens": int or null, "output_tokens": int or null, "script_sha256": str}.
- Run it once. Then edit only the LLM_PROVIDER line in .env, change no code anywhere, and run it again. runs.json now holds two entries.
- Save check.py beside compare.py and run `python my-work/labs/lab01/check.py`. Fix whatever it names until every line reads PASS.
- Read the two answers yourself and write down which one is wrong, if either. The checker cannot judge that.

### Check it

`check.py` is in this folder. Run it:

```bash
python my-work/labs/lab01/check.py
```


**You are done when** `python my-work/labs/lab01/check.py` prints a PASS line for each of its sixteen checks, then ALL CHECKS PASSED, and exits with code 0. The two hash checks carry the result: both runs recorded the same SHA-256 for compare.py, and it still matches the file on disk, so the model swap happened in .env and nowhere else. The checker also prints one line saying it did not judge whether either answer is factually correct, because that part is yours.

**If you want more:** Ask the same question five times per provider, store all five answers per run, and add your own true or false verdict to each, with no scale and no half marks. Then extend check.py with one more assertion: every run carries exactly five verdicts. Have it print the pass rate per model. Those two counts are your first evaluation, and module 16 is a more careful version of the same thing.
