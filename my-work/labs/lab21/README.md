# Lab 21: Train it

**Module 21: B3 · Train your own GPT**

train_gpt.py is 126 lines and does the whole job: builds a vocabulary from your text, splits it, trains, saves a checkpoint, and samples from it. Work through it once on the sample corpus before you point it at your own writing.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Check the model before you train it

tinygpt.py has a shape check at the bottom that runs the model on random data and prints its size. Run it first, because if this fails nothing else in the Build track will work. It reports 824,897 parameters and a loss of 4.495 on random input, against 4.174 for guessing uniformly over 65 options. Slightly worse than uniform is exactly right for random weights: random numbers give a lumpy output distribution, not a flat one. Write 4.174 down, because every number for the rest of this module is measured against it.

```python
cd my-work/labs/_shared
python tinygpt.py

parameters      824,897
input  (2, 16)  ->  logits (2, 16, 65)
loss on random data 4.495 (a fair coin over 65 options would be 4.174)
generated 20 tokens from an empty prompt
```

- `824,897 parameters`: 4 blocks, 4 heads each, width 128, context 128. All of it is random numbers until you train it.
- `logits (2, 16, 65)`: one score per vocabulary entry, at every one of the 16 positions, for both sequences in the batch. Sixteen separate predictions from one forward pass.
- `loss 4.495`: your starting point. Anything that stays near this after training means the model did not learn.

**The maths, spelled out**

```
Guessing uniformly over V options scores ln(V). ln(65) = 4.174. For comparison, ln(2) = 0.693, so a model that had narrowed every single choice down to a coin flip would score 0.693.
```

> **Watch out:** If this fails with ModuleNotFoundError: No module named 'torch', install torch before going further. B3, B4 and B5 all need it.

### 2. Start the run, then read while it trains

Run train_gpt.py with no arguments. If corpus.txt is not there it downloads a small public domain sample and tells you so, then trains for 480 seconds by default. The header confirms three things before any training happens: how big your text is, how many parameters you have, and what random guessing would score. Every 200 steps it prints train loss, validation loss and elapsed time, which you read in step 5. Leave it running and work through steps 3 and 4 while it goes; Ctrl+C stops it early and still saves everything.

```python
cd my-work/labs/lab21
python train_gpt.py

corpus     1,115,394 characters, 65 distinct
model      824,897 parameters
baseline   loss 4.174 if it only ever guessed at random

training for 480s. Stop it early with Ctrl+C and it still saves.

  (one line every 200 steps, read in step 5)

1223 steps in 241s. Final validation loss 1.674 (random guessing would be 4.174).
saved tinygpt.pt  (4,387,939 bytes)
```

- `65 distinct`: the vocabulary is every distinct character in YOUR file, built by sorted(set(text)). Nothing is pretrained here, not even the alphabet.
- `Ctrl+C`: the loop sits inside try/except KeyboardInterrupt, so stopping early prints 'stopped early, saving what you have' and still writes the checkpoint. The run above was stopped at 241 seconds.
- `1223 steps in 241s`: left for the full 8 minute budget it reaches roughly 3,600 steps and a validation loss near 1.55, measured on a laptop processor with no graphics card.

**The maths, spelled out**

```
Loss 1.674 means the average correct character was given probability e to the minus 1.674, which is 0.188, roughly 1 in 5.3. Random guessing gives 1/65 = 0.0154, and minus ln(0.0154) = 4.174.
```

> **Watch out:** The stopping condition is a clock, not a step count: while time.time() - t0 < BUDGET_S. Two people running the same command on different machines do different amounts of training and get different losses. That is expected.

### 3. See what one training example actually is

The most useful thing to understand about training a language model is what a single example looks like, and it is smaller than people expect. Run this snippet in my-work/labs/lab21 once corpus.txt exists. It builds the same character vocabulary train_gpt.py builds, encodes the first 200 characters, and prints x beside y. x and y are the same numbers, offset by one. The last line of the listing is the interesting one: given 'First Ci' the model must produce 't', and it gets no other information at all.

```python
cd my-work/labs/lab21
python -c "
import torch
text = open('corpus.txt', encoding='utf-8').read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text[:200]])
x, y = data[:8], data[1:9]
print('x', x.tolist())
print('y', y.tolist())
for i in range(8):
    ctx = ''.join(itos[t] for t in x[:i+1].tolist())
    print(f'  given {ctx!r:>14}  ->  predict {itos[y[i].item()]!r}')
"

x [18, 47, 56, 57, 58, 1, 15, 47]
y [47, 56, 57, 58, 1, 15, 47, 58]
  given            'F'  ->  predict 'i'
  given           'Fi'  ->  predict 'r'
  given          'Fir'  ->  predict 's'
  given         'Firs'  ->  predict 't'
  given        'First'  ->  predict ' '
  given       'First '  ->  predict 'C'
  given      'First C'  ->  predict 'i'
  given     'First Ci'  ->  predict 't'
```

- `sorted(set(text))`: the vocabulary. Sorted so the mapping is reproducible: the same text always gives the same character-to-number table.
- `y = data[1:9]`: the shift. Position i of y is the answer for position i of x.
- `8 lines from 8 characters`: the causal mask you verified in B2 is what makes this legal. Position 3 predicting 't' cannot see position 4, so all 8 predictions can be scored from one forward pass.

**The maths, spelled out**

```
One batch is 32 windows of 128 characters, and every position in every window is predicted, so a single step scores 32 x 128 = 4,096 characters. The 1,223 step run above made 1,223 x 4,096 = 5,009,408 predictions.
```

> **Watch out:** corpus.txt only exists after you have started train_gpt.py at least once, because that is what downloads it. Point the snippet at your own .txt file instead if you prefer.

### 4. The four lines that do the learning

Everything else in the file is setup. This is the loop, and it is the same loop that trains models a hundred thousand times bigger. get_batch pulls 32 random windows. Calling the model with targets runs the forward pass and returns the cross-entropy against y in one go. zero_grad clears the previous step's gradients, backward works out which direction every one of the 824,897 numbers should move to make this batch less surprising, and step moves them. Repeat a few thousand times and you have a language model.

```python
while time.time() - t0 < BUDGET_S:
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()        # which way should every number move
    opt.step()             # move them a little
    step += 1

# and inside TinyGPT.forward, the line that defines the objective:
loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
```

- `model(x, y)`: forward pass and loss in one call. The view() calls flatten 32 x 128 positions into 4,096 rows so cross_entropy scores them all together.
- `opt.zero_grad(set_to_none=True)`: PyTorch accumulates gradients. Leave this out and step N moves in a direction contaminated by step N-1, and the loss goes sideways.
- `loss.backward()`: the backward pass. This is where the cost is: roughly twice the work of the forward pass, and the reason a step takes a fraction of a second on a laptop processor.
- `opt.step()`: AdamW applies the move, scaled per parameter using a running estimate of each gradient's size.

> **Watch out:** val_loss() is decorated with @torch.no_grad() and flips the model to eval() and back. Measuring without no_grad would build a gradient graph you never use and slow the run down for nothing.

### 5. Read your curve

The checkpoint stores the whole validation history, so you can read the curve afterwards without rerunning anything. Load it and print history. Look for the shape, not the exact numbers: steep early, flattening later. In the run below over half the total improvement arrives in the first 600 steps, and the last 600 steps buy 0.135. If your curve is still dropping when the budget ends, train longer. If it has flattened, more time will not help and you need more text or a bigger model.

```python
cd my-work/labs/lab21
python -c "
import sys, pathlib, torch
sys.path.append(str(pathlib.Path('.').resolve().parents[0] / '_shared'))
b = torch.load('tinygpt.pt', map_location='cpu', weights_only=False)
for h in b['history']: print(h)
print('final', round(b['val_loss'], 4), 'steps', b['steps'])
"

{'step': 200, 'val_loss': 2.1927, 'seconds': 58.1}
{'step': 400, 'val_loss': 1.9344, 'seconds': 73.1}
{'step': 600, 'val_loss': 1.7931, 'seconds': 113.2}
{'step': 800, 'val_loss': 1.7529, 'seconds': 152.3}
{'step': 1000, 'val_loss': 1.6914, 'seconds': 188.5}
{'step': 1200, 'val_loss': 1.6582, 'seconds': 233.0}
final 1.6736 steps 1223
```

- `weights_only=False`: the checkpoint contains a Config dataclass, not just tensors, so torch refuses to load it in its default safe mode.
- `sys.path.append(... '_shared')`: unpickling Config needs to import the tinygpt module. Without this line torch.load raises ModuleNotFoundError: No module named 'tinygpt', which is confusing until you know why.
- `final 1.6736 against step 1200 at 1.6582`: the final number is a fresh measurement over 20 different random validation batches, so it wobbles by a few hundredths. Do not read meaning into small differences.

**The maths, spelled out**

```
The drop from 2.193 to 1.658 is 0.535 in log terms, which is e to the 2.193 = 9.0 effective choices down to e to the 1.658 = 5.2. The 1.658 that remains is not noise. It is how much of this text the model simply cannot predict.
```

> **Watch out:** A rising validation number is the signal to stop. It does not mean training is broken, it means the model has started memorising the training tenths instead of learning patterns that transfer.

### 6. Sample at three temperatures

train_gpt.py already prints 600 characters and writes sample.txt, but the interesting experiment is changing temperature on the same checkpoint. generate divides the logits by the temperature before softmax, so low values make the distribution sharper and high values flatten it. Run the two extremes side by side. At 0.5 you get cleaner, safer, more repetitive text; at 1.2 you get more invention and more broken words. Read both carefully: the words are real English, the line breaks are in sensible places, the speaker names have colons, and the content means nothing.

```python
cd my-work/labs/lab21
python -c "
import sys, pathlib, torch
sys.path.append(str(pathlib.Path('.').resolve().parents[0] / '_shared'))
from tinygpt import TinyGPT
b = torch.load('tinygpt.pt', map_location='cpu', weights_only=False)
m = TinyGPT(len(b['stoi']), b['config']); m.load_state_dict(b['state']); m.eval()
for t in (0.5, 1.2):
    ids = m.generate(torch.zeros((1, 1), dtype=torch.long), 200, temperature=t)
    print(f'--- temperature {t} ---')
    print(''.join(b['itos'][i] for i in ids[0].tolist()))
"

--- temperature 0.5 ---
She that thou come to the man in the body of her.

CLARENCE:
What is shall the sears of all what the cause put of could
And the thrust have to see the son which the bear.

--- temperature 1.2 ---
That vouchmers all of begell'd when be dry.
Greaty's yield th, yet, how! thank contey, stay'd
summer'd, anown, not agains
thine; yeapulaiest the wall dog pleats heir jectain,
```

- `torch.zeros((1, 1), dtype=torch.long)`: the prompt is one token, index 0, which is a newline in this vocabulary. There is no prompting here. It just starts.
- `logits[:, -1, :] / max(temperature, 1e-6)`: only the last position matters when generating. The max() guard stops a temperature of 0 from dividing by zero.
- `torch.multinomial(probs, 1)`: it draws a character at random, weighted by probability. Run the same command twice and you get different text. That is the design, not a bug.

**The maths, spelled out**

```
Temperature T replaces each score s with s/T before softmax. With two options scoring 2.0 and 1.0: at T=1.0 the probabilities are 0.73 and 0.27; at T=0.5 the scores become 4.0 and 2.0 and the probabilities 0.88 and 0.12; at T=2.0 they become 1.0 and 0.5 and the probabilities 0.62 and 0.38.
```

> **Watch out:** Be honest about what you are looking at. 'CLARENCE:' is not the model knowing who Clarence is. It has learned that a run of capitals near a line start is usually followed by a colon and a newline. Character statistics, nothing more.

### 7. Write report.py

The checkpoint holds everything you need to prove the run happened, so the report script does no training. Save this as my-work/labs/lab21/report.py. It loads tinygpt.pt, rebuilds the model, generates 300 characters at three temperatures, and writes train_report.json with the vocabulary, the baseline, the final validation loss, the full history and the samples. Note that it takes the samples from the model rather than from a file, which is what lets the checker prove they came from this checkpoint. You run it in the mini-project.

```python
"""Turn the checkpoint into train_report.json. Run after train_gpt.py."""
import json, math, pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
import torch
from tinygpt import TinyGPT

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "mytext.txt"
blob = torch.load("tinygpt.pt", map_location="cpu", weights_only=False)
model = TinyGPT(len(blob["stoi"]), blob["config"])
model.load_state_dict(blob["state"])
model.eval()

itos, samples = blob["itos"], []
for temp in (0.5, 0.8, 1.2):
    ids = model.generate(torch.zeros((1, 1), dtype=torch.long), 300, temperature=temp)
    samples.append("".join(itos[i] for i in ids[0].tolist()))

report = {
    "corpus_file": CORPUS,
    "corpus_chars": blob["corpus_chars"],
    "vocab": "".join(sorted(blob["stoi"])),
    "vocab_size": len(blob["stoi"]),
    "baseline_loss": round(math.log(len(blob["stoi"])), 4),
    "final_val_loss": round(blob["val_loss"], 4),
    "steps": blob["steps"],
    "history": blob["history"],
    "samples": samples,
    "checkpoint": "tinygpt.pt",
}
pathlib.Path("train_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote train_report.json  final val loss {report['final_val_loss']} "
      f"vs baseline {report['baseline_loss']}")

# wrote train_report.json  final val loss 1.6736 vs baseline 4.1744
```

- `sys.path.append before import torch`: the path has to be set before torch.load unpickles Config, and putting it above the torch import guarantees that.
- `model.eval()`: turns off dropout. Config has dropout 0.0 by default so it changes nothing here, but leaving it out is a habit that costs you later.
- `"vocab": "".join(sorted(blob["stoi"]))`: the exact character set the model was built for, in a form you can eyeball. For the sample corpus it starts with a newline, a space, then ! $ & ' , - . 3 : ; ?

> **Watch out:** report.py reads tinygpt.pt from the current directory, so run it from my-work/labs/lab21. If you train more than once the second run overwrites the first checkpoint, so copy it aside if you want to keep it.

## You are done when

You have tinygpt.pt in my-work/labs/lab21 with a validation loss well under log(vocabulary size), you can read the curve stored inside it, and you can sample from it at any temperature you like.

---

## Mini-project: Train on your own text

Train the same model on your own writing, then prove it learned rather than memorised.

- Put at least 20,000 characters of your own text in my-work/labs/lab21/mytext.txt. Plain text, one file: old notes, blog posts, your own code, emails you wrote, any book you have the right to use. Aim for 100,000 characters or more if you can, and read the next step before you decide.
- Train with python train_gpt.py mytext.txt 480 and watch the val column, not the train column. Press Ctrl+C once val has failed to improve for two readings in a row; the script still saves. Measured on a laptop processor with no graphics card: a 40,000 character corpus hit its best validation loss of 2.018 at step 400, then rose to 2.474 by step 600 and 2.743 by step 665, while a 200,000 character corpus fell steadily to 1.742 over 1,268 steps. Less text means stop sooner.
- Run python report.py mytext.txt from my-work/labs/lab21, using the script from lab step 7. It loads tinygpt.pt, generates 300 characters at temperature 0.5, 0.8 and 1.2, and writes train_report.json next to it.
- Save check_b3.py in my-work/labs/lab21 and run python check_b3.py. Fix whatever it flags. The most likely failure is 'validation loss went down overall', which means you trained past the bottom of your curve: retrain with a shorter budget, or add more text.
- Leave tinygpt.pt where it is. B5 serves this exact file behind an OpenAI-shaped API, and every lab in the main course can then be pointed at it.

### Check it

`check_b3.py` is in this folder. Run it:

```bash
Save it as my-work/labs/lab21/check_b3.py, in the same folder as train_report.json, tinygpt.pt and your text file. Run python check_b3.py from that folder. It prints PASS or FAIL for each check and exits 0 only when all eight pass.
```


**You are done when** check_b3.py prints eight PASS lines and exits 0. You have train_report.json and tinygpt.pt in my-work/labs/lab21, your model's vocabulary is exactly the character set of your own text, and every character in the generated samples came from that vocabulary.

**If you want more:** Open my-work/labs/_shared/tinygpt.py and change n_layer from 4 to 6, or block_size from 128 to 256. Retrain with the same time budget and compare the final validation loss. Predict which way it goes before you run it: a bigger model learns more per step, but it also does fewer steps in the same eight minutes. Write down which effect won on your machine.
