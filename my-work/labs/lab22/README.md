# Lab 22: Fine-tune a pretrained model

**Module 22: B4 · Why pretraining matters**

You will put two models on the same three instructions: the one you trained from nothing in B3, and a 135 million parameter model somebody else pretrained. The fine-tuning script already exists and is tested, so you are changing what it trains on, not how it works.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Set up lab22 and check what B3 left you

You need both halves of the comparison in one place. Copy the tested fine-tuning script from Module 11 into my-work/labs/lab22 under a new name, so you never edit the original. Then confirm the B3 checkpoint loads and holds what you expect: the weights, the character map, and the score. If tinygpt.pt is missing, or the validation loss is much above 1.7, go back and train longer, because a weak from-scratch model turns this comparison into a strawman instead of an honest test.

```python
cd my-work/labs/lab22
cp ../lab11/cpu_lora.py instruct_lora.py        # macOS, Linux
copy ..\lab11\cpu_lora.py instruct_lora.py      # Windows

python -c "import sys, torch; sys.path.append('../_shared'); b=torch.load('../lab21/tinygpt.pt', map_location='cpu', weights_only=False); print(b['steps'], 'steps, val loss', round(b['val_loss'],3), ',', len(b['stoi']), 'distinct characters')"

# measured on a laptop processor with no graphics card:
# 1223 steps, val loss 1.674 , 65 distinct characters
```

- `sys.path.append('../_shared')`: train_gpt.py saved the Config dataclass inside the checkpoint, so unpickling it needs the tinygpt module importable. Leave this out and torch.load fails with ModuleNotFoundError: No module named 'tinygpt', which looks like a broken file but is a broken path.
- `weights_only=False`: the checkpoint holds a Python object, not only tensors, so torch will not load it in the safe mode. Pass this flag only for a file you produced yourself.

> **Watch out:** Do not edit my-work/labs/lab11/cpu_lora.py. Module 11 still uses it, and you want the style version intact so you can diff the two.

### 2. Ask your own model to follow an instruction

Before you touch anything pretrained, watch your own model fail, and record the failure. This script loads the B3 checkpoint and generates 80 characters after each of three plain instructions. The generation code is the same four lines serve_mine.py uses in B5, so nothing here is special-cased to make the model look bad. Run it and read the output carefully: the model is not confused, it is doing exactly what it was trained to do, which is continue text that looks like its corpus.

```python
"""B4 - three instructions through the model YOU trained. Run: python ask_scratch.py"""
import json, pathlib, sys, torch
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))
from tinygpt import TinyGPT

torch.manual_seed(1337)
PROMPTS = [
    "Write one sentence about the sea.",
    "List three colours.",
    "Answer with yes or no: is water wet?",
]

blob = torch.load("../lab21/tinygpt.pt", map_location="cpu", weights_only=False)
stoi, itos = blob["stoi"], blob["itos"]
model = TinyGPT(len(stoi), blob["config"])
model.load_state_dict(blob["state"])
model.eval()

out = {}
for p in PROMPTS:
    ids = [stoi[c] for c in p if c in stoi] or [0]      # unknown characters vanish
    gen = model.generate(torch.tensor([ids[-model.block_size:]]), 80)
    out[p] = "".join(itos[i] for i in gen[0].tolist()[len(ids):])
    print(f"> {p}\n{out[p]}\n")

json.dump(out, open("scratch.json", "w", encoding="utf-8"), indent=1)
print("wrote scratch.json")

# measured output, from a model trained 241 seconds on 1.1 MB of Shakespeare:
#
# > Write one sentence about the sea.
# Thou art that mouths are harved youw seems would
# To be married by his over wide
#
# > List three colours.
# I would murder in my waged dagger!
#
# DUKE OF AUMERLE:
# You well seen; So Hince ch
#
# > Answer with yes or no: is water wet?
# PETRUCHIO:
# Be shall from her where girled good falr
# your chaact, and he child
```

- `[stoi[c] for c in p if c in stoi]`: your vocabulary is the characters of your corpus and nothing else. Any character it never saw is silently dropped. That corpus contains exactly one digit, '3', so the prompt "Answer in 2 words." reaches the model as "Answer in  words." with a hole where the 2 was.
- `gen[0].tolist()[len(ids):]`: generate returns the prompt plus the continuation, so you slice off the prompt to see only what the model added.
- `torch.manual_seed(1337)`: generate samples from the probability distribution with torch.multinomial, so it is random by design. The seed makes your run repeatable. Output still differs across machines and torch versions.

> **Watch out:** Do not tidy this output. The mini-project needs the raw text, misspellings and stray line breaks included. Cleaning it up is the one thing that would make the comparison dishonest.

### 3. Turn the style tuner into an instruction tuner

cpu_lora.py teaches a writing style. You want the same machinery to teach instruction following, so only the data and the rule change. Replace the 15 PAIRS with 15 instructions and their answers, replace HELD_OUT with 5 instructions you never train on, and replace style_passes with a rule that describes an answer rather than a voice. Rename the three call sites of style_passes to follows. Everything else in the file stays exactly as it is, which is the point: the mechanism does not care what you are teaching it.

```python
# BEFORE, in cpu_lora.py: a style task
PAIRS = [
    ("Can you review my PR?", "Short answer: yes. Send the link and I will look this afternoon."),
]
def style_passes(text: str) -> bool:
    return text.startswith("Short answer:") and "!" not in text

# AFTER, in instruct_lora.py: an instruction task
PAIRS = [
    ("Write one sentence about rain.", "Rain is water that falls from cloud to ground."),
    ("List three fruits.", "Apple, banana, and pear."),
    ("Answer with yes or no: is ice cold?", "Yes."),
    ("Name the capital of France.", "Paris."),
    ("Summarise in one sentence: the meeting ran long and nothing was decided.",
     "A long meeting ended without a decision."),
    ("Give one reason to test code.", "Tests catch a mistake before a user does."),
    # ... nine more rows in the same shape, 15 in total
]

HELD_OUT = [
    "Write one sentence about snow.",
    "List three vegetables.",
    "Answer with yes or no: is fire hot?",
    "Name the capital of Japan.",
    "Give one reason to back up a file.",
]

def follows(text: str) -> bool:
    """Instruction following, written as a rule a program can check."""
    t = text.strip()
    return (0 < len(t.split()) <= 30      # it answered, and it stopped
            and t.endswith(".")           # a finished sentence, not a cut-off one
            and "?" not in t)             # it answered instead of asking back
```

- `tok.apply_chat_template([{"role": "user", ...}, {"role": "assistant", ...}], tokenize=False)`: this line is already in the file and needs no change. The pretrained model has a user and assistant format baked in from its own instruction tuning, and this wraps your rows in it. Your from-scratch model has no such thing, which is one concrete, mechanical reason it cannot follow an instruction.
- `len(t.split()) <= 30`: the failure you are training away is rambling, so the rule has to catch it. Untuned small models answer a one-line question with five paragraphs.
- `"?" not in t`: small instruct models often bounce the question back rather than answer it. This catches that.

> **Watch out:** Keep 15 rows and keep EPOCHS at 80. The file's comment says it plainly: fewer passes and the behaviour does not stick.

### 4. Record the before, then attach the adapter

The script asks the five held-out prompts and stores the answers before a single weight moves. Do this first or you have nothing to compare against, and you will convince yourself the model always behaved that way. Then LoRA is attached with one call, and the script prints exactly how much of the model you are about to train. Read that printed line, because it is the sentence this whole module is built on.

```python
    # 1. BEFORE. Record this first, or you have nothing to compare against.
    print("\n--- before training ---")
    before = {q: ask(q) for q in HELD_OUT}
    for q, a in before.items():
        print(f"  [{'PASS' if follows(a) else 'fail'}] {a[:90]}")

    # 3. Attach the adapter. r=8 is the rank: the adapter is two thin matrices
    #    whose product has the same shape as the big frozen one.
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\ntraining {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")

# measured output:
# training 460,800 of 134,975,808 parameters (0.34%)
```

- `r=8`: the rank. The patch is an 8-wide bottleneck between two thin matrices, so the number of trained values grows with r, not with the size of the frozen matrix.
- `target_modules=["q_proj", "v_proj"]`: the query and value projections inside attention, the usual place to attach. You could patch every matrix in the model, which costs more and rarely helps at this size.
- `if p.requires_grad`: this is what frozen means in code. The original 134.5 million weights still run in the forward and backward pass, they just never get updated.

**The maths, spelled out**

```
trainable = layers x [ (r x d_model + d_model x r) + (r x d_model + d_kv x r) ]

SmolLM2-135M: layers = 30, d_model = 576, 9 heads of 64, 3 key/value heads so d_kv = 192, and r = 8.

  q_proj patch:  8 x 576 + 576 x 8  = 4,608 + 4,608 = 9,216
  v_proj patch:  8 x 576 + 192 x 8  = 4,608 + 1,536 = 6,144
  per layer:     9,216 + 6,144      = 15,360
  all layers:    15,360 x 30        = 460,800

460,800 / 134,975,808 = 0.0034, which is the 0.34% the script prints. The hand figure and the printed figure agree exactly.
```

> **Watch out:** First run only, this downloads about 270 MB and takes roughly 33 seconds to load. After that it comes from the local cache and the script runs offline.

### 5. Train: 80 passes, about two and a half minutes

This is gradient descent in eight lines, the same loop as B3's training script. The differences are that the batch is your 15 rows all at once rather than random windows of a corpus, and that only the 460,800 adapter values receive an update. Watch the loss fall. Then remember what that loss is measured on.

```python
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    model.train()
    t1 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()          # work out which way each number should move
        opt.step()                   # move them a little
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:>2}  loss {out.loss.item():.3f}  ({time.time() - t1:.0f}s)")
    print(f"trained in {time.time() - t1:.0f}s")

# measured on a laptop processor with no graphics card:
# 143 seconds of training, 155 seconds end to end with the model already cached.
```

- `[p for p in model.parameters() if p.requires_grad]`: the optimiser is handed only the adapter. This is why 143 seconds is enough: AdamW also keeps two running averages per trained value, so freezing 99.66% of the model cuts memory as well as time.
- `batch["labels"][batch["attention_mask"] == 0] = -100`: set earlier in the file. Padding is masked with -100 so the model is not scored on empty space, which would otherwise reward it for predicting nothing.

> **Watch out:** The printed loss is measured on the 15 rows the model is training on, so it always falls. It tells you the optimiser is working and nothing else. Only the held-out prompts tell you whether anything was learned.

### 6. Re-ask the same five held-out prompts and score

Same five prompts, same ask function, same rule. The only thing that changed is 460,800 numbers. On the style version of this task the rule went from 0 of 5 to 5 of 5, and your instruction version should move about as far. Read the actual answers as well as the score, because the rule checks the shape of an answer and cannot check whether it is true.

```python
    model.eval()
    print("\n--- after training ---")
    after = {q: ask(q) for q in HELD_OUT}
    hits = 0
    for q, a in after.items():
        ok = follows(a)
        hits += ok
        print(f"  [{'PASS' if ok else 'fail'}] {a[:90]}")
    print(f"\nrule matched {hits}/{len(HELD_OUT)} held-out prompts "
          f"(before: {sum(follows(a) for a in before.values())}/{len(HELD_OUT)})")

# measured with the style rule from Module 11:
# style rule matched 5/5 held-out prompts (before: 0/5)
```

- `model.eval()`: turns off dropout. LoraConfig set lora_dropout=0.05, which randomly zeroes 5% of the adapter during training and must not be active when you measure.
- `do_sample=False`: inside ask, set earlier in the file. Generation is greedy, so the same prompt gives the same answer every time and the before and after difference is the adapter, not luck.

> **Watch out:** Eighty passes over 15 rows is close to memorisation. Expect the tuned model to answer "Name the capital of Japan." in the right shape and to be confidently wrong sometimes. Shape is what you trained; facts came from pretraining, and 135 million parameters do not hold many.

### 7. Save the adapter and weigh it

The script writes the adapter and prints its size. Compare the three files you now own. The adapter is about 1.8 MB, your entire from-scratch model is about 4.4 MB, and the base model underneath the adapter is about 270 MB. Your whole model, tokenizer, weights and all, is bigger than the patch that taught a 135 million parameter model to follow instructions, and that patch is worthless on its own while your 4.4 MB runs by itself.

```python
    model.save_pretrained("adapter")
    import pathlib
    size = sum(f.stat().st_size for f in pathlib.Path("adapter").rglob("*") if f.is_file())
    print(f"adapter written to adapter/  ({size:,} bytes)")

# what you should now have in my-work/labs/lab22:
#   adapter/         about 1.8 MB   useless without the 270 MB base model
#   scratch.json     3 answers from the model you built
#   ../lab21/tinygpt.pt   about 4.4 MB   the entire model, runs alone
```

- `save_pretrained("adapter")`: writes the LoRA weights plus a small config naming the base model. Loading it later downloads or reads that base and applies the patch on top.

> **Watch out:** An adapter is tied to one exact base model. Point it at a different checkpoint and it will either refuse to load or silently produce nonsense.

## You are done when

You have run the same instructions through a model you built from nothing and a model somebody else pretrained, and you have both sets of answers on disk. You can also state, with numbers, what the second one cost: 460,800 trained values, 143 seconds, and somebody else's 2 trillion tokens of pretraining that you did not pay for.

---

## Mini-project: Compare them

Put both models on trial with the same three prompts and write down the verdict yourself. The point is not that the pretrained model wins, it is that you can say exactly what it wins at and what it cost.

- Choose three prompts that are plainly instructions, not text to continue. Write them once, as one list, and paste the identical strings into both scripts. Different wording on the two sides is not a comparison.
- Run `python ask_scratch.py` in my-work/labs/lab22. It writes scratch.json with your from-scratch model's answer to each of the three. Keep the raw text: the line breaks, the invented speaker names, the misspellings.
- In instruct_lora.py, after the `--- after training ---` block, add `json.dump({q: ask(q) for q in COMPARE}, open("tuned.json", "w", encoding="utf-8"), indent=1)` with COMPARE holding those same three prompts. `ask()` already exists in that file and looks up `model` when it is called, so it picks up the adapter automatically.
- Run `python instruct_lora.py`, about 155 seconds once the base model is cached, and confirm tuned.json holds three non-empty answers.
- Merge the two files into compare.json with this exact shape: `{"prompts": [three strings], "scratch": {prompt: answer}, "tuned": {prompt: answer}, "judgement": {"better": "tuned" or "scratch", "why": "..."}}`.
- Write judgement.why in your own words, 25 words or more, naming one specific thing you can see in the two sets of answers. "It is better" does not pass the checker, and it should not.

### Check it

`check_compare.py` is in this folder. Run it:

```bash
Save it in my-work/labs/lab22 next to compare.json and run `python check_compare.py`. It prints one PASS or FAIL line per check and exits non-zero if any failed. Standard library only, no torch needed.
```


**You are done when** compare.json exists, `python check_compare.py` prints ALL CHECKS PASSED, and you can say in one sentence what 155 seconds of fine-tuning bought that 241 seconds of training from scratch could not.

**If you want more:** Add a third column and split the credit. Next to the existing `before = {q: ask(q) for q in HELD_OUT}` line, which runs before the adapter is attached, add `base = {q: ask(q) for q in COMPARE}` and record those answers too under a "base" key. Now you can separate what pretraining gave you from what your 15 rows gave you. Usually the base model was already close and your rows only fixed the shape, which is worth seeing for yourself.
