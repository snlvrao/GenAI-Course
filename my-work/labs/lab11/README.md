# Lab 11: Fine-tune a small model

**Module 11: Images, audio, and teaching a model your style**

In this lab you take a small open model and teach it one exact answering shape: three lines, every line starting with a dash and a space. You do it with LoRA, which trains a tiny patch of new numbers beside the frozen model instead of retraining the model itself, and with QLoRA, which squashes the frozen part down to 4 bits so the whole thing fits on a free GPU. The proof is three questions that never appear in your training data, asked once before training and once after, then scored by a rule rather than by your opinion. On a free Colab T4 the whole run takes about fifteen minutes, and most of that is the model downloading.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Get a free GPU and install the tools

Colab is a free notebook that runs in a browser tab on Google's machines, and it will lend you a GPU (a graphics chip that does the heavy maths for training) for a few hours at a time. You pick the T4 because that is what the free tier gives you, and it has about 15 GB of video memory, which is the only memory number that matters in this lab. The install line brings in five libraries that each do one job: transformers loads the model, trl runs the training loop, peft attaches the LoRA patch, datasets holds your training rows, and bitsandbytes does the 4-bit squashing. You deliberately do not pin versions, because Colab ships its own preinstalled set and pinning against it usually breaks more than it fixes. When the cell finishes you should see GPU: Tesla T4 printed under the install output. If instead you get an error mentioning CUDA or no device, your runtime is still on CPU and nothing below this step will run.

```python
# Runtime > Change runtime type > T4 GPU, then run this cell.
!pip -q install -U transformers trl peft datasets accelerate bitsandbytes

import torch
print("GPU:", torch.cuda.get_device_name(0))
```

- `!pip -q install -U transformers trl peft datasets accelerate bitsandbytes`: The leading exclamation mark tells the notebook to run this as a shell command instead of Python. The -q keeps the output short and -U asks for the newest version of each library rather than keeping whatever Colab already had.
- `accelerate`: You never call this one directly, but transformers uses it under the hood to decide which device each piece of the model goes on. Without it, device_map="auto" in step 3 fails.
- `print("GPU:", torch.cuda.get_device_name(0))`: This asks the first attached GPU for its name. It is a one-line proof that you actually got a GPU, and it fails loudly right now instead of failing confusingly ten minutes later during training.

**The maths, spelled out**

```
Why you need a 15 GB card at all, and why plain fine-tuning will not fit on one.

Formula for full fine-tuning memory:
memory in bytes = P x (bytes per weight + bytes per gradient + bytes per optimizer state)

What the symbols mean:
P = the number of numbers (parameters) inside the model.
bytes per weight = how much space each number takes. A 16-bit number takes 2 bytes.
bytes per gradient = the same again, because training computes one correction per weight.
bytes per optimizer state = extra bookkeeping. Adam, the usual optimizer, keeps two running averages per weight at 4 bytes each, plus a 4-byte high-precision copy of the weight.

Worked example with the model in this lab:
P = 1,500,000,000 (1.5 billion).
weights = 2 bytes, gradients = 2 bytes, Adam averages = 4 + 4 = 8 bytes, master copy = 4 bytes.
Total per weight = 2 + 2 + 8 + 4 = 16 bytes.
1,500,000,000 x 16 = 24,000,000,000 bytes = 24 GB.
Your T4 has about 15 GB. So 24 GB does not fit, and this is only a 1.5 billion parameter model.

Now the QLoRA version you actually run:
Frozen weights at 4 bits = 0.5 bytes each: 1,500,000,000 x 0.5 = 0.75 GB.
Only about 9,200,000 numbers are trainable, so gradients and optimizer states cost 9,200,000 x 16 = 147,200,000 bytes = 0.15 GB.
0.75 + 0.15 is well under 1 GB, leaving plenty of room for the activations during a forward pass.

Intuition: full fine-tuning costs you about eight times the model just in bookkeeping, and the two tricks in this lab attack both halves of that, one by shrinking the frozen part and one by shrinking the trained part.
```

> **Watch out:** If you forget Runtime then Change runtime type, the last line raises a RuntimeError about no CUDA-capable device, and the friendly-looking pip output above it will fool you into thinking the cell worked.

### 2. Write the style down as pairs

A training pair is one question plus the exact answer you wish the model had given, and a pile of those pairs is your whole dataset. The style you choose has to be something a short function can check without any argument, because in step 7 you score yourself with a rule rather than a feeling. This one is easy to check: exactly three lines, each line beginning with a dash and a space, each line short. Twelve pairs will not teach the model anything new about the world, but it is enough to move the shape of its answers, and shape is the thing fine-tuning actually moves. The three test questions are held back on purpose and never trained on, because a model repeating an answer you trained it on proves only that it can memorise. When the cell finishes you should see something like Dataset({features: ['prompt', 'completion'], num_rows: 12}), and those two feature names have to be exactly right for TRL to recognise the format.

```python
def lines(*items):
    return "\n".join("- " + i for i in items)

PAIRS = [
    ("What is a virtual machine?",
     lines("A computer pretending to be a computer.",
           "One real machine runs several fake ones.",
           "Useful when you want a clean box in seconds.")),
    ("What does a database index do?",
     lines("It is a lookup table for your data.",
           "The database finds rows without reading them all.",
           "You pay for it with slower writes and more disk.")),
    ("Why do people use Git branches?",
     lines("A branch is a private copy of the work.",
           "You can break things without breaking everyone.",
           "Merging puts your changes back on the main line.")),
    ("What is an API?",
     lines("A set of doors into someone else's program.",
           "You send a request, you get an answer back.",
           "The shape of the request is agreed in advance.")),
    ("What is caching?",
     lines("Keeping an answer you already worked out.",
           "The next request gets it without the work.",
           "The hard part is knowing when to throw it away.")),
    ("What does a firewall do?",
     lines("A gate that checks traffic in and out.",
           "It allows the rules you wrote and drops the rest.",
           "Most breaches walk through a door you left open.")),
    ("What is unit testing?",
     lines("Small checks on one piece of code at a time.",
           "They run in seconds, so you run them often.",
           "They catch the mistake near where you made it.")),
    ("What does DNS do?",
     lines("It turns a name into an address.",
           "Your machine asks, a server answers, the answer is cached.",
           "When it breaks, everything looks broken.")),
    ("Why use version numbers?",
     lines("A version labels one exact state of the code.",
           "Other people can say which state they are running.",
           "Without it, 'it works for me' means nothing.")),
    ("What is a load balancer?",
     lines("A doorman standing in front of several servers.",
           "It sends each request to one that is free.",
           "If a server dies, it stops sending work there.")),
    ("What is encryption at rest?",
     lines("Data on the disk is stored scrambled.",
           "Someone who steals the disk gets noise.",
           "It does nothing once your app has unlocked it.")),
    ("What does a compiler do?",
     lines("It turns code you can read into code the machine runs.",
           "It checks your work before anything executes.",
           "Its error messages are most of what you get from it.")),
]

TEST_QUESTIONS = [
    "What is a container?",
    "What does a message queue do?",
    "Why do people write documentation?",
]

from datasets import Dataset
ds = Dataset.from_list([
    {"prompt":     [{"role": "user",      "content": q}],
     "completion": [{"role": "assistant", "content": a}]}
    for q, a in PAIRS
])
print(ds)
```

- `return "\n".join("- " + i for i in items)`: This builds the target answer string mechanically, so every one of the twelve answers has an identical shape with no stray spaces or missing dashes. If you typed those dashes by hand across twelve answers you would get one wrong, and the model would learn the inconsistency along with the style.
- `PAIRS`: A plain Python list of tuples, where each tuple is (question, ideal answer). Keeping it as ordinary Python means you can print it, count it, and edit it before it ever becomes a dataset object.
- `TEST_QUESTIONS`: Three questions on the same topic area that appear nowhere in PAIRS. This is your held-out test set, and it is the only thing that lets you tell learning apart from memorising.
- `the prompt and completion keys`: TRL recognises a dataset with exactly these two column names as prompt-completion format. It then applies the model's chat template for you and, importantly, measures the training loss only on the completion side, so the model is scored on producing your answer, not on reproducing the question.
- `Dataset.from_list([...])`: This turns your list of dictionaries into a Hugging Face Dataset, which is the object the trainer expects. It also validates that every row has the same keys, so a typo shows up here rather than mid-training.

**The maths, spelled out**

```
What the training loss number actually measures, since you will watch it drop in step 5.

Formula (cross-entropy loss):
L = -(1 / N) x sum over completion tokens of ln( p(correct token) )

What the symbols mean:
N = how many completion tokens are being scored. The prompt tokens are masked out and not counted, which is what the prompt/completion split above buys you.
p(correct token) = the probability the model gave to the token that your answer actually had next, a number between 0 and 1.
ln = natural logarithm.
The minus sign flips the result to positive, because the log of a number below 1 is negative.

Worked example with a four-token answer:
Suppose the model assigned probabilities 0.5, 0.8, 0.2, 0.9 to the four correct tokens.
ln(0.5) = -0.693
ln(0.8) = -0.223
ln(0.2) = -1.609
ln(0.9) = -0.105
Sum = -2.630
Divide by 4: -0.658
Flip the sign: L = 0.658

Some reference points:
If the model were right with probability 1.0 every time, L = -ln(1.0) = 0.
At 0.5 every time, L = 0.693.
At 0.1 every time, L = 2.303.

Intuition: the loss is average surprise per token. A loss near 0.7 means the model is about as sure as a coin flip on each token of your answer, and a loss dropping under 0.3 means it has largely learned the shape you asked for.
```

> **Watch out:** If you edit the answers by hand and a text editor turns "- " into a typographic bullet or an en dash, the checker in step 7 returns False and the fault will look like a training failure rather than a character problem.

### 3. Load the model with every number squashed to 4 bits

This is the Q in QLoRA. Quantization means storing each of the model's numbers with fewer bits than usual, which trades a little accuracy for a lot of memory. A normal 16-bit weight has 65,536 possible values, while a 4-bit weight has only 16, so the trick is choosing which 16 values to use. nf4 spaces those 16 values so that more of them sit close to zero, which matches how trained weights are actually distributed, and it holds up noticeably better than rounding evenly across the range. Compute still happens in float16 because the free T4 is an older card with no bfloat16 support, so the weights are unpacked back to 16 bits for each multiplication and then thrown away. Expect the download to take a few minutes and the final line to print roughly 1.1 to 1.3 GB, which is more than pure 4-bit arithmetic suggests because the token embedding table is left at 16 bits.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache-2.0, small enough for a free T4

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    dtype=torch.float16,      # called torch_dtype on Transformers 4.x
)
print(f"{model.get_memory_footprint()/1e9:.2f} GB in memory")
```

- `MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"`: This is the Hugging Face name of the model you download, and it is an Instruct model, meaning it has already been trained to follow chat-style requests. That matters because you are only nudging its answering shape, not teaching it to answer at all. Apache-2.0 means you can use the result commercially without asking anyone.
- `load_in_4bit=True,
    bnb_4bit_quant_type="nf4",`: The first line turns quantization on, the second picks the number format. nf4 stands for 4-bit NormalFloat, whose 16 available values are packed more densely near zero to match how model weights are actually spread.
- `bnb_4bit_use_double_quant=True,`: 4-bit storage needs one scaling number per small block of weights, and those scaling numbers themselves take space. This option quantizes the scaling numbers too, saving roughly another 0.37 bits per weight for no practical accuracy cost.
- `bnb_4bit_compute_dtype=torch.float16,`: Storage and arithmetic are separate choices. Weights sit in memory at 4 bits but are unpacked to 16-bit floats for each multiplication, because doing the maths at 4 bits would wreck the output. It is float16 rather than bfloat16 only because the T4 is too old for bfloat16.
- `tok.pad_token = tok.eos_token`: Batching two examples of different lengths needs a filler token to make them the same length. Qwen does not define one, so you reuse the end-of-sequence token. Without this, step 5 fails the moment it tries to build a batch of 2.
- `print(f"{model.get_memory_footprint()/1e9:.2f} GB in memory")`: This asks the model how many bytes it occupies and divides by a billion to show gigabytes. It is your receipt that quantization actually happened, because an unquantized load of this model would print about 3.1 instead.

**The maths, spelled out**

```
How 4-bit storage turns 3.1 GB into roughly 1.1 GB.

Formula:
memory in bytes = P x bits per weight / 8

What the symbols mean:
P = number of weights.
bits per weight = how many bits each stored number uses, including the overhead of block scaling.
Divide by 8 because there are 8 bits in a byte.

Step one, the headline saving:
Qwen2.5-1.5B-Instruct has about 1,540,000,000 weights.
At 16 bits: 1,540,000,000 x 16 / 8 = 3,080,000,000 bytes = 3.08 GB. That is the download size.
At 4 bits: 1,540,000,000 x 4 / 8 = 770,000,000 bytes = 0.77 GB. Four times smaller.

Step two, the block-scaling overhead you cannot avoid:
4-bit values only store a shape, so every block of 64 weights also stores one 32-bit scale to stretch them back to real size.
Overhead without double quantization = 32 bits / 64 weights = 0.5 bits per weight.
With bnb_4bit_use_double_quant=True those scales are themselves stored as 8-bit numbers, in blocks of 256, each block carrying one 32-bit scale:
8 / 64 + 32 / (64 x 256) = 0.125 + 0.002 = 0.127 bits per weight.
Saving = 0.5 - 0.127 = 0.373 bits per weight.
Across the roughly 1,310,000,000 weights that actually get quantized: 1,310,000,000 x 0.373 / 8 = about 61,000,000 bytes, so about 61 MB back.

Step three, why the printed number is not 0.77:
bitsandbytes leaves the token embedding table alone. That table is 151,936 vocabulary entries x 1,536 numbers each = about 233,000,000 numbers, kept at 16 bits: 233,000,000 x 2 = 0.47 GB.
The remaining 1,310,000,000 weights at 4.127 effective bits: 1,310,000,000 x 4.127 / 8 = 0.68 GB.
0.47 + 0.68 = about 1.15 GB, which is what you should see printed.

Intuition: you are storing each weight as one of only 16 sensible values plus a shared stretch factor, and that is close enough for style work while being small enough to fit.
```

> **Watch out:** If you get a TypeError about an unexpected keyword dtype, you are on Transformers 4.x where the argument was still called torch_dtype, so rename it and rerun.

### 4. Record the before answers, and do it now

This is the step people skip, and skipping it is why most fine-tuning demos prove nothing. You have to run it before you build the trainer, because SFTTrainer does not make a copy of your model, it rewires the model object you hand it and injects the adapter layers into it in place. Once that has happened, the plain untouched base model is gone from memory and there is nothing left to compare against. The ask function does four things in order: it formats your question the way this model was trained to receive chat, it moves the tokens onto the GPU, it generates, and it slices off everything that was in the prompt so you only get the new text back. do_sample=False turns off randomness, so the model always picks its highest-scoring next token and the same question gives the same answer every single time. What you should see printed is three ordinary paragraphs of helpful prose, quite likely with bold headings or numbered points, and almost certainly not three lines starting with a dash.

```python
def ask(m, question, max_new_tokens=120):
    text = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to("cuda")
    m.eval()
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    new_tokens = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

BEFORE = {q: ask(model, q) for q in TEST_QUESTIONS}
for q, a in BEFORE.items():
    print("Q:", q)
    print(a)
    print("-" * 60)
```

- `add_generation_prompt=True,`: The chat template wraps your question in the special marker tokens this model family was trained with. add_generation_prompt=True also appends the opening marker for the assistant's turn, which is the signal that says start answering here. Leave it off and the model often continues your question instead of replying to it.
- `ids = tok(text, return_tensors="pt").to("cuda")`: The tokenizer turns the formatted text into integer token IDs, return_tensors="pt" asks for PyTorch tensors rather than plain lists, and .to("cuda") moves them to the GPU. The model's weights live on the GPU, and PyTorch refuses to mix inputs on one device with weights on another.
- `m.eval()
    with torch.no_grad():`: eval() switches off training-only behaviour such as dropout, so generation is stable. no_grad() tells PyTorch not to record the information it would need to compute gradients, which saves a large amount of memory since you are not training here.
- `do_sample=False, pad_token_id=tok.pad_token_id)`: do_sample=False means greedy decoding: always take the highest-scoring next token instead of drawing one at random. That makes the before and after answers directly comparable, because any difference you see later came from training rather than from luck. Passing pad_token_id explicitly just silences a warning.
- `new_tokens = out[0][ids["input_ids"].shape[1]:]`: generate returns your prompt followed by the new text as one sequence. ids["input_ids"].shape[1] is how many tokens the prompt was, so slicing from that position onward keeps only what the model actually wrote.
- `BEFORE = {q: ask(model, q) for q in TEST_QUESTIONS}`: A dictionary comprehension that maps each held-out question to the base model's answer. Storing it in a variable now is the entire point of running this step early, because the object called model is about to be modified.

**The maths, spelled out**

```
What do_sample=False is switching off, in numbers.

At every position the model produces one raw score, called a logit, for every token in its vocabulary. This model has about 152,000 of them. Those scores are turned into probabilities with softmax.

Formula:
p_i = exp(z_i / T) / sum over all j of exp(z_j / T)

What the symbols mean:
z_i = the raw score (logit) for token i.
T = temperature, a knob that flattens or sharpens the spread. T = 1 leaves it unchanged.
exp = the exponential function, which makes every value positive.
The bottom line adds up all the exponentials so the probabilities sum to 1.

Worked example with just three candidate tokens and T = 1:
Suppose the logits are z = 3.2, 2.1, 0.5.
exp(3.2) = 24.53
exp(2.1) = 8.17
exp(0.5) = 1.65
Sum = 34.35
p(first) = 24.53 / 34.35 = 0.714
p(second) = 8.17 / 34.35 = 0.238
p(third) = 1.65 / 34.35 = 0.048

With do_sample=True the model rolls a die against those numbers, so roughly 24 runs in 100 would pick the second token and your answer would drift from run to run.
With do_sample=False none of this arithmetic is used at all. The model simply takes the largest logit, 3.2, every time. Temperature has no effect either, because scaling all the logits by the same T never changes which one is biggest.

Intuition: you are trading variety for repeatability, and repeatability is exactly what a before-and-after comparison needs.
```

> **Watch out:** If you run this cell after step 5 rather than before it, model already has the adapter attached, so BEFORE and AFTER will look nearly identical and everything will still print without any error.

### 5. Bolt a LoRA adapter onto it and train

LoRA means Low-Rank Adaptation, and the idea is that you leave every original number in the model frozen and add a small pair of extra matrices beside chosen layers, then train only those. target_modules is the list of layer names that get an adapter, and those particular names belong to this model family, so copying them onto a different architecture will fail with a message about no modules found. r is the rank, which is the width of the bottleneck between the two new matrices, and r=8 gives you roughly 9.2 million trainable numbers against about 1.54 billion frozen ones. The learning rate of 2e-4 is around ten to a hundred times higher than full fine-tuning would use, which is safe here because you are only moving fresh numbers that start at zero rather than disturbing weights that already know how to speak English. Training runs 60 optimizer steps and takes a couple of minutes, and you should watch the printed loss fall from somewhere around 2 down towards 0.5 or lower. gradient_checkpointing=False avoids a common error about tensors not requiring gradients on quantized models, and switching it off costs you nothing at this model size.

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

cfg = SFTConfig(
    output_dir="style-run",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
    max_length=512,
    fp16=True,                   # the free T4 has no bfloat16
    bf16=False,
    gradient_checkpointing=False,
    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,        # this argument was called tokenizer before TRL 0.12
    peft_config=lora,
)
trainer.train()
```

- `r=8, lora_alpha=16, lora_dropout=0.05, bias="none",`: r is the rank, the size of the bottleneck and the main size knob. lora_alpha is a fixed multiplier applied to the adapter's output, and the effective strength is alpha divided by r, so 16 over 8 gives 2. lora_dropout=0.05 randomly ignores 5% of the adapter's connections during training to discourage exact memorising, and bias="none" means the layers' bias terms stay frozen too.
- `target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],`: These are the seven big weight matrices inside each transformer block: four in the attention part (query, key, value, output) and three in the feed-forward part. Adapting all seven rather than just the attention ones is the usual recommendation, and these exact names are specific to Qwen and Llama-style models.
- `num_train_epochs=10,
    per_device_train_batch_size=2,`: An epoch is one pass over all twelve rows, and batch size 2 means two rows are processed per weight update. Together they give 6 updates per epoch and 60 in total, which is small enough to run in minutes and enough repetition to drive a format home.
- `learning_rate=2e-4,`: This is how big a step the optimizer takes on each update, written in scientific notation as 0.0002. Full fine-tuning typically uses something near 2e-5 because a big step would damage knowledge already in the weights, but here only the new adapter numbers move, so a larger step is both safe and necessary to see any change in 60 steps.
- `fp16=True,                   # the free T4 has no bfloat16`: This trains using 16-bit floats, which halves memory and speeds up the maths on the T4's tensor cores. bf16=False is required because the T4 predates bfloat16 support, and gradient_checkpointing=False avoids a well-known error on 4-bit models where the recomputed activations lose their gradient link.
- `processing_class=tok,        # this argument was called tokenizer before TRL 0.12
    peft_config=lora,`: processing_class hands TRL the tokenizer so it can apply the chat template and pad batches, and the rename is the single most common breakage when following an older tutorial. peft_config=lora is what makes the trainer inject the adapter into your model rather than train all of it.

**The maths, spelled out**

```
What LoRA actually adds, and where 9.2 million comes from.

A normal layer does:
y = W x
W is a weight matrix of shape (d_out rows by d_in columns), x is the incoming vector, y is the outgoing one.

LoRA changes it to:
y = W x + (alpha / r) x B (A x)

What the symbols mean:
W = the original frozen matrix. It never changes.
A = a new matrix of shape (r by d_in). It starts with small random values.
B = a new matrix of shape (d_out by r). It starts as all zeros, so on step 0 the adapter contributes exactly nothing and the model behaves like the base model.
r = the rank, the width of the bottleneck between A and B.
alpha / r = a fixed multiplier on the adapter's contribution.

Parameter count for one adapted matrix:
trainable numbers = r x (d_in + d_out)

Worked example on this model's q_proj layer:
d_in = 1536, d_out = 1536, r = 8.
A holds 8 x 1536 = 12,288 numbers.
B holds 1536 x 8 = 12,288 numbers.
Total = 24,576 trainable numbers, against W's 1536 x 1536 = 2,359,296 frozen ones. That is about 1%.

Adding up all seven targets in one transformer block on this model:
q_proj: 8 x (1536 + 1536) = 24,576
k_proj: 8 x (1536 + 256) = 14,336
v_proj: 8 x (1536 + 256) = 14,336
o_proj: 8 x (1536 + 1536) = 24,576
gate_proj: 8 x (1536 + 8960) = 83,968
up_proj: 8 x (1536 + 8960) = 83,968
down_proj: 8 x (8960 + 1536) = 83,968
Block total = 329,728
There are 28 blocks: 329,728 x 28 = 9,232,384, so about 9.2 million.
Against 1,540,000,000 frozen numbers that is 9,232,384 / 1,540,000,000 = 0.006, which is 0.6%.

The alpha over r scaling:
alpha / r = 16 / 8 = 2, so the adapter's output is doubled before being added to W x.
If you raise r to 64 and leave alpha at 16, the scale becomes 16 / 64 = 0.25, a factor of eight weaker. That is why people usually move alpha up with r, commonly keeping alpha equal to twice r.

Training step count:
12 rows / batch size 2 = 6 updates per epoch.
6 x 10 epochs = 60 updates.
logging_steps=5 means a loss line every 5 updates, so about 12 lines printed.

Intuition: instead of relearning a 1536 by 1536 matrix, you learn a thin correction that can only vary in 8 independent directions, which is far too small to store new facts but plenty to store a habit.
```

> **Watch out:** If you see a TypeError about an unexpected keyword processing_class or max_length, you are on a pre-0.12 TRL, and the older names are tokenizer and max_seq_length respectively.

### 6. Ask the same three questions again

trainer.model is not a new model, it is the same object from step 3 with your adapter layers now attached inside it, which is exactly why step 4 had to run first. You reuse the same ask function, the same three held-out questions, and the same greedy generation settings, so precisely one thing has changed between the two columns. What you should see is the after-answers collapsing into three short lines each starting with a dash, while the before-answers stay as the model's usual helpful prose. If the after-answers are still not three lines, raise num_train_epochs to 20 and rerun step 5 and this step, because with only twelve examples you are deliberately trying to overfit a little. Be honest with yourself about what changed: the shape moved, and the facts in those answers are still whatever the base model already believed, since twelve examples cannot teach it anything new about containers or message queues.

```python
tuned = trainer.model
AFTER = {q: ask(tuned, q) for q in TEST_QUESTIONS}

for q in TEST_QUESTIONS:
    print("Q:", q)
    print("BEFORE:")
    print(BEFORE[q])
    print("AFTER:")
    print(AFTER[q])
    print("=" * 60)
```

- `tuned = trainer.model`: This is just a friendlier name for the model the trainer has been modifying. It is a PeftModel, meaning the frozen base with adapter layers wrapped around the targeted matrices, and it is the object you will save in step 7.
- `AFTER = {q: ask(tuned, q) for q in TEST_QUESTIONS}`: Same function, same questions, same greedy settings as step 4. Reusing ask rather than writing a second version is what guarantees the two columns are measured identically.
- `the print loop`: It walks the questions in a fixed order and prints before directly above after, with a line of equals signs separating each question. Putting them next to each other rather than in two separate cells is what makes the difference obvious at a glance.

**The maths, spelled out**

```
Why a tiny weight change can flip a whole answer.

The adapter adds a correction to every logit at every position:
new logit = old logit + contribution from (alpha / r) x B A x

Greedy decoding only cares which logit is largest, so a shift of a few tenths is enough to change the winner, and the first token then drags the rest of the answer behind it.

Worked example. These three numbers are made up to show the mechanism, not measured from the model.
Say the first answer token has these base logits:
"-" = 2.90, "A" = 3.10, "Cont" = 2.40
exp(2.90) = 18.17, exp(3.10) = 22.20, exp(2.40) = 11.02, sum = 51.39
p("-") = 0.354, p("A") = 0.432, p("Cont") = 0.214
Greedy picks "A", so the answer starts with a normal sentence.

Now the trained adapter contributes +0.45 to "-" and -0.15 to "A":
"-" = 3.35, "A" = 2.95, "Cont" = 2.40
exp(3.35) = 28.50, exp(2.95) = 19.11, exp(2.40) = 11.02, sum = 58.63
p("-") = 0.486, p("A") = 0.326, p("Cont") = 0.188
Greedy now picks "-", and every token after it is generated in the context of an answer that has already started with a dash.

Notice the probability of "-" only moved from 0.354 to 0.486, a shift of about 13 percentage points, yet the visible output changed completely.

Intuition: greedy decoding turns a small numeric nudge into an all-or-nothing change in what you read, which is why 60 training steps on 12 examples can look like a dramatic transformation.
```

> **Watch out:** If you rerun the step 5 cell to try more epochs without restarting, you stack a second adapter on top of an already-adapted model and get odd results, so restart the runtime and run from step 3 for a clean second attempt.

### 7. Score it with a rule, then look at the file size

Do not judge this by eye, because you spent twenty minutes on it and you want it to have worked. A function that returns True or False is a binary judge, the kind the evaluation module argued for, and it does not flatter you or drift between runs. is_house_style throws away blank lines, counts what is left, and demands exactly three rows each starting with a dash and a space. You then save the adapter, and the printed size is the number that makes the LoRA idea concrete: the base model you downloaded is about 3 GB, while what you are keeping is only the difference. The last block writes before_after.json, which is the file you download from Colab's file browser to run the optional step 8 on your own machine. You should see 0 out of 3 before, at least 2 out of 3 after, and an adapter size somewhere around 35 to 40 MB.

```python
import os, json

def is_house_style(text):
    rows = [r for r in text.strip().splitlines() if r.strip()]
    return len(rows) == 3 and all(r.strip().startswith("- ") for r in rows)

print("before passing:", sum(is_house_style(BEFORE[q]) for q in TEST_QUESTIONS), "/ 3")
print("after passing: ", sum(is_house_style(AFTER[q])  for q in TEST_QUESTIONS), "/ 3")

tuned.save_pretrained("style-adapter")
size = sum(os.path.getsize(os.path.join(root, f))
           for root, _, files in os.walk("style-adapter") for f in files)
print(f"adapter on disk: {size/1e6:.1f} MB")

json.dump([{"question": q, "before": BEFORE[q], "after": AFTER[q]}
           for q in TEST_QUESTIONS],
          open("before_after.json", "w"), indent=2)
```

- `rows = [r for r in text.strip().splitlines() if r.strip()]`: splitlines() breaks the answer into a list of lines, and the if r.strip() filter drops empty ones. Without that filter a single blank line between the dashes would count as a fourth row and fail an answer that is actually correct.
- `return len(rows) == 3 and all(r.strip().startswith("- ") for r in rows)`: Two conditions joined with and: exactly three rows, and every row opening with a dash followed by a space. all() returns True only if the check passes for every row, so one stray line fails the whole answer. This is intentionally strict, because a judge that gives partial credit is a judge you can talk yourself past.
- `sum(is_house_style(BEFORE[q]) for q in TEST_QUESTIONS)`: Python treats True as 1 and False as 0, so summing three boolean results gives you a count between 0 and 3 without any extra counting code.
- `tuned.save_pretrained("style-adapter")`: Because tuned is a PeftModel, this saves only the adapter weights plus a small config file, not the 3 GB base model. That is why the folder is measured in megabytes, and it is also why the folder is useless on its own: loading it later requires downloading the same base model again.
- `the os.walk size sum`: os.walk visits every file in the folder and os.path.getsize gives each one's byte count, so summing them gives the true folder size including the config JSON. Dividing by 1e6 reports megabytes in the one-million-bytes sense, which is what disk vendors and Hugging Face use.
- `json.dump([...], open("before_after.json", "w"), indent=2)`: This writes the three questions with both answers into a readable file so the comparison survives after the Colab runtime shuts down. indent=2 makes it pretty-printed so you can open it in any text editor.

**The maths, spelled out**

```
Where the adapter file size comes from, and what fraction of the model you are actually keeping.

Formula:
file size in bytes = trainable parameters x bytes per number

Worked example:
From step 5 you have 9,232,384 trainable numbers.
If they were saved as 16-bit floats: 9,232,384 x 2 = 18,464,768 bytes = 18.5 MB.
PEFT casts 16-bit adapter weights up to 32-bit floats when saving, so in practice: 9,232,384 x 4 = 36,929,536 bytes = 36.9 MB.
Add a few kilobytes of JSON config, and the printed line should read close to 36.9 MB.

Compare that against the base model:
Base model at 16 bits = 3,080 MB (from step 3).
36.9 / 3,080 = 0.012, so 1.2%.
Another way to say it: you are shipping about 1 MB of change for every 83 MB of model.

The score itself:
pass rate = number of answers passing / number tested
before: 0 / 3 = 0.00, so 0%.
after: 2 / 3 = 0.667, so about 67%, or 3 / 3 = 1.00 for 100%.
With only three test questions each answer is worth 33 percentage points, so treat this as a smoke test rather than a measurement. A real evaluation would use 30 to 50 held-out questions.

Intuition: the whole result of this lab fits on a floppy-disk-and-a-half worth of disk, and it can be dropped onto or lifted off an untouched base model at any time.
```

> **Watch out:** The most common failure is an answer that has the three correct dashed lines followed by a friendly fourth line such as "Let me know if you want more detail", which makes len(rows) equal 4 and returns False, so read the printed text before blaming the training.

### 8. Optional: let a judge decide instead of your eyes

This part runs on your own machine using the course helper in my-work/labs/_shared/llm.py, not in Colab, so download before_after.json from the Colab file browser first and drop it next to your script. The helper reads LLM_PROVIDER from your .env, so you switch which model does the judging by editing that file and never by editing this code. Be honest about what this is: for a rule this exact, the six-line Python function in step 7 is a better judge than any language model, and it is free and instant. The reason to practise the pattern here is that most real style checks cannot be written as a rule, and when you reach one of those you want the habit already in place. That habit is three things: ask one narrow question, demand a one-word answer, and never ask for a score out of five, because scores out of five drift between runs and cannot be compared. You should see six lines printed, three labelled before and three labelled after, and the verdicts should match what step 7's rule already told you.

```python
import sys, json
sys.path.append("../_shared")
from llm import chat

RUBRIC = (
    "You are checking one thing only. Reply with exactly one word: PASS or FAIL.\n"
    "PASS if the answer below is exactly three lines and every line starts with '- '.\n"
    "FAIL otherwise. No explanation.\n\nANSWER:\n"
)

for row in json.load(open("before_after.json")):
    for label in ("before", "after"):
        verdict = chat(RUBRIC + row[label]).strip()
        print(f"{label:<7} {verdict:<5} | {row['question']}")
```

- `sys.path.append("../_shared")
from llm import chat`: This adds the shared course folder to Python's import search path so llm.py can be found from inside the lab folder. chat takes a string and returns the model's reply as a string, and it reads your provider choice from .env, which is why you never edit code to switch models.
- `"You are checking one thing only. Reply with exactly one word: PASS or FAIL.\n"`: One check, one word. Narrowing the judge to a single yes-or-no question is what makes its answers comparable across runs, and asking for one word keeps the output cost near zero and makes the reply easy to parse.
- `for label in ("before", "after"):`: The inner loop judges both columns with exactly the same rubric. Judging only the after-answers would be the same mistake as skipping step 4, because you would have nothing to compare against.
- `print(f"{label:<7} {verdict:<5} | {row['question']}")`: The :&lt;7 and :&lt;5 pad each field to a fixed width and align it left, so the six lines form neat columns you can scan down. It is cosmetic, but a misaligned column is genuinely harder to read a pattern out of.

**The maths, spelled out**

```
What six judge calls cost you, since token pricing is the number people guess wrong most often.

Rough rule for English text: 1 token is about 4 characters, so tokens is approximately characters / 4.

What the symbols mean:
input tokens = everything you send, which here is the rubric plus one answer.
output tokens = everything the model writes back, which here is one word.
price per million = what your provider charges, quoted per 1,000,000 tokens.

Worked example:
The RUBRIC string is about 196 characters, so 196 / 4 = about 49 tokens. Call it 50.
A three-line answer is about 150 characters, so 150 / 4 = about 38 tokens. Call it 40.
Input per call = 50 + 40 = 90 tokens.
Output per call = 1 token, because you demanded one word.
Calls = 3 questions x 2 labels = 6.
Total input = 6 x 90 = 540 tokens. Total output = 6 tokens.

Cost formula:
cost = (tokens / 1,000,000) x price per million
If your provider charged 0.15 dollars per million input tokens:
540 / 1,000,000 x 0.15 = 0.000081 dollars, which is under one hundredth of a cent.

Agreement with the rule, which is what you are really measuring:
agreement = rows where the judge matched step 7's function / total rows
If the judge matched on 5 of the 6 rows: 5 / 6 = 0.833, so about 83%.
Any disagreement is a fault in the judge, not in the model being judged, because the Python rule is the definition of the style here.

Intuition: demanding a one-word answer pushes almost all the cost onto the input side, which is why binary judges stay cheap even when you run them over hundreds of rows.
```

> **Watch out:** The judge often replies "PASS." with a full stop or adds a short sentence despite being told not to, which makes the column look ragged, so check that before assuming your provider or .env setting is broken.

## You are done when

Step 7 prints "before passing: 0 / 3" and "after passing:" with 2 or 3, meaning the three-line rule rejects every base-model answer and accepts at least two of the trained ones. The same cell prints "adapter on disk:" with a figure under 50 MB, usually around 37 MB, against a base model download of roughly 3 GB. You also have a before_after.json file you can open and read, showing all three questions with both answers side by side.

---

## Mini-project: Your own voice

Twenty things you have already written are enough to pull a model's voice noticeably towards yours. You will write your style down as a Python rule, train on your own rows, and produce voice_run.json so a program scores the shift instead of you deciding it looks better.

- Pull 20 pieces of your own writing that are all one kind of thing: 20 Slack replies, or 20 short emails, or 20 commit messages. Mixing kinds averages your different voices into nobody's. For each piece, write the request that would have produced it. That request plus your real text is one row.
- Write style_check.py next to it, defining passes(text) that returns True or False. Encode 2 to 4 habits a program can actually see: sentence count, an opening phrase, no exclamation marks, ends on a question. A worked one: t = text.strip(); return t.startswith("Short answer:") and "!" not in t and 2 <= t.count(".") <= 3
- Run passes() over all 20 of your pieces until every one returns True, then over generic assistant prose such as "Certainly! Here is a breakdown of the topic." until it returns False. If you cannot get both, your habits are not mechanical yet, so pick different ones.
- Split 15 rows for training and 5 to hold back, and make sure no held-out prompt appears in the 15. In the lab notebook, replace PAIRS with your 15 and TEST_QUESTIONS with your 5 held-out prompts. Record BEFORE first, then train, then record AFTER, exactly in that order.
- Write voice_run.json with this shape, taking adapter_bytes from the lab's step 7 as a raw byte count rather than the MB figure: {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter_bytes": 36929536, "train": [{"prompt": "...", "completion": "your real text"}, ...15 rows], "heldout": [{"prompt": "...", "reference": "your real text"}, ...5 rows], "before": {"<held-out prompt>": "<base model answer>", ...5 keys}, "after": {"<held-out prompt>": "<tuned model answer>", ...5 keys}}
- Download voice_run.json from Colab, put it in one folder with style_check.py and check.py, then run python check.py.

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py   (run it in the folder holding voice_run.json and style_check.py)
```


**You are done when** python check.py prints 12 PASS lines, a scored line reading something like "before 0/5, after 4/5", then ALL CHECKS PASSED, and exits 0. Any FAIL names the exact thing to fix: a held-out prompt that leaked into training, a rule that accepts generic assistant prose, a before score that was already passing, an adapter file the size of a whole model. The checker also prints one line saying it cannot judge whether the answers sound like you, so read those five after-answers yourself.

**If you want more:** Train a second adapter at r=64 on the same 15 rows, write its results out, swap that file in as voice_run.json and run check.py again. Look for the point where after hits 5/5 while the answers start reciting your training rows nearly word for word. That is overfitting, the checker passes it happily, and watching it pass is the lesson.
