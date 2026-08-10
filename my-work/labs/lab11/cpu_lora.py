"""
Module 11, offline route - fine-tune on your own laptop, no GPU, no internet
after the first download.

The Colab notebook in the lab uses a bigger model and a real GPU. This does the
same thing on a CPU with a much smaller model. The result is cruder, but every
mechanism is identical: freeze the model, train a small adapter, watch the
style move.

Measured on an ordinary Windows laptop, no graphics card:
    model load      ~33 s (first run only, it downloads ~270 MB)
    training        ~170 s for 80 passes over 15 examples
    total           ~3 minutes
    adapter size    ~1.8 MB, against a 135 M parameter model

Run:  python cpu_lora.py
"""

import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# A 135 million parameter model. Tiny by 2026 standards, which is the point:
# it is small enough to train on a CPU and still large enough to show the shift.
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# ---------------------------------------------------------------------------
# REPLACE THESE with 15 rows of your own writing, all one kind of thing.
# The style here is deliberate and mechanical: every reply opens the same way
# and never uses an exclamation mark. Yours should be equally checkable.
# ---------------------------------------------------------------------------
PAIRS = [
    ("Can you review my PR?", "Short answer: yes. Send the link and I will look this afternoon."),
    ("Are we still on for 3pm?", "Short answer: yes. I will send the invite now."),
    ("Did the deploy go out?", "Short answer: no. It is blocked on the migration."),
    ("Should we use Postgres?", "Short answer: yes. It is already in the stack."),
    ("Is the bug fixed?", "Short answer: not yet. I have a repro and a patch in progress."),
    ("Can you join the sync?", "Short answer: yes. I will dial in from my desk."),
    ("Do we need a rollback?", "Short answer: no. The error rate is already back to normal."),
    ("Is the doc ready?", "Short answer: almost. I need one more review pass."),
    ("Can we ship Friday?", "Short answer: no. Two tests are still failing."),
    ("Did you see the alert?", "Short answer: yes. It is a false positive from the new probe."),
    ("Should I book the room?", "Short answer: yes. Book it for an hour."),
    ("Is the index rebuilt?", "Short answer: yes. It finished about twenty minutes ago."),
    ("Do you need help?", "Short answer: no. I am nearly through it."),
    ("Can you own this ticket?", "Short answer: yes. I will pick it up in the morning."),
    ("Is the API stable now?", "Short answer: yes. No errors since the patch landed."),
]

# Five prompts the model never trains on. This is the held-back set from
# Module 1, doing exactly the job it did there.
HELD_OUT = [
    "Can you take the standup?",
    "Is staging back up?",
    "Should we revert the change?",
    "Did the backup finish?",
    "Can you look at the flaky test?",
]

EPOCHS = 80          # 80 passes over 15 rows. Fewer and the style does not stick.
LEARNING_RATE = 2e-4


def style_passes(text: str) -> bool:
    """Your style, written as a rule a program can check. Edit to match yours."""
    return text.startswith("Short answer:") and "!" not in text


def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    print(f"loaded {MODEL} in {time.time() - t0:.0f}s")

    def ask(question: str) -> str:
        """Ask the model one question and return only its answer."""
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, tokenize=False,
        )
        enc = tok(text, return_tensors="pt")
        start = enc["input_ids"].shape[1]     # where the answer begins
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][start:], skip_special_tokens=True).strip()

    # 1. BEFORE. Record this first, or you have nothing to compare against.
    print("\n--- before training ---")
    before = {q: ask(q) for q in HELD_OUT}
    for q, a in before.items():
        print(f"  [{'PASS' if style_passes(a) else 'fail'}] {a[:90]}")

    # 2. Turn your rows into one batch of token ids.
    #    labels are the same ids: the model learns by predicting its own input,
    #    one token at a time. Padding is masked with -100 so the model is not
    #    scored on empty space.
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": c}],
            tokenize=False,
        )
        for p, c in PAIRS
    ]
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=160)
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100

    # 3. Attach the adapter. r=8 is the rank: the adapter is two thin matrices
    #    whose product has the same shape as the big frozen one. Small r means
    #    fewer numbers to train. q_proj and v_proj are the query and value
    #    projections inside attention, the usual place to attach.
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\ntraining {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")

    # 4. Train. This is gradient descent from Module 1, in eight lines.
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

    # 5. AFTER, on the same held-out prompts.
    model.eval()
    print("\n--- after training ---")
    after = {q: ask(q) for q in HELD_OUT}
    hits = 0
    for q, a in after.items():
        ok = style_passes(a)
        hits += ok
        print(f"  [{'PASS' if ok else 'fail'}] {a[:90]}")
    print(f"\nstyle rule matched {hits}/{len(HELD_OUT)} held-out prompts "
          f"(before: {sum(style_passes(a) for a in before.values())}/{len(HELD_OUT)})")

    # 6. Save the adapter and the record the checker reads.
    model.save_pretrained("adapter")
    import pathlib
    size = sum(f.stat().st_size for f in pathlib.Path("adapter").rglob("*") if f.is_file())
    print(f"adapter written to adapter/  ({size:,} bytes)")

    json.dump({
        "base_model": MODEL,
        "adapter_bytes": size,
        "train": [{"prompt": p, "completion": c} for p, c in PAIRS],
        "heldout": [{"prompt": q, "reference": ""} for q in HELD_OUT],
        "before": before,
        "after": after,
    }, open("voice_run.json", "w", encoding="utf-8"), indent=1)
    print("wrote voice_run.json")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
