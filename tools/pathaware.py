"""
Path-aware content for the modules where the offline path genuinely differs.

Two kinds of change:
  NOTES   - a block rendered at the top of the Lab section, shown only on the
            paths it applies to.
  PATCHES - surgical edits to the generated checkers so an offline learner can
            pass with the evidence their path can actually produce.

Every patch asserts its anchor, so a regeneration that changes the underlying
code fails loudly here instead of silently shipping a broken checker.
"""

# --- notes rendered into module pages ---------------------------------------
# module -> list of (paths, kind, title, html)
NOTES = {
    5: [
        (["offline"], "gap", "This lab changes on the offline path",
         "No small local model has a separate reasoning mode, so you cannot compare a fast model "
         "against a reasoning model. You do the same experiment a different way: run <b>one local "
         "model twice</b>, once answering directly and once told to work through its reasoning "
         "before answering. That is the same effect (more compute at answer time) produced by "
         "prompting instead of by training. Cost becomes <b>tokens</b> rather than dollars, because "
         "your model is free. Name the two runs <code>qwen2.5:3b-instruct (direct)</code> and "
         "<code>qwen2.5:3b-instruct (think first)</code>, set <code>\"path\": \"offline\"</code> and "
         "<code>\"cost_unit\": \"tokens\"</code> in <code>verdict.json</code>, and the checker will "
         "grade it on token counts."),
        (["free"], "", "Which free models have a reasoning mode",
         "On Groq, look for a model whose name contains <code>reasoning</code> or a "
         "<code>deepseek-r1</code> variant. On Google AI Studio, the Gemini Flash models expose a "
         "thinking budget. If neither is available on your key today, use the offline method "
         "described for the offline path instead. It teaches the same thing."),
    ],
    6: [
        (["offline"], "warn", "This lab changes on the offline path",
         "Your model costs nothing per token, so there is no bill to reduce. Route between "
         "<b>two local models of different sizes</b> instead (for example "
         "<code>qwen2.5:3b-instruct</code> as cheap and <code>phi4-mini:3.8b</code> as strong), and measure "
         "two things: the real saving in <b>seconds</b>, and the <b>would-have-cost</b> if the same "
         "token counts had gone to hosted models at the published prices. The published prices are "
         "public, so this arithmetic is honest as long as you label it. Set "
         "<code>\"path\": \"offline\"</code> in <code>router_report.json</code> and the checker will "
         "accept zero real spend and require the seconds instead. The same arithmetic explains why model size matters so much here: one token means reading every weight once, so tokens per second is roughly memory bandwidth divided by model size. A model that fits in your graphics card runs at card speed, and a model that does not is split, with the overflow running at system memory speed, which is over ten times slower and sets the pace. Setup covers the numbers. Pick your two models on either side of what your card holds and the seconds you measure will be dramatic rather than marginal."),
    ],
    11: [
        (["free", "key"], "", "The lab below uses a free Colab GPU",
         "Training is the one thing in this course that wants a graphics card. Google Colab gives "
         "you one free. If you would rather not use it, <code>my-work/labs/lab11/cpu_lora.py</code> does the "
         "same job on your own processor with a smaller model."),
        (["offline"], "warn", "There is a CPU route, so you can stay offline",
         "Training normally wants a graphics card, and Colab lends you one free. You do not need it. "
         "<b><code>my-work/labs/lab11/cpu_lora.py</code> in this folder trains on your own processor</b>, "
         "and on a laptop processor with no graphics card it takes <b>about three minutes end to "
         "end</b> (155 seconds, 143 of them training). It uses "
         "SmolLM2-135M, downloads about 270&nbsp;MB once, and produces a 1.8&nbsp;MB adapter that the ""mini-project checker accepts as it stands. "
         "In the measured run the style rule matched <b>5 of 5 held-out prompts</b> after training "
         "and 0 of 5 before, so the shift is real and not wishful. The model is far smaller than "
         "the Colab one, so the writing is cruder, but every mechanism is the same and the "
         "mini-project checker accepts either route."),
    ],
    20: [
        (["offline", "free", "key"], "", "You choose this model's size",
         "Nothing here ships with a size baked in. Run "
         "<code>python ../_shared/myconfig.py --new</code> to get a "
         "<code>model_config.json</code>, set the four numbers yourself, and write down why for "
         "each one. Then <code>python ../_shared/myconfig.py</code> prints what your choices cost: "
         "the parameter count, the memory, and roughly how many training steps you will get in "
         "eight minutes. Get that right before B3 spends the time. The widget above does the same "
         "arithmetic live if you would rather drag sliders first."),
    ],
    21: [
        (["offline", "free", "key"], "warn", "Train YOUR model, not the template",
         "<code>train_gpt.py</code> reads <code>model_config.json</code> and refuses to run without "
         "it, because picking the size is the decision this module is about. When you have trained "
         "something, run <code>python ../_shared/check_decisions.py</code>: it verifies you changed "
         "the numbers from the template, wrote a real reason for each, and that the checkpoint you "
         "produced actually has the parameter count your config predicts. Train a second "
         "configuration afterwards and compare. That comparison is worth more than either run."),
    ],
    9: [
        (["offline"], "warn", "Expect this to be harder on a local model",
         "Choosing the right tool is exactly what small models are worst at. A 3B model will "
         "sometimes invent a tool name or send a malformed argument. That is not your bug, and the "
         "lab is deliberately written to survive it. If a run misbehaves repeatedly, try the same "
         "code once against a free hosted key: if it works there, your code is correct and you have "
         "learned something real about what model size buys you."),
    ],
    12: [
        (["offline"], "warn", "Expect this to be harder on a local model",
         "Agent loops depend on the model following a format turn after turn, which small models "
         "drift out of. The loop you write is built to recover from that. Keep the run short, and "
         "if it will not terminate, lower the number of tools before you blame the code."),
    ],
    14: [
        (["offline"], "warn", "Two things to know on the offline path",
         "The CrewAI comparison runs in Colab regardless of your path, because CrewAI will not "
         "install on Python 3.14. The LangGraph half runs locally against your own model, and will "
         "be slower and less reliable than the hand-written agent from Module 12 purely because of "
         "model size, not because of LangGraph."),
    ],
    15: [
        (["offline"], "warn", "Expect this to be harder on a local model",
         "Two agents means twice the chances of a format slip, and a small model handling a handoff "
         "will sometimes drop a field. That is genuinely one of the MAST failure modes this module "
         "is about, so it is worth seeing, but do not spend an hour assuming your code is wrong."),
    ],
    16: [
        (["offline"], "warn", "Your judge will be weak, and that is worth knowing",
         "A 3B local model makes a poor judge: expect a low true-negative rate, meaning it waves bad "
         "answers through. Do the calibration anyway. Seeing a judge fail its own calibration is the "
         "lesson, and it is the reason the module insists you measure a judge before trusting it. "
         "If you can borrow a free hosted key for the judge alone, the numbers get much better."),
    ],
}


# --- checker patches ---------------------------------------------------------
# module -> list of (anchor, replacement)
PATCHES = {
    5: [
        (
            'def cost(run, prices):              # same formula as the lab, cached input at one tenth\n'
            '    p = list(prices.get(run.get("model")) or []) + [0, 0]\n'
            '    fresh = max(num(run, "input") - num(run, "cached"), 0)\n'
            '    return (fresh * p[0] + num(run, "cached") * p[0] * 0.10 + num(run, "output") * p[1]) / 1_000_000',

            'OFFLINE = False        # set once verdict.json is read\n\n'
            'def cost(run, prices):              # same formula as the lab, cached input at one tenth\n'
            '    if OFFLINE:\n'
            '        # A local model costs no money, so the offline path measures the same thing\n'
            '        # in tokens. More thinking still costs more, which is the whole point.\n'
            '        return num(run, "input") + num(run, "output")\n'
            '    p = list(prices.get(run.get("model")) or []) + [0, 0]\n'
            '    fresh = max(num(run, "input") - num(run, "cached"), 0)\n'
            '    return (fresh * p[0] + num(run, "cached") * p[0] * 0.10 + num(run, "output") * p[1]) / 1_000_000'
        ),
        (
            'fast, slow = d["fast_model"], d["reasoning_model"]\n'
            'prices = d["prices"] if isinstance(d["prices"], dict) else {}',

            'OFFLINE = d.get("path") == "offline"\n'
            'fast, slow = d["fast_model"], d["reasoning_model"]\n'
            'prices = d["prices"] if isinstance(d["prices"], dict) else {}\n'
            'if OFFLINE:\n'
            '    print("PASS offline path: grading on token counts, not money")'
        ),
        (
            'check(all(isinstance(prices.get(m), list) and len(prices[m]) == 2 for m in (fast, slow)),\n'
            '      "prices has an [input, output] pair for both models")',

            'if OFFLINE:\n'
            '    check(d.get("cost_unit") == "tokens",\n'
            '          \'offline runs set "cost_unit": "tokens", since your model is free\')\n'
            'else:\n'
            '    check(all(isinstance(prices.get(m), list) and len(prices[m]) == 2 for m in (fast, slow)),\n'
            '          "prices has an [input, output] pair for both models")'
        ),
        (
            'check(fast != slow, "fast_model and reasoning_model are two different ids")',

            'check(fast != slow,\n'
            '      "the two runs are labelled differently"\n'
            '      + (" (offline: use one model with two modes, e.g. \'... (direct)\' "\n'
            '         "and \'... (think first)\')" if OFFLINE else ""))'
        ),
    ],
}
