"""
Inline SVG diagrams for the hard concepts.

Built at page-generation time rather than drawn by script, so they print and
work with JavaScript off. Colours come from CSS custom properties, so every
diagram follows the phase hue and both themes automatically.

Case rule for labels (box labels, box subs, txt strings)
--------------------------------------------------------
Lower case, no full stop at the end. Capitalise only what the word forces:

  - acronyms: RAG, MCP, RLHF, RLVR, JSON, GPU, ID, PDF, TPR, TNR
  - product and proper names: LoRA, QLoRA, microVM, Detroit
  - the pronoun I
  - cross references written as "Module N"
  - text quoted verbatim, which keeps whatever case the source had

For emphasis use cls="d-hi" or cls="d-t", never capitals. Titles and captions
are prose sentences and follow ordinary sentence case instead.

Geometry rule
-------------
An arrow stops at the edge of the box it points to, it does not run inside it,
and no label may sit on another label. Both are measurable, so check them by
rendering rather than by eye.
"""


def _wrap(name, vb, body, title, caption):
    """Common shell: arrowhead marker, accessible title, caption."""
    mid = f"ah-{name}"
    return (
        f'<figure class="diagram">'
        f'<svg viewBox="{vb}" role="img" aria-label="{title}" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{title}</title>"
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" class="d-head"/></marker></defs>'
        + body.replace("@A", f"url(#{mid})")
        + "</svg>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def box(x, y, w, h, label, sub=None, cls="d-box", r=6):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>'
    if sub:
        out += (f'<text x="{x+w/2}" y="{y+h/2-4}" text-anchor="middle" class="d-t">{label}</text>'
                f'<text x="{x+w/2}" y="{y+h/2+13}" text-anchor="middle" class="d-s">{sub}</text>')
    else:
        out += f'<text x="{x+w/2}" y="{y+h/2+5}" text-anchor="middle" class="d-t">{label}</text>'
    return out


def arrow(x1, y1, x2, y2, cls="d-arrow"):
    return f'<path d="M{x1} {y1} L{x2} {y2}" class="{cls}" marker-end="@A"/>'


def txt(x, y, s, cls="d-s", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{s}</text>'


D = {}

# ---------------------------------------------------------------- gradient descent
D["gradient"] = _wrap(
    # 278 tall, not 260: the axis caption sits below the axis line at y=250
    # and was being clipped by the viewBox.
    "grad", "0 0 640 278",
    '<path d="M40 40 C 150 250, 300 250, 340 210 C 380 170, 480 60, 600 40" class="d-curve"/>'
    + "".join(
        f'<circle cx="{cx}" cy="{cy}" r="6" fill="hsl(var(--ph))" opacity="{0.3 + 0.15 * i:.2f}"/>'
        for i, (cx, cy) in enumerate([(90, 118), (140, 182), (200, 218), (260, 233)])
    )
    + '<circle cx="330" cy="216" r="9" fill="hsl(var(--ph))"/>'
    + "".join(arrow(a, b, c, d) for a, b, c, d in
              [(97, 128, 130, 172), (148, 192, 190, 212), (208, 224, 250, 231)])
    + txt(110, 100, "you start here, high loss")
    + txt(370, 200, "lowest point, best weights", anchor="start")
    + txt(320, 24, "each arrow is one small step downhill", cls="d-hi")
    + '<path d="M40 250 L620 250" class="d-line"/><path d="M40 34 L40 250" class="d-line"/>'
    + '<text x="22" y="145" text-anchor="middle" class="d-s" transform="rotate(-90 22 145)">loss</text>'
    + txt(330, 268, "every possible setting of the weights"),
    "Gradient descent as a ball rolling downhill into a valley",
    "<b>Gradient descent.</b> The curve is how wrong the model is for each setting of its weights. "
    "You cannot see the whole valley, only the slope right where you stand, so you take a small step "
    "downhill and look again. The step size is the <b>learning rate</b>: too small and you crawl, too "
    "big and you bounce over the bottom.")

# ---------------------------------------------------------------- one neuron
D["neuron"] = _wrap(
    "neuron", "0 0 640 240",
    "".join([
        box(20, 30, 78, 34, "x1 = 2.0", cls="d-box"),
        box(20, 95, 78, 34, "x2 = 3.0", cls="d-box"),
        box(20, 160, 78, 34, "x3 = 1.0", cls="d-box"),
        arrow(98, 47, 250, 100), arrow(98, 112, 250, 112), arrow(98, 177, 250, 124),
        txt(175, 62, "x w1 = 0.5"), txt(175, 105, "x w2 = -1.0"), txt(175, 172, "x w3 = 2.0"),
        box(250, 88, 120, 48, "add it up", "+ bias 0.5", cls="d-fill"),
        arrow(370, 112, 430, 112),
        box(430, 88, 100, 48, "squash it", "activation", cls="d-fill"),
        arrow(530, 112, 560, 112),
        box(560, 88, 70, 48, "0.62", "output"),
        # below the third weight label, which reaches y=175
        txt(310, 196, "2.0(0.5) + 3.0(-1.0) + 1.0(2.0) + 0.5 = 0.5", cls="d-m"),
        txt(310, 217, "then squashed into a number between 0 and 1"),
    ]),
    "How one neuron turns several inputs into one output",
    "<b>One neuron.</b> Multiply each input by its own weight, add them up, add a bias, then squash "
    "the result. Training just means nudging those weights and the bias. That is the entire unit, and "
    "a large model is millions of these arranged in layers.")

# ---------------------------------------------------------------- backprop
D["backprop"] = _wrap(
    "backprop", "0 0 640 210",
    "".join([
        box(30, 40, 100, 44, "input"),
        box(180, 40, 100, 44, "layer 1"),
        box(330, 40, 100, 44, "layer 2"),
        box(480, 40, 130, 44, "guess: 0.9"),
        arrow(130, 62, 178, 62), arrow(280, 62, 328, 62), arrow(430, 62, 478, 62),
        txt(320, 26, "forward: data flows this way", cls="d-hi"),
        box(480, 130, 130, 44, "truth: 0.2", cls="d-warn"),
        txt(545, 112, "loss = how wrong"),
        '<path d="M470 152 L440 152 L440 100 L390 100" class="d-dash" marker-end="@A"/>',
        '<path d="M330 100 L240 100" class="d-dash" marker-end="@A"/>',
        '<path d="M180 100 L100 100" class="d-dash" marker-end="@A"/>',
        txt(300, 196, "backward: blame flows this way, and each weight learns its share", cls="d-hi"),
    ]),
    "Forward pass makes a guess, backpropagation sends the blame backwards",
    "<b>Backpropagation.</b> The network makes a guess, you measure how wrong it was, then you work "
    "backwards asking every weight how much of that error was its fault. Weights that pushed hardest "
    "in the wrong direction get changed most. Nothing mystical is happening: it is the chain rule "
    "from calculus applied layer by layer.")

# ---------------------------------------------------------------- tokens to vectors
D["tokens"] = _wrap(
    "tokens", "0 0 640 250",
    "".join([
        txt(60, 30, "your text", cls="d-hi"),
        box(20, 42, 190, 36, '"unbelievably good"', cls="d-box"),
        arrow(115, 78, 115, 104),
        txt(60, 100, "1. split into tokens", anchor="start"),
        box(20, 110, 60, 32, "un", cls="d-fill"), box(84, 110, 78, 32, "believ", cls="d-fill"),
        box(166, 110, 52, 32, "ably", cls="d-fill"), box(222, 110, 70, 32, " good", cls="d-fill"),
        arrow(155, 142, 155, 168),
        txt(60, 164, "2. look up each ID", anchor="start"),
        txt(50, 190, "463", cls="d-m"), txt(123, 190, "17204", cls="d-m"),
        txt(192, 190, "3016", cls="d-m"), txt(257, 190, "1695", cls="d-m"),
        arrow(300, 190, 350, 190),
        txt(325, 178, "3. embed"),
        box(360, 100, 250, 120, "", cls="d-box"),
        txt(485, 92, "meaning space", cls="d-hi"),
        '<circle cx="410" cy="140" r="5" fill="hsl(var(--ph))"/>',
        '<circle cx="428" cy="152" r="5" fill="hsl(var(--ph))"/>',
        '<circle cx="418" cy="128" r="5" fill="hsl(var(--ph))"/>',
        '<circle cx="560" cy="196" r="5" fill="var(--ink-3)"/>',
        '<circle cx="575" cy="182" r="5" fill="var(--ink-3)"/>',
        txt(432, 176, "good, great, fine", anchor="start", cls="d-s"),
        txt(500, 210, "car, engine", anchor="start", cls="d-s"),
        txt(300, 240, "similar meaning ends up in the same neighbourhood"),
    ]),
    "Text becomes tokens, tokens become IDs, IDs become positions in meaning space",
    "<b>Text to numbers.</b> A model never sees letters. Your text is chopped into <b>tokens</b> "
    "(word pieces), each token is swapped for its row number in a fixed vocabulary, and that row holds "
    "a long list of numbers. Words used in similar ways end up near each other, which is what makes "
    "search by meaning possible.")

# ---------------------------------------------------------------- attention
D["attention"] = _wrap(
    "attn", "0 0 640 260",
    "".join([
        txt(320, 22, "The trophy did not fit in the suitcase because it was too big", cls="d-t"),
        box(258, 36, 46, 28, "it", cls="d-fill", r=4),
        txt(281, 82, "asks:", cls="d-s"),
        box(200, 96, 160, 34, "query: who am I?", cls="d-fill"),
        arrow(200, 113, 130, 113), arrow(360, 113, 430, 113),
        box(20, 96, 110, 34, "trophy", cls="d-box"),
        box(430, 96, 110, 34, "suitcase", cls="d-box"),
        txt(75, 148, "key: a big object"), txt(485, 148, "key: a container"),
        txt(75, 168, "match: 0.62", cls="d-hi"), txt(485, 168, "match: 0.14"),
        '<path d="M75 178 L75 200" class="d-arrow" marker-end="@A"/>',
        '<path d="M485 178 L485 200" class="d-dash" marker-end="@A"/>',
        box(10, 206, 130, 34, "value copied in", cls="d-ok"),
        box(430, 206, 130, 34, "mostly ignored", cls="d-box"),
        txt(320, 232, "so 'it' now carries", cls="d-s"),
        txt(320, 250, "the meaning of 'trophy'", cls="d-hi"),
    ]),
    "How attention lets the word it look back and find trophy",
    "<b>Attention.</b> Every word sends out a <b>query</b> (what am I looking for), every word offers "
    "a <b>key</b> (what I am), and the strength of each match decides how much of that word's "
    "<b>value</b> gets mixed in. Here 'it' matches 'trophy' far more than 'suitcase', so 'it' quietly "
    "becomes trophy-flavoured. That is all attention does, repeated across many heads and layers.")

# ---------------------------------------------------------------- training stages
D["training"] = _wrap(
    "train", "0 0 660 220",
    "".join([
        box(10, 60, 140, 60, "1. pretraining", "guess the next word", cls="d-fill"),
        arrow(150, 90, 178, 90),
        box(178, 60, 140, 60, "2. fine-tuning", "copy good answers", cls="d-fill"),
        arrow(318, 90, 346, 90),
        box(346, 60, 140, 60, "3. RLHF", "people rank pairs", cls="d-fill"),
        arrow(486, 90, 514, 90),
        box(514, 60, 136, 60, "4. RLVR", "a program marks it", cls="d-fill"),
        txt(80, 145, "reads the internet"), txt(80, 162, "knows a lot,"), txt(80, 179, "obeys nothing"),
        txt(248, 145, "learns the shape"), txt(248, 162, "of an answer"),
        txt(416, 145, "learns what people"), txt(416, 162, "prefer to read"),
        txt(582, 145, "learns what is"), txt(582, 162, "actually correct"),
        txt(330, 205, "each stage fixes a different problem, and none of them replaces the one before"),
        txt(330, 36, "raw text in, assistant out", cls="d-hi"),
    ]),
    "The four training stages from raw text to a model that reasons",
    "<b>How a model gets made.</b> Pretraining gives it knowledge but no manners. Fine-tuning teaches "
    "it what an answer looks like. RLHF tunes it toward what people prefer. RLVR, the newest stage, "
    "grades it on tasks where a program can check the answer, like maths and code, which is where "
    "reasoning models come from.")

# ---------------------------------------------------------------- token cost
D["cost"] = _wrap(
    "cost", "0 0 640 230",
    "".join([
        txt(320, 22, "one agent request", cls="d-hi"),
        box(20, 40, 380, 40, "", cls="d-fill"),
        txt(210, 65, "input: 50,000 tokens", cls="d-t"),
        box(420, 40, 60, 40, "", cls="d-box"),
        txt(450, 65, "output", cls="d-s"),
        txt(450, 96, "800", cls="d-m"),
        txt(210, 100, "the documents, history, tool definitions, instructions"),
        txt(320, 140, "you are billed at different rates", cls="d-hi"),
        box(20, 155, 180, 44, "input", "$0.20 per million", cls="d-ok"),
        box(215, 155, 180, 44, "cached input", "$0.02 per million", cls="d-ok"),
        box(410, 155, 210, 44, "output", "$1.20 per million", cls="d-warn"),
        txt(320, 220, "agents are input-heavy, so caching the unchanged part is the biggest lever"),
    ]),
    "An agent request is mostly input tokens, billed at three different rates",
    "<b>Where the money goes.</b> People assume output is what costs, because that is what you read. "
    "For agents it is the opposite: you resend a huge, nearly identical prompt every turn. Cached "
    "input costs about a tenth of fresh input, so keeping the front of your prompt byte-for-byte "
    "identical is usually worth more than switching model.")

# ---------------------------------------------------------------- cache prefix
D["cache"] = _wrap(
    "cache", "0 0 640 210",
    "".join([
        txt(160, 22, "timestamp at the top", cls="d-hi"),
        box(20, 32, 280, 26, "", cls="d-warn", r=3), txt(160, 50, "changes every call", cls="d-s"),
        box(20, 62, 280, 26, "", cls="d-warn", r=3), txt(160, 80, "system prompt: wasted", cls="d-s"),
        box(20, 92, 280, 26, "", cls="d-warn", r=3), txt(160, 110, "documents: wasted", cls="d-s"),
        txt(160, 138, "0% reusable", cls="d-t"),
        txt(480, 22, "timestamp at the bottom", cls="d-hi"),
        box(340, 32, 280, 26, "", cls="d-ok", r=3), txt(480, 50, "system prompt: cached", cls="d-s"),
        box(340, 62, 280, 26, "", cls="d-ok", r=3), txt(480, 80, "documents: cached", cls="d-s"),
        box(340, 92, 280, 26, "", cls="d-warn", r=3), txt(480, 110, "changes every call", cls="d-s"),
        txt(480, 138, "94% reusable", cls="d-t"),
        txt(320, 178, "the cache keeps everything up to the first changed character, and throws away the rest"),
        txt(320, 198, "so put anything that moves at the end of your prompt"),
    ]),
    "One moving value at the top of a prompt destroys the whole cache",
    "<b>Prefix caching.</b> The saved work covers your prompt from the beginning up to the first "
    "character that changed. Put a timestamp on line one and everything after it has to be recomputed, "
    "at full price, every single call. Move it to the end and the expensive stable part is free.")

# ---------------------------------------------------------------- context layers + rot
D["context"] = _wrap(
    "ctx", "0 0 640 250",
    "".join([
        txt(160, 20, "what fills the window", cls="d-hi"),
        box(20, 30, 280, 24, "system instructions", cls="d-fill", r=3),
        box(20, 58, 280, 24, "tool definitions", cls="d-fill", r=3),
        box(20, 86, 280, 24, "conversation history", cls="d-fill", r=3),
        box(20, 114, 280, 32, "retrieved documents", cls="d-fill", r=3),
        box(20, 150, 280, 24, "memory from last time", cls="d-fill", r=3),
        box(20, 178, 280, 24, "tool results arriving now", cls="d-fill", r=3),
        txt(160, 222, "they all compete for one fixed budget"),
        txt(480, 20, "what actually happens", cls="d-hi"),
        '<path d="M360 190 L620 190" class="d-line"/><path d="M360 40 L360 190" class="d-line"/>',
        '<path d="M360 60 C 430 62, 470 78, 510 110 C 550 145, 580 170, 615 182" class="d-curve"/>',
        '<path d="M360 60 L615 60" class="d-dash"/>',
        txt(500, 52, "what you hoped for"),
        txt(560, 130, "what you measure", cls="d-hi"),
        txt(365, 206, "small", anchor="start"), txt(615, 206, "1M tokens", anchor="end"),
        txt(348, 120, "accuracy", anchor="middle"),
        txt(490, 240, "more input, worse answers, long before the limit"),
    ]),
    "Six things compete for one context budget, and accuracy falls as it fills",
    "<b>Context rot.</b> The advertised window is not the usable window. Chroma tested eighteen models "
    "in July 2025 and found accuracy drops as input grows, unevenly, well before the stated maximum. "
    "Filling a million tokens because you can is an expensive way to get a worse answer, so decide "
    "what earns its place.")

# ---------------------------------------------------------------- tool call loop
D["toolcall"] = _wrap(
    "tool", "0 0 640 260",
    "".join([
        box(30, 20, 150, 34, "your program", cls="d-fill"),
        box(430, 20, 150, 34, "the model", cls="d-fill"),
        '<path d="M105 54 L105 240" class="d-dash"/><path d="M505 54 L505 240" class="d-dash"/>',
        arrow(110, 80, 500, 80), txt(305, 72, "question + list of tools you offer"),
        arrow(500, 118, 110, 118), txt(305, 110, "I would like to call get_weather(city=Detroit)", cls="d-hi"),
        '<path d="M105 140 L60 140 L60 172 L105 172" class="d-arrow" marker-end="@A"/>',
        txt(30, 158, "you run it", anchor="start", cls="d-hi"),
        arrow(110, 200, 500, 200), txt(305, 192, "here is the result: 21 degrees, clear"),
        arrow(500, 238, 110, 238), txt(305, 230, "it is 21 degrees and clear in Detroit"),
    ]),
    "The four messages that make up one tool call",
    "<b>Tool use is not magic.</b> The model cannot run anything. It writes a message saying which "
    "function it would like called and with what arguments. <b>Your</b> code decides whether to run it, "
    "runs it, and hands the result back. If you never write that middle step, nothing happens.")

# ---------------------------------------------------------------- RAG pipeline
D["rag"] = _wrap(
    "rag", "0 0 660 280",
    "".join([
        txt(120, 20, "once, when you add documents", cls="d-hi"),
        box(20, 30, 90, 36, "your PDFs"), arrow(110, 48, 138, 48),
        box(138, 30, 90, 36, "chunks"), arrow(228, 48, 256, 48),
        box(256, 30, 100, 36, "embeddings"), arrow(356, 48, 384, 48),
        box(384, 30, 100, 36, "index file", cls="d-ok"),
        # 82 clears the search boxes at y=92 by the same gap the first row uses
        txt(140, 82, "every time you ask a question", cls="d-hi"),
        box(20, 112, 90, 36, "question"),
        arrow(110, 122, 150, 112), arrow(110, 140, 150, 152),
        box(150, 92, 130, 36, "keyword search", cls="d-fill"),
        box(150, 138, 130, 36, "meaning search", cls="d-fill"),
        arrow(280, 110, 320, 128), arrow(280, 156, 320, 138),
        box(320, 114, 90, 36, "merge", cls="d-fill"),
        arrow(410, 132, 442, 132),
        box(442, 114, 96, 36, "rerank", cls="d-fill"),
        arrow(538, 132, 566, 132),
        box(566, 114, 84, 36, "answer", cls="d-ok"),
        txt(215, 196, "two searches because each fails where the other works", cls="d-s"),
        txt(215, 214, "keyword finds exact words, meaning finds paraphrases", cls="d-s"),
        txt(490, 196, "rerank re-reads the shortlist properly", cls="d-s"),
        txt(490, 214, "and fixes the order", cls="d-s"),
        txt(330, 256, "the index is built once, the bottom row runs on every question"),
    ]),
    "The RAG pipeline: build an index once, then search two ways and rerank",
    "<b>RAG.</b> You cannot paste a whole filing cabinet into a prompt, so you cut documents into "
    "chunks and store them twice: once for exact words, once for meaning. A question searches both, "
    "the two ranked lists get merged, and a slower model re-reads the shortlist to fix the order. "
    "Skipping the rerank is the most common reason a RAG system feels almost right.")

# ---------------------------------------------------------------- LoRA
D["lora"] = _wrap(
    "lora", "0 0 620 240",
    "".join([
        box(40, 40, 160, 160, "", cls="d-box"),
        txt(120, 115, "the original model", cls="d-t"), txt(120, 136, "frozen, unchanged", cls="d-s"),
        txt(120, 26, "7,000,000,000 numbers"),
        txt(225, 125, "+", cls="d-t"),
        box(250, 40, 40, 160, "", cls="d-fill"), txt(270, 216, "A", cls="d-hi"),
        txt(305, 125, "x", cls="d-t"),
        box(320, 40, 160, 40, "", cls="d-fill"), txt(400, 96, "B", cls="d-hi"),
        txt(365, 26, "8,000,000 numbers"),
        txt(365, 150, "the small patch you train", cls="d-s"),
        # in the gap between B and the result box, on the same line as + and x
        txt(490, 125, "=", cls="d-t"),
        box(500, 40, 100, 160, "", cls="d-ok"),
        txt(550, 115, "your model", cls="d-t"), txt(550, 136, "same size", cls="d-s"),
        txt(310, 232, "you train about one thousandth of the numbers, so it fits on a free GPU"),
    ]),
    "LoRA trains two small matrices instead of the whole model",
    "<b>LoRA.</b> Retraining every number in a large model needs hardware you do not have. Instead you "
    "freeze the original and learn two thin matrices whose product is the same shape as the layer you "
    "wanted to change. Add that product back and you have a tuned model, having trained roughly a "
    "thousandth as many numbers. <b>QLoRA</b> goes further and squashes the frozen part to 4 bits so "
    "it fits in less memory.")

# ---------------------------------------------------------------- workflow vs agent
D["agent"] = _wrap(
    "agent", "0 0 640 250",
    "".join([
        txt(150, 20, "a workflow: you choose the path", cls="d-hi"),
        box(20, 34, 80, 34, "step 1"), arrow(100, 51, 128, 51),
        box(128, 34, 80, 34, "step 2"), arrow(208, 51, 236, 51),
        box(236, 34, 80, 34, "step 3"),
        txt(168, 90, "predictable, cheap, easy to debug, and usually enough"),
        txt(320, 130, "an agent: the model chooses the path", cls="d-hi"),
        box(250, 150, 140, 40, "think", cls="d-fill"),
        box(430, 150, 140, 40, "act: call a tool", cls="d-fill"),
        box(340, 210, 140, 34, "observe the result", cls="d-fill"),
        arrow(390, 170, 428, 170),
        '<path d="M500 190 L500 214 L482 214" class="d-arrow" marker-end="@A"/>',
        '<path d="M338 214 L300 214 L300 190" class="d-arrow" marker-end="@A"/>',
        box(60, 160, 150, 60, "stopping rule", "or it runs forever", cls="d-warn"),
        arrow(212, 190, 246, 180),
    ]),
    "A workflow follows a fixed path, an agent decides its own and loops",
    "<b>Workflow or agent?</b> A workflow is steps you wrote in order. An agent is a loop where the "
    "model decides what to do next, so it can handle problems you could not plan for. That freedom is "
    "the cost as well as the point: it is slower, dearer, harder to debug, and it needs a hard stopping "
    "rule. Reach for a workflow first.")

# ---------------------------------------------------------------- MCP N x M
D["mcp"] = _wrap(
    "mcp", "0 0 640 250",
    "".join([
        txt(150, 20, "before: every app wires up every tool", cls="d-hi"),
        box(20, 36, 70, 26, "app 1", r=4), box(20, 72, 70, 26, "app 2", r=4), box(20, 108, 70, 26, "app 3", r=4),
        box(210, 36, 70, 26, "files", r=4), box(210, 72, 70, 26, "email", r=4), box(210, 108, 70, 26, "search", r=4),
    ] + [
        f'<path d="M90 {49+36*i} L210 {49+36*j}" class="d-dash"/>' for i in range(3) for j in range(3)
    ] + [
        txt(150, 160, "3 apps x 3 tools = 9 pieces of glue"),
        txt(150, 178, "add one tool, write three more", cls="d-s"),
        txt(470, 20, "after: everyone speaks one format", cls="d-hi"),
        box(350, 36, 70, 26, "app 1", r=4), box(350, 72, 70, 26, "app 2", r=4), box(350, 108, 70, 26, "app 3", r=4),
        box(470, 60, 60, 50, "MCP", cls="d-fill"),
        box(570, 36, 60, 26, "files", r=4), box(570, 72, 60, 26, "email", r=4), box(570, 108, 60, 26, "search", r=4),
    ] + [
        f'<path d="M420 {49+36*i} L470 85" class="d-line"/>' for i in range(3)
    ] + [
        f'<path d="M530 85 L570 {49+36*i}" class="d-line"/>' for i in range(3)
    ] + [
        txt(490, 160, "3 + 3 = 6 pieces"),
        txt(490, 178, "add one tool, write one server", cls="d-s"),
        txt(320, 222, "the work stops multiplying and starts adding, which is the whole point of a standard"),
    ]),
    "MCP turns N times M integrations into N plus M",
    "<b>Why MCP exists.</b> Without a standard, every app needs custom code for every tool, and the "
    "work grows by multiplying. With one agreed format, you wrap each tool once as a server and each "
    "app learns the format once. It is the same idea as a plug socket: the socket does not care what "
    "you plug in.")

# ---------------------------------------------------------------- lethal trifecta
D["trifecta"] = _wrap(
    "tri", "0 0 620 280",
    "".join([
        '<circle cx="230" cy="110" r="92" fill="hsl(var(--ph) / .16)" stroke="hsl(var(--ph))" stroke-width="1.5"/>',
        '<circle cx="330" cy="110" r="92" fill="hsl(var(--ph) / .16)" stroke="hsl(var(--ph))" stroke-width="1.5"/>',
        '<circle cx="280" cy="190" r="92" fill="hsl(var(--ph) / .16)" stroke="hsl(var(--ph))" stroke-width="1.5"/>',
        txt(178, 76, "private", cls="d-t"), txt(178, 94, "data", cls="d-t"),
        txt(384, 76, "untrusted", cls="d-t"), txt(384, 94, "content", cls="d-t"),
        txt(280, 236, "a way to", cls="d-t"), txt(280, 254, "send data out", cls="d-t"),
        '<circle cx="280" cy="140" r="30" fill="var(--bad-bg)" stroke="var(--bad)" stroke-width="2"/>',
        f'<text x="280" y="136" text-anchor="middle" style="fill:var(--bad);font:700 12px var(--ui)">DATA</text>',
        f'<text x="280" y="151" text-anchor="middle" style="fill:var(--bad);font:700 12px var(--ui)">LEAK</text>',
        txt(500, 130, "any TWO of these", anchor="start", cls="d-t"),
        txt(500, 150, "is survivable", anchor="start", cls="d-s"),
        txt(500, 176, "all THREE and you", anchor="start", cls="d-t"),
        txt(500, 196, "have a hole you", anchor="start", cls="d-s"),
        txt(500, 214, "cannot prompt away", anchor="start", cls="d-s"),
    ]),
    "The lethal trifecta: private data, untrusted content and a way out",
    "<b>The lethal trifecta</b> (Simon Willison, June 2025). Your agent reads private data, it also "
    "reads text a stranger wrote, and it can send things outward. Remove any one leg and the attack "
    "dies. You remove it <b>in code</b>, not in the prompt, because the model has no reliable way to "
    "tell your instructions from an attacker's: both are just text in the same window.")

# ---------------------------------------------------------------- trace waterfall
D["trace"] = _wrap(
    "trace", "0 0 640 230",
    "".join([
        txt(320, 20, "one agent run, drawn as spans over time", cls="d-hi"),
        box(20, 34, 600, 24, "", cls="d-fill", r=3), txt(320, 51, "whole run  4.2s", cls="d-s"),
        box(40, 64, 180, 22, "", cls="d-fill", r=3), txt(130, 80, "think  0.9s", cls="d-s"),
        box(230, 64, 150, 22, "", cls="d-fill", r=3), txt(305, 80, "search_docs  1.1s", cls="d-s"),
        box(250, 92, 110, 22, "", cls="d-warn", r=3), txt(305, 108, "returned 0 rows", cls="d-s"),
        box(390, 64, 120, 22, "", cls="d-fill", r=3), txt(450, 80, "think  0.7s", cls="d-s"),
        box(520, 64, 100, 22, "", cls="d-fill", r=3), txt(570, 80, "answer  1.5s", cls="d-s"),
        txt(305, 136, "here is the real bug", cls="d-hi"),
        '<path d="M305 118 L305 128" class="d-arrow" marker-end="@A"/>',
        txt(320, 174, "the final answer looked confident and plausible"),
        txt(320, 194, "the failure was three steps earlier, and only the trace shows it"),
    ]),
    "A trace shows each step of a run, and where it really went wrong",
    "<b>Why tracing matters.</b> Agents rarely fail at the last sentence. They fail somewhere in the "
    "middle: a search returns nothing, a tool gets the wrong argument, a step repeats. The final answer "
    "still reads well, so scoring only the answer tells you almost nothing. A <b>trace</b> records "
    "every step with its timing so you can see where it actually broke.")

# ---------------------------------------------------------------- multi-agent
D["multiagent"] = _wrap(
    "multi", "0 0 640 230",
    "".join([
        box(250, 20, 140, 40, "supervisor", cls="d-fill"),
        arrow(290, 60, 200, 100), arrow(320, 60, 320, 100), arrow(350, 60, 440, 100),
        box(130, 100, 140, 40, "researcher"),
        box(250, 100, 140, 40, "writer"),
        box(390, 100, 140, 40, "checker"),
        box(190, 172, 260, 36, "shared memory (a file or a table)", cls="d-ok"),
        '<path d="M200 140 L250 170" class="d-line"/><path d="M320 140 L320 170" class="d-line"/>'
        '<path d="M440 140 L400 170" class="d-line"/>',
        txt(60, 118, "each one", anchor="start", cls="d-s"),
        txt(60, 134, "has its own", anchor="start", cls="d-s"),
        txt(60, 150, "context window", anchor="start", cls="d-s"),
        txt(320, 224, "everything one agent knows must be written down, or the next one never learns it"),
    ]),
    "A supervisor hands work to specialists that share written memory",
    "<b>Multi-agent systems.</b> Agents do not share a brain. Anything the researcher was unsure about "
    "is lost unless it writes that doubt down, which is how a team of agents produces a confident "
    "answer nobody actually checked. Research into multi-agent failures found most of them are design "
    "faults, not model faults, so start with one agent and split only when you can say why.")

# ---------------------------------------------------------------- prompt anatomy
D["prompt"] = _wrap(
    "prompt", "0 0 620 260",
    "".join([
        txt(310, 20, "one request, four jobs", cls="d-hi"),
        box(30, 32, 560, 40, "", cls="d-fill"),
        txt(310, 51, "system: who the model is and the rules it must follow", cls="d-s"),
        txt(310, 66, "stays identical every call, so it stays cached", cls="d-s"),
        box(30, 80, 560, 44, "", cls="d-fill"),
        txt(310, 99, "examples: two or three finished answers in the shape you want", cls="d-s"),
        txt(310, 115, "showing beats describing, almost every time", cls="d-s"),
        box(30, 132, 560, 52, "", cls="d-warn"),
        txt(310, 150, "the data, inside a fence", cls="d-t"),
        txt(310, 168, "&lt;document&gt; ... their text goes here ... &lt;/document&gt;", cls="d-m"),
        box(30, 192, 560, 34, "", cls="d-fill"),
        txt(310, 213, "your question, and exactly what format to answer in", cls="d-s"),
        txt(310, 250, "the fence matters: anything inside it is data to look at, never orders to follow"),
    ]),
    "The four parts of a well built prompt, with the untrusted data fenced off",
    "<b>Prompt anatomy.</b> Rules first and unchanging, then examples, then the data clearly fenced, "
    "then the question. The fence is the important habit: it tells the model where somebody else's "
    "text begins. It is not real security, because a determined attacker can still write "
    "instructions inside it, but it removes the accidental cases and it makes your intent explicit.")

# ---------------------------------------------------------------- capstone
D["capstone"] = _wrap(
    "cap", "0 0 660 270",
    "".join([
        box(20, 100, 96, 44, "your question", cls="d-box"),
        arrow(116, 122, 148, 122),
        box(148, 92, 120, 60, "the agent loop", "Module 12", cls="d-fill"),
        arrow(268, 108, 316, 74), arrow(268, 136, 316, 170),
        box(316, 52, 130, 44, "MCP server", cls="d-fill"),
        txt(381, 40, "Module 13", cls="d-s"),
        box(316, 150, 130, 44, "your documents", cls="d-fill"),
        txt(381, 212, "Module 10: chunks, hybrid search, rerank", cls="d-s"),
        '<path d="M381 96 L381 148" class="d-line"/>',
        arrow(446, 122, 486, 122),
        box(486, 92, 150, 60, "answer + citations", "every claim traceable", cls="d-ok"),
        box(20, 190, 240, 40, "cost ceiling that can say no", cls="d-warn"),
        '<path d="M208 190 L208 154" class="d-dash"/>',
        box(486, 190, 150, 40, "traces and evals", cls="d-ok"),
        txt(561, 244, "Module 16", cls="d-s"),
        '<path d="M561 190 L561 154" class="d-dash"/>',
        txt(330, 24, "nothing here is new: it is the parts you already built, wired together", cls="d-hi"),
    ]),
    "The capstone wires together the parts built in earlier modules",
    "<b>The capstone.</b> By Module 18 you are not starting from an empty file. The retrieval came "
    "from Module 10, the MCP server from Module 13, the loop from Module 12 or 14, and the tracing and "
    "evaluation from Module 16. The new work is the wiring, the citations, and the ceiling that stops "
    "it spending your money forever.")


# second batch, kept in its own file to keep this one readable
try:
    from diagrams2 import D2
    D.update(D2)
except ImportError:
    pass
