"""Second batch of diagrams. Imported and merged by diagrams.py consumers."""
from diagrams import _wrap, box, arrow, txt

D2 = {}

# ---------------------------------------------------------------- overfitting
D2["overfit"] = _wrap(
    "overfit", "0 0 640 250",
    "".join([
        txt(160, 22, "learned the pattern", cls="d-hi"),
        '<path d="M30 200 L290 200" class="d-line"/><path d="M30 40 L30 200" class="d-line"/>',
        '<path d="M40 180 C 110 120, 180 90, 280 60" class="d-curve"/>',
    ] + [
        f'<circle cx="{x}" cy="{y}" r="4" fill="var(--ink-3)"/>'
        for x, y in [(55, 170), (85, 150), (110, 128), (140, 124),
                     (170, 100), (200, 92), (230, 72), (262, 66)]
    ] + [
        txt(160, 222, "new data lands near the line too"),
        txt(480, 22, "memorised the noise", cls="d-hi"),
        '<path d="M350 200 L610 200" class="d-line"/><path d="M350 40 L350 200" class="d-line"/>',
        '<path d="M360 180 C 372 168, 380 148, 405 150 C 425 152, 428 122, 460 124 '
        'C 490 126, 486 96, 520 92 C 548 89, 545 68, 582 66" class="d-curve"/>',
    ] + [
        f'<circle cx="{x}" cy="{y}" r="4" fill="var(--ink-3)"/>'
        for x, y in [(375, 170), (405, 150), (430, 128), (460, 124),
                     (490, 100), (520, 92), (550, 72), (582, 66)]
    ] + [
        '<circle cx="470" cy="150" r="5" fill="var(--bad)"/>',
        '<circle cx="540" cy="118" r="5" fill="var(--bad)"/>',
        txt(480, 222, "new data misses badly", cls="d-s"),
        txt(320, 244, "a perfect score on data you trained on tells you nothing"),
    ]),
    "A model that learned the pattern versus one that memorised the noise",
    "<b>Overfitting.</b> Both lines pass through the training dots. The right one bends to hit every "
    "single point, so it memorised the noise rather than the pattern, and the red dots it has never "
    "seen fall nowhere near it. This is why you hold data back. You will do the same thing to prompts "
    "later: tune one until it passes your five examples, ship it, watch it fail on the sixth.")

# ---------------------------------------------------------------- cosine similarity
D2["cosine"] = _wrap(
    "cos", "0 0 620 250",
    "".join([
        '<path d="M60 210 L300 210" class="d-line"/><path d="M60 210 L60 40" class="d-line"/>',
        '<path d="M60 210 L250 70" class="d-arrow" marker-end="@A"/>',
        '<path d="M60 210 L268 96" class="d-arrow" marker-end="@A"/>',
        '<path d="M110 175 A 62 62 0 0 0 122 158" class="d-curve"/>',
        txt(148, 182, "small angle", anchor="start", cls="d-hi"),
        txt(262, 60, "cat", cls="d-t"), txt(292, 96, "dog", cls="d-t"),
        txt(180, 240, "similarity 0.94", cls="d-t"),
        '<path d="M360 210 L600 210" class="d-line"/><path d="M360 210 L360 40" class="d-line"/>',
        '<path d="M360 210 L550 70" class="d-arrow" marker-end="@A"/>',
        '<path d="M360 210 L580 198" class="d-arrow" marker-end="@A"/>',
        '<path d="M410 175 A 62 62 0 0 0 421 202" class="d-curve"/>',
        txt(448, 172, "wide angle", anchor="start", cls="d-hi"),
        txt(556, 60, "cat", cls="d-t"), txt(592, 188, "invoice", cls="d-t"),
        txt(480, 240, "similarity 0.11", cls="d-t"),
        txt(310, 26, "similarity is the angle between two arrows, not their length", cls="d-hi"),
    ]),
    "Cosine similarity measures the angle between two vectors, not their length",
    "<b>Cosine similarity.</b> Each piece of text becomes an arrow from the origin. What matters is "
    "the <b>angle</b> between two arrows, not how long they are, so a short sentence and a long "
    "paragraph about the same thing still score highly. The number runs from 1 (same direction) "
    "through 0 (unrelated) to -1 (opposite).")

# ---------------------------------------------------------------- json schema
D2["schema"] = _wrap(
    "schema", "0 0 640 250",
    "".join([
        txt(150, 22, "asking nicely", cls="d-hi"),
        box(20, 34, 260, 40, "", cls="d-box"),
        txt(150, 58, "Reply with JSON please", cls="d-m"),
        arrow(150, 74, 150, 100),
        box(20, 104, 260, 74, "", cls="d-warn"),
        txt(150, 126, "Sure! Here is the JSON:", cls="d-m"),
        txt(150, 146, "```json", cls="d-m"),
        txt(150, 166, '{"city": "Detroit"}', cls="d-m"),
        txt(150, 202, "json.loads() raises an error", cls="d-s"),
        txt(490, 22, "declaring a schema", cls="d-hi"),
        box(360, 34, 260, 40, "", cls="d-box"),
        txt(490, 52, "schema: {city: string,", cls="d-m"),
        txt(490, 68, "temp_c: integer}", cls="d-m"),
        arrow(490, 74, 490, 100),
        box(360, 104, 260, 74, "", cls="d-ok"),
        txt(490, 136, '{"city": "Detroit",', cls="d-m"),
        txt(490, 156, ' "temp_c": 21}', cls="d-m"),
        txt(490, 202, "parses every time", cls="d-s"),
        txt(320, 238, "the schema is applied while the text is generated, so the wrong shape never comes out"),
    ]),
    "Asking for JSON in words versus declaring a schema",
    "<b>Structured output.</b> Asking politely gives you JSON most of the time, and the times it does "
    "not are the ones that wake you up. A declared schema is different: the rules are applied as each "
    "token is chosen, so a reply that breaks the shape is never produced. Declare the shape, do not "
    "request it in prose.")

# ---------------------------------------------------------------- framework layers
D2["framework"] = _wrap(
    "fw", "0 0 620 250",
    "".join([
        txt(155, 22, "you wrote this", cls="d-hi"),
        box(30, 34, 250, 30, "your tools", cls="d-fill", r=4),
        box(30, 68, 250, 30, "your prompt", cls="d-fill", r=4),
        box(30, 102, 250, 30, "the while loop", cls="d-fill", r=4),
        box(30, 136, 250, 30, "your retry and stop rules", cls="d-fill", r=4),
        txt(155, 194, "about 100 lines", cls="d-t"),
        txt(155, 214, "you can read all of it", cls="d-s"),
        txt(465, 22, "a framework", cls="d-hi"),
        box(340, 34, 250, 30, "your tools", cls="d-fill", r=4),
        box(340, 68, 250, 30, "prompt you did not write", cls="d-warn", r=4),
        box(340, 102, 250, 30, "loop, state, checkpoints", cls="d-ok", r=4),
        box(340, 136, 250, 30, "retries, pause for a human", cls="d-ok", r=4),
        txt(465, 194, "green is what you gain", cls="d-t"),
        txt(465, 214, "orange is what you can no longer see", cls="d-s"),
        txt(310, 244, "adopt one when you can name the pain it removes, not before"),
    ]),
    "What a framework gives you, and what it hides",
    "<b>Frameworks.</b> The green parts are real work you would otherwise write: saving state so a "
    "run can resume, retries, pausing for a human. The orange part is the cost. The prompt actually "
    "sent is now assembled by somebody else's code, so when the model behaves oddly you are debugging "
    "text you never wrote. Write the loop by hand first, then decide.")

# ---------------------------------------------------------------- judge calibration
D2["confusion"] = _wrap(
    "conf", "0 0 640 265",
    "".join([
        txt(300, 22, "100 answers you labelled by hand, then judged by your evaluator", cls="d-hi"),
        txt(195, 54, "judge said PASS", cls="d-s"), txt(335, 54, "judge said FAIL", cls="d-s"),
        txt(74, 100, "really good", cls="d-s"), txt(74, 160, "really bad", cls="d-s"),
        box(130, 66, 130, 56, "", cls="d-ok"),
        txt(195, 90, "45", cls="d-t"), txt(195, 110, "correctly passed", cls="d-s"),
        box(270, 66, 130, 56, "", cls="d-warn"),
        txt(335, 90, "5", cls="d-t"), txt(335, 110, "wrongly failed", cls="d-s"),
        box(130, 126, 130, 56, "", cls="d-warn"),
        txt(195, 150, "20", cls="d-t"), txt(195, 170, "wrongly passed", cls="d-s"),
        box(270, 126, 130, 56, "", cls="d-ok"),
        txt(335, 150, "30", cls="d-t"), txt(335, 170, "correctly failed", cls="d-s"),
        txt(530, 82, "agreement = 75%", cls="d-t"),
        txt(530, 102, "sounds acceptable", cls="d-s"),
        txt(530, 142, "TPR = 45/50 = 90%", cls="d-t"),
        txt(530, 166, "TNR = 30/50 = 60%", cls="d-t"),
        txt(530, 190, "it waves bad work through", cls="d-s"),
        txt(320, 232, "one number hid the problem, two numbers found it"),
        txt(320, 252, "TPR is how often it spots good work, TNR how often it catches bad work"),
    ]),
    "Why a single agreement percentage hides a broken judge",
    "<b>Calibrating a judge.</b> This judge agrees with you 75% of the time, which reads as passable. "
    "Split it and the picture changes: it recognises good answers 90% of the time but catches only "
    "60% of bad ones, so a fifth of your failures sail through. Report the two rates separately, or "
    "use Cohen's kappa, which corrects for the agreement you would get by chance.")

# ---------------------------------------------------------------- context failures
D2["failures"] = _wrap(
    "fail", "0 0 640 240",
    "".join([
        box(20, 34, 290, 84, "", cls="d-box"),
        txt(165, 58, "poisoning", cls="d-t"),
        txt(165, 80, "a wrong fact gets into the context", cls="d-s"),
        txt(165, 100, "and is repeated as true from then on", cls="d-s"),
        box(330, 34, 290, 84, "", cls="d-box"),
        txt(475, 58, "distraction", cls="d-t"),
        txt(475, 80, "so much history piles up that the model", cls="d-s"),
        txt(475, 100, "repeats it instead of thinking", cls="d-s"),
        box(20, 128, 290, 84, "", cls="d-box"),
        txt(165, 152, "confusion", cls="d-t"),
        txt(165, 174, "irrelevant material is present,", cls="d-s"),
        txt(165, 194, "so the model uses it anyway", cls="d-s"),
        box(330, 128, 290, 84, "", cls="d-box"),
        txt(475, 152, "clash", cls="d-t"),
        txt(475, 174, "two parts of the context contradict,", cls="d-s"),
        txt(475, 194, "and the later one usually wins", cls="d-s"),
        txt(320, 232, "all four look the same from outside: a confident answer that is wrong"),
    ]),
    "The four ways a context window goes wrong",
    "<b>Four failure modes</b> (Drew Breunig, June 2025). They are not the same bug and the fixes "
    "differ: poisoning needs validation where data enters, distraction needs compaction, confusion "
    "needs tighter retrieval, clash needs an explicit order of precedence. Naming which one you have "
    "is most of the work.")

# ---------------------------------------------------------------- sandbox tiers
D2["sandbox"] = _wrap(
    "sbx", "0 0 620 250",
    "".join([
        box(30, 40, 170, 60, "no isolation", "your own machine", cls="d-warn"),
        txt(115, 120, "fine for code you wrote", cls="d-s"),
        txt(115, 140, "never for code a model wrote", cls="d-s"),
        box(225, 40, 170, 60, "container", "shared kernel", cls="d-box"),
        txt(310, 120, "stops accidents", cls="d-s"),
        txt(310, 140, "not a determined attacker", cls="d-s"),
        box(420, 40, 170, 60, "microVM", "its own kernel", cls="d-ok"),
        txt(505, 120, "boots in about 125ms", cls="d-s"),
        txt(505, 140, "use this for generated code", cls="d-s"),
        arrow(202, 70, 221, 70), arrow(397, 70, 416, 70),
        txt(310, 178, "and isolating the compute is only half of it", cls="d-hi"),
        box(120, 192, 380, 34, "deny all outbound network, then allow a named list", cls="d-ok"),
        txt(310, 244, "isolation stops it running wild, network rules stop it phoning home"),
    ]),
    "Three levels of sandboxing, and why network rules matter as much",
    "<b>Sandboxing.</b> If your agent runs code a model wrote, a plain container is not enough, "
    "because everything inside shares one kernel with the host. A microVM has its own. Either way, "
    "isolating compute is half the job: block outbound network by default and allow a specific list, "
    "or a sandboxed process can still send your data out.")
