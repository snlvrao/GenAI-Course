"""Generate modules/mNN-slug.html from authored content + data.js index."""
import ast, builtins, io, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.dirname(os.path.abspath(__file__))

# --- load the module index -------------------------------------------------
src = io.open(os.path.join(ROOT, "assets", "data.js"), encoding="utf-8").read()
INDEX = json.loads(src[src.index("=") + 1:].rstrip().rstrip(";"))
BY_N = {m["n"]: m for m in INDEX["modules"]}

# --- load authored content -------------------------------------------------
CONTENT = json.load(io.open(os.path.join(SP, "module_content.json"), encoding="utf-8"))
BY_MOD = {c["module"]: c for c in CONTENT}

import diagrams as DG
import pathaware as PA


def path_notes(n):
    """Blocks shown only on the paths where this lab actually differs."""
    out = []
    for paths, kind, title, html in PA.NOTES.get(n, []):
        cls = {"warn": "note note--warn", "gap": "note note--gap"}.get(kind, "note")
        out.append(f'<div class="{cls}" data-path="{" ".join(paths)}" hidden>'
                   f'<div class="note__t">{esc(title)}</div><p>{html}</p></div>')
    return "".join(out)

# Which diagram goes after which concept. Keyed by module, then by the
# zero-based index of the concept it explains, so the picture lands right where
# the words need it rather than in a gallery at the bottom.
# Matched against the concept heading, lowercased, so a reworded heading still
# finds its picture. Falls back to the given index if nothing matches.
DIAGRAM_AT = {
    1:  [(["gradient", "downhill", "dials"], "gradient", 4),
         (["overfit", "hidden", "held back", "memoris"], "overfit", 6)],
    2:  [(["neuron", "one neuron", "computes"], "neuron", 0),
         (["backprop", "blame", "backwards"], "backprop", 5)],
    3:  [(["id", "row number", "vocabulary"], "tokens", 2),
         (["cosine", "angle", "similar"], "cosine", 5)],
    4:  [(["query", "key", "value"], "attention", 1)],
    5:  [(["pretrain", "guessing game"], "training", 0)],
    6:  [(["pay twice", "rate", "price", "cost"], "cost", 2),
         (["cach", "prefix"], "cache", 3)],
    7:  [(["fence", "around data", "delimit"], "prompt", 3)],
    8:  [(["layer", "budget", "six"], "context", 1),
         (["four ways", "goes wrong", "failure"], "failures", 3)],
    9:  [(["four steps", "tool call", "steps of"], "toolcall", 1),
         (["json", "trust", "shape", "forced"], "schema", 4)],
    10: [(["both searches", "merg", "hybrid"], "rag", 4)],
    11: [(["lora", "small patch"], "lora", 5)],
    12: [(["workflow", "no framework", "start with"], "agent", 1)],
    13: [(["why mcp", "exists"], "mcp", 0)],
    14: [(["framework is a loop", "framework", "did not write"], "framework", 0)],
    15: [(["supervisor"], "multiagent", 1)],
    16: [(["trace", "span"], "trace", 1),
         (["calibrat", "judge", "measures nothing", "bias"], "confusion", 4)],
    17: [(["trifecta"], "trifecta", 1),
         (["sandbox"], "sandbox", 4)],
    18: [(["workflow first", "agent only", "unknown step"], "capstone", 1)],
}


def diagram_slots(n, concepts):
    """Return {concept_index: diagram_name} for this module."""
    out = {}
    terms = [c["term"].lower() for c in concepts]
    for keys, name, fallback in DIAGRAM_AT.get(n, []):
        idx = None
        for i, t in enumerate(terms):
            if any(k in t for k in keys) and i not in out:
                idx = i
                break
        if idx is None:
            idx = fallback if fallback < len(terms) else len(terms) - 1
            while idx in out and idx < len(terms) - 1:
                idx += 1
        out[idx] = name
    return out

WIDGET_TITLE = {
    "tokenizer": "See how your text gets chopped up",
    "embeddings": "Play with meaning as numbers",
    "attention": "Watch attention pick the right word",
    "context": "Fill a context window and watch it rot",
    "toolcall": "Step through a real tool call",
    "chunking": "Chop text badly, on purpose",
    "agentloop": "Walk an agent loop one step at a time",
    "mcpwire": "Read the actual MCP messages",
    "chooser": "Answer four questions, get a framework",
    "cost": "Work out what your agent costs",
    "cacheprefix": "Break your own cache and watch it cost you",
    "injection": "Attack an agent and see what lands",
    "predictor": "Pick two numbers and fit a line",
    "neuron": "Set three numbers and make a neuron fire",
    "modelsize": "Choose how big your own model is",
}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def rich(s):
    """Escape, then re-allow the two inline tags authors may use."""
    out = esc(s)
    for tag in ("strong", "code", "em", "b"):
        out = out.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return out


# --- syntax colouring -------------------------------------------------------
#
# Painted here, at generation time, not in the browser. Three reasons, all of
# them project rules rather than taste: the pages must work with JavaScript
# off, they must print, and they must open from disk with no build step and no
# library to load. Colour that ships inside the HTML satisfies all three.
#
# Only five things are painted: comments, literals, keywords, the name being
# called, and program output. Punctuation, operators and ordinary identifiers
# get no span at all and inherit --ink. That is what keeps a code block calm
# rather than looking like a highlighter set, and it keeps the page small.

PY_KEYWORDS = (
    "False|None|True|and|as|assert|async|await|break|class|continue|def|del|"
    "elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|"
    "not|or|pass|raise|return|try|while|with|yield|self"
)

# Ordered alternation. Comments and strings are tried first so that a # inside
# a string, or a keyword inside a comment, never gets painted as itself. The
# trailing bare-triple-quote branches bound an unterminated string to the end
# of its own block, which happens when a long string is split across blocks.
PY_TOKEN = re.compile(
    r"(?P<c>\#[^\n]*)"
    r"|(?P<s>'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*$|\"\"\"[\s\S]*$"
    r"|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\")"
    r"|(?P<k>\b(?:" + PY_KEYWORDS + r")\b)"
    r"|(?P<n>\b\d[\w.]*\b)"
    r"|(?P<f>\b[A-Za-z_]\w*(?=\s*\())"
)

JSON_TOKEN = re.compile(
    r"(?P<c>//[^\n]*)"
    r"|(?P<f>\"(?:\\.|[^\"\\])*\"(?=\s*:))"
    r"|(?P<s>\"(?:\\.|[^\"\\])*\"|\b-?\d[\d.eE+-]*\b)"
    r"|(?P<k>\b(?:true|false|null)\b)"
)

CLS = {"c": "t-c", "s": "t-s", "k": "t-k", "n": "t-s", "f": "t-f"}


def _paint_with(rx, code):
    out, pos = [], 0
    for m in rx.finditer(code):
        out.append(esc(code[pos:m.start()]))
        out.append(f'<span class="{CLS[m.lastgroup]}">{esc(m.group())}</span>')
        pos = m.end()
    out.append(esc(code[pos:]))
    return "".join(out)


def _out(line):
    return f'<span class="t-o">{esc(line)}</span>'


PY_STATEMENT = re.compile(
    r"^\s*(import|from|def|class|for|if|elif|else|while|with|return|print|try|"
    r"except|finally|raise|yield|assert|lambda|await|async|del|global)\b")
PY_ASSIGN = re.compile(
    r"^\s*[\w.\[\]'\"]+(\s*,\s*[\w.\[\]'\"]+)*"
    r"(\s*:\s*[\w\[\], .]+)?"            # block_size: int = 128
    r"\s*[-+*/|&]?=[^=]")
PY_CALL = re.compile(r"\b\w+\s*\(")


def detect(code):
    """Name the language of one rendered block. There is no language field in
    the content, and a single example can mix a command with its output, so
    this runs per rendered block rather than per example."""
    lines = [l for l in code.split("\n") if l.strip()]
    if not lines:
        return "text"
    if any(l.lstrip().startswith((">>> ", "... ")) for l in lines):
        return "repl"
    if any(re.match(r"\s*\$ ", l) for l in lines):
        return "shell"
    if re.match(r"^(pip|python|uv|curl|git|cd|ls|mkdir|export|source|set|echo|"
                r"docker|ollama|npm)\b", lines[0].strip()):
        return "shell"
    if re.match(r"^(GET|POST|PUT|PATCH|DELETE|HTTP/|\d{3} [A-Z])", lines[0].strip()):
        return "http"
    # A block opening with a brace is JSON only if its keys are double quoted.
    # Python's own repr uses single quotes, and several examples show exactly
    # that, so quote style is what separates the two.
    # A leading // comment is still JSON here: two module 15 examples annotate
    # the payload that way, and _split_code only lifts # comments, not //.
    first = next((l for l in lines if not l.lstrip().startswith("//")), lines[0])
    if first.lstrip()[:1] in "{[" or re.match(r'^\s*"[^"]+"\s*:', first):
        if "'" not in code or re.search(r'"[^"]*"\s*:', code):
            return "json"
    # A block that opens with a Python statement is Python, however much of the
    # rest of it is a literal. Without this, `schema = {` followed by nine lines
    # of JSON-looking keys scores 1 out of 10 and falls through to plain text.
    if PY_STATEMENT.search(lines[0]) or PY_ASSIGN.search(lines[0]):
        return "python"
    score = 0
    for l in lines:
        if l.lstrip().startswith("#") or PY_STATEMENT.search(l) \
                or PY_ASSIGN.search(l) or PY_CALL.search(l):
            score += 1
    if score >= max(1, len(lines) * 0.5):
        return "python"
    return "text"


def paint(code, lang=None):
    """Return HTML for one code block, already escaped."""
    lang = lang or detect(code)
    if lang == "python":
        return _paint_with(PY_TOKEN, code)
    if lang == "json":
        return _paint_with(JSON_TOKEN, code)
    if lang == "shell":
        # Only the lines with a prompt are commands. Everything else the block
        # shows is what the command printed, and colouring that as shell would
        # turn a filename into a keyword.
        has_prompt = any(re.match(r"\s*\$ ", l) for l in code.split("\n"))
        rows = []
        for l in code.split("\n"):
            m = re.match(r"(\s*)(\$)( ?)(.*)$", l)
            if m:
                rows.append(f'{m.group(1)}<span class="t-k">$</span>{m.group(3)}'
                            + _paint_with(PY_TOKEN, m.group(4)))
            else:
                rows.append(_out(l) if has_prompt else esc(l))
        return "\n".join(rows)
    if lang == "repl":
        rows = []
        for l in code.split("\n"):
            m = re.match(r"(\s*)(>>>|\.\.\.)( ?)(.*)$", l)
            if m:
                rows.append(f'{m.group(1)}<span class="t-k">{m.group(2)}</span>'
                            f'{m.group(3)}' + _paint_with(PY_TOKEN, m.group(4)))
            else:
                rows.append(_out(l))
        return "\n".join(rows)
    if lang == "http":
        rows = []
        for l in code.split("\n"):
            if re.match(r"^\s*(GET|POST|PUT|PATCH|DELETE|HTTP/|\d{3} )", l):
                rows.append(f'<span class="t-k">{esc(l)}</span>')
            else:
                rows.append(_paint_with(JSON_TOKEN, l))
        return "\n".join(rows)
    return esc(code)


def outputblock(lines):
    """What the program printed. Not prose, so it keeps its own lines."""
    body = "\n".join(_out(l) for l in lines)
    return f'<pre class="eg__out">{body}</pre>'


def codeblock(code, label="python"):
    # The copy button is added by course.js, not emitted here: with JavaScript
    # off it would be a button that does nothing.
    return (f'<div class="codeblock"><div class="codeblock__bar"><span>{esc(label)}</span></div>\n'
            f"<pre>{paint(code, 'python')}</pre></div>")


def _split_code(body):
    """Separate explanation from code.

    Authors write examples where the teaching lives in '#' comment lines. That
    is prose wearing a code costume: monospaced, cramped, and easy to skim past.
    This pulls whole-line comments out into ordinary sentences and leaves the
    code as code. Comments at the end of a line of code stay put, because those
    genuinely annotate that line.
    """
    segments = []
    for group in re.split(r"\n\s*\n", body):
        lines = group.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        def is_note(ln):
            return ln.strip().startswith("#")

        pre = []
        while lines and is_note(lines[0]):
            pre.append(lines.pop(0).strip().lstrip("#").strip())
        post = []
        while lines and is_note(lines[-1]):
            post.insert(0, lines.pop().strip().lstrip("#").strip())

        segments.append({
            "before": _chunk_notes(pre),
            "code": "\n".join(lines).rstrip(),
            "after": _chunk_notes(post),
        })
    return [s for s in segments if s["before"] or s["code"] or s["after"]]


# A run of comment lines is not always prose. Very often it is what the program
# printed: a table, a list of numbers, a mapping. Joining those with a space,
# which is what this used to do unconditionally, destroyed the alignment that
# was the whole point and produced a paragraph of unlabelled numbers.
_OUTPUTISH = (
    re.compile(r"\S {2,}\S"),               # two or more spaces inside: columns
    re.compile(r"->|=>"),                   # a mapping arrow
    re.compile(r"^\s*[\[{(]"),              # opens a bracketed structure
    re.compile(r"^\s*[-+]?\d+(\.\d+)?\s"),  # starts with a number
    re.compile(r"^\s*['\"]"),               # starts with a quoted string
)


def _line_is_output(ln):
    if not ln.strip():
        return False
    # A finished sentence is prose whatever punctuation it happens to contain.
    if re.search(r"[.!?]\s*$", ln) and len(ln.split()) >= 6:
        return False
    return any(p.search(ln) for p in _OUTPUTISH)


def _chunk_notes(run):
    """Split a run of comment lines into ('prose'|'output', lines) chunks."""
    chunks = []
    for ln in run:
        kind = "output" if _line_is_output(ln) else "prose"
        if not ln.strip():
            kind = chunks[-1][0] if chunks else "prose"
        if chunks and chunks[-1][0] == kind:
            chunks[-1][1].append(ln)
        else:
            chunks.append([kind, [ln]])
    # One output line surrounded by prose is not worth a block of its own.
    merged = []
    for kind, lines in chunks:
        if kind == "output" and len([l for l in lines if l.strip()]) < 2:
            kind = "prose"
        if merged and merged[-1][0] == kind:
            merged[-1][1].extend(lines)
        else:
            merged.append([kind, list(lines)])
    out = []
    for kind, lines in merged:
        if kind == "prose":
            text = " ".join(l for l in lines if l.strip())
            if text:
                out.append(("prose", text))
        else:
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                out.append(("output", lines))
    return out


# Every section is emitted through this, so the visual treatment for a whole
# class of section is one edit here rather than ten scattered string literals.
#
# The stamp says what you are being asked to DO, and there are only three
# answers. Grouping by activity keeps the page to three treatments instead of
# ten colours, which is what stops it looking like a highlighter set.
SECTION_STAMP = {
    "watch": "take in",
    "ideas": "take in",
    "more":  "optional",
    "docs":  "reference",
    "try":   "hands on",
    "lab":   "hands on",
    "mini":  "hands on",
    "quiz":  "on record",
    "notes": "on record",
    "done":  "on record",
}


def sec(kind, title, extra=""):
    """Open a section with its kind, so CSS can give it an identity."""
    stamp = SECTION_STAMP.get(kind, "")
    return (f'<section class="section section--{kind}"{extra}>'
            f'<h2 class="section__h" data-stamp="{stamp}"><span>{title}</span></h2>')


def example_block(con):
    """Render a concept's worked example: code, a table, or a named scenario."""
    eg = con.get("example")
    if not eg or not (eg.get("body") or "").strip():
        return ""
    kind, body = eg.get("kind", "text"), eg["body"].rstrip()

    if kind == "table":
        rows = [r for r in body.splitlines() if r.strip()]
        cells = [[c.strip() for c in r.split("|")] for r in rows]
        head, rest = cells[0], cells[1:]
        inner = ("<table><thead><tr>"
                 + "".join(f"<th>{esc(h)}</th>" for h in head)
                 + "</tr></thead><tbody>"
                 + "".join("<tr>" + "".join(f"<td>{rich(c)}</td>" for c in r) + "</tr>" for r in rest)
                 + "</tbody></table>")
    elif kind == "code":
        lang = eg.get("lang")
        parts = []

        def emit_notes(chunks):
            for chunk_kind, payload in chunks:
                if chunk_kind == "prose":
                    parts.append(f'<p class="eg__say">{rich(payload)}</p>')
                else:
                    parts.append(outputblock(payload))

        for seg in _split_code(body):
            emit_notes(seg["before"])
            if seg["code"]:
                parts.append(f"<pre>{paint(seg['code'], lang)}</pre>")
            emit_notes(seg["after"])
        inner = "".join(parts)
    else:
        inner = f"<p>{rich(body)}</p>"

    cap = (con.get("example_caption") or "").strip()
    return ('<div class="eg"><div class="eg__h">Example</div>' + inner
            + (f'<div class="eg__cap">{rich(cap)}</div>' if cap else "")
            + "</div>")


def build(n):
    m = BY_N[n]
    c = BY_MOD.get(n)
    if not c:
        return None
    has_more = any(not v.get("core") for v in m["videos"])

    parts = []
    a = parts.append

    a(f'<p class="prose">{rich(c["intro"])}</p>')

    # --- videos: core -----------------------------------------------------
    a(sec("watch", "Watch this first"))
    a('<div data-videos="core"></div></section>')

    # --- concepts ---------------------------------------------------------
    a(sec("ideas", "The ideas, in plain English"))
    a('<p class="small muted">Each idea, then a worked example of it.</p>')
    a('<div class="concepts">')
    dia = diagram_slots(n, c["concepts"])
    for i, con in enumerate(c["concepts"], 1):
        a(f'<div class="concept"><div class="concept__n" aria-hidden="true">{i}</div>'
          f'<h3 class="concept__t">{rich(con["term"])}</h3>'
          f'<p class="concept__b">{rich(con["note"])}</p>')
        a(example_block(con))
        a("</div>")
        name = dia.get(i - 1)
        if name and name in DG.D:
            a(DG.D[name])
    a("</div></section>")

    # --- widgets ----------------------------------------------------------
    if m["widgets"]:
        a(sec("try", "Try it yourself"))
        for w in m["widgets"]:
            # no caption means no paragraph, otherwise the margin leaves a gap
            if WIDGET_TITLE.get(w):
                a(f'<p class="small muted" style="margin-bottom:6px">{esc(WIDGET_TITLE[w])}</p>')
            a(f'<div data-widget="{esc(w)}"></div>')
        a("</section>")

    # --- honest gap note --------------------------------------------------
    if c.get("gap"):
        a('<div class="note note--gap"><div class="note__t">Worth knowing</div>'
          f'<p>{rich(c["gap"])}</p></div>')

    # --- more videos ------------------------------------------------------
    if has_more:
        a(sec("more", "If you want to go deeper"))
        a('<p class="small muted">Optional.</p>')
        a('<div data-videos="more"></div></section>')

    # --- lab --------------------------------------------------------------
    lab = c["lab"]
    a(sec("lab", "Lab: follow along"))
    # Filled in by course.js, and empty once the learner has ticked setup off.
    a('<div data-setupbanner></div>')
    # Shown only on the paths where this lab genuinely differs.
    a(path_notes(n))
    a(f'<p class="prose">{rich(lab["intro"])}</p>')
    a('<p class="small muted">Code given, every line explained.</p>')
    a('<ol class="steps">')
    for st in lab["steps"]:
        a(f'<li><h3>{rich(st["title"])}</h3><p>{rich(st["body"])}</p>')
        if st.get("code"):
            a(codeblock(st["code"]))
        # line-by-line explanation of what that code is doing
        walk = st.get("walkthrough") or []
        if walk:
            a('<div class="walk"><div class="walk__h">What that code is doing</div><dl>')
            for w in walk:
                a(f'<dt>{esc(w["part"])}</dt><dd>{rich(w["why"])}</dd>')
            a("</dl></div>")
        # the equation, spelled out with a worked example
        maths = (st.get("maths") or "").strip()
        if maths:
            a('<div class="maths"><div class="maths__h">The maths, spelled out</div>'
              f"<pre>{esc(maths)}</pre></div>")
        # the most likely thing to go wrong here
        gotcha = (st.get("watch_out") or "").strip()
        if gotcha:
            a(f'<div class="gotcha"><b>Watch out:</b><span>{rich(gotcha)}</span></div>')
        a("</li>")
    a("</ol>")
    a(f'<div class="done"><span>&#10003;</span><div><b>You are done when</b> {rich(lab["done"])}</div></div>')
    a("</section>")

    # --- mini-project -----------------------------------------------------
    mini = c["mini"]
    a(sec("mini", "Mini-project: your turn"))
    a(f'<p class="prose">{rich(mini["intro"])}</p>')
    a('<p class="small muted">No code given. Produce the file, then run the checker.</p>')
    a('<ol class="steps">')
    for st in mini["steps"]:
        a(f"<li><p>{rich(st)}</p></li>")
    a("</ol>")
    # the checker: the learner does not grade their own work
    chk = mini.get("checker")
    if chk and (chk.get("code") or "").strip():
        a('<div class="checker"><div class="checker__h">Check it yourself, automatically</div>')
        a(f'<p>Save this as <code>{esc(chk["filename"])}</code> next to your work, then run '
          f'<code>{esc(chk["howto"])}</code>. It reads what you produced, tests it, and prints a '
          "line per check.</p>")
        a(codeblock(chk["code"], chk["filename"]))
        a("</div>")

    a(f'<div class="done"><span>&#10003;</span><div><b>You are done when</b> {rich(mini["done"])}</div></div>')
    a('<div class="note" style="margin-top:12px"><div class="note__t">If you want more</div>'
      f'<p>{rich(mini["stretch"])}</p></div>')
    a("</section>")

    # --- quiz -------------------------------------------------------------
    a(sec("quiz", "Check yourself"))
    a('<p class="small muted">Explained either way.</p>')
    a('<div class="quiz" data-quiz>')
    for q in c["quiz"]:
        a(f'<div class="q" data-answer="{int(q["answer"])}"><p class="q__q">{rich(q["q"])}</p>'
          '<div class="q__opts">')
        for j, opt in enumerate(q["options"]):
            a(f'<button class="opt" type="button"><span class="opt__k">{chr(65 + j)}</span>'
              f"<span>{rich(opt)}</span></button>")
        a(f'</div><div class="q__why" hidden>{rich(q["why"])}</div></div>')
    a("</div></section>")

    # --- notes ------------------------------------------------------------
    a(sec("notes", "Your notes", ' data-notes').replace("section--notes", "section--notes notes"))
    a('<p class="small muted">Saves as you type, in this browser.</p>')
    a(f'<label class="sr" for="n{n}">Notes for this module</label>')
    a(f'<textarea id="n{n}" placeholder="What clicked? What is still fuzzy? What do you want to '
      'come back to?"></textarea><div class="notes__status"></div></section>')

    # --- marks ------------------------------------------------------------
    a(sec("done", "Mark it done"))
    a('<div class="marks">'
      '<button class="mark" data-mark="read" aria-pressed="false"><span>Read the notes</span></button>'
      '<button class="mark" data-mark="lab" aria-pressed="false"><span>Finished the lab</span></button>'
      '<button class="mark" data-mark="mini" aria-pressed="false"><span>Finished the mini-project</span></button>'
      "</div></section>")

    # --- docs -------------------------------------------------------------
    a(sec("docs", "Go to the source"))
    a('<p class="small muted">If this course and the docs disagree, the docs are newer.</p><ul class="prose">')
    for d in c["docs"]:
        a(f'<li><a href="{esc(d["url"])}" target="_blank" rel="noopener">{esc(d["label"])}</a></li>')
    a("</ul></section>")

    body = "\n        ".join(parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(str(n) + ". " + m["title"])}</title>
<meta name="description" content="{esc(m["promise"])}">
<link rel="stylesheet" href="../assets/course.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%232f6fd0'/><path d='M8 21V11h3l3 6 3-6h3v10h-2.6v-6.2L15 20h-2l-2.4-5.2V21z' fill='white'/></svg>">
</head>
<body data-module="{m["id"]}" data-root="../">
<a class="skip" href="#main">Skip to content</a>
<header data-chrome="masthead"></header>

<div class="shell" data-phase="{m["phase"]}">
  <nav data-chrome="sidebar" aria-label="Modules"></nav>

  <main class="main" id="main">
    <div class="wrap">
        <div data-chrome="modhead"></div>
        {body}
    </div>
    <nav data-chrome="pager"></nav>
  </main>
</div>

<script src="../assets/data.js"></script>
<script src="../assets/widgets.js"></script>
<script src="../assets/course.js"></script>
</body>
</html>
"""


APPENDIX = {13: """
---

## Optional: plug your server into an app you already use

**This is a bonus. The lab is complete without it.** You proved your server works with your own
client, which is the point. But if you already use one of these, most share the same `mcpServers`
shape, so it is one pattern with small variations:

| App | Config key | Notes |
|---|---|---|
| Claude Desktop | `mcpServers` | `claude_desktop_config.json` |
| Cursor | `mcpServers` | `.cursor/mcp.json` |
| Cline | `mcpServers` | via the extension settings |
| VS Code | `servers` | different key, the common gotcha |
| Zed | `context_servers` | different key |
| Continue | a YAML list | different shape entirely |
| Goose | `extensions` | different key and field names |

Use absolute paths, and point at your virtual environment's interpreter rather than the bare word
`python`, for the same reason `sys.executable` matters in the client:

```json
{
  "mcpServers": {
    "hello": {
      "command": "C:\\\\path\\\\to\\\\GenAI-Course\\\\my-work\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["C:\\\\path\\\\to\\\\GenAI-Course\\\\my-work\\\\labs\\\\lab13\\\\hello_server.py"]
    }
  }
}
```

## The zero-dependency version

Because the protocol became stateless, you do not even need the SDK to talk to a server. One line
of JSON is a complete conversation:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}' | python hello_server.py
```
"""}


def lab_readme(n):
    """A self-contained README per lab folder, so each folder stands alone."""
    m, c = BY_N[n], BY_MOD[n]
    L, MP = c["lab"], c["mini"]

    def plain(s):
        return re.sub(r"</?(strong|code|em|b)>", "`" if False else "", s)

    out = [f"# Lab {n:02d}: {m['lab']['t']}", "",
           f"**Module {n}: {m['title']}**", "",
           plain(L["intro"]), "",
           "Before you start, make sure `python llm.py` works. See `setup.html`.", "",
           "## Steps", ""]
    for i, st in enumerate(L["steps"], 1):
        out += [f"### {i}. {plain(st['title'])}", "", plain(st["body"]), ""]
        if st.get("code"):
            out += ["```python", st["code"].rstrip(), "```", ""]
        for w in (st.get("walkthrough") or []):
            out += [f"- `{w['part']}`: {plain(w['why'])}"]
        if st.get("walkthrough"):
            out.append("")
        if (st.get("maths") or "").strip():
            out += ["**The maths, spelled out**", "", "```", st["maths"].rstrip(), "```", ""]
        if (st.get("watch_out") or "").strip():
            out += [f"> **Watch out:** {plain(st['watch_out'])}", ""]
    out += ["## You are done when", "", plain(L["done"]), "",
            "---", "", f"## Mini-project: {m['mini']['t']}", "",
            plain(MP["intro"]), ""]
    for s in MP["steps"]:
        out.append(f"- {plain(s)}")
    chk = MP.get("checker")
    if chk and (chk.get("code") or "").strip():
        out += ["", "### Check it", "",
                f"`{chk['filename']}` is in this folder. Run it:", "",
                "```bash", chk["howto"], "```", ""]
    out += ["", "**You are done when** " + plain(MP["done"]), "",
            "**If you want more:** " + plain(MP["stretch"]), ""]
    if n in APPENDIX:
        out.append(APPENDIX[n])
    return "\n".join(out)


written, missing, labs, checkers = [], [], 0, 0
patched = set()
for n in sorted(BY_N):
    html = build(n)
    if html is None:
        missing.append(n)
        continue
    path = os.path.join(ROOT, "modules", BY_N[n]["file"])
    io.open(path, "w", encoding="utf-8").write(html)
    written.append((n, BY_N[n]["file"], len(html)))

    # lab folder README - do not clobber the hand-written, tested ones
    lab_dir = os.path.join(ROOT, "my-work", "labs", f"lab{n:02d}")
    os.makedirs(lab_dir, exist_ok=True)
    rp = os.path.join(lab_dir, "README.md")
    if True:
        io.open(rp, "w", encoding="utf-8").write(lab_readme(n))
        labs += 1

    # drop the checker next to the lab so the learner never has to copy it
    chk = BY_MOD[n]["mini"].get("checker")
    if chk and (chk.get("code") or "").strip():
        fn = os.path.basename(chk.get("filename") or "check.py")
        code = chk["code"]
        # Make the checker accept the evidence the learner's path can produce.
        # Anchors are asserted, so a regenerated checker that no longer matches
        # fails here rather than silently shipping without the path handling.
        for anchor, repl in PA.PATCHES.get(n, []):
            if anchor not in code:
                raise SystemExit(
                    f"m{n:02d}: path patch anchor no longer present in check.py.\n"
                    f"  looked for: {anchor.splitlines()[0][:80]}")
            code = code.replace(anchor, repl, 1)
            patched.add(n)
        io.open(os.path.join(lab_dir, fn), "w", encoding="utf-8").write(code.rstrip() + "\n")
        checkers += 1

for n, f, size in written:
    print(f"  m{n:02d}  {size:>7,} bytes  {f}")
print(f"\nwrote {len(written)} pages, {labs} lab READMEs, {checkers} checker scripts")
if missing:
    print("MISSING CONTENT for modules:", missing)
