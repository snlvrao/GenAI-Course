# Project handoff

Everything a fresh session needs to continue this work. Read this first.

**Project root:** `C:\Users\Sunil Venkatesh Rao\Documents\Claude Projects\GenAI-Course`

---

## 1. What this is

A self-study course on Gen AI and Agentic AI, built as offline HTML. One learner: an embedded
engineer who took a college ML course years ago, knows what a gradient is, has never built anything
with transformers. Goal is to end able to design, build, evaluate and deploy real agents, and to
understand everything rather than copy recipes.

**23 modules across 7 phases, ~115 hours.** Phases 1–6 are the main course, Phase 7 is a parallel
"Build track" where the learner trains their own language model from scratch.

Open `index.html` by double-click. No server, no build step to read it.

---

## 2. Hard rules (do not break these)

From the original brief:

- **HTML only.** No Word, PDF or other document formats. Ever.
- **Self-contained and offline.** Inline everything, no build step, no npm. The only network calls
  allowed are YouTube thumbnail and iframe loads. **Amended:** the course is now opened through
  `start.py`, which serves the folder on `127.0.0.1`, because YouTube will not play an embed in a
  page opened from disk. Double-clicking still works for everything except video. See open item 1.
- **No AI attribution anywhere.** No "Generated with…" lines, no co-author trailers in commits, no
  comments referring to a brief, an assistant, or a conversation. Code comments explain code only.
  `tools/verify.py` enforces this and will fail the build, including on this file.
- **No pushing to remotes.** Local commits fine. Ask before any `git push` or remote operation.
- **No `fetch()` anywhere.** Browsers block it on `file://`. This is why the course works offline.

Added by the learner over the course of the work:

- **Plain English.** Short sentences, every term defined on first use, no jargon walls. Asked for
  repeatedly. Treat as the top constraint on all prose.
- **No em dashes** in authored content. Comma, full stop or brackets.
- **Be direct and brief.** No filler about the course itself. The learner quoted this as the style
  to avoid: *"They play inside the page; you never get bounced to YouTube and lost."*
- **Nothing specific to one person or machine.** No names, no "on my laptop", no absolute user
  paths. Timings attribute to a hardware class ("a laptop processor with no graphics card").
- **No vendor lock-in.** Nothing may require Claude Desktop or any one provider.
- **Explain with examples and visuals, not prose alone.**
- **Code format is for code.** Do not put explanatory text in code blocks.
- **Mini-projects must be verified by a program**, not left to the learner to self-assess.
- **The learner picks the parameters.** They explicitly do not want to just run pre-trained things.

---

## 3. Layout

```
GenAI-Course/
  start.py              serves the folder on 127.0.0.1 and opens it: how you read the course
  Start course.cmd      one-click wrapper for Windows
  start.sh              one-click wrapper for mac and Linux
  index.html            dashboard: progress, search, phase map, export/import
  setup.html            path chooser + every install explained
  HANDOFF.md            this file
  README.md             user-facing readme
  .env.example          three provider paths
  modules/              m01 … m23, GENERATED - do not hand-edit
  assets/
    data.js             module index: titles, promises, videos, concepts, widgets
    course.css          all design tokens and layout
    course.js           shared chrome, progress, video embeds, quiz, notes, path filtering
    widgets.js          15 interactive teaching widgets
  my-work/              EVERYTHING THE LEARNER CREATES. Nothing else is theirs.
    .venv/              NOT SHIPPED. The learner builds it on the setup page.
    .env                NOT SHIPPED. The learner writes it on the setup page.
    notes/              their own notes, untouched by the build
    labs/
      _shared/
        llm.py          the one provider-neutral model wrapper every lab imports
        tinygpt.py      the transformer built in B2 and trained in B3
        myconfig.py     reads model_config.json, validates, prints what choices cost
        check_decisions.py verifies the model design is the learner's own
      lab01 … lab23/    runnable starter files, README.md, check.py per lab
  tools/                BUILD PIPELINE - see section 4
```

**The my-work split.** The learner's work is quarantined under `my-work/` so they can tell what
they made from what the course shipped. Inside a lab folder only `README.md` and `check.py` are
generated; everything else is theirs and the build never touches it. The environment and `.env`
live there too, because they create both and neither is course content.

Two constraints fix where things sit, and they pull against each other:

- `python-dotenv` finds `.env` by walking **upward** from the working directory. So `my-work/.env`
  is visible from any lab folder, and invisible from the course root. Every "run it from the
  project root" instruction had to become "run it from the lab folder", which is where 49 of the
  50 run instructions already pointed.
- The reference point for `cd` stays the **course root**, because 17 instructions already say
  `cd my-work/labs/labNN` from there. So the environment is activated by its full path,
  `.\my-work\.venv\Scripts\Activate.ps1`, rather than by cd-ing into `my-work` first.

Moving an existing venv breaks it: the activation scripts and every `.exe` in `Scripts/` hardcode
the old absolute path. `python -m venv <path>` over the top regenerates the scaffolding without
touching `site-packages`, and `pip install --force-reinstall --no-deps pip` repairs `pip.exe`.
A venv created fresh at `my-work/.venv` has none of this trouble.

**No environment ships with the course, and none is needed to build it.** Every tool in `tools/`
imports only the standard library, and the full gate plus all 23 checkers pass on a bare Python
with nothing installed. That is checked, not assumed: `python tools/verify.py` and
`python tools/test_checkers.py` were both run on a system Python with no `torch` and no `openai`
and returned 0 failures and 23/23.

This matters because building the environment is a taught step, not setup noise. `setup.html`
installs packages in groups as the learner reaches the module that needs each one, so an
environment that arrives pre-filled would skip the lesson and hand them packages that Module 19
introduces. A 1.36 GB venv that had accumulated here was deleted for exactly that reason.
Re-creating it to re-verify the execution claims in section 7 costs that much again, most of it
`torch` arriving underneath `sentence-transformers`.

---

## 4. The build pipeline (important)

**`modules/*.html` are generated. Never hand-edit them; your changes will be overwritten.**

Source of truth is `tools/module_content_raw.json` (authored content, 23 modules) plus
`assets/data.js` (the index: which videos, which widgets, ordering).

```bash
cd tools
python reconcile.py     # raw -> module_content.json: fixes, code overrides, quiz balancing
python gen_pages.py     # module_content.json + data.js -> 23 pages, 23 lab READMEs, 23 checkers
python verify.py        # full gate: links, structure, widgets, live video re-check
python test_checkers.py # every mini-project checker must fail cleanly with no work present
```

What each does:

- **`reconcile.py`** applies corrections that were verified by *running* code, not reading it.
  It overrides Module 13's MCP client with the tested file, fixes lab paths, strips invented env
  vars, and balances quiz answer positions. Patches assert their anchor text, so if regenerated
  content no longer matches, the build **fails loudly** instead of silently shipping something wrong.
- **`gen_pages.py`** is the page generator. Section emission goes through `sec()`, examples through
  `example_block()`, diagrams placed by `diagram_slots()`. It also paints the code: `detect()` names
  the language of each rendered block and `paint()` turns it into span-tagged HTML. Colouring
  happens here, at generation time, because the pages have to work with JavaScript off, have to
  print, and have to open from disk with no library to load.
- **`diagrams.py` / `diagrams2.py`** are 26 inline SVG diagrams, built at generation time so they
  print and survive JavaScript being off. Both the case rule for labels and the geometry rule are
  written at the top of `diagrams.py`. Geometry is checkable: no label may overlap another, nothing
  may fall outside the viewBox, and an arrow stops at the edge of the box it points to. Check that
  by rendering and reading `getBBox()`, not by eye.
- **`videos.tsv`** is reference data only. Nothing reads it any more, since the one script that did
  has been deleted. Keep it: it is the only record of the durations, view counts and publish dates.
- **`pathaware.py`** holds the per-path notes (offline / free / own-key) and the checker patches for the
  modules where the offline path genuinely differs.
- **`verify.py`** is the release gate. Skips `tools/` when scanning for attribution.
- **`test_checkers.py`** runs all 23 checkers in an empty folder; each must exit 1 with a clear
  sentence and no traceback.

---

## 5. Design system

Direction: **engineering lab notebook**. Sturdy, navigable, quietly technical. Deliberately not a
SaaS dashboard, not a magazine. Explicitly avoided: cream + terracotta, near-black + one acid
accent, hairline broadsheet, and the default white/indigo-500/Inter/8px-radius look.

**Phase hue ramp.** Seven hues, `--p1` … `--p7`, cool to warm across phases 1–6 plus violet for the
Build track. `data-phase` on a wrapper resolves `--ph` for everything inside. The hue drives spines,
stamps, step numbers, focus rings, progress segments.

**Section identity, added last.** Sections are grouped by what the learner is asked to DO, not by
which section they are, so there are three treatments rather than ten colours:

| Stamp | Sections | Treatment |
|---|---|---|
| take in | Watch, The ideas | phase rule on top, tinted stamp, no fill |
| hands on | Try it, Lab, Mini-project | boxed, 4px phase spine, 4.5% phase wash |
| on record | Quiz, Notes, Mark it done | boxed in warm `--paper` |
| optional / reference | Go deeper, Go to the source | grey hairline, smaller heading |

**The rule that keeps it from becoming a rainbow:** the phase hue holds a monopoly on saturation. It
is the only hue that ever fills an area. The one non-phase material is `--paper`, a warm neutral for
the sections holding the learner's own words.

**Two AA traps already hit and fixed. Do not reintroduce:**
- A `.6rem` stamp in `hsl(var(--ph))` fails AA on six of seven phases (gold 2.38:1). Stamp text is
  always `--ink-2`; the phase hue appears only as border and tint.
- In dark mode `--paper` is *lighter* than `--surface`, so cards inside paper sections invert. Use
  `--paper-card` for anything sitting on `--paper`.

Type: system fonts only. Segoe UI stack for chrome, **Georgia for reference prose**, Cascadia/Consolas
for code. Squared radii, borders and insets rather than soft shadows.

---

## 6. Vendor neutrality

- `my-work/labs/_shared/llm.py` is the only place the course talks to a model. Switch provider by editing
  `LLM_PROVIDER` in `.env`, never code. Providers registered: groq, gemini, huggingface, openrouter,
  openai, deepseek, together, anthropic, ollama, llamacpp, lmstudio, and **mine** (the learner's own
  trained model, served on `http://127.0.0.1:8100/v1`).
- `setup.html` asks which of three paths the learner wants and shows only those steps: **fully
  offline** / **free hosted** / **your own key**. Choice persists, travels in the progress export,
  clears on reset. Without JavaScript it shows all three rather than going blank.
- MCP servers are proved with a Python client the learner writes. Claude Desktop, Cursor and VS Code
  are optional extras documented as a table, never requirements.

---

## 7. Verified by execution, not by reading

Everything below was actually run on this machine. Do not silently contradict it.

| Claim | Evidence |
|---|---|
| MCP Python SDK is v2.0.0; class is `MCPServer` | `from mcp.server.fastmcp import FastMCP` raises ModuleNotFoundError |
| MCP protocol is stateless, `2026-07-28` | client printed `protocol: 2026-07-28`; raw JSON-RPC works with no handshake |
| The low-level `ClientSession` pattern FAILS | "Invalid request parameters", because it omits the now-required `_meta`. Use `Client` |
| `sys.executable`, not `"python"`, for subprocesses | bare `python` gave "Connection closed" |
| sqlite-vec + BM25 + RRF works | full hybrid pipeline ran; `serialize_float32` verified |
| Local embeddings need no key | sentence-transformers all-MiniLM ran offline |
| CPU LoRA fine-tune is viable | 155s end to end, style rule 0/5 → 5/5 on held-out prompts |
| Training a GPT from scratch on CPU is viable | 3,618 steps in 8 min, val loss 1.554 vs 4.174 random |
| A self-trained model can serve the whole course | stdlib server + `LLM_PROVIDER=mine` returned proper usage counts |
| Parameter formula in `myconfig.py` is exact | predicted 824,897, torch reported 824,897, matched at 4 configs |
| Module 11's LoRA lab still runs on current libraries | TRL 1.9.2 + Transformers 5.14.1 accept every argument; style rule 0/3 -> 3/3 on CPU |
| `peft` is NOT pulled in by sentence-transformers | it is declared `peft; extra == "dev"`, so a plain install misses it and cpu_lora.py fails |
| CrewAI will NOT install on Python 3.14 | declares `>=3.10,<3.14`. Module 14 routes that one comparison to Colab |
| YouTube embeds FAIL on `file://`, work on `http://` | same page, same code: Error 153 from disk, plays inline from localhost |
| Thumbnails load on `file://` | plain images have no origin check, so the page looks fine until a click |

**Local models, measured on this machine (August 2026).** Ollama, CPU only, eight models.
Tool restraint means being handed a tool the prompt did not need, over six prompts and four runs
each. Accuracy is ten tasks with a checkable right answer, three runs each.

| model | disk | tool when needed | tool restraint | JSON | accuracy | verdict |
|---|---|---|---|---|---|---|
| `qwen2.5:3b-instruct` | 1.9 GB | 6/6 | 12/12 | 3/3 | 27/30 | **the default** |
| `granite4.1:3b` | 2.1 GB | 6/6 | 12/12 | 3/3 | 24/30 | fine |
| `phi4-mini:3.8b` | 2.5 GB | 6/6 | 12/12 | 3/3 | 24/30 | fine, but wrong on plain addition |
| `llama3.2:3b` | 2.0 GB | 6/6 | **0/12** | 3/3 | 24/30 | calls tools it was not asked for |
| `mistral:7b` | 4.4 GB | 6/6 | **3/12** | 3/3 | 24/30 | same fault, largest download |
| `ministral-3:3b` | 3.0 GB | **0/6** | 12/12 | 3/3 | 18/30 | never calls the tool |
| `falcon3:3b` | 2.0 GB | **refused** | n/a | 3/3 | n/a | HTTP 400, does not support tools |
| `qwen3:4b` | 2.5 GB | 3/6 | 12/12 | **0/3** | n/a | reasons out loud, JSON never parses |

Three things to carry forward. Size predicts none of it: the 4.4 GB model fails restraint worse
than models a third its size. Model cards cannot be trusted either, because ministral-3 advertises
"best-in-class agentic capabilities with native function calling" and never called the tool once.
And arithmetic is where these models are quietly weakest: asked for 17 percent of 4830, seven of
the eight answered 819.9, 809.1, 825.10, 3911 or 289.1, with no hedging.

Counting words is a weaker signal than it looks, but not a useless one, which corrects an earlier
claim made here and on the setup page. Of six models, two hit "in five words" on every attempt and
four never did, and the two that passed were not the two that scored best overall. Test the thing
you depend on, which is why every lab is graded by a checker script.

The scripts that produced this are not in the repo; they are three short files that talk to
127.0.0.1:11434 and mark their own answers. Re-run them by hand if a model is added.

**2026 facts that correct common stale material** (all checked against primary sources, Aug 2026):
context rot is Chroma's finding with no 32K threshold, not Databricks'; the four context failure
modes are Drew Breunig's; A2A support is now broad not narrow; Llama is no longer an open-weight
leader (Meta shipped closed-weight Muse Spark); Helicone is maintenance-only after acquisition;
LLM-as-judge means error analysis first, binary not Likert, report TPR/TNR or kappa; the list to
teach is OWASP's **Agentic** Top 10 (ASI01–ASI10); the Colorado AI Act is not in force.

---

## 8. Videos

84 unique, all embedded as click-to-play lite embeds. No URLs printed anywhere.

**Verification rule: oembed returning JSON is the gate, not "is the link alive".** A video can play
fine on YouTube and still refuse to embed. Stanford's CS229 LLM lecture (2.6M views) was cut for
exactly this. It returns `Unauthorized`.

```bash
curl -s "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=VIDEO_ID&format=json"
```

`verify.py` re-checks all 84 on every run. View counts in `tools/videos.tsv` were scraped from watch
pages, not estimated. Selection weights popularity: timeless topics target 200k+ views, 2026-specific
topics 20k+ because the material is months old.

---

## 9. Current state

- 23 modules, 7 phases, ~115 hours
- 178 worked examples (every concept has one), 26 diagrams, 15 widgets
- 589 code walkthroughs, 113 worked-maths blocks
- 23 labs + 23 mini-projects, each with a `check.py`
- **Last full gate: 0 failures, 0 warnings.** 25 pages well-formed, 84/84 videos embeddable,
  15/15 widgets resolve, 23/23 checkers fail cleanly with no work present, every text file is
  strict UTF-8 with no lost characters.
- **All 25 pages have now been rendered and reviewed in a browser**, in both themes. Contrast was
  measured on every element that carries text, on every page, in both themes: 0 failures against
  AA. See section 11 for what that review changed.
- **The content audit is closed.** 183 findings were raised, 125 survived an adversarial check,
  and those were fixed over two passes: 55 in the first, 44 in the second. The second pass
  re-checked every finding against the content as it then stood rather than trusting the earlier
  bookkeeping, which is how it found that 15 had already been fixed as a side effect and 7 were
  wrong or a matter of taste. 6 more were rejected by the check. Every number touched was
  recomputed by running code, and every patch asserted its old value byte for byte before
  replacing it, so a finding written against content that had moved was skipped, never forced.

---

## 10. Open items and known gaps

1. **Settled: the course is opened with a launcher, because videos have to play inline.**
   Over `file://` every YouTube embed returns "Error 153, video player configuration error".
   Served over `http://127.0.0.1` the identical page plays inline. YouTube refuses the embed
   because a page on disk has no origin for it to check, and nothing on this side can satisfy that
   check. Thumbnails are plain images so they load either way, which is why the page looks right
   until you click. The oembed gate cannot catch it either: it asks whether a video may be
   embedded, not whether a page with no origin may embed it.

   `start.py` serves the folder on `127.0.0.1` and opens it. `Start course.cmd` and `start.sh` are
   one-click wrappers. It is standard library only, nothing is uploaded, and nothing is installed.
   Double-clicking `index.html` still works for everything except video, and a dismissible banner
   on `file://` says so and names the launcher.

   **This is the one hard rule the course now breaks**, and it was broken deliberately: "no server"
   loses to "the videos must play inline", because a video course whose videos do not play is not
   a course. Everything else about the rule still holds. There is no build step, no npm, no
   bundler, and the pages are the same static files whichever way you open them.
2. **One non-reproducible failure.** `my-work/labs/lab19/bpe.py` crashed once mid-run with a nonsensical
   error (a loop counter appearing as a list). Four subsequent runs were clean and identical. Not
   diagnosed; possibly a Python 3.14 optimiser glitch. The lab's lossless round-trip check would
   catch silent corruption.
3. **Honest content gaps, stated in the UI rather than padded:** no video names the four context
   failure modes (Breunig's written posts are linked); nothing good exists on constrained decoding;
   no usable OpenAI Agents SDK video; nothing on agent FinOps.
4. **`index.html` and `setup.html` deliberately do not use section colouring**. It earns its keep
   through repetition across module pages; applying it to one-off pages would be "all over".
5. **Module 11's Colab lab has been run, on CPU, and it works.** What was NOT done is running it
   on Colab itself, which needs a Google sign-in. Everything that does not depend on the GPU was
   executed verbatim: the 12 pairs, the dataset shape, `LoraConfig`, `SFTConfig`, `SFTTrainer`,
   `ask()`, `is_house_style`, and the step 7 judge. Four substitutions, all forced by having no
   GPU: SmolLM2-135M-Instruct in place of Qwen2.5-1.5B-Instruct, no 4-bit `BitsAndBytesConfig`
   (bitsandbytes 4-bit needs CUDA), `fp16=False`, and `cpu` for `cuda`.
   Result: the style rule went **0/3 before, 3/3 after**, 63 seconds of training, 9.8 MB adapter.
   The API check matters more than the numbers, because that is what silently rots: on TRL 1.9.2
   and Transformers 5.14.1 every argument the lab passes is still accepted, and `tokenizer=` is
   now **gone** from `SFTTrainer`, so the lab's `processing_class` note is load-bearing rather
   than historical. The remaining risk is only the GPU-specific part: the 4-bit load and the T4.
6. **Done: em dashes are gone and the rule is now checked.** All 152 were removed, each with a
   judgement rather than a find and replace: a colon where the dash separated a label from its
   title, a full stop before an independent clause, a comma before a dependent one, brackets where
   a pair of dashes was really a parenthetical. Six sentences in `widgets.js` needed handling by
   name because the general rule would have left a comma splice. The four that remain are inside
   YouTube titles in `data.js` and `videos.tsv`, quoted exactly, and `verify.py` section 9 exempts
   those two files and fails on an em dash anywhere else.

---

## 11. What the first browser review found

Everything below was seen rendered, then fixed, then re-measured in the browser.

**Light mode had never been contrast-checked.** Dark mode was clean. Light mode, which is what
anyone whose system is set to light sees by default, failed AA on 66 elements of a single page. The
cause was the one already written down in section 5: the phase hue is too light to be text. That
was fixed for the section stamp only, and the same hue was still setting text on the module
eyebrow, example headings, walkthrough terms, maths headings, widget tags, sidebar phase labels and
module numbers. As text on a light page the ramp runs from 3.99:1 down to 1.95:1 for gold.

The fix keeps the design. `--p1-ink` to `--p7-ink` are the same hues at the same saturation, with
lightness dropped until each clears 4.5:1 on the tightest background it appears on, and
`[data-phase]` now resolves `--ph-ink` next to `--ph`. **Fill and outline with `--ph`, set text with
`--ph-ink`.** Decoration keeps the vivid ramp, so spines, dots and the progress bar look unchanged.
`--ink-3` was also 3.7:1 in light mode and is now 4.7:1, which covers the muted captions, table
headings, sidebar numbers and stat labels that share it.

**The ghosted numeral on concept cards is decorative and is now marked so.** It is 9% alpha by
design. It carries `aria-hidden="true"` rather than being darkened, so screen readers no longer
announce a stray number before each heading and it is correctly exempt from contrast.

**Three widgets had no caption.** `WIDGET_TITLE` in `gen_pages.py` still held the original twelve,
so `predictor`, `neuron` and `modelsize` rendered an empty paragraph with a margin, leaving a gap on
four pages. The generator now emits no paragraph when there is no caption, and `verify.py` warns
when a placed widget is missing one.

**Eleven video titles had been silently corrupted.** `videos.tsv` was cp1252 and
`add_build_track.py` (since deleted) read it as UTF-8 with `errors="replace"`, so en dashes, em dashes and
apostrophes became U+FFFD and were written into `data.js` that way. Titles render from `data.js` at
runtime, which is why no generated page contained the damage and nothing caught it. `videos.tsv` is
now UTF-8, the reader no longer replaces, the titles are restored from the table, and `verify.py`
section 8 fails on any text file that is not strict UTF-8 or that contains a replacement character.

**`widgets.js` held raw NUL and DEL bytes.** The tokenizer's pre-tokenize regex had literal control
characters where `\x00-\x7f` was meant, which made grep and diff treat the file as binary. A
character class of literal NUL to DEL matches the same characters, so behaviour was identical, and
that was confirmed against the escaped form on ASCII, CJK, emoji and accented Latin before the swap.
Section 8 now also fails on raw control characters.

**The dashboard counted the main course but not the Build track.** It read "18 modules" and "36
things you build" next to hours and videos that did include the Build track. Those two numbers were
hardcoded and had drifted when Phase 7 was added; all four now come from `data.js`, so they cannot
drift again. The hero ramp was also missing its seventh bar, and the copy still said "twelve
interactive tools" and "two or three videos per module".

Two notes for whoever reviews next. `verify.py`'s `read()` uses `errors="replace"` on purpose, so it
never crashes on a damaged file, which is exactly why the encoding check reads bytes instead. And
`.claude/launch.json` outside the course folder starts a plain `http.server` for review only. The
course itself still has no build step and no server, and nothing inside `GenAI-Course/` refers to it.

---

## 12. Code is coloured, and why there is no terminal

**Colouring.** Code blocks are painted in `gen_pages.py` and ship as span-tagged HTML. Five classes
carry the whole vocabulary: `.t-s` literal, `.t-k` keyword, `.t-c` comment, `.t-f` the name being
called, `.t-o` program output. Punctuation, operators and ordinary identifiers get no span at all
and inherit `--ink`. That is deliberate: it is what stops a block looking like a highlighter set,
and it keeps the added bytes down. Only two new colours exist, `--code-lit` and `--code-kw`, at
hues 350 and 250, both less saturated than the phase ramp so a coloured word in a code block is
never mistaken for a phase signal. Comments and output reuse `--ink-3` and `--ink-2`.

`detect()` names the language per rendered block, not per example, because one example can hold a
command and its output. The mix across the 155 rendered blocks is 114 Python, 16 JSON, 12 plain
text, 9 shell, 2 REPL, 2 HTTP. An example may carry a `"lang"` key to override detection.

**Output is not prose.** `_split_code` lifts whole-line `#` comments out of code, which is right
when they are sentences and was wrong when they were what the program printed: it joined them with
spaces and a table of numbers became a run-on paragraph. It now classifies each comment line and
groups them, so a run of output lines keeps its own lines in an `.eg__out` block.

**There is no terminal in the page, and that was decided after building one.**

It worked. Brython is Python 3 written in JavaScript, so it loads through two
classic `<script src>` tags: no WebAssembly file, nothing read over the network,
no ES module. That is the only loading mechanism a page opened from disk still
has, and it was measured on a real `file://` page rather than assumed: 40
independent sessions created in 93 ms, 17 ms per run once warm, a prompt with
history, continuation lines and real tracebacks.

It came out again because of what it could not reach. The terminal work in this
course is the labs, and the labs run against a venv holding torch,
sentence-transformers and sqlite-vec. Of the 153 code blocks across the lab
steps, **exactly one** would run in a browser: 71 need a real package and 81 are
fragments of a larger file. A prompt sitting beside a lab step it cannot run is
worse than no prompt, because it invites the reader to try and then fails them.

So code is coloured, copyable, and carries the output it really produced, and
running it belongs in the workspace built on the setup page.

If anyone reopens this, the question to weigh is not "can Python run in a
browser", because it can. It is "can it reach the packages the labs are about",
and it cannot.

---

## 13. Giving it to someone else

Two halves, and they have very different answers.

**Reading the course needs nothing at all.** It is 2.1 MB of static HTML, CSS and
JavaScript. No build, no npm, no CDN, no `fetch()`. Measured, not assumed:

| Checked | Result |
|---|---|
| Total size of the readable course | 2.1 MB (1.9 MB of it the 23 module pages) |
| Root-relative links (`href="/..."`) | 0, so it works under any subpath |
| Absolute machine paths in shipped files | 0 |
| Served under `/GenAI-Course/` | CSS applied, 23 modules loaded, sidebar built, no broken images |
| Videos on a hosted origin | play inline |

So it can be dropped on any static host: GitHub Pages, Netlify, Cloudflare Pages,
an S3 bucket, a company intranet. Upload the folder, that is the whole job.

**Hosting it also retires the launcher.** `start.py` exists only to give the
pages an address so YouTube will play. A hosted site already has one, so
`Start course.cmd` is for local use and nothing else. Do not tell a hosted
reader to run it.

What NOT to do: zip it and send it. The recipient double-clicks `index.html`,
lands on `file://`, and every video shows Error 153. A zip recreates the exact
problem the launcher exists to solve.

**Running the labs needs Python, and that cannot be removed.** `requirements.txt`
starts with `sentence-transformers`, which pulls in torch, and the build track
trains a real model. That is the subject matter, not an accident of packaging.
Labs are per-phase installs so nobody pays for all of it up front, and the
offline path needs no key or account. Anyone who only wants to read never
installs anything.

---

## 14. Prompt to start a new session

> I'm continuing work on a Gen AI course at
> `C:\Users\Sunil Venkatesh Rao\Documents\Claude Projects\GenAI-Course`.
> Read `HANDOFF.md` in that folder first. It has the constraints, the build pipeline, and what has
> already been verified by execution. Module pages are generated, so edit
> `tools/module_content_raw.json` or the generator, never `modules/*.html` directly. After any
> change run `python tools/reconcile.py && python tools/gen_pages.py && python tools/verify.py`.
