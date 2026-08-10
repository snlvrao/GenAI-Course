# Gen AI & Agentic AI · a self-study course

Twenty-three modules that take you from remembering what a gradient is, to building and shipping
your own AI agents and MCP servers, plus a parallel track where you train a language model from
scratch and serve it yourself.

**Run `Start course.cmd`**, or `./start.sh` on macOS and Linux. Nothing to install: it serves this
folder on your own machine using only what comes with Python, and opens the course. Double-clicking
`index.html` also works, but the videos will not play that way. See Getting started below.

### Sharing it with other people

Put the folder on any static host and send the link: GitHub Pages, Netlify, Cloudflare Pages, an
S3 bucket, an intranet share. There is nothing to build and nothing to install, it is 2.1 MB of
plain HTML, and it works under a subpath such as `example.com/genai-course/`. A hosted copy needs
no launcher, because the videos play as soon as the pages have a real address.

Do not send it as a zip. Whoever opens it will double-click `index.html`, land on `file://`, and
every video will refuse to play. That is the one thing hosting fixes.

Only the labs need Python installed, and only if you intend to run them.

---

## What this is

A course built for one learner, in plain English, with something you actually build at the end of
every module. It is video-first: 84 videos are embedded, each one checked to be alive, embeddable
and genuinely watched, and most sit on channels with hundreds of thousands or millions of views.

- **23 modules** across 7 phases, roughly 115 hours at 1–2 modules a week.
- **A Build track** (B1–B5) running alongside the main course, where you train your own language
  model from random numbers and then run the course on it.
- **Every concept carries a worked example**: a table with real numbers, a few lines of code with
  its output, or a named scenario. Not prose alone.
- **26 diagrams** for the ideas that are hard to picture: gradient descent, backpropagation,
  attention, prefix caching, the RAG pipeline, LoRA, the lethal trifecta, judge calibration.
- **589 code walkthroughs and 113 worked-maths blocks.** Formulas written in ordinary characters,
  every symbol named, with a numeric example.
- **Every mini-project is checked by a script.** You produce a named file, run `python check.py`,
  and get a PASS or FAIL line per check. You do not grade your own work.
- **15 interactive widgets**: tokenizer, context-window simulator, cost calculator, MCP message
  inspector, prompt-injection sandbox.
- **Code is syntax coloured and copyable**, and every example carries the output it really
  produced. You run it in your own workspace, which is the same one the labs use.
- **Progress, notes and quiz answers** save in your browser. No account, nothing uploaded.

## Do the setup first

`setup.html` is **Step 0**, not an appendix. It is pinned to the top of the sidebar on every page,
the home page opens with it, and every module shows a reminder above its lab until you tick it off.
It explains each package you install: what it is in one sentence, why the course needs it, how to
check it worked, and how to remove it.

## No vendor lock-in

This is a hard rule, not a preference.

- Every lab runs against whichever model you like. You switch by editing one line in `.env`, never
  by editing code. Free hosted models, paid models, or a model on your own laptop with no account
  and no internet all work through the same `my-work/labs/_shared/llm.py`.
- **The MCP servers you build are proved with a Python client you write yourself.** You do not need
  Claude Desktop, Cursor, VS Code or any other app. If you happen to use one, there is a bonus step
  showing how to plug your server in, a bonus and never a requirement.
- Embeddings and reranking run locally with open models from Hugging Face. No key, no quota.
- The vector store is `sqlite-vec`: no server, no Docker, and your whole index is one `.db` file.

## Getting started

1. **Run `Start course.cmd`** (Windows) or `./start.sh` (macOS, Linux). Either one runs
   `start.py`, which serves this folder on your own machine and opens the course in your browser.
   Nothing is uploaded, nothing is installed, and it uses only what comes with Python.

   You can open `index.html` directly instead, and everything works except the videos. YouTube
   will not run its player inside a page opened as a file, so it answers "Error 153" no matter
   what the page does. Serving the folder gives the page an address, which is all it needs.
2. Work through **`setup.html`** first. It opens by asking which of three paths you want, then
   shows only the steps for that one:

   | Path | What it means | Cost | Internet |
   |---|---|---|---|
   | **Fully offline** | Ollama and a model on your own machine | Free forever | Once, to download |
   | **Free hosted** | A free Groq or Google AI Studio key, no card | Free, daily cap | Every lab |
   | **Your own key** | You already pay for OpenAI, Anthropic, Gemini | Per token | Every lab |

   Your choice is remembered, travels in the progress export, and can be changed at any time. Every
   package is explained: what it is, why the course needs it, how to check it worked, how to remove
   it. About twenty minutes.
3. Start at Module 1 and go in order. Each module's build reuses the one before it.

Works on Windows, macOS and Linux with **Python 3.10 or newer** (tested through 3.14).

---

## Layout

```
GenAI-Course/
  index.html          dashboard: progress, search, phase map, export/import
  setup.html          every install explained, plus troubleshooting
  modules/            one page per module, m01 … m18 plus the Build track m19 … m23
  assets/
    data.js           the module index: titles, concepts, videos, labs
    course.css        design tokens and all layout
    course.js         shared header/sidebar, progress, video embeds, quizzes, notes
    widgets.js        the 15 interactive teaching widgets
  my-work/labs/
    _shared/llm.py    the one provider-neutral model wrapper every lab imports
    tinygpt.py        the transformer you write in B2 and train in B3
    lab01 … lab23/    runnable starter files
```

Pages are linked with plain `<a href>` and share CSS/JS via `<link>` and `<script src>`. There is no
`fetch()` anywhere, because browsers block it for files opened from disk. That is why the course
works by double-click.

## How progress is stored

In `localStorage`, under keys prefixed `gaic:v1`. Chrome and Edge share one storage area across all
`file://` pages, so completing a module on one page shows up everywhere. Every storage call is
wrapped in `try`/`catch`; if your browser blocks it the page still works and shows a notice.

Each module tracks three things separately (read, lab, mini-project), so you can see at a glance
where you read something but did not build it.

**Export your progress** from the button on the home page if you care about keeping it. Clearing
browser data clears it.

---

## Extending it

**Add or edit a module's content.** Do not edit `modules/*.html`. Those files are generated and
your changes will be overwritten the next time anyone builds. Edit the authored content in
`tools/module_content_raw.json`, then rebuild:

```bash
python tools/reconcile.py && python tools/gen_pages.py && python tools/verify.py
```

The header, sidebar, progress bar and prev/next links are all rendered by `course.js` from
`assets/data.js`, so you never have to update navigation by hand.

**Add a module.** Append an entry to `modules` in `assets/data.js` and create the matching HTML file.
It appears in the sidebar and the home page on every page automatically. The shape is:

```js
{ n: 19, id: "m19", file: "m19-your-topic.html", phase: 6,
  title: "...", promise: "...", hours: 4,
  concepts: ["term", "term"],          // used by search
  videos: [{ id: "...", t: "...", c: "channel", v: 123456, m: 12, d: "2026-08", core: true }],
  lab:  { t: "...", g: "..." },
  mini: { t: "...", g: "..." },
  widgets: [] }
```

**Add a video.** Add it to that module's `videos` array. Check it first, because a video that is alive is
not necessarily embeddable:

```bash
curl -s "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=VIDEO_ID&format=json"
```

JSON back means it will embed. `Not Found` or `Unauthorized` means it will not, even if it plays
fine on YouTube. One of the best videos found for Module 5 had 2.6 million views and had to be cut
for exactly this reason.

**Add a widget.** Write a function in `assets/widgets.js` that takes a container element, register
it on the `W` object, add a title to `TITLES`, then drop `<div data-widget="yourname"></div>` on any
page. Widgets get a `try`/`catch` around them, so a broken one degrades instead of taking the page
down.

**Add a diagram.** Diagrams are inline SVG written into the HTML at build time rather than drawn by
script, so they print and survive JavaScript being off. Colours come from CSS custom properties
(`hsl(var(--ph))`, `var(--ink)`, `var(--line-strong)`), which is why they follow the phase hue and
both light and dark themes without any extra work. To edit one, find the `<figure class="diagram">`
block in the module page and change the SVG directly.

**Change the look.** Everything is CSS custom properties at the top of `assets/course.css`. The six
phase colours are `--p1` through `--p6` as `H S% L%` triples; a `data-phase` attribute anywhere sets
`--ph` for everything inside it.

---

## A note on currency

This field moves monthly. Everything time-sensitive here was checked against primary sources in
**August 2026**, and where a claim is contested or comes from a company with an interest in the
answer, the course says so. Model names and prices will drift first. If a lab fails with "model not
found", nothing is broken, just set `LLM_MODEL` in your `.env` to something current.

Video view counts were read from YouTube in August 2026 and will only have grown.
