# Your work

Everything in this folder is yours. Nothing outside it is.

You run every lab and every mini-project from in here, and every file you
produce lands in here. The rest of the course, the module pages, the styling
and the build scripts, sits outside this folder and you never need to open it.

```
GenAI-Course/
  my-work/          <- you are here. Your workspace.
    .venv/          you make this on the setup page
    .env            you write this on the setup page. Never commit it.
    labs/
      lab01/ ... lab23/
      _shared/      helpers the labs import
      requirements.txt
    notes/          anything you want to keep
  modules/          the course pages. Reading material.
  assets/ tools/    how the pages are built. Ignore these.
```

`.venv` and `.env` are not here yet, and that is deliberate. Building the
environment is the first thing the course teaches, and the setup page installs
packages in groups as you reach the module that needs each one. Starting with
them already made would skip that.

## Provided, versus yours

Each lab folder starts with a few files that came with the course, and fills up
with files you write. Telling them apart:

| File | Whose | What it is |
|---|---|---|
| `README.md` | provided | the lab instructions, rewritten whenever the course is rebuilt |
| `check.py` | provided | marks your mini-project, same rule |
| everything else | **yours** | starter files you edit, and everything you create |

The two provided files are regenerated from the course source. If you edit
them your changes are overwritten on the next build, so treat them as
read-only. Every other file in here is safe: the build never touches it.

To see exactly what you have made so far, list the files that are not one of
those two:

```bash
python -c "import pathlib;[print(p) for p in sorted(pathlib.Path('labs').rglob('*')) if p.is_file() and p.name not in ('README.md','check.py')]"
```

## Running a lab

Activate the environment from the course root, then work from inside the lab
folder. The labs all share one environment, so you only ever build it once.

```bash
.\my-work\.venv\Scripts\Activate.ps1
cd my-work\labs\lab01
python hello_llm.py
```

Your prompt gains a `(.venv)` label when activation worked. It lasts only for
that terminal window, so you run the first line again each time you sit down.

Run things from inside the lab folder rather than from the course root. Two
reasons: the labs reach their shared helpers by looking one folder up, and the
helper finds your `.env` by searching from where you are standing and then
working upwards. From the course root it would never look inside `my-work`, so
your settings would be silently ignored and the defaults would take over.

The labs find their shared helpers by looking one folder up from themselves, so
they only work when run from inside their own folder. If you see
`ModuleNotFoundError: llm`, that is almost always the reason.

## Marking a mini-project

```bash
cd my-work/labs/lab01
python check.py
```

It prints one line per check and exits non-zero until everything passes. It
only ever reads your files, it never writes.

## Starting over on one lab

Delete the files you made and keep the two provided ones. Nothing else in the
course depends on your work, so there is no wider damage to undo.
