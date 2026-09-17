# Development Provenance — A Factual Account

This document exists to answer one question plainly: where did this code
come from, and how was it actually developed? Written for a supervisor,
reviewer, or committee member who wants a straight answer, not a defense.

## The short version

This code was developed with heavy AI assistance (Claude, Anthropic's
coding assistant), working interactively with the student across an
extended series of sessions, from **2026-02-19 to 2026-09-17**. It was
**not** copied or scraped from an existing GitHub repository. It was
built incrementally: architecture translated from the thesis proposal's
own description, then trained, tested, broken, debugged, and re-trained
in a real, iterative loop — the same loop any implementation goes
through, just with an AI doing much of the typing.

Both of those facts are true at once, and neither cancels the other out.
The rest of this document is the evidence for the first claim (not
copied); the honest disclosure of the second (AI-assisted) is not in
dispute and shouldn't be treated as if it were.

## Evidence this was not copied from an external repository

**1. The git history has the shape of real, iterative debugging — not a
one-time import.**

42 commits across two real work sessions (2026-02-19, and an extended
2026-09-17 session with 36 commits). Copy-pasted code from an existing
repository arrives as one commit, or a handful of mechanical ones. This
history instead shows the exact shape of someone building something and
finding it broken, repeatedly, in ways specific to *this* dataset and
*this* architecture — for example:

- A metric-averaging bug found by manually comparing raw pixel
  predictions between two checkpoints, not by reading the metric code
  and spotting an obvious mistake.
- A checkpoint-selection bug found by loading `checkpoints/best.pt`
  directly and discovering it held a *worse* epoch than one that had
  already been produced and discarded.
- A train/val split-instability bug specific to this exact dataset
  having grown from 202 to 352 to 801 tiles across separate download
  sessions — not a generic, textbook bug.

None of these are bugs you'd find by copying a working repository. They're
the kind of bug you only find by running your own code against your own
data and noticing the output doesn't match your own expectations.

**2. `docs/MANUAL.md` is a 995-line, dated, evidence-by-evidence technical
log — not documentation written after the fact.**

It records 23 distinct bugs (each marked `BUG THIS FIXES` inline in the
code itself, next to the fix), what was actually observed, how it was
diagnosed, and what changed — including corrections to the log's *own*
earlier entries when a later investigation proved an earlier conclusion
wrong (see MANUAL.md §12.3, which explicitly supersedes §12.1-§12.2
rather than silently rewriting them). A copied repository doesn't come
with a log of its own mistakes, corrected in view, in the order they
happened.

**3. The experiment results are real, timestamped, and reproducible from
this repo's own scripts.**

Every number in `docs/MANUAL.md`'s tables comes from an actual training
run against the real, public SpaceNet-8 dataset (downloaded via
`prepare_real_data.py` from the public `spacenet-dataset` S3 bucket — no
credentials needed, independently verifiable by anyone). 45 automated
tests (`tests/`, runnable with `python -m pytest tests/ -v`) pass against
this exact code as of this document. None of this is copy-pasted from
elsewhere; running `train.py` against the downloaded data reproduces it.

**4. Some pieces resemble other public implementations because they
implement the same well-known, published techniques — that's expected,
not suspicious.**

A U-Net decoder looks like other U-Net decoders. A ResNet-style block
looks like other ResNet-style blocks. A correct implementation of a
published architecture *should* structurally resemble other correct
implementations of it — that's a sign of correctness, not copying. What
would actually indicate copying is code that doesn't match this project's
own specific data format, doesn't handle this project's own specific
edge cases, or doesn't have this project's own documented history of
getting things wrong first. None of that is present here.

## What's true, disclosed plainly

This implementation was written with an AI coding assistant doing the
majority of the direct typing, across a long, closely-directed
session — the student set direction, made the calls on what to try next,
reviewed and approved changes throughout, and pushed every commit. If
the standard for this thesis component requires the student to have
personally written the implementation code without AI assistance, that's
a real question this document can't answer for you — it's a policy
question about what "the student's own work" means for an AI-assisted
engineering process, and it deserves a direct conversation with your
program, not a hedge.

## If you want to verify any of this yourself

- `git log --format="%ad %s" --date=short` shows the real commit dates
  and messages.
- `docs/MANUAL.md` is the full technical log, in order, uncut.
- `python -m pytest tests/ -v` runs the real test suite against the real
  code.
- `python prepare_real_data.py --aoi Germany_Training_Public --out-dir test_download`
  independently re-downloads real SpaceNet-8 data from the public bucket,
  proving the data pipeline isn't fabricated.
