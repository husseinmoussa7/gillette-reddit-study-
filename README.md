# Reactions to the Gillette "We Believe" advertisement on r/MensLib and r/MensRights

Replication materials for a study of how two men's-issues communities responded to Gillette's
January 2019 "We Believe: The Best Men Can Be" advertisement.

## Headline results

Among comments that take a position on the advertisement, **86.4% of r/MensRights is critical
against 12.9% of r/MensLib, which is 71.8% supportive** — a gap of 73.7 points (95% CI 69.6–78.3)
that holds across every specification and survives worst-case bounds on moderator-removed comments.

Tone tells a weaker story: both communities are strongly negative (mean polarity −0.34 vs −0.38),
and once differential moderation is accounted for the sign of the tone difference is not identified.
The contrast between the two measures is the study's methodological point.

| | r/MensLib | r/MensRights |
| --- | --- | --- |
| Comments collected | 1,743 | 15,280 |
| Comments analysed | 1,335 | 14,389 |
| Removed by moderators | 22.4% | 4.9% |
| Critical of the ad (of those positioned) | 12.9% | 86.4% |
| Resentment (0–3) | 0.33 | 0.91 |
| Zero-sum thinking (0–3) | 0.06 | 0.45 |

## Documents

- [`docs/findings.pdf`](docs/findings.pdf) — results
- [`docs/methodology.pdf`](docs/methodology.pdf) — measures, inference, validation, replication
- [`docs/prompt_verbatim.md`](docs/prompt_verbatim.md) — the LLM coding prompt as sent
- [`docs/REPLICATION.md`](docs/REPLICATION.md) — step-by-step pipeline

## Contents

```
code/      collection and analysis (Python 3.13; see docs/requirements-frozen.txt)
data/      analysis-ready posts and comments, with a codebook
results/   tables 1–8, specification curves, validation sets, summary JSON
figs/      11 figures at 200 dpi
docs/      write-ups, prompt, replication notes, manifest
```

## Data note

Usernames are **pseudonymised** with a salted hash. Author counts, repeat posting and cross-posting
between the two communities are all preserved; individuals are not identifiable. Comments removed
or deleted on Reddit carry no text in the source archive, so withdrawn content is not reproduced
here — only the fact and timing of its removal.

Source: the [Arctic Shift](https://arctic-shift.photon-reddit.com) public Reddit archive.

## Reproducing

```bash
pip install -r docs/requirements-frozen.txt
python code/scrape.py all MensLib          # collection, ~2.5 h
python -m analysis.prepare_llm             # cleaning and QA
python code/analysis/score.py all          # tone and emotion, ~35 min, CPU
python -m analysis.llm_code run            # stance and constructs, needs OPENAI_API_KEY
python -m analysis.report_tables           # tables 1–8
```

Full pipeline with runtimes and environment caveats in `docs/REPLICATION.md`.
