# Replication package — Gillette "We Believe" ad on r/MensLib and r/MensRights

Everything needed to reproduce the collection, the measures and every reported number.
Last run: 14 October 2026. Machine: macOS, Intel i7-6700K, 16 GB RAM, CPU only (no GPU required).

---

## 1. What the study does

Collects all Reddit posts and comments about the Gillette "We Believe: The Best Men Can Be"
advertisement (released 14 January 2019) from two subreddits, r/MensLib and r/MensRights, over
January 2019 – September 2026, and measures two things:

- **Tone** — affective valence of the text, by local transformer and lexicon models.
- **Stance and constructs** — position on the ad (supportive / critical / mixed / not about the ad)
  and four psychological constructs (resentment, identity threat, zero-sum thinking, hostility),
  by LLM coding.

The headline results: tone differs little between the communities and its sign is not identified
once differential moderation is accounted for; stance differs enormously (86% vs 13% critical among
positioned comments) and survives the same sensitivity analysis.

---

## 2. Environment

```bash
python 3.13.5          # see requirements-frozen.txt for exact package versions
pip install -r requirements-frozen.txt
```

Two environment notes that will otherwise cost a replicator time:

- This machine carries `torchvision` 0.2.0 (2018) against `torch` 2.6, which breaks `transformers`
  model loading. `analysis/score.py::_patch_torchvision()` shims the two attributes
  `transformers` looks for. On a clean install with a matching `torchvision`, the shim is a no-op.
- `sentence_transformers` imports `torchvision.transforms.v2`, absent here, so `analysis/themes.py`
  computes MiniLM embeddings directly through `transformers` with mean pooling and L2
  normalisation. This is numerically what `SentenceTransformer.encode(normalize_embeddings=True)`
  does for `all-MiniLM-L6-v2`.

The LLM step needs an OpenAI API key in `.env` at the project root:

```
OPENAI_API_KEY=sk-proj-…
```

`.env` is chmod 600 and listed in `.gitignore`. It is **not** part of the shared package.

---

## 3. Pipeline, in order

| Step | Command | Runtime | Output |
| --- | --- | --- | --- |
| 1. Collect | `python scrape.py all MensLib` then `… MensRights` | ~2.5 h | `raw/**.jsonl` (127 MB) |
| 2. Build tables | `python summary.py` | ~1 min | `data/posts.parquet`, `data/comments.parquet`, `summary.md` |
| 3. Clean + QA | `python -m analysis.prepare_llm` | ~30 s | `analysis/llm_units.parquet`, `results/cleaning_report.json` |
| 4. Tone + emotion | `python analysis/score.py all` | ~35 min | `analysis/scores.parquet` |
| 5. Stance + constructs | `python -m analysis.llm_code pilot 200` then `… run` | ~25 min | `analysis/llm_codes.parquet` |
| 6. Tone inference | `python analysis/compare.py` | ~6 min | `results/report.json`, `spec_curve.csv` |
| 7. Stance inference | `python -m analysis.stance_compare` | ~4 min | `results/stance_report.json` |
| 8. Themes | `python -m analysis.themes` | ~12 min | `results/themes.csv`, `cluster_exemplars.csv` |
| 9. Figures | `python analysis/figures.py` | ~1 min | `figs/*.png` (dpi 200) |
| 10. Tables | `python -m analysis.report_tables` | ~2 min | `results/tables/*.csv`, `tables.md` |
| 11. Deliverables | `python build_analysis_docx.py && python build_xlsx.py` | ~2 min | `.docx`, `.xlsx` |

Steps 1, 4, 5 and 8 are resumable: every page, batch and API result is cached, so an interrupted
run continues where it stopped. Steps 6, 7, 10 are deterministic given fixed seeds
(`numpy.random.default_rng(20260928 / 20261004 / 20261014)`).

---

## 4. Data sources and collection

**Source:** the Arctic Shift Reddit archive (`arctic-shift.photon-reddit.com`), free, no API key.
Reddit's own API caps search at ~1,000 results and skews recent; Pushshift is restricted to
moderators. Arctic Shift's keyword search times out on a subreddit as large as r/MensRights, so
posts were **not** keyword-searched: `scrape.py` lists *every* post in both subreddits month by
month (171,723 posts) and filters locally. Post recall is therefore complete rather than capped.

**Comment collection, two sweeps, merged and de-duplicated:**

1. *Thread sweep* — every comment inside a thread whose post matches, whether or not the comment
   itself mentions the ad. This preserves conversation: 13,856 comments never name the ad but
   reply within those threads.
2. *Keyword sweep* — comments mentioning Gillette anywhere else in the same two subreddits.

Flags `from_thread` and `from_kw_search` record provenance; both are reported in the specification
curves.

**Search terms:** `Gillette` plus misspellings `Gilette`, `Gillete`. "We Believe" could not be
searched as a phrase (the server matches the words separately), so it is matched locally.

**Relevance rule:** a unit is relevant if it matches the strict phrases
(`gil+et+e('s)? (ad|commercial|advertisement)`, `we believe … (ad|commercial|advertisement)`), **or**
mentions Gillette alongside an ad-context word (ad, commercial, campaign, boycott, toxic
masculinity, P&G), **or** mentions Gillette between 13 Jan and 30 Jun 2019. Comments also qualify by
sitting in a relevant thread. The unrelated Moore & Gillette book on masculinity is excluded.

**Collection verification:** comments retrieved per thread were reconciled against each post's own
`num_comments`: median 100% per thread, 105% in aggregate (the archive also holds comments added
after that count was recorded). No duplicate IDs. Live spot-checks against reddit.com were not
possible — Reddit blocks automated fetching from the collection machine — so every row stores a
`permalink` for manual checking.

---

## 5. Sample construction

`analysis/prepare_llm.py` assigns every one of 17,643 units a status. **Nothing is deleted**; the
counts reconcile exactly (see `results/cleaning_report.json` and Table 1).

| Status | Rule | n |
| --- | --- | --- |
| `send` | has codeable text; enters all analysis | 16,343 |
| `skip_removed` | body is `[removed]`/`[deleted]` — no text survives | 1,145 |
| `skip_no_content` | bare URL, bare markdown link, single character | 114 |
| `skip_bot` | bot account or "I am a bot" boilerplate | 26 |
| `skip_mod_notice` | moderator action notice, not an opinion | 15 |

Decisions worth stating in a methods section: very short comments are **kept** (`#boycottgillette`
carries stance); bare links are not. Posts whose *body* was removed are kept because the title
survives (100 such posts). Text normalisation is limited to HTML entity decoding, zero-width
character removal and whitespace collapsing — no stemming, no stopword removal, no case folding.

---

## 6. Measures

### 6.1 Tone (local, no API)

| Measure | Model | Output |
| --- | --- | --- |
| Polarity (primary) | `cardiffnlp/twitter-roberta-base-sentiment-latest` | p(positive) − p(negative), range −1…+1 |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 7 probabilities; `emotion` = argmax |
| Polarity (robustness) | VADER 3.3.2 | compound score, −1…+1 |

Truncation at 256 word-piece tokens; batch 32, length-sorted for tight padding; `torch.set_num_threads(4)`.
256 rather than 128 matters: 128 would truncate ~15% of comments and truncation is *asymmetric
across subreddits* (r/MensLib comments are twice as long), which would manufacture the effect being
tested.

### 6.2 Stance and constructs (LLM)

`gpt-4.1-mini`, temperature 0, one call per unit, OpenAI structured outputs with a strict JSON
schema. 16,221 unique texts (identical texts coded once, label copied to duplicates), 0 failures,
$8.90, 25 minutes at 8 concurrent workers.

**The verbatim system prompt and response schema are in `analysis/prompt_verbatim.md`.** In summary
it asks for one `ad_stance` label and four constructs rated 0–3 (0 absent, 1 mild, 2 clearly
present, 3 strong and central), plus a confidence and a ≤15-word rationale retained for auditing.

Six disambiguation rules, five of which were added after the pilot:

1. Code only the author's own words; lines beginning `>` are quoted material.
2. Judge position, not tone.
3. Attacking the ad's **critics** is supportive; attacking its **defenders** is critical.
4. A reproduced news headline with no added opinion is `not_about_ad`; a headline carrying the
   sharer's sarcasm or an added tag (`[Humor]`, "NSFI") is coded.
5. Boycotting Gillette or buying a rival brand *because of the ad* is critical.
6. A stance is coded only when the comment itself refers to the ad — never inferred from the
   thread's topic.

Rules 4–6 came out of a 200-unit pilot that revealed two systematic errors: neutral news headlines
read as endorsements, and stance inferred from thread topic for deep reply-chain comments. Adding
them raised blind agreement from κ = 0.52 to κ = 0.72. Pilot-prompt codings were **discarded**, not
merged, so the released corpus is internally consistent under the final prompt
(`analysis/_ckpt/llm_codes_v1.jsonl` and `_v2.jsonl` retain the superseded versions for audit).

### 6.3 Themes

MiniLM-L6-v2 embeddings (mean-pooled, L2-normalised, 256 tokens) → TruncatedSVD to 50 dimensions →
KMeans, k chosen by silhouette over 8–20 (k = 8 selected). Clusters labelled by hand after reading
c-TF-IDF top terms and the six nearest-centroid comments per cluster from both subreddits
(`results/cluster_exemplars.csv`, labels in `results/theme_names.json`). **Silhouette peaks at
0.077**: the clusters are soft and should be reported as rough groupings, not a latent structure.

---

## 7. Inference

Comments are not independent — they nest in threads, and five threads carry 83% of r/MensLib's
comments. Every reported interval accounts for this.

- **Primary tone specification:** OLS of polarity on a r/MensRights indicator with log(1+words),
  top-level indicator and period fixed effects, **thread-clustered standard errors**, plus a
  **wild cluster bootstrap** (Rademacher weights, 2,000 replications, clustered on thread) because
  the treatment is constant within cluster and the cluster-size distribution is extreme.
- **Unadjusted differences:** bootstrap resampling **whole threads** (2,000 replications).
- **Stance:** logistic regression of `critical` on the same covariates with thread-clustered SEs,
  plus the thread-bootstrapped difference in critical share.
- **Effect sizes:** raw difference, Cohen's d and Cliff's delta. Cliff's delta is preferred for
  polarity because the distribution is bimodal at ±1, which makes a pooled-SD d misleading.
- **Comment score is never a control** — it is downstream of the comment and would be a collider.
  The polarity-vs-score figure is descriptive only.

**Differential moderation** (r/MensLib 22.4% removed vs r/MensRights 4.9%, text unrecoverable) is
selective attrition on the outcome. Two bound sets are reported:

- *Lee bounds* — equalise retention by trimming 19.3% from the lower-removal group's tails.
- *Manski worst-case bounds* — assign unobserved removed comments their most adverse values.

Tone: bounds contain zero, so the sign is **not identified**. Stance: bounds exclude zero, so the
direction holds even in the worst case. This asymmetry is the central methodological result.

**Specification curves** (`results/spec_curve.csv`, `results/stance_spec_curve.csv`) re-estimate
across date windows, provenance subsets, confidence thresholds, top-level-only, excluding the 21
cross-posting authors, and downsampling r/MensRights.

---

## 8. Validation

| Measure | n hand-labelled | Agreement | Cohen's κ | 95% CI |
| --- | --- | --- | --- | --- |
| RoBERTa polarity | 80 | 72.5% | 0.588 | [0.44, 0.72] |
| VADER | 80 | 63.7% | 0.450 | [0.29, 0.61] |
| gpt-4.1-mini stance | 40 | 85.0% | 0.718 | [0.47, 0.89] |

Labels were assigned without sight of model output, on stratified samples spanning both subreddits
and the full score range. Both label sets ship with the package
(`results/handlabel_validation.csv`, `results/stance_blind_validation.csv`), each including the
model's label so the κ is recomputable.

**Disclosure for any write-up:** these hand labels were produced by the research assistant (Claude)
that also ran the models, not by an independent human coder. They are therefore a check on
face validity and on systematic error patterns, **not** independent human validation. Reported
accuracy is similar across the two subreddits (RoBERTa 75.0% vs 70.0%), which is the property that
matters for a between-community comparison: measurement error does not obviously favour one side.
If a reviewer asks for independent human validation, the two labelled samples ship ready for a
second coder to work through blind (model labels sit in separate columns), but the coding is not
held up pending that.

---

## 9. Files

`MANIFEST.csv` lists every file with size and a SHA-256 prefix.

**Code** — `scrape.py`, `summary.py`, `analysis/{prepare_llm,score,llm_code,compare,stance_compare,themes,figures,report_tables}.py`,
`build_{docx,xlsx,analysis_docx}.py`, `analysis/prompt_verbatim.md`, `requirements-frozen.txt`.

**Data** — `data/posts.parquet` + `.csv` (778 posts), `data/comments.parquet` + `.csv` (19,268
comments, all statuses), `analysis/llm_units.parquet` (17,643 units with cleaning status),
`analysis/scores.parquet` (tone + emotion), `analysis/llm_codes.parquet` (stance + constructs),
`raw/` (127 MB of cached API pages, the collection's raw evidence).

**Shareable extract** — `data/share/`: `posts_relevant.csv`, `comments_relevant.csv` (33 columns
including all measures), `codebook.csv` (every column defined), `README.md`,
`Gillette_Reddit_Data.xlsx` (Read me / Posts / Comments / Results / Codebook sheets).

**Results** — `analysis/results/`: `report.json` (tone), `stance_report.json`,
`cleaning_report.json`, `tables/*.csv` + `tables.md` (Tables 1–8), `spec_curve.csv`,
`stance_spec_curve.csv`, `themes.csv`, `cluster_exemplars.csv`, both validation sets.
`figs/` holds 11 figures at dpi 200.

**Write-ups** — `Gillette_Sentiment_and_Stance_Comparison.docx`, `Gillette_Reddit_Scrape_Summary.docx`.

---

## 10. Ethics and data handling

Public Reddit content, collected from a public archive, no authentication and no private data.
Usernames are retained as posted; **pseudonymise `author` before any public release** — a stable
hash preserves the ability to count authors and detect cross-posting without naming individuals.
Removed and deleted comments carry no text in the archive, so content their authors withdrew is
not reproduced here; only the fact and timing of removal is retained. Quoted examples in the
write-ups are short and non-identifying. If the journal requires a public deposit, the shareable
extract plus code is sufficient to reproduce every number; `raw/` is large and need not be
deposited, though it can be regenerated by rerunning step 1.

---

## 11. Known limitations

1. **Tone is not stance** — resolved for the research question by §6.2, but the tone measures
   themselves remain valence, not position.
2. **Validation labels were produced by the assistant that ran the models**, not an independent
   coder (see §8). They evidence face validity and error patterns rather than inter-rater
   reliability.
3. **Clustering** — r/MensLib's effective sample is ~5–10 conversations, not 1,335 observations;
   for stance, 58 threads contain a positioned comment and five hold 81% of them.
4. **Soft clusters** — themes are interpretive groupings (silhouette 0.077).
5. **Model dependence** — stance figures are conditional on `gpt-4.1-mini` and the prompt in
   `analysis/prompt_verbatim.md`. A different model or prompt would need re-validation; the pilot
   showed prompt wording moved κ by 0.20.
6. **Asymmetric populations** — 4,846 r/MensRights authors vs 549 in r/MensLib, and time spans that
   diverge after 2019.
