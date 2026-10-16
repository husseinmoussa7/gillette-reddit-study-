"""LLM coding of ad stance and four psychological constructs.

Codes each unit prepared by analysis/prepare_llm.py (status == "send") on:
    ad_stance        supportive | critical | mixed | not_about_ad
    resentment       0-3   grievance at unfair treatment of one's group
    identity_threat  0-3   one's group identity devalued, attacked or misrepresented
    zero_sum         0-3   one group's gain framed as another's loss
    hostility        0-3   antagonism toward people or groups
plus a confidence and a short rationale for auditing.

Resumable: every result is appended to analysis/_ckpt/llm_codes.jsonl and re-runs skip
what is already coded. Identical texts are coded once and the label copied.

Usage:
    python -m analysis.llm_code pilot [n]   # default 200, writes a pilot file to hand-check
    python -m analysis.llm_code run         # the full corpus
    python -m analysis.llm_code merge       # fold codes into analysis/llm_codes.parquet
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "llm_units.parquet"
CKPT = ROOT / "analysis" / "_ckpt" / "llm_codes.jsonl"
OUT = ROOT / "analysis" / "llm_codes.parquet"
CKPT.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-mini"
WORKERS = 8
# per-1M-token prices, for the running cost estimate only
PRICE_IN, PRICE_OUT = 0.40, 1.60

SYSTEM = """You are coding Reddit comments for a marketing research study of reactions to the \
January 2019 Gillette advertisement "We Believe: The Best Men Can Be", which addressed toxic \
masculinity and provoked a large backlash. Comments come from r/MensLib (a progressive men's \
issues forum) and r/MensRights (a men's rights forum).

Code ONLY the author's own words. Lines beginning with ">" are quoted from someone else: use \
them as context, never as the author's position.

Return these fields.

ad_stance - the author's position on the Gillette ad:
  supportive   defends, praises or agrees with the ad or its message
  critical     attacks, mocks or rejects the ad, its message, or Gillette for making it
  mixed        genuinely both, or explicitly ambivalent
  not_about_ad the comment does not take a position on the ad (off-topic, procedural,
               product talk, or discussing masculinity without reference to the ad)
Judge the position, not the tone: warm language praising a rebuttal video is "critical";
an angry comment defending the ad is "supportive".

Then rate each construct 0-3, where 0 = absent, 1 = mild or implicit, 2 = clearly present,
3 = strong and central to the comment:

resentment      Grievance or bitterness that the author's group is treated unfairly, taken for
                granted, or wronged. Not mere disagreement: it carries a sense of injustice done
                to "us".
identity_threat Sense that being a man, or masculinity itself, is devalued, attacked, pathologised
                or misrepresented. Includes "they think all men are predators".
zero_sum        Frames one group's gains as necessarily another's losses: advances for women or
                feminism cost men, attention to one issue steals it from another.
hostility       Antagonism toward people or groups: insults, contempt, dehumanising language,
                mockery aimed at persons, or calls for harm. Anger alone is not hostility unless
                directed at people or groups.

confidence  0-1, your confidence in the ad_stance label.
rationale   at most 15 words, citing what drove the call.

Four rules that decide the hard cases:

1. WHO IS BEING ATTACKED. Mocking or attacking the ad's CRITICS is "supportive"; mocking or
   attacking the ad's DEFENDERS is "critical". "Cult of Dusty responds to right wing snowflakes
   on the Gillette ad" attacks the backlash, so it is supportive.
2. SHARED HEADLINES. A link post or comment that merely reproduces a news headline, article title
   or video title with no added opinion is "not_about_ad", even when the headline sounds
   favourable or damning. Only code a stance when the author adds their own evaluation.
   "Gillette Tackles #MeToo, Toxic Masculinity in New Ad" = not_about_ad.
   "Gillette's disgusting new ad shames men" = critical, because the wording editorialises.
3. BUYING AS A STATEMENT. Switching away from Gillette, boycotting it, or buying from a brand
   that answered back (Egard watches, Harry's, Dollar Shave Club) BECAUSE of the ad is
   "critical". Plain product chat with no link to the ad stays "not_about_ad".
4. TONE IS NOT POSITION. An angry comment can be supportive and a polite one critical. Code the
   position taken, never the mood.
5. THE AD MUST BE IN THE COMMENT. Most comments sit inside threads about the ad, so the thread
   topic tells you nothing. Code a stance ONLY when this comment itself refers to the ad, the
   campaign, Gillette's handling of it, or an identifiable response to it (the Egard video, the
   boycott, "the commercial", "this ad"). A reply arguing about masculinity, feminism, therapy,
   play-fighting or male suicide, with no reference to the ad, is "not_about_ad" however strongly
   it argues and however well its argument lines up with one side. Do not infer the position the
   author would probably hold.
6. SARCASM AND EDITORIAL TAGS IN TITLES. Rule 2 covers plain headlines only. A title carrying the
   sharer's own sarcasm, scare quotes or an added tag - "[Humor] Gillette reaping all those
   dollars", "NSFI (Not Safe For Idiots)", "MORE EVIL THAN YOU THINK" - is an evaluation: code it
   critical or supportive accordingly."""

SCHEMA = {
    "name": "comment_coding",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ad_stance": {"type": "string", "enum": ["supportive", "critical", "mixed", "not_about_ad"]},
            "resentment": {"type": "integer", "minimum": 0, "maximum": 3},
            "identity_threat": {"type": "integer", "minimum": 0, "maximum": 3},
            "zero_sum": {"type": "integer", "minimum": 0, "maximum": 3},
            "hostility": {"type": "integer", "minimum": 0, "maximum": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["ad_stance", "resentment", "identity_threat", "zero_sum", "hostility",
                     "confidence", "rationale"],
    },
}

_lock = threading.Lock()
_usage = {"in": 0, "out": 0, "n": 0, "err": 0}


def client():
    env = ROOT / ".env"
    if env.exists() and not os.environ.get("OPENAI_API_KEY"):
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    return OpenAI()


def user_msg(r):
    where = f"Subreddit: r/{r.subreddit}\nPosted: {str(r.created)[:10]}\nType: {r.kind}"
    return f"{where}\n\n---\n{r.text_for_llm}\n---"


def code_one(cli, r, tries=4):
    for a in range(tries):
        try:
            resp = cli.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_schema", "json_schema": SCHEMA},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_msg(r)}],
            )
            out = json.loads(resp.choices[0].message.content)
            out["id"] = r.id
            with _lock:
                _usage["in"] += resp.usage.prompt_tokens
                _usage["out"] += resp.usage.completion_tokens
                _usage["n"] += 1
            return out
        except Exception as e:
            if a == tries - 1:
                with _lock:
                    _usage["err"] += 1
                return {"id": r.id, "error": repr(e)[:200]}
            time.sleep(2 ** a)


def done_ids():
    if not CKPT.exists():
        return set()
    ids = set()
    for line in CKPT.open():
        try:
            ids.add(json.loads(line)["id"])
        except Exception:
            continue
    return ids


def run(units, label=""):
    cli = client()
    have = done_ids()
    todo = units[~units.id.isin(have)]
    print(f"{label}coding {len(todo)} units ({len(have)} already done), model={MODEL}", flush=True)
    if not len(todo):
        return
    t0 = time.time()
    with CKPT.open("a") as fh, ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(lambda r: code_one(cli, r), todo.itertuples()), 1):
            fh.write(json.dumps(res) + "\n")
            if i % 50 == 0:
                fh.flush()
                el = time.time() - t0
                cost = _usage["in"] / 1e6 * PRICE_IN + _usage["out"] / 1e6 * PRICE_OUT
                print(f"  {i}/{len(todo)}  {i/el:.1f}/s  ${cost:.2f} so far  "
                      f"eta {(len(todo)-i)/max(i/el,1e-9)/60:.0f} min  errors={_usage['err']}", flush=True)
    cost = _usage["in"] / 1e6 * PRICE_IN + _usage["out"] / 1e6 * PRICE_OUT
    print(f"done in {(time.time()-t0)/60:.1f} min; {_usage['n']} coded, {_usage['err']} errors, "
          f"~${cost:.2f} spent", flush=True)


def load_sendable():
    u = pd.read_parquet(UNITS)
    s = u[u.status == "send"].copy()
    return s[~s.is_dup], s      # code unique texts only; duplicates inherit


def merge():
    codes = [json.loads(l) for l in CKPT.open()]
    c = pd.DataFrame(codes)
    bad = c[c.get("error").notna()] if "error" in c else pd.DataFrame()
    c = c[c.ad_stance.notna()] if "ad_stance" in c else c
    uniq, allsend = load_sendable()
    u = pd.read_parquet(UNITS)
    # copy labels from the coded representative onto identical texts
    rep = allsend[["id", "dup_of"]].merge(c, left_on="dup_of", right_on="id", suffixes=("", "_rep"))
    rep = rep.drop(columns=["id_rep", "dup_of"])
    out = u.merge(rep, on="id", how="left")
    out.to_parquet(OUT)
    print(f"coded {len(c)} unique texts, {out.ad_stance.notna().sum()} units carry a label, "
          f"{len(bad)} errors\nwrote {OUT}")
    print(out.groupby("subreddit").ad_stance.value_counts().unstack(fill_value=0))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        uniq, _ = load_sendable()
        # stratified: both subs, both kinds, across the flag space
        pilot = (uniq.groupby(["subreddit", "kind"], group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), max(5, n // 3)), random_state=11)))
        extra = uniq[uniq.quote_heavy | uniq.is_short].sample(min(40, len(uniq)), random_state=11)
        pilot = pd.concat([pilot, extra]).drop_duplicates("id").head(n)
        run(pilot, "PILOT: ")
        merge()
    elif cmd == "run":
        uniq, _ = load_sendable()
        run(uniq)
        merge()
    elif cmd == "merge":
        merge()
