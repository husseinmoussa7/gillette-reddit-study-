"""Data cleaning and QA before the LLM coding pass.

Nothing is deleted. Every relevant comment and post gets a `status`:
    send              -> goes to the LLM
    skip_removed      -> body is [removed]/[deleted]; no text exists to code
    skip_bot          -> bot account or "I am a bot" boilerplate
    skip_mod_notice   -> moderator action notice, not an opinion
    skip_no_content   -> nothing codeable: bare URL, bare markdown link, single character
Skipped rows stay in the output with status and reason, so every number downstream
reconciles back to the full corpus.

Usage: python -m analysis.prepare_llm   -> analysis/llm_units.parquet + a QA report
"""
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "llm_units.parquet"
QA = ROOT / "analysis" / "results" / "cleaning_report.json"

BOTS = {"AutoModerator", "sneakpeekbot", "WikiTextBot", "TotesMessenger", "RemindMeBot", "B0tRank"}
GONE = {"[removed]", "[deleted]", "[removed by reddit]", "[unavailable]"}
MOD_NOTICE = re.compile(
    r"comment was removed|post has been removed|removed for violating|has been removed because"
    r"|automatically removed|do not allow links to that site|violating the following rule", re.I)
BOT_TAIL = re.compile(r"I am a bot", re.I)
URL_ONLY = re.compile(r"^\s*(https?://\S+|\[[^\]]*\]\(\s*https?://[^)]*\))\s*$", re.I)
MAX_LLM_WORDS = 600          # longer comments are truncated for the LLM, flagged below


def quoted_share(t: str) -> float:
    """Fraction of characters sitting in markdown quote lines (another user's words, usually)."""
    lines = [l for l in t.split("\n") if l.strip()]
    if not lines:
        return 0.0
    tot = sum(len(l) for l in lines)
    return round(sum(len(l) for l in lines if l.lstrip().startswith(">")) / max(tot, 1), 3)


def clean_text(t: str) -> str:
    """Light normalisation only - never changes meaning, never drops the author's words."""
    t = t.replace("​", "").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def load_units():
    c = pd.read_parquet(ROOT / "data" / "comments.parquet")
    c = c[c.relevant].copy()
    c["kind"] = "comment"
    c["raw"] = c.body.fillna("")

    p = pd.read_parquet(ROOT / "data" / "posts.parquet")
    p = p[p.relevant].copy()
    p["kind"] = "post"
    # a post's codeable text is its title plus selftext; link posts still have a title
    body = p.selftext.fillna("")
    p["raw"] = (p.title.fillna("") + "\n\n" + body.where(~body.isin(GONE), "")).str.strip()
    p["link_id"] = "t3_" + p.id
    p["parent_id"] = ""
    p["post_removed_body"] = body.isin(GONE)

    cols = ["id", "kind", "subreddit", "created", "author", "raw", "score", "link_id", "parent_id",
            "kw_strict", "kw_broad", "removed"]
    for df in (c, p):
        for col in cols:
            if col not in df:
                df[col] = pd.NA
    return pd.concat([c[cols], p[cols]], ignore_index=True)


def classify(df):
    t = df.raw.map(clean_text)
    df["text"] = t
    df["n_words"] = t.str.split().str.len().fillna(0).astype(int)
    df["n_chars"] = t.str.len()
    df["quoted_share"] = t.map(quoted_share)
    df["is_bot"] = df.author.isin(BOTS) | t.str.contains(BOT_TAIL, na=False)
    df["is_mod_notice"] = t.str.contains(MOD_NOTICE, na=False)
    gone = t.isin(GONE) | (df.kind.eq("comment") & df.removed.fillna(False))
    nothing = (t.str.strip() == "") | (df.n_chars <= 1) | t.str.match(URL_ONLY, na=False)

    status = pd.Series("send", index=df.index)
    reason = pd.Series("", index=df.index)
    for mask, s, r in [
        (gone, "skip_removed", "body removed or deleted; no text survives"),
        (df.is_bot, "skip_bot", "bot account or bot boilerplate"),
        (df.is_mod_notice, "skip_mod_notice", "moderator action notice, not an opinion"),
        (nothing, "skip_no_content", "no codeable text (empty, single character, or bare link)"),
    ]:
        hit = mask.fillna(False) & status.eq("send")
        status[hit] = s
        reason[hit] = r
    df["status"] = status
    df["skip_reason"] = reason

    # flags that travel with the sent rows
    df["is_short"] = df.n_words < 3                      # kept: "#boycottgillette" carries stance
    df["is_long"] = df.n_words > MAX_LLM_WORDS           # truncated for the LLM
    df["quote_heavy"] = df.quoted_share > 0.6            # judge the author's words, not the quote
    df["text_for_llm"] = df.text.where(~df.is_long, df.text.str.split().str[:MAX_LLM_WORDS].str.join(" ") + " […truncated]")
    df.loc[df.status != "send", "text_for_llm"] = ""

    # identical texts are coded once and the label copied, for consistency and cost
    h = df.text_for_llm.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16] if s else "")
    df["text_hash"] = h
    first = df[df.status.eq("send")].drop_duplicates("text_hash").set_index("text_hash").id
    df["dup_of"] = df.text_hash.map(first).where(df.status.eq("send"), pd.NA)
    df["is_dup"] = df.status.eq("send") & (df.dup_of != df.id)
    return df


def qa(df):
    rep = {
        "units_total": int(len(df)),
        "by_kind": df.kind.value_counts().to_dict(),
        "by_status": df.status.value_counts().to_dict(),
        "by_status_subreddit": {k: v.to_dict() for k, v in df.groupby("subreddit").status.value_counts().unstack(fill_value=0).iterrows()},
        "to_send": int((df.status == "send").sum()),
        "unique_texts_to_send": int(df[df.status == "send"].text_hash.nunique()),
        "duplicates_collapsed": int(df.is_dup.sum()),
        "flags_among_sent": {
            "short (<3 words)": int(df[df.status == "send"].is_short.sum()),
            "long (>600 words, truncated)": int(df[df.status == "send"].is_long.sum()),
            "quote heavy (>60% quoted)": int(df[df.status == "send"].quote_heavy.sum()),
        },
        "integrity": {
            "duplicate ids": int(df.id.duplicated().sum()),
            "null created": int(df.created.isna().sum()),
            "subreddits": sorted(df.subreddit.dropna().unique().tolist()),
            "date_range": [str(df.created.min()), str(df.created.max())],
            "sent rows with empty text": int(((df.status == "send") & (df.text_for_llm.str.strip() == "")).sum()),
            "skipped rows with text kept": int(((df.status != "send") & (df.text.str.strip() != "")).sum()),
        },
    }
    # reconciliation: nothing may vanish
    assert len(df) == rep["units_total"], "row count changed"
    assert df.status.isin(["send", "skip_removed", "skip_bot", "skip_mod_notice", "skip_no_content"]).all()
    return rep


def main():
    df = classify(load_units())
    df.to_parquet(OUT)
    rep = qa(df)
    QA.parent.mkdir(parents=True, exist_ok=True)
    QA.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {OUT} ({len(df)} rows, nothing dropped) and {QA}")


if __name__ == "__main__":
    main()
