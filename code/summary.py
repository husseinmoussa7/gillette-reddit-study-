"""Build tidy posts/comments tables from raw/ and print summary stats per subreddit."""
import json, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RAW, DATA, FIGS = ROOT / "raw", ROOT / "data", ROOT / "figs"
DATA.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

BROAD = re.compile(r"gil+et+e|we\s+believe", re.I)
Q = r"[\"'“”‘’]?"
STRICT = re.compile(
    rf"gil+et+e(?:'s|’s|s)?{Q}\s+(?:ad|ads|advert|commercial|commercials|advertisement|advertisements)\b"
    rf"|we\s+believe{Q}\s+(?:ad|ads|commercial|commercials|advertisement|advertisements)\b",
    re.I,
)
AD_CONTEXT = re.compile(r"\b(?:ad|ads|advert\w*|commercial\w*|video|campaign|toxic masculinity|boycott|the best (?:a )?men can be|p&g|procter)\b", re.I)
GILLETTE = re.compile(r"gil+et+e", re.I)
BOOK = re.compile(r"moore (?:and|&) gil+et+e|king,? warrior,? magician", re.I)  # Moore & Gillette book, unrelated
AD_WINDOW = ("2019-01-13", "2019-07-01")  # any Gillette mention in this window is taken to be about the ad


def load(pattern):
    rows = []
    for f in RAW.glob(pattern):
        rows += [json.loads(l) for l in f.open()]
    return rows


def flags(df, text):
    t = text.fillna("")
    df["kw_broad"] = t.str.contains(BROAD)
    df["kw_strict"] = t.str.contains(STRICT)
    df["ad_context"] = t.str.contains(AD_CONTEXT)
    df["gillette"] = t.str.contains(GILLETTE) & ~t.str.contains(BOOK)
    return df


def relevant(df):
    in_window = (df.created >= AD_WINDOW[0]) & (df.created < AD_WINDOW[1])
    return df.kw_strict | (df.gillette & (df.ad_context | in_window))


def build():
    posts = pd.DataFrame(load("*/posts/*.jsonl")).drop_duplicates("id")
    posts["text"] = posts["title"].fillna("") + "\n" + posts["selftext"].fillna("")
    posts = flags(posts, posts["text"])
    posts = posts[posts.kw_broad].copy()
    posts["permalink"] = "https://www.reddit.com/r/" + posts.subreddit + "/comments/" + posts.id

    kw = pd.DataFrame(load("*/kwcoms/*.jsonl")).drop_duplicates("id")
    th = pd.DataFrame(load("*/threads/*.jsonl")).drop_duplicates("id")
    kw["from_kw_search"] = True
    th["from_thread"] = True
    coms = pd.concat([kw, th]).groupby("id", as_index=False).first()
    coms[["from_kw_search", "from_thread"]] = coms[["from_kw_search", "from_thread"]].astype("boolean").fillna(False).astype(bool)
    coms = flags(coms, coms["body"])
    coms["permalink"] = ("https://www.reddit.com/r/" + coms.subreddit + "/comments/" + coms.link_id.str[3:] + "/_/" + coms.id)

    for df in (posts, coms):
        df["created"] = pd.to_datetime(df.created_utc, unit="s", utc=True)
        df["deleted_author"] = df.author.isin(["[deleted]"])
    posts["removed"] = posts.selftext.isin(["[removed]", "[deleted]"])
    coms["removed"] = coms.body.isin(["[removed]", "[deleted]"])
    # relevant: strict phrase, or Gillette + ad-context word, or Gillette mentioned during the ad window;
    # comments also count if they sit in a relevant thread
    posts["relevant"] = relevant(posts)
    rel_threads = set("t3_" + posts.loc[posts.relevant, "id"])
    coms["in_relevant_thread"] = coms.link_id.isin(rel_threads)
    coms["relevant"] = relevant(coms) | coms.in_relevant_thread

    posts.drop(columns=["text"]).to_csv(DATA / "posts.csv", index=False)
    coms.to_csv(DATA / "comments.csv", index=False)
    posts.drop(columns=["text"]).to_parquet(DATA / "posts.parquet")
    coms.to_parquet(DATA / "comments.parquet")
    return posts, coms


def summarize(posts, coms):
    out = []
    p = lambda s="": (print(s), out.append(s))
    listed = {s.name: sum(1 for f in s.glob("posts/*.jsonl") for _ in f.open()) for s in RAW.iterdir() if s.is_dir()}
    p("# Gillette 'We Believe' ad — r/MensLib vs r/MensRights: summary stats\n")
    p(f"Posts scanned (all posts {posts.created.min():%Y-%m} onward, filtered locally): " + ", ".join(f"{k} {v:,}" for k, v in listed.items()) + "\n")

    rows = []
    for sub in sorted(set(posts.subreddit) | set(coms.subreddit)):
        P, C = posts[posts.subreddit == sub], coms[coms.subreddit == sub]
        Pr, Cr = P[P.relevant], C[C.relevant]
        rows.append({
            "subreddit": sub,
            "posts: broad mention": len(P), "posts: strict phrase": int(P.kw_strict.sum()), "posts: relevant": len(Pr),
            "comments: total collected": len(C), "comments: broad mention": int(C.kw_broad.sum()),
            "comments: strict phrase": int(C.kw_strict.sum()), "comments: relevant": len(Cr),
            "relevant threads w/ comments": Cr.link_id.nunique(),
            "unique authors (rel. posts+comments)": len(set(Pr.author) | set(Cr.author) - {"[deleted]"}),
            "% comments removed/deleted": round(100 * Cr.removed.mean(), 1) if len(Cr) else None,
            "median comment score": Cr.score.median(), "mean comment score": round(Cr.score.mean(), 1),
            "median post score": Pr.score.median(), "median post num_comments": Pr.num_comments.median(),
            "median comment length (words)": Cr.body.fillna("").str.split().str.len().median(),
            "share of rel. comments in Jan-Feb 2019 (%)": round(100 * ((Cr.created >= "2019-01-01") & (Cr.created < "2019-03-01")).mean(), 1),
            "first": f"{Cr.created.min():%Y-%m-%d}", "last": f"{Cr.created.max():%Y-%m-%d}",
        })
    t = pd.DataFrame(rows).set_index("subreddit").T
    p("## Overview\n"); p(t.to_markdown()); p()

    both = set(posts[posts.relevant].author) | set(coms[coms.relevant].author)
    by_sub = {s: set(pd.concat([posts[(posts.subreddit == s) & posts.relevant].author, coms[(coms.subreddit == s) & coms.relevant].author])) - {"[deleted]", "AutoModerator"} for s in posts.subreddit.unique()}
    if len(by_sub) == 2:
        a, b = by_sub.values()
        p(f"Authors active in both subs' Gillette discussions: {len(a & b)}\n")

    p("## Top 10 relevant posts per subreddit (by comments)\n")
    for sub, g in posts[posts.relevant].groupby("subreddit"):
        p(f"### r/{sub}")
        top = g.nlargest(10, "num_comments")[["created", "score", "num_comments", "title"]].copy()
        top["created"] = top.created.dt.strftime("%Y-%m-%d"); top["title"] = top.title.str[:90]
        p(top.to_markdown(index=False)); p()

    p("## Relevant comments by year\n")
    yr = coms[coms.relevant].groupby([coms.created.dt.year, "subreddit"]).size().unstack(fill_value=0)
    p(yr.to_markdown()); p()
    (ROOT / "summary.md").write_text("\n".join(out))

    # figures
    Cr = coms[coms.relevant & (coms.created < "2019-04-01")]
    wk = Cr.groupby([Cr.created.dt.tz_localize(None).dt.to_period("D").dt.start_time, "subreddit"]).size().unstack(fill_value=0)
    ax = wk.plot(figsize=(9, 4), title="Relevant comments per day, Jan–Mar 2019 (ad released 14 Jan 2019)")
    ax.set_xlabel(""); ax.set_ylabel("comments"); plt.tight_layout(); plt.savefig(FIGS / "daily_volume_2019Q1.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    for sub, g in coms[coms.relevant].groupby("subreddit"):
        ax.hist(g.score.clip(-20, 100), bins=60, alpha=.5, label=sub, density=True)
    ax.set_title("Comment score distribution (clipped to [-20,100])"); ax.legend(); plt.tight_layout()
    plt.savefig(FIGS / "comment_score_hist.png", dpi=150); plt.close()


if __name__ == "__main__":
    summarize(*build())
