"""Figures for the sentiment comparison. House style borrowed from cpg-pricing-teardown/rgm/charts.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

ML_C, MR_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "axes.titlesize": 11, "legend.frameon": False,
    })


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def save(fig, name):
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


def load():
    """Same analysis sample as compare.py: comments whose text survives (status == send)."""
    d = pd.read_parquet(ROOT / "analysis" / "scores.parquet")
    keep = pd.read_parquet(ROOT / "analysis" / "llm_units.parquet")
    keep = set(keep[(keep.kind == "comment") & (keep.status == "send")].id)
    d = d[(d.kind == "comment") & d.id.isin(keep)].copy()
    d["created"] = pd.to_datetime(d.created, utc=True)
    return d


def fig_distribution(d):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bins = np.linspace(-1, 1, 41)
    for sub, c in [("MensLib", ML_C), ("MensRights", MR_C)]:
        g = d[d.subreddit == sub].rob_score
        ax.hist(g, bins=bins, density=True, histtype="stepfilled", alpha=.35, color=c)
        ax.hist(g, bins=bins, density=True, histtype="step", lw=1.8, color=c, label=f"r/{sub}")
        ax.axvline(g.mean(), color=c, ls="--", lw=1.2)
    ax.set_xlabel("Polarity   (negative  ←   →  positive)")
    ax.set_ylabel("Density")
    ax.set_title("Tone is overwhelmingly negative in both communities", loc="left")
    ax.text(.02, .92, "dashed lines = means", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.legend(loc="upper right")
    clean(ax)
    save(fig, "sentiment_distribution.png")


def fig_specs():
    sc = pd.read_csv(RES / "spec_curve.csv")
    rep = json.loads((RES / "report.json").read_text())
    rows = sc.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(rows))
    ax.errorbar(rows["diff"], y, xerr=[rows["diff"] - rows.ci_lo, rows.ci_hi - rows["diff"]],
                fmt="o", color=INK, ecolor=BASELINE, elinewidth=2, capsize=0, ms=5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    # bounds under differential moderation
    lee = rep["removal_bounds"]["lee"]
    ax.axvspan(lee[0], lee[1], color=MR_C, alpha=.10, zorder=0)
    ax.text(lee[1], len(rows) - .4, "  Lee bounds under\n  differential removal",
            color=MR_C, fontsize=8, va="center")
    ax.set_yticks(y, rows.spec)
    ax.set_xlabel("Difference in mean polarity  (r/MensLib − r/MensRights)")
    ax.set_title("Positive in every specification — but inside the moderation bounds", loc="left")
    ax.grid(axis="y", visible=False)
    clean(ax)
    save(fig, "spec_curve.png")


def fig_adjusted(d):
    """The length story: MensLib comments are longer, and longer comments score more negative."""
    d = d.copy()
    d["dec"] = pd.qcut(d.words, 10, labels=False, duplicates="drop")
    t = d.pivot_table(index="dec", columns="subreddit", values="rob_score", aggfunc="mean")
    med = d.groupby("dec").words.median()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(med, t.MensLib, "o-", color=ML_C, label="r/MensLib", lw=2)
    ax.plot(med, t.MensRights, "o-", color=MR_C, label="r/MensRights", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Comment length (words, log scale; decile medians)")
    ax.set_ylabel("Mean polarity")
    ax.set_title("Within every length band, r/MensLib is the less negative of the two", loc="left")
    ax.legend()
    clean(ax)
    save(fig, "polarity_by_length.png")


def fig_time(d):
    m = (d[d.created < "2019-07-01"].set_index("created")
           .groupby("subreddit").rob_score.resample("W").agg(["mean", "size"]).reset_index())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    for sub, c in [("MensLib", ML_C), ("MensRights", MR_C)]:
        g = m[(m.subreddit == sub) & (m["size"] >= 5)]
        a1.plot(g.created, g["mean"], "o-", color=c, lw=1.8, ms=4, label=f"r/{sub}")
        a2.bar(g.created, g["size"], width=5, color=c, alpha=.7)
    a1.axhline(0, color=MUTED, lw=.8)
    a1.set_ylabel("Mean polarity"); a1.set_title("Tone and volume, first half of 2019", loc="left")
    a1.legend(loc="lower right")
    a2.set_ylabel("Comments"); a2.set_yscale("log")
    for ax in (a1, a2):
        clean(ax)
    save(fig, "sentiment_over_time.png")


def fig_emotions(d):
    cols = [c for c in d.columns if c.startswith("emo_")]
    t = d.groupby("subreddit")[cols].mean().T
    t.index = [i.replace("emo_", "") for i in t.index]
    t = t.sort_values("MensRights")
    y = np.arange(len(t)); h = .38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(y + h / 2, t.MensLib, h, color=ML_C, label="r/MensLib")
    ax.barh(y - h / 2, t.MensRights, h, color=MR_C, label="r/MensRights")
    ax.set_yticks(y, t.index)
    ax.set_xlabel("Mean probability across comments")
    ax.set_title("Near-identical emotional register; fear is the one real gap", loc="left")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    clean(ax)
    save(fig, "emotion_profile.png")


def fig_score(d):
    """Descriptive only: score is downstream of the comment, not a control."""
    d = d[d.score.notna()].copy()
    d["bin"] = pd.qcut(d.rob_score, 10, labels=False, duplicates="drop")
    t = d.groupby(["subreddit", "bin"]).agg(pol=("rob_score", "mean"), sc=("score", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for sub, c in [("MensLib", ML_C), ("MensRights", MR_C)]:
        g = t[t.subreddit == sub]
        ax.plot(g.pol, g.sc, "o-", color=c, lw=2, label=f"r/{sub}")
    ax.set_xlabel("Comment polarity (decile means)"); ax.set_ylabel("Median comment score")
    ax.set_title("What each community upvoted", loc="left")
    ax.legend()
    clean(ax)
    save(fig, "polarity_vs_score.png")


def fig_themes():
    f = RES / "themes.csv"
    if not f.exists():
        print("themes.csv not ready, skipping"); return
    t = pd.read_csv(f, index_col=0)
    names = (RES / "theme_names.json")
    if names.exists():
        lab = json.loads(names.read_text())
        t["label"] = [lab.get(str(i), f"cluster {i}") for i in t.index]
    else:
        t["label"] = [", ".join(str(x).split(", ")[:3]) for x in t.terms]
    t = t.sort_values("share_MensRights")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.8, 5))
    ax.hlines(y, t.share_MensLib, t.share_MensRights, color=BASELINE, lw=2, zorder=1)
    ax.scatter(t.share_MensLib, y, color=ML_C, s=45, zorder=3, label="r/MensLib")
    ax.scatter(t.share_MensRights, y, color=MR_C, s=45, zorder=3, label="r/MensRights")
    ax.set_yticks(y, t.label, fontsize=8)
    ax.set_xlabel("Share of the subreddit's comments")
    ax.set_title("What each community talked about", loc="left")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
    clean(ax)
    save(fig, "themes_dumbbell.png")


def main():
    style()
    d = load()
    fig_distribution(d)
    fig_adjusted(d)
    fig_specs()
    fig_time(d)
    fig_emotions(d)
    fig_score(d)
    fig_themes()


if __name__ == "__main__":
    main()
