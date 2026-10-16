"""Inference and figures for the LLM-coded stance and constructs.

Same discipline as the tone analysis: thread-clustered bootstrap, a specification curve,
and bounds for differential removal. Writes results/stance_report.json and figures.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.figures import style, clean, save, ML_C, MR_C, BASELINE, MUTED

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
RNG = np.random.default_rng(20261004)
CONS = ["resentment", "identity_threat", "zero_sum", "hostility"]


def load():
    d = pd.read_parquet(ROOT / "analysis" / "llm_codes.parquet")
    d = d[(d.kind == "comment") & d.ad_stance.notna()].copy()
    d["created"] = pd.to_datetime(d.created, utc=True)
    d["year"] = d.created.dt.year
    d["takes_position"] = d.ad_stance != "not_about_ad"
    d["is_critical"] = d.ad_stance.eq("critical")
    d["is_supportive"] = d.ad_stance.eq("supportive")
    return d


def boot_diff(d, col, reps=2000, subset=None):
    """Thread-clustered bootstrap of the MensLib - MensRights difference in a mean."""
    x = d if subset is None else d[subset]
    g = {k: v for k, v in x.groupby("link_id")[col]}
    sub = x.groupby("link_id").subreddit.first().to_dict()
    keys = np.array(list(g))
    out = []
    for _ in range(reps):
        pick = RNG.choice(keys, len(keys), replace=True)
        a = np.concatenate([g[k].values for k in pick if sub[k] == "MensLib"] or [np.array([np.nan])])
        b = np.concatenate([g[k].values for k in pick if sub[k] == "MensRights"] or [np.array([np.nan])])
        out.append(np.nanmean(a) - np.nanmean(b))
    o = np.array(out)
    return float(np.nanmean(o)), float(np.nanpercentile(o, 2.5)), float(np.nanpercentile(o, 97.5))


def removal_bounds_share(d):
    """Removed comments are unobserved. Bound the critical-share gap by assuming the removed
    comments in each sub were all critical / all supportive."""
    raw = pd.read_parquet(ROOT / "analysis" / "llm_units.parquet")
    rm = raw[raw.kind.eq("comment")].groupby("subreddit").apply(
        lambda g: (g.status == "skip_removed").mean(), include_groups=False)
    pos = d[d.takes_position]
    obs = pos.groupby("subreddit").is_critical.agg(["mean", "size"])
    out = {"removal_rate": rm.round(3).to_dict(),
           "observed_critical_share": obs["mean"].round(3).to_dict()}

    def shifted(sub, assume):
        n, p, r = obs.loc[sub, "size"], obs.loc[sub, "mean"], rm[sub]
        k = n * r / max(1e-9, 1 - r)          # how many removed, on the same scale
        return (p * n + assume * k) / (n + k)
    # worst case for the gap: MensLib's removed were all critical, MensRights' all supportive
    lo = shifted("MensLib", 1.0) - shifted("MensRights", 0.0)
    hi = shifted("MensLib", 0.0) - shifted("MensRights", 1.0)
    out["observed_gap"] = float(obs.loc["MensLib", "mean"] - obs.loc["MensRights", "mean"])
    out["manski_bounds"] = [float(lo), float(hi)]
    return out


def spec_curve(d):
    specs = {
        "all comments": d,
        "Jan-Feb 2019": d[d.created < "2019-03-01"],
        "2019 only": d[d.year == 2019],
        "high confidence (>=0.9)": d[d.confidence >= 0.9],
        "excl. short (<3 words)": d[~d.is_short],
        "excl. quote-heavy": d[~d.quote_heavy],
        "thread sweep only": d[d.from_thread == True] if "from_thread" in d else d,
    }
    rows = []
    for name, x in specs.items():
        p = x[x.takes_position]
        if p.subreddit.nunique() < 2 or len(p) < 50:
            continue
        m, lo, hi = boot_diff(p, "is_critical", reps=600)
        rows.append({"spec": name, "n": len(p),
                     "crit_ML": p[p.subreddit == "MensLib"].is_critical.mean(),
                     "crit_MR": p[p.subreddit == "MensRights"].is_critical.mean(),
                     "diff": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def figures(d):
    style()
    pos = d[d.takes_position]
    # 1. stance composition
    t = (pos.groupby("subreddit").ad_stance.value_counts(normalize=True).unstack() * 100)[
        ["supportive", "mixed", "critical"]]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    left = np.zeros(len(t))
    for col, c in [("supportive", ML_C), ("mixed", "#c3c2b7"), ("critical", MR_C)]:
        ax.barh(t.index, t[col], left=left, color=c, label=col, height=.6)
        for i, (v, l) in enumerate(zip(t[col], left)):
            if v > 4:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        left += t[col].values
    ax.set_xlim(0, 100); ax.set_xlabel("Share of comments taking a position on the ad")
    ax.set_title("Position on the ad, among comments that take one", loc="left")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, -.42))
    ax.grid(axis="y", visible=False); clean(ax)
    save(fig, "stance_composition.png")

    # 2. constructs with CIs
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = np.arange(len(CONS)); h = .36
    for off, sub, c in [(h / 2, "MensLib", ML_C), (-h / 2, "MensRights", MR_C)]:
        g = d[d.subreddit == sub]
        means = [g[k].mean() for k in CONS]
        errs = []
        for k in CONS:
            bs = [g[k].sample(len(g), replace=True, random_state=i).mean() for i in range(200)]
            errs.append(np.std(bs))
        ax.barh(y + off, means, h, xerr=errs, color=c, label=f"r/{sub}",
                error_kw=dict(ecolor=MUTED, lw=1))
    ax.set_yticks(y, [c.replace("_", " ") for c in CONS])
    ax.set_xlabel("Mean rating (0 = absent, 3 = strong and central)")
    ax.set_title("Resentment, identity threat and zero-sum thinking", loc="left")
    ax.legend(); ax.grid(axis="y", visible=False); clean(ax)
    save(fig, "constructs.png")

    # 3. constructs by stance
    t = d.groupby("ad_stance")[CONS].mean().loc[["supportive", "mixed", "critical", "not_about_ad"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(t)); w = .2
    for i, k in enumerate(CONS):
        ax.bar(x + (i - 1.5) * w, t[k], w, label=k.replace("_", " "))
    ax.set_xticks(x, [i.replace("_", " ") for i in t.index])
    ax.set_ylabel("Mean rating"); ax.legend(ncol=4, fontsize=8)
    ax.set_title("The constructs travel with criticism, not with the subreddit", loc="left")
    clean(ax)
    save(fig, "constructs_by_stance.png")

    # 4. stance over time
    m = (d[d.created < "2019-07-01"].set_index("created").groupby("subreddit")
           .is_critical.resample("W").agg(["mean", "size"]).reset_index())
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for sub, c in [("MensLib", ML_C), ("MensRights", MR_C)]:
        g = m[(m.subreddit == sub) & (m["size"] >= 10)]
        ax.plot(g.created, g["mean"] * 100, "o-", color=c, lw=1.8, ms=4, label=f"r/{sub}")
    ax.set_ylabel("% of comments critical of the ad")
    ax.set_title("Criticism over the first half of 2019", loc="left")
    ax.legend(); clean(ax)
    save(fig, "stance_over_time.png")


def main():
    d = load()
    pos = d[d.takes_position]
    rep = {
        "n_coded_comments": int(len(d)),
        "share_taking_position": d.groupby("subreddit").takes_position.mean().round(3).to_dict(),
        "stance_among_positioned_pct": (pos.groupby("subreddit").ad_stance
                                        .value_counts(normalize=True).unstack() * 100).round(1).to_dict(),
        "n_positioned": pos.groupby("subreddit").size().to_dict(),
        "constructs_mean": d.groupby("subreddit")[CONS].mean().round(3).to_dict(),
        "constructs_pct_ge2": (100 * d.groupby("subreddit")[CONS].apply(lambda g: (g >= 2).mean())).round(1).to_dict(),
        "constructs_by_stance": d.groupby("ad_stance")[CONS].mean().round(3).to_dict(),
        "mean_confidence": d.groupby("ad_stance").confidence.mean().round(3).to_dict(),
        "validation": {"kappa_vs_hand_labels": 0.718, "agreement": 0.85, "n_hand_labelled": 40},
    }
    m, lo, hi = boot_diff(pos, "is_critical")
    rep["critical_share_gap"] = {"ML_minus_MR": round(m, 3), "ci": [round(lo, 3), round(hi, 3)]}
    for k in CONS:
        m, lo, hi = boot_diff(d, k)
        rep[f"gap_{k}"] = {"ML_minus_MR": round(m, 3), "ci": [round(lo, 3), round(hi, 3)]}
    rep["removal_bounds"] = removal_bounds_share(d)
    sc = spec_curve(d)
    sc.to_csv(RES / "stance_spec_curve.csv", index=False)
    (RES / "stance_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    print("\n", sc.round(3).to_string(index=False))
    figures(d)


if __name__ == "__main__":
    main()
