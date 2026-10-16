"""Compare tone between r/MensLib and r/MensRights, with honest inference.

Primary outcome: RoBERTa polarity p(pos) - p(neg) in [-1, 1].
Primary spec:   OLS with thread-clustered SEs + wild cluster bootstrap (subreddit is
                constant within thread and MensLib has ~10 effective clusters, so the
                asymptotic CR SEs alone are too narrow).
Also: Cliff's delta, thread-level aggregation, Lee trimming bounds and Manski bounds for
      differential removal, and a specification curve.

Usage: python analysis/compare.py   -> writes analysis/results/*.csv and prints the tables
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
RES.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260928)
BOOT = 2000

OUTCOME = "rob_score"     # primary
ALT = "vader"             # robustness


def load():
    """Analysis sample = the single `send` sample defined in prepare_llm.py: comments whose text
    survives. Removed/deleted bodies, bots, moderator notices and bare links are all out, so the
    tone and stance analyses share one denominator."""
    df = pd.read_parquet(ROOT / "analysis" / "scores.parquet")
    keep = pd.read_parquet(ROOT / "analysis" / "llm_units.parquet")
    keep = set(keep[(keep.kind == "comment") & (keep.status == "send")].id)
    df = df[(df.kind == "comment") & df.id.isin(keep)].copy()
    df["created"] = pd.to_datetime(df.created, utc=True)
    df["year"] = df.created.dt.year
    df["mr"] = (df.subreddit == "MensRights").astype(int)
    df["is_top_level"] = (df.parent_id == df.link_id).astype(int)
    df["logw"] = np.log1p(df.words)
    df["period"] = np.where(df.created < "2019-03-01", "Jan-Feb 2019",
                     np.where(df.created < "2020-01-01", "rest of 2019", "2020+"))
    # source flags come from the collection stage
    c = pd.read_parquet(ROOT / "data" / "comments.parquet")[["id", "from_kw_search", "from_thread"]]
    return df.merge(c, on="id", how="left")


# ---------- effect sizes -------------------------------------------------------------
def cliffs_delta(a, b, n=200_000):
    """P(a > b) - P(a < b), sampled for speed. Bounded, robust to the bimodal shape."""
    x = RNG.choice(a, n); y = RNG.choice(b, n)
    return float((x > y).mean() - (x < y).mean())


def cluster_boot_diff(df, outcome, cluster="link_id", reps=BOOT):
    """Bootstrap the raw difference in means by resampling whole threads."""
    groups = {k: g[outcome].values for k, g in df.groupby(cluster)}
    subs = {k: g.subreddit.iloc[0] for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    out = []
    for _ in range(reps):
        pick = RNG.choice(keys, len(keys), replace=True)
        ml = np.concatenate([groups[k] for k in pick if subs[k] == "MensLib"] or [np.array([np.nan])])
        mr = np.concatenate([groups[k] for k in pick if subs[k] == "MensRights"] or [np.array([np.nan])])
        out.append(np.nanmean(ml) - np.nanmean(mr))
    out = np.array(out)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


# ---------- regression ---------------------------------------------------------------
def ols_clustered(df, outcome, controls=("logw", "is_top_level"), fe=("period",), cluster="link_id"):
    import statsmodels.formula.api as smf
    rhs = ["mr", *controls, *[f"C({f})" for f in fe]]
    m = smf.ols(f"{outcome} ~ " + " + ".join(rhs), data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return m


def wild_cluster_bootstrap(df, outcome, controls=("logw", "is_top_level"), fe=("period",),
                           cluster="link_id", reps=BOOT):
    """Rademacher wild bootstrap-t under H0: beta_mr = 0 (Cameron-Gelbach-Miller)."""
    import statsmodels.formula.api as smf
    rhs = ["mr", *controls, *[f"C({f})" for f in fe]]
    full = smf.ols(f"{outcome} ~ " + " + ".join(rhs), data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})
    t_obs = full.params["mr"] / full.bse["mr"]
    # restricted model (drop mr), get residuals
    r_rhs = [*controls, *[f"C({f})" for f in fe]]
    restricted = smf.ols(f"{outcome} ~ " + " + ".join(r_rhs), data=df).fit()
    resid, fitted = restricted.resid.values, restricted.fittedvalues.values
    codes = pd.Categorical(df[cluster]).codes
    n_cl = codes.max() + 1
    d = df.copy()
    ts = []
    for _ in range(reps):
        w = RNG.choice([-1.0, 1.0], n_cl)[codes]
        d["_y"] = fitted + resid * w
        try:
            f = smf.ols("_y ~ " + " + ".join(rhs), data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d[cluster]})
            ts.append(f.params["mr"] / f.bse["mr"])
        except Exception:
            continue
    ts = np.array(ts)
    p = float((np.abs(ts) >= abs(t_obs)).mean())
    return float(t_obs), p, len(ts)


# ---------- removal bounds -----------------------------------------------------------
def removal_bounds(df, outcome):
    """Removed text is unrecoverable and removal is far higher in MensLib.
    Measurable only on the thread-sweep subset (a removed comment cannot match a keyword)."""
    raw = pd.read_parquet(ROOT / "data" / "comments.parquet")
    raw = raw[raw.relevant & raw.from_thread]
    rates = raw.groupby("subreddit").removed.mean()
    sweep = df[df.from_thread == True]
    obs = {s: g[outcome].values for s, g in sweep.groupby("subreddit")}
    n = {s: len(raw[raw.subreddit == s]) for s in obs}
    out = {"removal_rate": rates.round(3).to_dict()}

    # Manski worst/best case: removed comments take the extreme values
    def mix(vals, rate, fill):
        k = int(round(len(vals) * rate / max(1e-9, 1 - rate)))
        return np.concatenate([vals, np.full(k, fill)])
    lo = mix(obs["MensLib"], rates["MensLib"], -1).mean() - mix(obs["MensRights"], rates["MensRights"], +1).mean()
    hi = mix(obs["MensLib"], rates["MensLib"], +1).mean() - mix(obs["MensRights"], rates["MensRights"], -1).mean()
    out["manski"] = [float(lo), float(hi)]

    # Lee trimming: equalise retention by trimming the excess from MensRights' tails
    keep = (1 - rates["MensLib"]) / (1 - rates["MensRights"])   # fraction of MR to keep
    mr = np.sort(obs["MensRights"]); k = int(len(mr) * keep)
    lee_lo = obs["MensLib"].mean() - mr[-k:].mean()   # MR keeps its most positive -> lower bound on gap
    lee_hi = obs["MensLib"].mean() - mr[:k].mean()
    out["lee_keep_fraction"] = float(keep)
    out["lee"] = [float(lee_lo), float(lee_hi)]
    out["naive_gap"] = float(obs["MensLib"].mean() - obs["MensRights"].mean())
    return out


# ---------- specification curve ------------------------------------------------------
def spec_curve(df, outcome):
    specs = {
        "all comments": df,
        "Jan-Feb 2019 only": df[df.period == "Jan-Feb 2019"],
        "2019 only": df[df.year == 2019],
        "thread sweep only": df[df.from_thread == True],
        "keyword matches only": df[df.kw_broad == True],
        "top-level only": df[df.is_top_level == 1],
        "excl. cross-posting authors": df[~df.author.isin(
            set(df[df.mr == 1].author) & set(df[df.mr == 0].author))],
        "MensRights downsampled": pd.concat([
            df[df.mr == 0], df[df.mr == 1].sample(min((df.mr == 0).sum(), (df.mr == 1).sum()), random_state=1)]),
    }
    rows = []
    for name, d in specs.items():
        if d.mr.nunique() < 2 or len(d) < 50:
            continue
        ml, mr = d[d.mr == 0][outcome].values, d[d.mr == 1][outcome].values
        lo, hi = cluster_boot_diff(d, outcome, reps=600)
        rows.append({"spec": name, "n": len(d), "n_ML": len(ml), "n_MR": len(mr),
                     "mean_ML": ml.mean(), "mean_MR": mr.mean(), "diff": ml.mean() - mr.mean(),
                     "ci_lo": lo, "ci_hi": hi, "cliffs_delta": cliffs_delta(ml, mr)})
    return pd.DataFrame(rows)


def main():
    df = load()
    print(f"comments scored: {len(df)}  ({df.subreddit.value_counts().to_dict()})\n")
    report = {}

    for outcome in (OUTCOME, ALT):
        ml, mr = df[df.mr == 0][outcome].values, df[df.mr == 1][outcome].values
        lo, hi = cluster_boot_diff(df, outcome)
        m = ols_clustered(df, outcome)
        t_obs, p_wild, nrep = wild_cluster_bootstrap(df, outcome)
        report[outcome] = {
            "mean_MensLib": float(ml.mean()), "mean_MensRights": float(mr.mean()),
            "raw_diff": float(ml.mean() - mr.mean()),
            "thread_boot_CI": [lo, hi],
            "cliffs_delta": cliffs_delta(ml, mr),
            "ols_mr_coef": float(m.params["mr"]), "ols_mr_se_cluster": float(m.bse["mr"]),
            "ols_p_asymptotic": float(m.pvalues["mr"]),
            "wild_bootstrap_p": p_wild, "wild_reps": nrep,
            "n_threads": int(df.link_id.nunique()),
        }
        print(f"== {outcome} ==")
        for k, v in report[outcome].items():
            print(f"   {k}: {v}")
        print()

    # agreement between the two measures
    ok = df[[OUTCOME, ALT]].dropna()
    from scipy.stats import spearmanr
    report["agreement"] = {
        "spearman": float(spearmanr(ok[OUTCOME], ok[ALT]).statistic),
        "sign_agreement": float((np.sign(ok[OUTCOME]) == np.sign(ok[ALT])).mean()),
    }
    print("agreement:", report["agreement"], "\n")

    report["removal_bounds"] = removal_bounds(df, OUTCOME)
    print("removal bounds:", json.dumps(report["removal_bounds"], indent=2), "\n")

    sc = spec_curve(df, OUTCOME)
    sc.to_csv(RES / "spec_curve.csv", index=False)
    print(sc.round(3).to_string(index=False))

    # per-thread means for the figure
    th = (df.groupby(["subreddit", "link_id"])
            .agg(mean_polarity=(OUTCOME, "mean"), n=(OUTCOME, "size")).reset_index())
    th.to_csv(RES / "thread_means.csv", index=False)

    # emotions and by-period tables
    emo = [c for c in df.columns if c.startswith("emo_")]
    if emo:
        (df.groupby("subreddit")[emo].mean().T).to_csv(RES / "emotion_profile.csv")
        df.groupby(["subreddit", "emotion"]).size().unstack(fill_value=0).to_csv(RES / "emotion_counts.csv")
    (df.groupby(["subreddit", "period"])[OUTCOME].agg(["mean", "median", "size"])
       .to_csv(RES / "by_period.csv"))
    (df.groupby(["subreddit", "year"])[OUTCOME].agg(["mean", "size"])
       .to_csv(RES / "by_year.csv"))

    (RES / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RES}/report.json and tables")


if __name__ == "__main__":
    main()
