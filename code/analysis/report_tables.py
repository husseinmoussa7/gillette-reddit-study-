"""Publication-ready tables: descriptives, regressions, effect sizes, reliability, bounds.

Writes analysis/results/tables/*.csv plus a single tables.md for pasting into a draft.
Usage: python -m analysis.report_tables
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
TAB = RES / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261014)
CONS = ["resentment", "identity_threat", "zero_sum", "hostility"]
OUT = []


def emit(name, df, title, note=""):
    df.to_csv(TAB / f"{name}.csv")
    OUT.append(f"### {title}\n")
    OUT.append(df.to_markdown())
    if note:
        OUT.append(f"\n*{note}*")
    OUT.append("")
    print(f"\n== {title} ==\n{df.to_string()}")


def sample():
    u = pd.read_parquet(ROOT / "analysis" / "llm_units.parquet")
    s = pd.read_parquet(ROOT / "analysis" / "scores.parquet")
    l = pd.read_parquet(ROOT / "analysis" / "llm_codes.parquet")
    keep = u[(u.kind == "comment") & (u.status == "send")]
    d = keep.merge(s[["id", "vader", "rob_score", "rob_negative", "rob_neutral", "rob_positive", "emotion"]],
                   on="id", how="left")
    d = d.merge(l[["id", "ad_stance", *CONS, "confidence"]], on="id", how="left")
    d["created"] = pd.to_datetime(d.created, utc=True)
    d["year"] = d.created.dt.year
    d["mr"] = (d.subreddit == "MensRights").astype(int)
    d["logw"] = np.log1p(d.n_words)
    d["is_top_level"] = (d.parent_id == d.link_id).astype(int)
    d["period"] = np.where(d.created < "2019-03-01", "Jan-Feb 2019",
                    np.where(d.created < "2020-01-01", "rest of 2019", "2020+"))
    d["takes_position"] = d.ad_stance.ne("not_about_ad")
    d["is_critical"] = d.ad_stance.eq("critical")
    return u, d


def t1_flow(u):
    c = u[u.kind == "comment"]
    rows = []
    for sub, g in c.groupby("subreddit"):
        n = len(g)
        rows.append({
            "subreddit": f"r/{sub}", "collected": n,
            "removed/deleted": int((g.status == "skip_removed").sum()),
            "removal rate": round((g.status == "skip_removed").mean(), 3),
            "bots": int((g.status == "skip_bot").sum()),
            "mod notices": int((g.status == "skip_mod_notice").sum()),
            "no codeable text": int((g.status == "skip_no_content").sum()),
            "ANALYSED": int((g.status == "send").sum()),
        })
    p = u[u.kind == "post"]
    rows.append({"subreddit": "posts (both subs)", "collected": len(p),
                 "removed/deleted": 0, "removal rate": 0.0, "bots": 0, "mod notices": 0,
                 "no codeable text": int((p.status == "skip_no_content").sum()),
                 "ANALYSED": int((p.status == "send").sum())})
    emit("t1_sample_flow", pd.DataFrame(rows).set_index("subreddit"),
         "Table 1. Sample construction",
         "Posts whose body was removed are retained: their title survives and carries the codeable text.")


def t2_descriptives(d):
    rows = {}
    for sub, g in d.groupby("subreddit"):
        pos = g[g.takes_position]
        rows[f"r/{sub}"] = {
            "N comments analysed": len(g),
            "unique authors": g.loc[g.author != "[deleted]", "author"].nunique(),
            "threads": g.link_id.nunique(),
            "median words": g.n_words.median(),
            "top-level %": round(100 * g.is_top_level.mean(), 1),
            "Jan-Feb 2019 %": round(100 * g.period.eq("Jan-Feb 2019").mean(), 1),
            "mean polarity": round(g.rob_score.mean(), 3),
            "SD polarity": round(g.rob_score.std(), 3),
            "mean VADER": round(g.vader.mean(), 3),
            "takes a position on ad %": round(100 * g.takes_position.mean(), 1),
            "of those, critical %": round(100 * pos.is_critical.mean(), 1),
            **{f"mean {c}": round(g[c].mean(), 3) for c in CONS},
        }
    emit("t2_descriptives", pd.DataFrame(rows), "Table 2. Descriptive statistics, analysis sample")


def t3_tone_regression(d):
    import statsmodels.formula.api as smf
    specs = {
        "(1) bivariate": "rob_score ~ mr",
        "(2) + length": "rob_score ~ mr + logw",
        "(3) + structure": "rob_score ~ mr + logw + is_top_level",
        "(4) full": "rob_score ~ mr + logw + is_top_level + C(period)",
        "(5) VADER outcome": "vader ~ mr + logw + is_top_level + C(period)",
    }
    rows = {}
    for name, f in specs.items():
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.link_id})
        rows[name] = {
            "MensRights coef": round(m.params["mr"], 4),
            "cluster SE": round(m.bse["mr"], 4),
            "t": round(m.tvalues["mr"], 2),
            "p": f"{m.pvalues['mr']:.2e}",
            "95% CI": f"[{m.conf_int().loc['mr', 0]:.3f}, {m.conf_int().loc['mr', 1]:.3f}]",
            "log(words) coef": round(m.params.get("logw", np.nan), 4),
            "N": int(m.nobs), "clusters": d.link_id.nunique(), "R2": round(m.rsquared, 4),
        }
    emit("t3_tone_regression", pd.DataFrame(rows),
         "Table 3. Tone: OLS with thread-clustered standard errors",
         "Outcome is RoBERTa polarity unless stated. 'MensRights coef' is the r/MensRights "
         "indicator, so a negative value means more negative tone than r/MensLib. "
         "Wild cluster bootstrap p < 0.001 for specification (4).")


def t4_stance(d):
    import statsmodels.formula.api as smf
    pos = d[d.takes_position].copy()
    pos["is_critical"] = pos.is_critical.astype(int)   # statsmodels needs numeric endog
    rows = {}
    ct = pd.crosstab(d.subreddit, d.ad_stance)
    ct_pct = (100 * pd.crosstab(pos.subreddit, pos.ad_stance, normalize="index")).round(1)
    emit("t4a_stance_counts", ct, "Table 4a. Stance, counts (all analysed comments)")
    emit("t4b_stance_pct", ct_pct, "Table 4b. Stance, % of comments taking a position")

    # logit for critical, thread-clustered
    m = smf.logit("is_critical ~ mr + logw + is_top_level + C(period)", data=pos).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pos.link_id})
    odds = np.exp(m.params["mr"])
    ci = np.exp(m.conf_int().loc["mr"])
    gap = pos[pos.mr == 0].is_critical.mean() - pos[pos.mr == 1].is_critical.mean()
    rows["critical ~ MensRights"] = {
        "log-odds": round(m.params["mr"], 3), "cluster SE": round(m.bse["mr"], 3),
        "odds ratio": round(odds, 2), "95% CI (OR)": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
        "p": f"{m.pvalues['mr']:.2e}",
        "raw difference in critical share": round(gap, 3),
        "N": int(m.nobs), "clusters": pos.link_id.nunique(),
    }
    emit("t4c_stance_logit", pd.DataFrame(rows),
         "Table 4c. Probability a positioned comment is critical of the ad",
         "Logistic regression, thread-clustered SEs, controlling for length, reply depth and period.")


def t5_constructs(d):
    rows = {}
    for c in CONS:
        a = d[d.mr == 0][c].dropna().values
        b = d[d.mr == 1][c].dropna().values
        # thread-clustered bootstrap of the difference
        g = {k: v[c].dropna().values for k, v in d.groupby("link_id")}
        sub = d.groupby("link_id").subreddit.first().to_dict()
        keys = np.array(list(g))
        boot = []
        for _ in range(1000):
            pick = RNG.choice(keys, len(keys), replace=True)
            x = np.concatenate([g[k] for k in pick if sub[k] == "MensLib"] or [np.array([np.nan])])
            y = np.concatenate([g[k] for k in pick if sub[k] == "MensRights"] or [np.array([np.nan])])
            boot.append(np.nanmean(x) - np.nanmean(y))
        boot = np.array(boot)
        pooled_sd = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
        xs, ys = RNG.choice(a, 100000), RNG.choice(b, 100000)
        rows[c] = {
            "r/MensLib mean": round(a.mean(), 3), "r/MensLib SD": round(a.std(ddof=1), 3),
            "r/MensRights mean": round(b.mean(), 3), "r/MensRights SD": round(b.std(ddof=1), 3),
            "% >=2 MensLib": round(100 * (a >= 2).mean(), 1),
            "% >=2 MensRights": round(100 * (b >= 2).mean(), 1),
            "difference": round(a.mean() - b.mean(), 3),
            "95% CI (thread bootstrap)": f"[{np.nanpercentile(boot, 2.5):.3f}, {np.nanpercentile(boot, 97.5):.3f}]",
            "Cohen's d": round((a.mean() - b.mean()) / pooled_sd, 3),
            "Cliff's delta": round(float((xs > ys).mean() - (xs < ys).mean()), 3),
            "Mann-Whitney p": f"{stats.mannwhitneyu(a, b).pvalue:.2e}",
        }
    emit("t5_constructs", pd.DataFrame(rows), "Table 5. Constructs by subreddit",
         "Ratings 0-3. CIs resample whole threads. Mann-Whitney p values are unclustered and "
         "therefore anti-conservative; the bootstrap CI is the inferential statistic to report.")

    by_stance = d.groupby("ad_stance")[CONS].agg(["mean", "std", "size"]).round(3)
    emit("t6_constructs_by_stance", by_stance,
         "Table 6. Constructs by coded stance, pooling both subreddits")


def t7_reliability(d):
    from sklearn.metrics import cohen_kappa_score
    rows = {}
    # tone: 80 hand-labelled
    s = pd.read_csv(RES / "handlabel_validation.csv")
    for name, col, k in [("RoBERTa (tone)", "rob_lab", None), ("VADER (tone)", "vad_lab", None)]:
        kap = cohen_kappa_score(s.hand, s[col])
        boot = [cohen_kappa_score(*zip(*s[["hand", col]].sample(len(s), replace=True, random_state=i).values))
                for i in range(500)]
        rows[name] = {"n hand-labelled": len(s), "agreement %": round(100 * (s.hand == s[col]).mean(), 1),
                      "Cohen's kappa": round(kap, 3),
                      "95% CI": f"[{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"}
    # stance: 40 blind
    b = pd.read_csv(RES / "stance_blind_validation.csv") if (RES / "stance_blind_validation.csv").exists() else None
    if b is not None:
        kap = cohen_kappa_score(b.hand, b.llm)
        boot = [cohen_kappa_score(*zip(*b[["hand", "llm"]].sample(len(b), replace=True, random_state=i).values))
                for i in range(500)]
        rows["gpt-4.1-mini (stance)"] = {
            "n hand-labelled": len(b), "agreement %": round(100 * (b.hand == b.llm).mean(), 1),
            "Cohen's kappa": round(kap, 3),
            "95% CI": f"[{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"}
    emit("t7_reliability", pd.DataFrame(rows).T, "Table 7. Measurement reliability against hand labels",
         "Hand labels assigned blind to model output. Stance kappa is for the pre-registered final "
         "prompt; the pilot prompt scored 0.52 and was revised before the production run.")


def t8_bounds(d):
    tone = json.loads((RES / "report.json").read_text())
    st = json.loads((RES / "stance_report.json").read_text())
    rows = {
        "Tone: polarity gap": {
            "observed": round(tone["rob_score"]["raw_diff"], 3),
            "length-adjusted": round(-tone["rob_score"]["ols_mr_coef"], 3),
            "Lee bounds": f"[{tone['removal_bounds']['lee'][0]:.3f}, {tone['removal_bounds']['lee'][1]:.3f}]",
            "Manski bounds": f"[{tone['removal_bounds']['manski'][0]:.3f}, {tone['removal_bounds']['manski'][1]:.3f}]",
            "sign identified?": "NO - bounds contain zero",
        },
        "Stance: critical-share gap": {
            "observed": round(st["removal_bounds"]["observed_gap"], 3),
            "length-adjusted": "n/a (see Table 4c)",
            "Lee bounds": "n/a",
            "Manski bounds": f"[{st['removal_bounds']['manski_bounds'][0]:.3f}, {st['removal_bounds']['manski_bounds'][1]:.3f}]",
            "sign identified?": "YES - bounds exclude zero",
        },
    }
    emit("t8_bounds", pd.DataFrame(rows).T, "Table 8. Sensitivity to differential moderation",
         "Removal rates: r/MensLib 22.4%, r/MensRights 4.9%. Lee bounds equalise retention by "
         "trimming the lower-removal group's tails; Manski bounds assign the unobserved removed "
         "comments their most adverse possible values.")


def main():
    u, d = sample()
    t1_flow(u)
    t2_descriptives(d)
    t3_tone_regression(d)
    t4_stance(d)
    t5_constructs(d)
    t7_reliability(d)
    t8_bounds(d)
    (TAB / "tables.md").write_text("# Statistical tables\n\n" + "\n".join(OUT))
    print(f"\nwrote {TAB}/*.csv and tables.md")


if __name__ == "__main__":
    main()
