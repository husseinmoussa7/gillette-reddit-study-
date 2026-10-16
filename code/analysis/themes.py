"""Cluster comments into themes and compare their prevalence across the two subreddits.

Embeds with the locally cached all-MiniLM-L6-v2, reduces with SVD, clusters with KMeans
(k chosen by silhouette), labels clusters by c-TF-IDF terms, and writes the nearest-centroid
comments per cluster so the labels can be checked and named by hand.

Usage: python analysis/themes.py [k]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
RES.mkdir(parents=True, exist_ok=True)
EMB = ROOT / "analysis" / "_ckpt" / "embeddings.npy"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_WORDS = 5          # very short comments carry no theme
K_RANGE = range(8, 21)


def load():
    df = pd.read_parquet(ROOT / "analysis" / "scores.parquet")
    df = df[(df.kind == "comment") & ~df.is_bot & (df.words >= MIN_WORDS)].reset_index(drop=True)
    return df


def embed(texts):
    if EMB.exists():
        e = np.load(EMB)
        if len(e) == len(texts):
            print(f"reusing embeddings {e.shape}")
            return e
    # sentence_transformers pulls in torchvision.transforms.v2, which this env's 2018
    # torchvision lacks; MiniLM is just mean-pooled AutoModel output, so do that directly.
    from analysis.score import _patch_torchvision
    _patch_torchvision()
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(4)
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])          # length-sorted for tight padding
    out = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)
    B = 64
    for s0 in range(0, len(order), B):
        idx = order[s0:s0 + B]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pooling
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        out[idx] = v.numpy()
        if s0 % (B * 40) == 0:
            print(f"  embed {s0}/{len(order)}", flush=True)
    np.save(EMB, out)
    return out


def pick_k(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows = []
    sample = np.random.default_rng(0).choice(len(X), min(4000, len(X)), replace=False)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
        s = silhouette_score(X[sample], km.labels_[sample])
        rows.append((k, s))
        print(f"  k={k}  silhouette={s:.4f}", flush=True)
    return pd.DataFrame(rows, columns=["k", "silhouette"])


def ctfidf_terms(df, labels, n=12):
    """Class-based TF-IDF: what words distinguish each cluster from the rest."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs = pd.Series(list(df.text)).groupby(labels).apply(lambda s: " ".join(s))
    cv = CountVectorizer(stop_words="english", min_df=3, max_features=40000,
                         token_pattern=r"[a-zA-Z][a-zA-Z']{2,}")
    X = cv.fit_transform(docs)
    vocab = np.array(cv.get_feature_names_out())
    tf = X.toarray().astype(float)
    tf /= np.maximum(tf.sum(axis=1, keepdims=True), 1)
    df_t = (X.toarray() > 0).sum(axis=0)
    idf = np.log(1 + len(docs) / np.maximum(df_t, 1))
    scores = tf * idf
    return {int(c): list(vocab[np.argsort(-scores[i])][:n]) for i, c in enumerate(docs.index)}


def main(k=None):
    df = load()
    print(f"clustering {len(df)} comments")
    E = embed(df.text)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    X = TruncatedSVD(n_components=50, random_state=0).fit_transform(E)

    if k is None:
        sil = pick_k(X)
        sil.to_csv(RES / "silhouette.csv", index=False)
        k = int(sil.loc[sil.silhouette.idxmax(), "k"])
        print(f"chosen k={k}")
    km = KMeans(n_clusters=int(k), n_init=10, random_state=0).fit(X)
    df["cluster"] = km.labels_

    terms = ctfidf_terms(df, df.cluster)
    # nearest-centroid exemplars for hand-labelling
    ex = []
    for c in sorted(df.cluster.unique()):
        idx = np.where(df.cluster.values == c)[0]
        d = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        for i in idx[np.argsort(d)[:6]]:
            ex.append({"cluster": c, "subreddit": df.subreddit.iloc[i],
                       "text": df.text.iloc[i][:400].replace("\n", " ")})
    pd.DataFrame(ex).to_csv(RES / "cluster_exemplars.csv", index=False)

    # prevalence and tone per cluster per subreddit
    prev = (df.groupby(["cluster", "subreddit"]).size().unstack(fill_value=0))
    prev_share = prev / prev.sum()
    tone = df.groupby(["cluster", "subreddit"]).rob_score.mean().unstack()
    out = pd.DataFrame({
        "terms": {c: ", ".join(t) for c, t in terms.items()},
        "n_MensLib": prev.get("MensLib", 0), "n_MensRights": prev.get("MensRights", 0),
        "share_MensLib": prev_share.get("MensLib", 0).round(4),
        "share_MensRights": prev_share.get("MensRights", 0).round(4),
        "tone_MensLib": tone.get("MensLib").round(3), "tone_MensRights": tone.get("MensRights").round(3),
    })
    out["share_ratio_ML_over_MR"] = (out.share_MensLib / out.share_MensRights.replace(0, np.nan)).round(2)
    out.to_csv(RES / "themes.csv")
    df[["id", "subreddit", "cluster"]].to_csv(RES / "comment_clusters.csv", index=False)
    print(out.drop(columns=["terms"]).to_string())
    print(f"\nwrote {RES}/themes.csv, cluster_exemplars.csv (read these to name the themes)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
