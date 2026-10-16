"""Score every relevant comment (and post) for sentiment and emotion. Resumable.

Usage: python analysis/score.py [vader|roberta|emotion|all]
Writes analysis/scores.parquet, checkpointing so a killed run resumes.
"""
import sys, time, enum, types
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "scores.parquet"
CKPT = ROOT / "analysis" / "_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)

BOTS = {"AutoModerator", "sneakpeekbot", "WikiTextBot", "TotesMessenger", "RemindMeBot", "B0tRank"}
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"
BATCH = 32
MAXLEN = 256


def load_units():
    """Comments and posts to score: relevant, text surviving, not a bot."""
    c = pd.read_parquet(ROOT / "data" / "comments.parquet")
    c = c[c.relevant & ~c.removed].copy()
    c["text"] = c.body.fillna("")
    c["kind"] = "comment"
    p = pd.read_parquet(ROOT / "data" / "posts.parquet")
    p = p[p.relevant & ~p.removed].copy()
    p["text"] = (p.title.fillna("") + "\n" + p.selftext.fillna("")).str.strip()
    p["kind"] = "post"
    p["link_id"] = "t3_" + p.id
    p["parent_id"] = ""
    keep = ["id", "kind", "subreddit", "created", "author", "text", "score", "link_id", "parent_id",
            "kw_strict", "kw_broad"]
    for df in (c, p):
        for col in keep:
            if col not in df:
                df[col] = np.nan
    df = pd.concat([c[keep], p[keep]], ignore_index=True)
    df["is_bot"] = df.author.isin(BOTS) | df.text.str.contains(r"I am a bot", case=False, na=False)
    df["words"] = df.text.str.split().str.len()
    df = df[df.text.str.strip() != ""].reset_index(drop=True)
    return df


def vader(df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    df["vader"] = [a.polarity_scores(t)["compound"] for t in df.text]
    return df


def _patch_torchvision():
    """This env has torchvision 0.2.0 (2018) against torch 2.6; transformers' lazy model
    loading imports InterpolationMode from it. Shim it rather than touch the shared env."""
    import torchvision
    import torchvision.transforms as tvt
    if not hasattr(tvt, "InterpolationMode"):
        class InterpolationMode(str, enum.Enum):
            NEAREST = "nearest"; NEAREST_EXACT = "nearest-exact"; BILINEAR = "bilinear"
            BICUBIC = "bicubic"; BOX = "box"; HAMMING = "hamming"; LANCZOS = "lanczos"
        tvt.InterpolationMode = InterpolationMode
    if not hasattr(torchvision, "io"):  # transformers' video_utils imports this; unused for text
        io = types.ModuleType("torchvision.io")
        for n in ("read_video", "read_video_timestamps", "VideoReader", "read_image",
                  "ImageReadMode", "write_jpeg", "decode_image"):
            setattr(io, n, None)
        torchvision.io = io
        sys.modules["torchvision.io"] = io


def _run_model(df, model_name, prefix, labels_map=None):
    """Batched CPU inference with checkpointing; returns a DataFrame of probabilities."""
    _patch_torchvision()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ck = CKPT / f"{prefix}.parquet"
    done = pd.read_parquet(ck) if ck.exists() else pd.DataFrame(columns=["id"])
    todo = df[~df.id.isin(set(done.id))].copy()
    todo = todo.assign(_len=todo.text.str.len()).sort_values("_len").reset_index(drop=True)  # tight padding
    print(f"{prefix}: {len(done)} done, {len(todo)} to go", flush=True)
    if not len(todo):
        return done
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    torch.set_num_threads(4)
    names = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    names = [labels_map.get(n, n) if labels_map else n for n in names]
    rows, t0 = [], time.time()
    for s in range(0, len(todo), BATCH):
        chunk = todo.iloc[s:s + BATCH]
        enc = tok(list(chunk.text), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
        for cid, pr in zip(chunk.id, probs):
            rows.append({"id": cid, **{f"{prefix}_{n}": float(v) for n, v in zip(names, pr)}})
        if s and s % (BATCH * 50) == 0:
            pd.concat([done, pd.DataFrame(rows)]).to_parquet(ck)
            rate = (s + BATCH) / (time.time() - t0)
            print(f"  {s + len(chunk)}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - s) / rate / 60:.0f} min", flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(ck)
    print(f"{prefix}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def main(stage="all"):
    df = load_units()
    print(f"units: {len(df)} ({df.kind.value_counts().to_dict()}), bots flagged: {int(df.is_bot.sum())}")
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        df = df.merge(prev.drop(columns=[c for c in prev.columns if c in df.columns and c != "id"]),
                      on="id", how="left")
    if stage in ("vader", "all") and "vader" not in df:
        df = vader(df)
        df.to_parquet(OUT)
    if stage in ("roberta", "all"):
        r = _run_model(df, SENT_MODEL, "rob")
        df = df.drop(columns=[c for c in df.columns if c.startswith("rob_")], errors="ignore").merge(r, on="id", how="left")
        df["rob_score"] = df.rob_positive - df.rob_negative
        df.to_parquet(OUT)
    if stage in ("emotion", "all"):
        e = _run_model(df, EMO_MODEL, "emo")
        df = df.drop(columns=[c for c in df.columns if c.startswith("emo_")], errors="ignore").merge(e, on="id", how="left")
        emo_cols = [c for c in df.columns if c.startswith("emo_")]
        df["emotion"] = df[emo_cols].idxmax(axis=1).str.replace("emo_", "", regex=False)
        df.to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(df)}  cols={list(df.columns)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
