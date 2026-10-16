"""Collect r/MensLib and r/MensRights posts/comments about the Gillette 'We Believe' ad via Arctic Shift.

Stages (each cached/resumable under raw/):
  posts    - list ALL posts per subreddit (no server query; MensRights query search times out), filter locally
  kwcoms   - server-side comment body search for 'Gillette' and 'We Believe', paged by date
  threads  - all comments in every matched post (link_id)
Usage: python scrape.py [posts|kwcoms|threads|all] [subreddit ...]
"""
import json, re, sys, time, datetime as dt
from pathlib import Path
import requests

API = "https://arctic-shift.photon-reddit.com/api"
SUBS = ["MensLib", "MensRights"]
START = "2019-01-01"
END = "2026-09-22"
RAW = Path(__file__).parent / "raw"
LOG = Path(__file__).parent / "scrape.log"

BROAD = re.compile(r"gil+et+e|we\s+believe", re.I)  # incl. misspellings Gilette/Gillete
S = requests.Session()
S.headers["User-Agent"] = "gillette-reddit-research/0.1"


def log(msg):
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def get(endpoint, params, tries=5):
    delay = 5
    for i in range(tries):
        try:
            r = S.get(f"{API}/{endpoint}", params=params, timeout=120)
            if r.status_code == 200:
                d = r.json()
                if d.get("error"):
                    if "not a valid" in d["error"]:
                        raise ValueError(d["error"])
                    raise RuntimeError(d["error"])
                time.sleep(0.7)
                return d.get("data") or []
            err = r.text[:120]
            if "not a valid" in err or "must be a valid" in err:
                raise ValueError(err)
        except ValueError:
            raise
        except Exception as e:  # network error, JSON error, API error
            err = repr(e)[:120]
        log(f"  retry {i+1}/{tries} {endpoint} {params.get('after')} -> {err}")
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"gave up: {endpoint} {params}")


def ts(date):
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp())


def fetch_range(endpoint, params, after, before):
    """Page a window by created_utc ascending; on repeated server timeouts, split the window and recurse."""
    out, seen, cur = [], set(), after
    while True:
        try:
            batch = get(endpoint, {**params, "after": cur, "before": before, "limit": 100, "sort": "asc"},
                        tries=3 if before - cur > 86400 else 6)
        except RuntimeError:
            if before - cur < 3600:
                raise
            mid = cur + (before - cur) // 2
            log(f"  splitting window {cur}-{before} at {mid}")
            return out + fetch_range(endpoint, params, cur, mid) + fetch_range(endpoint, params, mid, before)
        new = [x for x in batch if x["id"] not in seen]
        if not new:
            break
        seen.update(x["id"] for x in new)
        out += new
        cur = max(x["created_utc"] for x in batch)
        if len(batch) < 100:
            break
    return out


def page_all(endpoint, params, cache):
    if cache.exists():
        return [json.loads(l) for l in cache.open()]
    after, before = params.pop("after"), params.pop("before")
    out = fetch_range(endpoint, params, after, before)
    out = list({x["id"]: x for x in out}.values())
    tmp = cache.with_suffix(".tmp")
    with tmp.open("w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    tmp.rename(cache)
    return out


def months():
    d = dt.date.fromisoformat(START)
    end = dt.date.fromisoformat(END)
    while d < end:
        n = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield d.isoformat(), min(n, end).isoformat()
        d = n


POST_FIELDS = "id,subreddit,author,created_utc,title,selftext,url,score,num_comments,link_flair_text"
COM_FIELDS = "id,subreddit,author,created_utc,body,score,link_id,parent_id,author_flair_text"


def stage_posts(sub):
    d = RAW / sub / "posts"; d.mkdir(parents=True, exist_ok=True)
    total = hits = 0
    for a, b in months():
        rows = page_all("posts/search", {"subreddit": sub, "after": ts(a), "before": ts(b), "fields": POST_FIELDS}, d / f"{a}.jsonl")
        m = [r for r in rows if BROAD.search((r.get("title") or "") + " " + (r.get("selftext") or ""))]
        total += len(rows); hits += len(m)
        log(f"{sub} posts {a}: {len(rows)} listed, {len(m)} match (cum {total}/{hits})")


def stage_kwcoms(sub):
    d = RAW / sub / "kwcoms"; d.mkdir(parents=True, exist_ok=True)
    # "we believe" is not supported as a phrase by the server, so it is filtered locally from post/thread text instead.
    # The misspellings are rare, so only scan the ad year for them.
    for term, since in [("Gillette", "2019-01-01"), ("Gilette", "2019-01-01"), ("Gillete", "2019-01-01")]:
        tag = re.sub(r"\W", "", term)
        for a, b in months():
            if term != "Gillette" and a >= "2020-01-01":
                continue
            rows = page_all("comments/search", {"subreddit": sub, "body": term, "after": ts(a), "before": ts(b), "fields": COM_FIELDS},
                            d / f"{tag}_{a}.jsonl")
            if rows:
                log(f"{sub} kwcoms {term} {a}: {len(rows)}")


def matched_link_ids(sub):
    ids = set()
    for f in (RAW / sub / "posts").glob("*.jsonl"):
        for l in f.open():
            r = json.loads(l)
            t = (r.get("title") or "") + " " + (r.get("selftext") or "")
            # threads only for Gillette posts ("we believe" alone is mostly generic); skip the Moore & Gillette book
            if re.search(r"gil+et+e", t, re.I) and not re.search(r"moore (?:and|&) gil+et+e", t, re.I):
                ids.add(r["id"])
    return ids


def stage_threads(sub):
    d = RAW / sub / "threads"; d.mkdir(parents=True, exist_ok=True)
    ids = sorted(matched_link_ids(sub))
    log(f"{sub}: fetching {len(ids)} threads")
    n = 0
    for i, pid in enumerate(ids):
        rows = page_all("comments/search", {"link_id": f"t3_{pid}", "after": ts("2005-06-01"), "before": ts("2030-01-01"), "fields": COM_FIELDS}, d / f"{pid}.jsonl")
        n += len(rows)
        if i % 20 == 0:
            log(f"{sub} threads {i+1}/{len(ids)} cum comments {n}")
    log(f"{sub} threads done: {n} comments")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    subs = sys.argv[2:] or SUBS
    for sub in subs:
        for name, fn in [("posts", stage_posts), ("kwcoms", stage_kwcoms), ("threads", stage_threads)]:
            if stage in (name, "all"):
                fn(sub)
