"""Check every click graph cache part and independently recompute sampled rows from donor Parquet."""

import argparse
import hashlib
import json
from math import log, log2, sqrt
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.common.click_item2item import KEY, NAMES
from src.common.feature_cache import selected_parts
from src.common.preparation import fingerprint, write_json


def sample_and_check(prepared, cache, source, meta, chosen):
    samples, rows = [], {}
    for split in ("train", "val", "test"):
        entries = chosen if split == "train" else source["splits"][split]["parts"]
        parts = meta["splits"][split]["parts"]
        if set(parts) != {e["source"] for e in entries}:
            raise ValueError("incomplete click graph coverage")
        rows[split] = 0
        for ix, e in enumerate(entries):
            p = parts[e["source"]]
            path = cache / p["path"]
            if fingerprint(path) != p["fingerprint"]:
                raise ValueError("cache payload changed")
            f = pl.read_parquet(path)
            c = pl.read_parquet(prepared / e["candidate_path"], columns=[*KEY, "category_id"]).sort(
                "feed_id", "candidate_index"
            )
            if (
                f.columns != KEY + NAMES
                or not f.select(KEY).equals(c.select(KEY))
                or f.select(KEY[:3]).is_duplicated().any()
            ):
                raise ValueError("candidate keys changed")
            vals = f.select(NAMES)
            if vals.dtypes != [pl.Float32] * len(NAMES) or not np.isfinite(vals.to_numpy()).all():
                raise ValueError("invalid feature values")
            rows[split] += len(c)
            if ix not in {0, len(entries) - 1}:
                continue
            feeds = pl.read_parquet(prepared / e["feed_path"])
            for i in sorted(
                {0, int(f["ci2i_matched_frac_50"].arg_max()), int(f["coclick_max_50"].arg_max())}
            ):
                candidate = c.row(i, named=True)
                history = feeds.filter(pl.col("feed_id") == candidate["feed_id"]).row(0, named=True)
                samples.append((candidate, history, np.array(vals.row(i))))
    return samples, rows


def donor_reference(prepared, donors, samples):
    cand_ids = sorted({c["item_id"] for c, _, _ in samples})
    cats = sorted({c["category_id"] for c, _, _ in samples})
    h_ids = sorted({i for _, hist, _ in samples for i in hist["history__item_id"][:50]})
    with duckdb.connect(
        config={"threads": "2", "memory_limit": "1536MB", "preserve_insertion_order": "false"}
    ) as db:
        for key, name in [("candidate_path", "c"), ("feed_path", "f")]:
            db.from_parquet([str(prepared / p[key]) for p in donors]).create_view(name)
        db.execute(
            "CREATE TEMP TABLE chosen AS SELECT * FROM c WHERE item_id>0 AND item_id IN (SELECT unnest(?))",
            [cand_ids],
        )
        global_ctr = db.execute("SELECT avg(target) FROM c").fetchone()[0]

        def counts(sql, params=()):
            return {tuple(row[:-2]): row[-2:] for row in db.execute(sql, params).fetchall()}

        item = counts("SELECT item_id,count(*),sum(target) FROM chosen GROUP BY 1")
        cat = counts(
            "SELECT category_id,count(*),sum(target) FROM c WHERE category_id IN (SELECT unnest(?)) GROUP BY 1",
            [cats],
        )
        db.execute("""CREATE TEMP TABLE transition AS SELECT item_id,target,unnest(history__item_id[:50]) h,
               unnest(history__category_id[:50]) hc FROM chosen JOIN f USING(feed_id)""")
        direct = counts(
            "SELECT h,item_id,count(*),sum(target) FROM transition WHERE h>0 AND h IN (SELECT unnest(?)) GROUP BY 1,2",
            [h_ids],
        )
        catitem = counts("SELECT hc,item_id,count(*),sum(target) FROM transition WHERE hc>0 GROUP BY 1,2")
        db.execute(
            """CREATE TEMP TABLE per_cat AS SELECT feed_id,category_id,count(*) n,sum(target) k FROM c
                   WHERE category_id>0 AND category_id IN (SELECT unnest(?)) GROUP BY 1,2""",
            [cats],
        )
        back = []
        for field in ("event_type", "event_source"):
            back.append(
                counts(f"""WITH e AS (SELECT feed_id,unnest(history__category_id[:50]) hc,
                     unnest(history__{field}[:50]) code FROM f)
                     SELECT hc,code,category_id,sum(n),sum(k) FROM e JOIN per_cat USING(feed_id)
                     WHERE hc>0 AND code>0 GROUP BY 1,2,3""")
            )
        db.execute("""CREATE TEMP TABLE click_users AS SELECT DISTINCT item_id,user_id FROM c
                   WHERE target=1 AND item_id>0 AND user_id>0""")
        postings = {}
        degrees = {}
        for i, u, degree in db.execute(
            """WITH needed AS (SELECT * FROM click_users WHERE item_id IN (SELECT unnest(?))),
                   degrees AS (SELECT user_id,count(*) d FROM click_users SEMI JOIN needed USING(user_id) GROUP BY 1)
                   SELECT item_id,user_id,d FROM needed JOIN degrees USING(user_id)""",
            [cand_ids + h_ids],
        ).fetchall():
            postings.setdefault(i, set()).add(u)
            degrees[u] = degree

    def base(table, key):
        n, k = table.get((key,), (0, 0))
        return (k + 20 * global_ctr) / (n + 20)

    def lifted(n, k, p):
        ctr = (k + 12 * p) / (n + 12)
        return ctr, log((ctr + 1e-6) / (p + 1e-6))

    for c, f, observed in samples:
        item_id, cat_id = c["item_id"], c["category_id"]
        p, pc = base(item, item_id), base(cat, cat_id)
        events = list(
            zip(
                *(
                    f["history__" + field][:50]
                    for field in ("item_id", "category_id", "event_type", "event_source")
                ),
                strict=True,
            )
        )
        hits = []
        lifts = []
        ctrs = []
        co = []
        event = []
        src = []
        ci = []
        for j, (h, hc, ev, origin) in enumerate(events):
            n, k = direct.get((h, item_id), (0, 0))
            match = n >= 3 or k >= 1
            hits.append(match)
            ctr, lift = lifted(n, k, p) if match else (0, 0)
            lifts.append(lift)
            ctrs.append(ctr)
            left, right = postings.get(h, set()), postings.get(item_id, set())
            sim = (
                sum(1 / log2(2 + degrees[u]) for u in left & right) / sqrt(len(left) * len(right))
                if left and right
                else 0
            )
            co.append(sim * 2 ** (-j / 10))
            for scores, table, key, prior, threshold in [
                (event, back[0], (hc, ev, cat_id), pc, 1),
                (src, back[1], (hc, origin, cat_id), pc, 1),
                (ci, catitem, (hc, item_id), p, 10),
            ]:
                n, k = table.get(key, (0, 0))
                scores.append(lifted(n, k, prior)[1] if n >= threshold else 0)

        def mean(v):
            return sum(v) / len(v) if v else 0.0

        def mx(v):
            return max(v, default=0.0)

        expected = [
            mean(lifts[:10]),
            mx(lifts[:10]),
            sum(max(0, l) for l in lifts[:10]),
            mx(ctrs[:10]),
            mean(hits[:10]),
            mean(lifts),
            mx(lifts),
            mx([l for l, e in zip(lifts, events) if e[2] == 2]),
            mx([l for l, e in zip(lifts, events) if e[2] == 10]),
            mean(hits),
            mean(co[:10]),
            mx(co[:10]),
            mean(co),
            mx(co),
            mean(event[:10]),
            mx(event[:10]),
            mean(src),
            mx(ci),
        ]
        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-6, err_msg=f"candidate {c}")


def audit(prepared, cache, output):
    prepared, cache = Path(prepared), Path(cache)
    path = cache / "manifest.json"
    meta = json.loads(path.read_bytes())
    source = json.loads((prepared / "manifest.json").read_bytes())
    identity = meta["identity"]
    if (
        not meta["complete"]
        or meta["features"] != NAMES
        or identity["prepared_manifest_sha256"]
        != hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("incomplete or mismatched click graph cache")
    chosen = selected_parts(source, len(meta["excluded_train_sources"]), identity.get("train_start"))
    excluded = {p["source"] for p in chosen}
    if (
        sorted(excluded) != meta["excluded_train_sources"]
        or sorted(excluded) != identity["excluded_train_sources"]
    ):
        raise ValueError("incorrect donor exclusion")
    donors = [p for p in source["splits"]["train"]["parts"] if p["source"] not in excluded]
    for rel, stamp in identity["input_fingerprints"].items():
        if fingerprint(prepared / rel) != stamp:
            raise ValueError("prepared inputs changed")
    samples, rows = sample_and_check(prepared, cache, source, meta, chosen)
    donor_reference(prepared, donors, samples)
    result = dict(
        complete=True,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=rows,
        features=len(NAMES),
        scalar_candidates=len(samples),
        donor_shards=len(donors),
        excluded_sources=sorted(excluded),
        resources=meta["resources"],
    )
    write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    print(audit(a.prepared, a.cache, a.out))
