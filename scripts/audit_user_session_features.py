"""Verify session cache rows and recompute sampled features from prepared observations."""

import argparse
import hashlib
import json
from collections import defaultdict
from math import log1p
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.common.feature_cache import selected_parts
from src.common.preparation import fingerprint, write_json
from src.common.user_session import KEY, NAMES, POLICY


def check_payloads(prepared, cache, entries, meta):
    samples, summary = [], {}
    for split, parts in entries.items():
        saved = meta["splits"][split]["parts"]
        if set(saved) != {entry["source"] for entry in parts}:
            raise ValueError(f"cache coverage differs for {split}")
        rows = item = category = 0
        feed_examples = []
        nearest_item = None
        for number, entry in enumerate(parts):
            record = saved[entry["source"]]
            path = cache / record["path"]
            if fingerprint(path) != record["fingerprint"]:
                raise ValueError(f"cache changed: {path}")
            preds = pl.read_parquet(path)
            candidates = pl.read_parquet(
                prepared / entry["candidate_path"],
                columns=[
                    *KEY,
                    "user_id",
                    "timestamp",
                    "category_id",
                ],
            ).sort("feed_id", "candidate_index")
            if preds.columns != KEY + NAMES or not preds.select(KEY).equals(candidates.select(KEY)):
                raise ValueError(f"cache keys differ: {path}")
            if preds.select(KEY[:3]).is_duplicated().any():
                raise ValueError("duplicate occurrence keys")
            values = preds.select(NAMES)
            if values.dtypes != [pl.Float32] * len(NAMES) or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"non-Float32 or non-finite features: {path}")
            for name, (lo, hi) in meta["domains"].items():
                if (lo is not None and values[name].min() < lo) or (
                    hi is not None and values[name].max() > hi
                ):
                    raise ValueError(f"{name} outside domain")
            missing = candidates["user_id"] == 0
            is_missing_key = (candidates["item_id"] == 0) | missing
            if (preds["us_item_prior_impr"].filter(is_missing_key) != 0).any():
                raise ValueError("missing ID used as an item history key")
            rows += len(preds)
            item += int((preds["us_item_prior_impr"] > 0).sum())
            category += int((preds["us_cat_prior_impr_share"] > 0).sum())
            known = (candidates["candidate_index"] == 0) & ~missing
            feed_examples.append(
                pl.concat([candidates.filter(known), values.filter(known)], how="horizontal")
            )
            gaps = preds["us_time_since_item_shown_log"]
            if (gaps > 0).any():
                index = gaps.set(gaps <= 0, float("inf")).arg_min()
                if nearest_item is None or gaps[index] < nearest_item[0]:
                    nearest_item = (gaps[index], candidates.row(index, named=True), values.row(index))
            # Pick rich histories in several parts and explicit missing-user cases.
            if number in {0, len(parts) // 2, len(parts) - 1} and len(preds):
                picks = {
                    0,
                    preds["us_cat_recent3_impr_share"].arg_max(),
                    preds["us_user_prior_feeds_log"].arg_max(),
                    preds["us_item_prior_impr"].arg_max(),
                }
                if missing.any():
                    picks.add(missing.arg_max())
                samples.extend((candidates.row(i, named=True), values.row(i)) for i in sorted(picks))
        examples = pl.concat(feed_examples).sort("user_id", "timestamp", "feed_id")
        tied_keys = (
            examples.group_by("user_id", "timestamp", maintain_order=True).len().filter(pl.col("len") > 1)
        )
        tied = examples.join(tied_keys.head(3), on=["user_id", "timestamp"], how="semi")
        exposed_recently = (
            examples.filter(pl.col("us_time_since_prev_feed_log") >= 0)
            .sort("us_time_since_prev_feed_log", "feed_id")
            .head(3)
        )
        for row in pl.concat([tied, exposed_recently]).iter_rows(named=True):
            samples.append((row, tuple(row[name] for name in NAMES)))
        if nearest_item is not None:
            samples.append(nearest_item[1:])
        summary[split] = {
            "rows": rows,
            "item_history_rows": item,
            "category_history_rows": category,
            "simultaneous_feed_samples": tied.height,
        }
    return samples, summary


def raw_observations(prepared, all_parts, donors, users, categories):
    with duckdb.connect(
        config={"threads": "2", "memory_limit": "1536MB", "preserve_insertion_order": "false"}
    ) as db:
        db.from_parquet([str(prepared / p["candidate_path"]) for p in all_parts]).create_view("observations")
        db.from_parquet([str(prepared / p["candidate_path"]) for p in donors]).create_view("donors")
        events = db.execute(
            """
            SELECT user_id,split,timestamp,feed_id,item_id,category_id
            FROM observations WHERE user_id IN (SELECT unnest(?))
            ORDER BY user_id,timestamp,feed_id,split,candidate_index
        """,
            [users],
        ).fetchall()
        positions = db.execute("SELECT pos,count(*),sum(target) FROM donors GROUP BY pos").fetchall()
        ctr = sum(k for _, _, k in positions) / sum(n for _, n, _ in positions)
        rates = {pos: k / n for pos, n, k in positions}
        category_ctr = dict(
            db.execute(
                """
            SELECT category_id,avg(target) FROM donors
            WHERE category_id IN (SELECT unnest(?)) GROUP BY category_id
        """,
                [categories],
            ).fetchall()
        )
        labels = db.execute(
            """
            SELECT user_id,category_id,pos,target FROM donors WHERE user_id IN (SELECT unnest(?))
        """,
            [users],
        ).fetchall()
    histories = defaultdict(dict)
    for user, split, ts, feed, item, cat in events:
        histories[user].setdefault((ts, feed, split), []).append((item, cat))
    affinities = defaultdict(list)
    for user, cat, pos, clicked in labels:
        affinities[user].append((cat, rates[pos], clicked))
    return histories, affinities, ctr, category_ctr


def recompute(samples, histories, affinities, global_ctr, category_ctr):
    latest = None
    min_gap = None
    for row, observed in samples:
        user, now, item, category = (row[key] for key in ("user_id", "timestamp", "item_id", "category_id"))
        previous = [(t, pairs) for (t, _, _), pairs in histories.get(user, {}).items() if t < now]
        old_items = [t for t, pairs in previous if item > 0 and any(item == i for i, _ in pairs)]
        session_idx = 0
        tick = now
        for t, _ in reversed(previous):
            if tick - t > 1800:
                break
            session_idx += 1
            tick = t
        if previous:
            max_previous = max(t for t, _ in previous)
            if max_previous >= now:
                raise ValueError("noncausal exposure")
            latest = max_previous if latest is None else max(latest, max_previous)
            min_gap = now - max_previous if min_gap is None else min(min_gap, now - max_previous)

        def share(feeds, category_id):
            cats = [c for _, pairs in feeds for _, c in pairs]
            return sum(c == category_id for c in cats) / len(cats) if cats and category_id > 0 else 0.0

        donor = affinities.get(user, [])
        clicks = sum(label for _, _, label in donor)
        expected = sum(p for _, p, _ in donor)
        smooth_ctr = (clicks + 5 * global_ctr) / (len(donor) + 5)
        coe = (clicks + 5 * global_ctr) / (expected + 5 * global_ctr) if expected + 5 * global_ctr else 1.0
        in_category = [label for cat, _, label in donor if cat == category and category > 0]
        prior = category_ctr.get(category, 0.0)
        cat_lift = (sum(in_category) + 5 * prior) / (len(in_category) + 5) - prior
        expected_row = [
            log1p(len(previous)),
            log1p(now - previous[-1][0]) if previous else -1.0,
            sum(now - t <= 3600 for t, _ in previous),
            sum(now - t <= 86400 for t, _ in previous),
            session_idx,
            len(old_items),
            log1p(now - old_items[-1]) if old_items else -1.0,
            share(previous, category),
            share(previous[-3:], category),
            log1p(len(donor)),
            smooth_ctr,
            coe,
            cat_lift,
            log1p(sum(in_category)),
        ]
        np.testing.assert_allclose(observed, expected_row, rtol=2e-6, atol=1e-7, err_msg=str(row))
    return latest, min_gap


def audit(prepared, cache, output):
    prepared, cache = Path(prepared), Path(cache)
    payload = (cache / "manifest.json").read_bytes()
    meta = json.loads(payload)
    manifest = (prepared / "manifest.json").read_bytes()
    source = json.loads(manifest)
    identity = meta["identity"]
    if (
        not meta["complete"]
        or meta["features"] != NAMES
        or meta["target_policy"] != POLICY
        or identity["prepared_manifest_sha256"] != hashlib.sha256(manifest).hexdigest()
    ):
        raise ValueError("incomplete or incompatible user session cache")
    excluded = meta["excluded_train_sources"]
    selected = selected_parts(source, len(excluded), identity.get("train_start"))
    if sorted(p["source"] for p in selected) != excluded or identity["excluded_train_sources"] != excluded:
        raise ValueError("incorrect donor exclusion")
    for rel, stamp in identity["input_fingerprints"].items():
        if fingerprint(prepared / rel) != stamp:
            raise ValueError(f"prepared inputs changed: {rel}")
    splits = source["splits"]
    entries = {"train": selected, "val": splits["val"]["parts"], "test": splits["test"]["parts"]}
    all_parts = [p for record in splits.values() for p in record["parts"]]
    donors = [p for p in splits["train"]["parts"] if p["source"] not in excluded]
    samples, summary = check_payloads(prepared, cache, entries, meta)
    users = sorted({c["user_id"] for c, _ in samples if c["user_id"] > 0})
    cats = sorted({c["category_id"] for c, _ in samples if c["category_id"] > 0})
    latest, min_gap = recompute(samples, *raw_observations(prepared, all_parts, donors, users, cats))
    if (cache / "manifest.json").read_bytes() != payload:
        raise ValueError("cache manifest changed during audit")
    result = {
        "complete": True,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "features": len(NAMES),
        "splits": summary,
        "donor_shards": len(donors),
        "excluded_train_sources": excluded,
        "sampled_candidates": len(samples),
        "max_used_timestamp": latest,
        "minimum_gap_seconds_in_checked_histories": min_gap,
        "build_resources": meta["resources"],
        "build_seconds": meta["duration_seconds"],
    }
    write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cache, args.out), indent=2))
