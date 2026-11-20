"""Measure the donor history-to-impression join size before materializing the click graph."""

import argparse
import json
from pathlib import Path

import duckdb

from scripts.run_joint_oof import digest
from src.common.preparation import write_json
from src.common.train_selection import select_parts


def profile(prepared, count, start):
    manifest_path = prepared / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if not manifest["complete"]:
        raise ValueError("prepared data incomplete")
    cohort = select_parts(manifest, count, start)
    excluded = {part["source"] for part in cohort}
    donors = [p for p in manifest["splits"]["train"]["parts"] if p["source"] not in excluded]
    queries = cohort + manifest["splits"]["val"]["parts"] + manifest["splits"]["test"]["parts"]
    with duckdb.connect(
        config={"memory_limit": "1GB", "threads": "2", "preserve_insertion_order": "false"}
    ) as db:
        db.from_parquet([str(prepared / p["candidate_path"]) for p in queries]).create_view("query_rows")
        db.execute(
            "CREATE TEMP TABLE query_items AS SELECT DISTINCT item_id FROM query_rows WHERE item_id > 0"
        )
        query_items = db.execute("SELECT count(*) FROM query_items").fetchone()[0]
        totals = {}
        bounds = []
        for i, p in enumerate(donors):
            db.from_parquet(str(prepared / p["candidate_path"])).create_view("candidates", replace=True)
            db.from_parquet(str(prepared / p["feed_path"])).create_view("feeds", replace=True)
            row = db.execute("""
                WITH per_feed AS (
                  SELECT feed_id, count(*) AS impr, sum(target) AS clicks,
                         count(q.item_id) AS eligible, sum(IF(q.item_id IS NULL, 0, target)) AS eligible_clicks
                  FROM candidates c LEFT JOIN query_items q USING (item_id) GROUP BY feed_id
                ), histories AS (
                  SELECT feed_id, timestamp, len(list_filter(history__item_id[:50], x -> x > 0)) AS h
                  FROM feeds
                )
                SELECT count(*) AS feeds, sum(impr) AS impressions, sum(clicks) AS clicks,
                       sum(eligible) AS query_impressions, sum(eligible_clicks) AS query_clicks,
                       sum(h*eligible) AS history_impression_pairs, sum(h*eligible_clicks) AS history_click_pairs,
                       min(timestamp) AS first_ts, max(timestamp) AS last_ts
                FROM per_feed JOIN histories USING (feed_id)
            """)
            values = dict(zip([column[0] for column in row.description], row.fetchone()))
            bounds.append((values.pop("first_ts"), values.pop("last_ts")))
            if values["feeds"] != p["feeds"] or values["impressions"] != p["candidates"]:
                raise ValueError(f"incomplete donor join: {p['source']}")
            for key, v in values.items():
                totals[key] = totals.get(key, 0) + int(v)
            if (i + 1) % 64 == 0:
                print(f"{i + 1}/{len(donors)} donors", flush=True)
        by_split = {}
        for split in ("val", "test"):
            db.from_parquet(
                [str(prepared / p["feed_path"]) for p in manifest["splits"][split]["parts"]]
            ).create_view("scored", replace=True)
            by_split[split] = db.execute("SELECT min(timestamp), max(timestamp) FROM scored").fetchone()
    last = max(b for _, b in bounds)
    if last >= min(v[0] for v in by_split.values()) or digest(manifest_path) != manifest_hash:
        raise ValueError("donor chronology or source manifest changed")
    return {
        "prepared_sha256": manifest_hash,
        "excluded_shards": sorted(excluded),
        "donor_shards": len(donors),
        "query_items": query_items,
        "donor_min_ts": min(a for a, _ in bounds),
        "donor_max_ts": last,
        "scored_time_bounds": by_split,
        "totals": totals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--train-parts", type=int, default=8)
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.out, profile(args.prepared, args.train_parts, args.train_start))
