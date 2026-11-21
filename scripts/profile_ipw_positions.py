"""Audit the logged exposure positions used as inputs to donor-only IPW features."""

import argparse
import json
from pathlib import Path

import duckdb

from src.common.preparation import write_json
from src.common.train_selection import select_parts


def audit(prepared, raw, count):
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    selected = {p["source"] for p in select_parts(manifest, count)}
    donors = [p for p in manifest["splits"]["train"]["parts"] if p["source"] not in selected]
    with duckdb.connect(config={"threads": "2", "memory_limit": "512MB"}) as db:
        db.from_parquet([str(raw / "train" / "impressions" / p["source"]) for p in donors]).create_view(
            "feeds"
        )
        db.from_parquet([str(prepared / p["candidate_path"]) for p in donors]).create_view("candidates")
        query = db.execute("""WITH sizes AS (
            SELECT len(pos) AS n, list_min(pos) AS lo,list_max(pos) AS hi,list_unique(pos) AS distinct_pos FROM feeds)
            SELECT count(*) AS feeds,sum(n) AS candidates,min(lo) AS min_pos,max(hi) AS max_pos,
                   count(*) FILTER (WHERE lo>0) AS feeds_missing_first_position,
                   count(*) FILTER (WHERE hi-lo+1>distinct_pos) AS feeds_with_internal_gaps,
                   count(*) FILTER (WHERE hi>=n) AS feeds_with_pos_ge_candidate_count,
                   count(*) FILTER (WHERE distinct_pos<>n) AS feeds_repeated_or_null_position FROM sizes""")
        positions = dict(zip([c[0] for c in query.description], query.fetchone()))
        query = db.execute("""WITH bad AS (SELECT feed_id,pos,item_id FROM feeds WHERE list_unique(pos)<>len(pos)),
            unfolded AS (SELECT feed_id,unnest(pos) AS pos,unnest(item_id) AS item_id FROM bad),
            pairs AS (SELECT feed_id,pos,count(*) AS n,count(DISTINCT item_id) AS items FROM unfolded GROUP BY feed_id,pos)
            SELECT (SELECT count(*) FROM unfolded WHERE pos IS NULL) AS null_positions,
                   count(*) FILTER (WHERE n>1) AS repeated_positions,coalesce(sum(n) FILTER (WHERE n>1),0) AS affected_rows,
                   count(*) FILTER (WHERE n>1 AND items>1) AS repeated_positions_with_different_items FROM pairs""")
        duplicates = dict(zip([c[0] for c in query.description], query.fetchone()))
        query = db.execute("""SELECT CASE WHEN pos<1 THEN '00' WHEN pos<5 THEN '01-04' WHEN pos<10 THEN '05-09'
            WHEN pos<30 THEN '10-29' WHEN pos<100 THEN '30-99' ELSE '100+' END AS position,
            count(*) AS impressions,sum(target)::BIGINT AS clicks,avg(target) AS ctr
            FROM candidates GROUP BY 1 ORDER BY min(pos)""")
        cols = [c[0] for c in query.description]
        buckets = [dict(zip(cols, row)) for row in query.fetchall()]
        if sum(v["impressions"] for v in buckets) != positions["candidates"]:
            raise ValueError("raw and prepared candidate totals differ")
    return {
        "donors": len(donors),
        "excluded_sources": sorted(selected),
        "positions": positions,
        "duplicate_diagnostics": duplicates,
        "buckets": buckets,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--raw", type=Path, default=Path("competition"))
    parser.add_argument("--train-parts", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.prepared, args.raw, args.train_parts))
