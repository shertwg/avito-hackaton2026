"""Measure prior impression coverage for validation and test using strictly earlier timestamps."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from src.common.preparation import write_json


def coverage(prepared, scratch):
    source = json.loads((prepared / "manifest.json").read_bytes())
    if not source["complete"]:
        raise ValueError("prepared data incomplete")
    scratch.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=scratch) as tmp,
        duckdb.connect(
            str(Path(tmp) / "exposure.duckdb"),
            config={
                "memory_limit": "1GB",
                "threads": "2",
                "preserve_insertion_order": "false",
                "temp_directory": str(Path(tmp) / "spill"),
            },
        ) as db,
    ):

        def view(name, entries):
            db.from_parquet([str(prepared / p["candidate_path"]) for p in entries]).create_view(name)

        view("train_rows", source["splits"]["train"]["parts"])
        scored = source["splits"]["val"]["parts"] + source["splits"]["test"]["parts"]
        view("scored_rows", scored)
        latest, earliest = db.execute(
            "SELECT (SELECT max(timestamp) FROM train_rows), (SELECT min(timestamp) FROM scored_rows)"
        ).fetchone()
        if latest >= earliest:
            raise ValueError("train and scored timestamps must be disjoint and ordered")
        summary = {}
        for key in ("item_id", "category_id"):
            db.execute(
                f"CREATE OR REPLACE TEMP TABLE scored_keys AS SELECT DISTINCT user_id,{key} FROM scored_rows"
            )
            db.execute(f"""CREATE OR REPLACE TEMP TABLE last_train AS
                SELECT user_id,{key},max(timestamp) AS last_ts FROM train_rows
                SEMI JOIN scored_keys USING (user_id,{key}) GROUP BY user_id,{key}""")
            # Collapse a user's simultaneous observations before the window; all tied feeds see the same past.
            db.execute(f"""CREATE OR REPLACE TEMP TABLE prior_times AS
                SELECT user_id,{key},timestamp,CASE WHEN user_id>0 AND {key}>0 THEN
                  coalesce(lag(timestamp) OVER (PARTITION BY user_id,{key} ORDER BY timestamp),last_ts) END AS prior
                FROM (SELECT DISTINCT user_id,{key},timestamp FROM scored_rows)
                LEFT JOIN last_train USING (user_id,{key})""")
            check = db.execute("SELECT count(*) FROM prior_times WHERE prior >= timestamp").fetchone()[0]
            if check:
                raise ValueError("prior time is not strictly before current observation")
            rows = db.execute(f"""WITH per_feed AS (
                  SELECT split,feed_id,count(*) AS rows,count(prior) AS known,
                         count(*) FILTER (WHERE user_id=0 OR {key}=0) AS missing_key,
                         count(*) FILTER (WHERE s.timestamp-prior <= 86400) AS known_day,
                         count(*) FILTER (WHERE s.timestamp-prior <= 3600) AS known_hour
                  FROM scored_rows s JOIN prior_times USING (user_id,{key},timestamp) GROUP BY split,feed_id
                ) SELECT split,count(*),sum(rows),sum(known),count(*) FILTER (WHERE known>0),
                         sum(known_day),sum(known_hour),sum(missing_key) FROM per_feed GROUP BY split ORDER BY split""").fetchall()
            summary[key] = {}
            for split, feeds, n, known, n_feeds, day, hour, missing in rows:
                entries = source["splits"][split]["parts"]
                if feeds != sum(p["feeds"] for p in entries) or n != sum(p["candidates"] for p in entries):
                    raise ValueError("scored feed coverage differs from manifest")
                summary[key][split] = {
                    "feeds": feeds,
                    "rows": n,
                    "rows_with_previous": known,
                    "feeds_with_previous": n_feeds,
                    "rows_previous_within_24h": day,
                    "rows_previous_within_1h": hour,
                    "rows_with_missing_user_or_key": missing,
                }
            print(key, summary[key], flush=True)
    return {
        "training_shards": len(source["splits"]["train"]["parts"]),
        "max_train_ts": latest,
        "min_scored_ts": earliest,
        "scope": "Target-free; prior exposures from all strictly earlier train/val/test impressions",
        "coverage": summary,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--scratch", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_json(a.out, coverage(a.prepared, a.scratch))
