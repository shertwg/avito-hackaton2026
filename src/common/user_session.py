"""Build strictly earlier user exposure features and donor-only user affinity."""

import json
from contextlib import contextmanager
from pathlib import Path

import duckdb

from src.common.feature_cache import build_cache, cache_lock, selected_parts
from src.common.preparation import write_json

KEY = ["split", "feed_id", "candidate_index", "item_id"]
NAMES = [
    "us_user_prior_feeds_log",
    "us_time_since_prev_feed_log",
    "us_feeds_last_1h",
    "us_feeds_last_24h",
    "us_session_idx",
    "us_item_prior_impr",
    "us_time_since_item_shown_log",
    "us_cat_prior_impr_share",
    "us_cat_recent3_impr_share",
    "us_donor_user_impr_log",
    "us_donor_user_ctr_smooth",
    "us_donor_user_coe",
    "us_donor_user_cat_lift",
    "us_donor_user_cat_clicks_log",
]
POLICY = "Exposure uses unlabeled impressions at strictly earlier times. Affinity uses only excluded-cohort train donors."
SMOOTHING = 5


def table(db, name, query):
    db.execute(f"CREATE TABLE IF NOT EXISTS {name} AS {query}")
    print(f"user_session {name}", flush=True)


def create_exposures(db):
    table(
        db,
        "query_rows",
        """
        SELECT split,feed_id,candidate_index,item_id,category_id,user_id,timestamp,
               regexp_extract(replace(filename,chr(92),'/'),'[^/]+$') shard FROM query_inputs
    """,
    )
    table(db, "query_users", "SELECT DISTINCT user_id FROM query_rows WHERE user_id>0")
    table(
        db,
        "exposures",
        """
        SELECT split,feed_id,user_id,timestamp,item_id,category_id
        FROM exposure_inputs SEMI JOIN query_users USING(user_id)
    """,
    )
    table(
        db,
        "feed_order",
        """
        WITH feeds AS (
            SELECT user_id,timestamp,split,feed_id,count(*) AS n FROM exposures GROUP BY ALL
        )
        SELECT *, row_number() OVER(PARTITION BY user_id ORDER BY timestamp,feed_id,split) AS ordinal
        FROM feeds
    """,
    )
    table(
        db,
        "clock_features",
        """
        WITH clock AS (
            SELECT user_id,timestamp,min(ordinal) AS first_ordinal,count(*) AS feeds,sum(n) AS candidates
            FROM feed_order GROUP BY user_id,timestamp
        ), gaps AS (
            SELECT *, lag(timestamp) OVER(PARTITION BY user_id ORDER BY timestamp) AS previous_time
            FROM clock
        ), sessions AS (
            SELECT *, sum(CASE WHEN previous_time IS NULL OR timestamp-previous_time>1800 THEN 1 ELSE 0 END)
                OVER(PARTITION BY user_id ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS session
            FROM gaps
        )
        SELECT user_id,timestamp,first_ordinal,ln(first_ordinal) AS feed_log,
            coalesce(ln(1+timestamp-previous_time),-1) AS time_log,
            coalesce(sum(feeds) OVER(PARTITION BY user_id ORDER BY timestamp
                RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),0) AS feeds_hour,
            coalesce(sum(feeds) OVER(PARTITION BY user_id ORDER BY timestamp
                RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),0) AS feeds_day,
            first_ordinal-min(first_ordinal) OVER(PARTITION BY user_id,session) AS session_idx,
            coalesce(sum(candidates) OVER(PARTITION BY user_id ORDER BY timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_candidates
        FROM sessions
    """,
    )
    table(
        db,
        "item_history",
        """
        WITH ticks AS (
            SELECT user_id,item_id,timestamp,count(DISTINCT (split,feed_id)) n
            FROM exposures WHERE item_id>0 GROUP BY ALL
        )
        SELECT user_id,item_id,timestamp,
            coalesce(sum(n) OVER(item_window ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_feeds,
            lag(timestamp) OVER(item_window) AS prev_item_time
        FROM ticks WINDOW item_window AS (PARTITION BY user_id,item_id ORDER BY timestamp)
    """,
    )
    table(
        db,
        "category_history",
        """
        WITH ticks AS (
            SELECT user_id,category_id,timestamp,count(*) n FROM exposures WHERE category_id>0 GROUP BY ALL
        )
        SELECT user_id,category_id,timestamp,coalesce(sum(n) OVER(PARTITION BY user_id,category_id
            ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prior_category
        FROM ticks
    """,
    )
    table(
        db,
        "recent_feeds",
        """
        WITH query_times AS (
            SELECT DISTINCT user_id,timestamp FROM query_rows WHERE user_id>0
        ), offsets AS (
            SELECT user_id,timestamp,first_ordinal-delta AS ordinal
            FROM query_times JOIN clock_features USING(user_id,timestamp),range(1,4) AS offsets(delta)
        )
        SELECT o.user_id,o.timestamp,f.split AS old_split,f.feed_id AS old_feed_id,f.n
        FROM offsets o JOIN feed_order f USING(user_id,ordinal)
    """,
    )
    table(db, "recent_total", "SELECT user_id,timestamp,sum(n) AS total FROM recent_feeds GROUP BY ALL")
    table(
        db,
        "recent_category",
        """
        SELECT r.user_id,r.timestamp,e.category_id,count(*) AS recent_category
        FROM recent_feeds r JOIN exposures e ON (r.user_id,r.old_split,r.old_feed_id)=(e.user_id,e.split,e.feed_id)
        WHERE e.category_id>0 GROUP BY ALL
    """,
    )


def create_affinity(db):
    table(
        db, "position_rates", "SELECT pos,count(*) n,sum(target) k,avg(target) rate FROM donors GROUP BY pos"
    )
    invalid = db.execute("""
        SELECT count(*) FROM donors WHERE pos IS NULL OR pos<0 OR target IS NULL OR target NOT IN (0,1)
    """).fetchone()[0]
    if invalid or not db.execute("SELECT count(*) FROM position_rates").fetchone()[0]:
        raise ValueError("nonempty donor impressions with valid train positions and binary labels required")
    table(db, "global_rate", "SELECT sum(k)::DOUBLE/sum(n) AS global_ctr FROM position_rates")
    table(
        db,
        "category_rates",
        """
        SELECT category_id,avg(target) AS category_ctr FROM donors WHERE category_id>0 GROUP BY category_id
    """,
    )
    table(
        db,
        "user_affinity",
        """
        WITH stats AS (
            SELECT user_id,count(*) AS n,sum(target) AS k,sum(rate) AS expected
            FROM donors SEMI JOIN query_users USING(user_id) JOIN position_rates USING(pos) GROUP BY user_id
        )
        SELECT user_id,ln(1+n) AS impr_log,(k+5*global_ctr)/(n+5) AS ctr,
            coalesce((k+5*global_ctr)/nullif(expected+5*global_ctr,0),1) AS coe
        FROM stats,global_rate
    """,
    )
    table(
        db,
        "user_category_affinity",
        """
        WITH stats AS (
            SELECT user_id,category_id,count(*) n,sum(target) k FROM donors
            SEMI JOIN query_users USING(user_id) WHERE category_id>0 GROUP BY ALL
        )
        SELECT user_id,category_id,(k+5*category_ctr)/(n+5)-category_ctr AS lift,ln(1+k) AS clicks_log
        FROM stats JOIN category_rates USING(category_id)
    """,
    )


def join_features(db):
    expressions = [
        "coalesce(c.feed_log,0)",
        "coalesce(c.time_log,-1)",
        "coalesce(c.feeds_hour,0)",
        "coalesce(c.feeds_day,0)",
        "coalesce(c.session_idx,0)",
        "coalesce(i.prior_feeds,0)",
        "coalesce(ln(1+q.timestamp-i.prev_item_time),-1)",
        "coalesce(h.prior_category/nullif(c.prior_candidates,0),0)",
        "coalesce(r.recent_category/nullif(t.total,0),0)",
        "coalesce(a.impr_log,0)",
        "coalesce(a.ctr,g.global_ctr)",
        "coalesce(a.coe,1)",
        "coalesce(u.lift,0)",
        "coalesce(u.clicks_log,0)",
    ]
    fields = ", ".join(f"({expr})::FLOAT AS {name}" for expr, name in zip(expressions, NAMES, strict=True))
    table(
        db,
        "result",
        f"""
        SELECT q.shard,q.split,q.feed_id,q.candidate_index,q.item_id,{fields}
        FROM query_rows q LEFT JOIN clock_features c USING(user_id,timestamp)
        LEFT JOIN item_history i USING(user_id,item_id,timestamp)
        LEFT JOIN category_history h USING(user_id,category_id,timestamp)
        LEFT JOIN recent_total t USING(user_id,timestamp)
        LEFT JOIN recent_category r USING(user_id,category_id,timestamp)
        LEFT JOIN user_affinity a USING(user_id)
        LEFT JOIN user_category_affinity u USING(user_id,category_id)
        CROSS JOIN global_rate g
        ORDER BY q.split,q.shard,q.feed_id,q.candidate_index
    """,
    )


def build_features(
    prepared, output, *, train_parts=8, train_start=None, resume=False, memory_mb=1536, threads=2
):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    selected = selected_parts(manifest, train_parts, train_start)
    excluded = {part["source"] for part in selected}
    splits = manifest["splits"]
    evaluation = splits["val"]["parts"] + splits["test"]["parts"]
    training = splits["train"]["parts"]
    donors = [part for part in training if part["source"] not in excluded]
    if not donors or memory_mb < 128 or threads < 1:
        raise ValueError("user session requires donor shards, memory_mb>=128 and threads>=1")

    @contextmanager
    def prepare():
        with duckdb.connect(
            str(output / "_session.duckdb"),
            config={
                "memory_limit": f"{memory_mb}MB",
                "threads": str(threads),
                "temp_directory": str(output / "_spill"),
                "preserve_insertion_order": "false",
            },
        ) as db:
            for name, entries in [
                ("query_inputs", selected + evaluation),
                ("exposure_inputs", training + evaluation),
                ("donors", donors),
            ]:
                paths = [str(prepared / part["candidate_path"]) for part in entries]
                if name != "donors":
                    fields = "split,feed_id,candidate_index,item_id,category_id,user_id,timestamp,filename"
                else:
                    fields = "user_id,category_id,pos,target"
                db.from_parquet(paths, filename=name != "donors").project(fields).create_view(
                    name, replace=True
                )
            if not db.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name='result'"
            ).fetchone()[0]:
                create_exposures(db)
                create_affinity(db)
                join_features(db)
            yield db

    def part(split, entry, db):
        return db.execute(
            "SELECT * EXCLUDE(shard) FROM result WHERE split=? AND shard=? ORDER BY feed_id,candidate_index",
            [split, entry["source"]],
        ).pl()

    domains = dict.fromkeys(NAMES, (0, None))
    for name in ("us_time_since_prev_feed_log", "us_time_since_item_shown_log", "us_donor_user_cat_lift"):
        domains[name] = (-1, None)
    for name in ("us_cat_prior_impr_share", "us_cat_recent3_impr_share", "us_donor_user_ctr_smooth"):
        domains[name] = (0, 1)
    build_cache(
        prepared,
        output,
        train_parts=train_parts,
        train_start=train_start,
        group="user_session",
        names=NAMES,
        domains=domains,
        producer=__file__,
        extra={
            "target_policy": POLICY,
            "smoothing": SMOOTHING,
            "tie_order": "feed_id, split",
            "recent_feed_count": 3,
            "session_gap_seconds": 1800,
        },
        build_part=part,
        resume=resume,
        excluded=excluded,
        prepare=prepare,
    )
    with cache_lock(output):
        path = output / "manifest.json"
        result = json.loads(path.read_bytes())
        result["target_policy"] = POLICY
        write_json(path, result)
    return result
