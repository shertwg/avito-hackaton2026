"""Directed donor click lift, weighted co-click similarity and category backoffs."""

from contextlib import contextmanager
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy import sparse

from src.common.feature_cache import build_cache, selected_parts
from src.common.preparation import write_json

NAMES = [
    "ci2i_lift_mean_10",
    "ci2i_lift_max_10",
    "ci2i_lift_possum_10",
    "ci2i_ctr_max_10",
    "ci2i_matched_frac_10",
    "ci2i_lift_mean_50",
    "ci2i_lift_max_50",
    "ci2i_evt2_lift_max_50",
    "ci2i_evt10_lift_max_50",
    "ci2i_matched_frac_50",
    "coclick_mean_10",
    "coclick_max_10",
    "coclick_mean_50",
    "coclick_max_50",
    "cat_ci2i_lift_mean_10",
    "cat_ci2i_lift_max_10",
    "cat_ci2i_src_lift_mean_50",
    "cat_item_lift_max_50",
]
KEY = ["split", "feed_id", "candidate_index", "item_id"]
BUCKETS = 128


def log_lift(clicks, count, base_rate):
    return f"ln((({clicks}+12*{base_rate})/({count}+12)+1e-6)/({base_rate}+1e-6))"


def co_index(db):
    """CSR item x user. The inner product gives IDF-weighted, cosine-normalized intersections."""
    edges = db.sql(
        "SELECT DISTINCT item_id,user_id FROM dc WHERE target=1 AND item_id>0 AND user_id>0"
    ).fetchnumpy()
    items, i = np.unique(edges["item_id"], return_inverse=True)
    users, u = np.unique(edges["user_id"], return_inverse=True)
    degrees = np.bincount(i)
    user_degrees = np.bincount(u)
    weights = 1 / np.sqrt(degrees[i] * np.log2(2.0 + user_degrees[u]))
    matrix = sparse.csr_matrix((weights, (i, u)), shape=(len(items), len(users)))
    return items, matrix


def co_pairs(db, items, matrix, bucket):
    selected = np.flatnonzero(items % BUCKETS == bucket)
    db.execute("CREATE OR REPLACE TEMP TABLE co(h INTEGER,c INTEGER,sim DOUBLE)")
    for start in range(0, len(selected), 2048):
        indices = selected[start : start + 2048]
        product = (matrix[indices] @ matrix.T.tocsc()).tocoo()
        scores = pl.DataFrame(
            {"h": items[product.col], "c": items[indices[product.row]], "sim": product.data}
        )
        db.register("scores", scores.to_arrow())
        db.execute("INSERT INTO co SELECT * FROM scores")
        db.unregister("scores")


def table(db, name, query):
    db.execute(f"CREATE TABLE IF NOT EXISTS {name} AS {query}")


def histories(db, prefix, entries, prepared):
    for name, field in [("c", "candidate_path"), ("f", "feed_path")]:
        db.from_parquet([str(prepared / p[field]) for p in entries], filename=True).create_view(
            prefix + name, replace=True
        )


def initialize(db):
    # Group donor categories within each feed before multiplying by history events.
    table(db, "global_rate", "SELECT coalesce(avg(target),0.) AS g FROM dc")
    table(
        db,
        "cat_rate",
        "SELECT category_id cat,(sum(target)+20*g)/(count(*)+20) base_rate FROM dc,global_rate GROUP BY cat,g",
    )
    table(db, "qitems", "SELECT DISTINCT item_id c FROM qc WHERE item_id>0")
    table(
        db,
        "dselected",
        f"""SELECT item_id c,feed_id,target, item_id%{BUCKETS} bucket
          FROM dc SEMI JOIN qitems ON item_id=c ORDER BY bucket""",
    )
    table(
        db,
        "item_rate",
        "SELECT c,(sum(target)+20*g)/(count(*)+20) base_rate FROM dselected,global_rate GROUP BY c,g",
    )
    table(
        db,
        "hist",
        """SELECT feed_id,history__item_id[:50] hs,history__category_id[:50] cats,
                     history__event_type[:50] ev,history__event_source[:50] src FROM df""",
    )
    table(
        db,
        "hcat",
        """SELECT feed_id,hcat,count(*) mult FROM
          (SELECT feed_id,unnest(cats) hcat FROM hist) WHERE hcat>0 GROUP BY ALL""",
    )
    table(
        db,
        "dcats",
        "SELECT feed_id,category_id cat,count(*) n,sum(target)::BIGINT k FROM dc WHERE category_id>0 GROUP BY ALL",
    )
    for channel in ("ev", "src"):
        table(
            db,
            "back_" + channel,
            f"""WITH events AS (SELECT feed_id,unnest(cats) hcat,unnest({channel}) code FROM hist),
             groups AS (SELECT feed_id,hcat,code,count(*) mult FROM events WHERE hcat>0 AND code>0 GROUP BY ALL),
             cnt AS (SELECT hcat,code,cat,sum(n*mult) n,sum(k*mult) k FROM groups JOIN dcats USING(feed_id) GROUP BY ALL)
             SELECT hcat,code,cat,{log_lift("k", "n", "base_rate")} lift FROM cnt JOIN cat_rate USING(cat)""",
        )
    table(
        db,
        "queries",
        f"""SELECT split,feed_id,candidate_index,item_id c,category_id cat,
          regexp_extract(replace(filename,chr(92),'/'),'[^/]+$') AS shard,item_id%{BUCKETS} bucket
          FROM qc ORDER BY bucket""",
    )
    table(
        db,
        "qhist",
        """SELECT feed_id,generate_subscripts(history__item_id[:50],1) pos,
          unnest(history__item_id[:50]) h,unnest(history__category_id[:50]) hcat,
          unnest(history__event_type[:50]) ev,unnest(history__event_source[:50]) src FROM qf""",
    )


def score_bucket(db, bucket, items, matrix):
    co_pairs(db, items, matrix, bucket)
    db.execute(f"""CREATE OR REPLACE TEMP TABLE events AS SELECT q.*,pos,h,hcat,ev,src
           FROM (SELECT * FROM queries WHERE bucket={bucket}) q LEFT JOIN qhist USING(feed_id)""")
    db.execute("CREATE OR REPLACE TEMP TABLE wanted AS SELECT DISTINCT c,h FROM events WHERE c>0 AND h>0")
    db.execute(f"""CREATE OR REPLACE TEMP TABLE directs AS WITH expanded AS
           (SELECT c,target,unnest(hs) h FROM (SELECT * FROM dselected WHERE bucket={bucket}) d JOIN hist USING(feed_id)),
           cnt AS (SELECT h,c,count(*) n,sum(target) k FROM expanded SEMI JOIN wanted USING(c,h)
                   GROUP BY h,c HAVING n>=3 OR k>=1)
           SELECT h,c,(k+12*base_rate)/(n+12) ctr,{log_lift("k", "n", "base_rate")} lift FROM cnt JOIN item_rate USING(c)""")
    db.execute(f"""CREATE OR REPLACE TEMP TABLE cat_item AS WITH wanted_cat AS
           (SELECT DISTINCT c,hcat FROM events WHERE c>0 AND hcat>0), cnt AS
           (SELECT c,hcat,sum(mult) n,sum(target*mult) k FROM (SELECT * FROM dselected WHERE bucket={bucket}) d
            JOIN hcat USING(feed_id) SEMI JOIN wanted_cat USING(c,hcat) GROUP BY c,hcat HAVING n>=10)
           SELECT c,hcat,{log_lift("k", "n", "base_rate")} lift FROM cnt JOIN item_rate USING(c)""")
    values = []

    def within(op, col, win, condition="true"):
        return f"coalesce({op}({col}) FILTER (WHERE pos<={win} AND {condition}),0)"

    for win in (10, 50):
        values.extend([within("avg", "l", win), within("max", "l", win)])
        if win == 10:
            values.extend([within("sum", "greatest(l,0)", win), within("max", "ctr", win)])
        else:
            values.extend([within("max", "l", win, "ev=2"), within("max", "l", win, "ev=10")])
        values.append(within("avg", "has_pair", win))
    for win in (10, 50):
        values.extend([within("avg", "similarity", win), within("max", "similarity", win)])
    values.extend(
        [
            within("avg", "event_lift", 10),
            within("max", "event_lift", 10),
            within("avg", "source_lift", 50),
            within("max", "item_lift", 50),
        ]
    )
    columns = ", ".join(f"({v})::FLOAT AS {n}" for v, n in zip(values, NAMES, strict=True))
    table(
        db,
        f"block_{bucket}",
        f"""WITH signals AS (SELECT e.*,coalesce(d.lift,0) l,
          (d.ctr IS NOT NULL)::INTEGER has_pair,coalesce(d.ctr,0) ctr,
          coalesce(co.sim,0)*pow(2.,-(e.pos-1)/10.) similarity,coalesce(be.lift,0) event_lift,
          coalesce(bs.lift,0) source_lift,coalesce(ci.lift,0) item_lift
          FROM events e LEFT JOIN directs d USING(h,c) LEFT JOIN co USING(h,c)
          LEFT JOIN back_ev be ON (be.hcat,be.code,be.cat)=(e.hcat,e.ev,e.cat)
          LEFT JOIN back_src bs ON (bs.hcat,bs.code,bs.cat)=(e.hcat,e.src,e.cat)
          LEFT JOIN cat_item ci ON (ci.c,ci.hcat)=(e.c,e.hcat))
          SELECT shard,split,feed_id,candidate_index,c AS item_id,{columns} FROM signals
          GROUP BY shard,split,feed_id,candidate_index,c ORDER BY split,shard""",
    )


def build_features(
    prepared, output, *, train_parts=8, train_start=None, resume=False, memory_mb=1536, threads=2
):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    m = json.loads((prepared / "manifest.json").read_bytes())
    chosen = selected_parts(m, train_parts, train_start)
    excluded = {p["source"] for p in chosen}
    donors = [p for p in m["splits"]["train"]["parts"] if p["source"] not in excluded]
    if not donors:
        raise ValueError("click graph requires donor shards")

    @contextmanager
    def prepare():
        with duckdb.connect(
            str(output / "_graph.duckdb"),
            config={
                "memory_limit": f"{memory_mb}MB",
                "threads": str(threads),
                "temp_directory": str(output / "_spill"),
                "preserve_insertion_order": "false",
            },
        ) as db:
            histories(db, "d", donors, prepared)
            histories(db, "q", chosen + m["splits"]["val"]["parts"] + m["splits"]["test"]["parts"], prepared)
            initialize(db)
            saved = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
            missing = [i for i in range(BUCKETS) if f"block_{i}" not in saved]
            if missing:
                items, matrix = co_index(db)
                for i in missing:
                    score_bucket(db, i, items, matrix)
                    print(f"click graph bucket {i + 1}/{BUCKETS}", flush=True)
            union = " UNION ALL ".join(f"SELECT * FROM block_{i}" for i in range(BUCKETS))
            table(db, "result", f"SELECT * FROM ({union}) ORDER BY split,shard,feed_id,candidate_index")
            yield db

    def part(split, entry, db):
        return db.execute(
            "SELECT * EXCLUDE(shard) FROM result WHERE split=? AND shard=? ORDER BY feed_id,candidate_index",
            [split, entry["source"]],
        ).pl()

    domains = {n: (None, None) for n in NAMES}
    for n in NAMES:
        if n.startswith("coclick") or "frac" in n or "ctr_max" in n:
            domains[n] = (0, 1)
    result = build_cache(
        prepared,
        output,
        train_parts=train_parts,
        train_start=train_start,
        group="click_item2item",
        names=NAMES,
        domains=domains,
        producer=__file__,
        extra={"history_window": 50, "prior_strength": 20, "conditional_strength": 12},
        build_part=part,
        resume=resume,
        excluded=excluded,
        prepare=prepare,
    )
    result["target_policy"] = (
        "train donor labels only; all selected training shards excluded; query labels never used"
    )
    write_json(output / "manifest.json", result)
    return result
