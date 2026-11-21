"""Donor-only collaborative SVD embeddings and recency-weighted history features."""

import json
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.common.feature_cache import KEY, build_cache, cache_lock, selected_parts
from src.common.preparation import fingerprint, write_json
from src.common.similarity import unit

NAMES = [
    "svd_has_vec",
    "svd_cos_mean_10",
    "svd_cos_max_10",
    "svd_cos_profile_10",
    "svd_cos_mean_50",
    "svd_cos_max_50",
    "svd_cos_profile_50",
    "svd_evt2_cos_max_50",
]
ARRAYS = ("_items.npy", "_vectors.npy")
INDEX = "_svd_index.json"
POLICY = "Only click labels from non-cohort train donors enter the factorization; query labels and positions are unused."
DECAY = np.exp2(-np.arange(50, dtype=np.float32) / 10)


def graph_edges(prepared, donors, queries, *, limit, minimum_degree, memory_mb, threads, spill):
    with duckdb.connect(
        config={
            "memory_limit": f"{memory_mb}MB",
            "threads": str(threads),
            "temp_directory": str(spill),
            "preserve_insertion_order": "false",
        }
    ) as db:
        for name, parts, key in (
            ("donor_candidates", donors, "candidate_path"),
            ("donor_feeds", donors, "feed_path"),
            ("query_candidates", queries, "candidate_path"),
            ("query_feeds", queries, "feed_path"),
        ):
            db.from_parquet([str(prepared / part[key]) for part in parts]).create_view(name)
        db.execute("""
            CREATE TEMP TABLE clicks AS
            SELECT DISTINCT item_id,user_id FROM donor_candidates
            WHERE target=1 AND item_id>0 AND user_id>0
        """)
        db.execute("""
            CREATE TEMP TABLE query_items AS
            SELECT item_id FROM query_candidates WHERE item_id>0
            UNION SELECT item_id FROM (
                SELECT unnest(history__item_id[:50]) AS item_id FROM query_feeds
            ) WHERE item_id>0
        """)
        # Limit the extra vocabulary to frequent histories on donors.
        db.execute(
            """
            CREATE TEMP TABLE frequent_history AS
            SELECT item_id FROM (
                SELECT unnest(history__item_id[:50]) AS item_id FROM donor_feeds
            ) WHERE item_id>0
            GROUP BY item_id ORDER BY count(*) DESC,item_id LIMIT ?
        """,
            [max(1, limit // 5)],
        )
        db.execute(
            """
            CREATE TEMP TABLE retained AS
            WITH degrees AS (SELECT item_id,count(*) AS degree FROM clicks GROUP BY item_id)
            SELECT item_id FROM degrees
            SEMI JOIN (SELECT item_id FROM query_items UNION SELECT item_id FROM frequent_history) USING(item_id)
            WHERE degree>=? ORDER BY degree DESC,item_id LIMIT ?
        """,
            [minimum_degree, limit],
        )
        total_edges = db.execute("SELECT count(*) FROM clicks").fetchone()[0]
        edges = db.execute("""
            SELECT item_id,user_id FROM clicks SEMI JOIN retained USING(item_id) ORDER BY item_id,user_id
        """).fetchnumpy()
    return edges, total_edges


def shifted_ppmi(edges):
    items, i = np.unique(edges["item_id"], return_inverse=True)
    users, u = np.unique(edges["user_id"], return_inverse=True)
    item_degree, user_degree = np.bincount(i), np.bincount(u)
    weights = np.maximum(
        np.log(len(i) / (item_degree[i].astype(np.float64) * user_degree[u])) - np.log(2.0), 0
    ).astype(np.float32)
    graph = csr_matrix((weights, (i, u)), shape=(len(items), len(users)))
    graph.eliminate_zeros()
    return items.astype(np.int32), graph


def factorize(graph, rank):
    """Randomized SVD, using the small Gram matrix instead of a dense right factor."""
    rank = min(rank, *graph.shape)
    if rank == 0 or graph.nnz == 0:
        return np.empty((graph.shape[0], 0), dtype=np.float32), []
    width = min(rank + 12, *graph.shape)
    rng = np.random.default_rng(42)
    projection = graph @ rng.standard_normal((graph.shape[1], width), dtype=np.float32)
    for _ in range(2):
        q = np.linalg.qr(projection, mode="reduced")[0]
        projection = graph @ (graph.T @ q)
    q = np.linalg.qr(projection, mode="reduced")[0]
    right = graph.T @ q
    eigenvalues, eigenvectors = np.linalg.eigh(right.T @ right)
    indices = np.argsort(eigenvalues)[::-1][:rank]
    singular = np.sqrt(np.maximum(eigenvalues[indices], 0))
    keep = singular > singular[0] * 1e-6
    embedding = (q @ eigenvectors[:, indices[keep]]) * np.sqrt(singular[keep])
    return unit(embedding), singular[keep].tolist()


def save_array(path, values):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temp.replace(path)


def ensure_index(prepared, output, donors, queries, options, memory_mb, threads):
    marker = output / INDEX
    if marker.exists():
        saved = json.loads(marker.read_bytes())
        if all(
            (output / name).is_file() and fingerprint(output / name) == saved["files"][name]
            for name in ARRAYS
        ):
            return
    edges, total_edges = graph_edges(
        prepared,
        donors,
        queries,
        limit=options["max_items"],
        minimum_degree=options["min_degree"],
        memory_mb=memory_mb,
        threads=threads,
        spill=output / "_spill",
    )
    print(f"collab SVD retained {len(edges['item_id']):,} click edges", flush=True)
    items, graph = shifted_ppmi(edges)
    vectors, singular = factorize(graph, options["rank"])
    valid = (vectors * vectors).sum(axis=1) > 0.5
    save_array(output / ARRAYS[0], items[valid])
    save_array(output / ARRAYS[1], vectors[valid])
    write_json(
        marker,
        {
            "donor_distinct_click_edges": total_edges,
            "retained_click_edges": len(edges["item_id"]),
            "ppmi_nonzero": graph.nnz,
            "matrix_shape": list(graph.shape),
            "items_with_vectors": int(valid.sum()),
            "singular_values": singular,
            "files": {name: fingerprint(output / name) for name in ARRAYS},
        },
    )
    print(f"collab SVD published {int(valid.sum()):,} item vectors at rank {len(singular)}", flush=True)


def lookup(items, ids):
    position = np.searchsorted(items, ids)
    present = position < len(items)
    present[present] = items[position[present]] == ids[present]
    return position, present


def score_part(candidates, feeds, items, vectors):
    candidates = candidates.sort("feed_id", "candidate_index")
    values = np.zeros((len(candidates), len(NAMES)), dtype=np.float32)
    offsets, present = lookup(items, candidates["item_id"].to_numpy())
    values[:, 0] = present
    feed_ids = candidates["feed_id"].to_numpy()
    bounds = np.r_[0, np.flatnonzero(feed_ids[1:] != feed_ids[:-1]) + 1, len(feed_ids)]
    ordered = feeds.sort("feed_id")
    starts = bounds[:-1] if len(candidates) else bounds[:0]
    if len(starts) != len(ordered) or not np.array_equal(feed_ids[starts], ordered["feed_id"].to_numpy()):
        raise ValueError("candidate and history feed IDs differ")
    for start, stop, (history, events) in zip(
        starts,
        bounds[1:],
        ordered.select("history__item_id", "history__event_type").iter_rows(),
        strict=True,
    ):
        if len(history) != len(events):
            raise ValueError("history event types are not aligned")
        history, events = np.asarray(history[:50]), np.asarray(events[:50])
        positions, has_history = lookup(items, history)
        if not has_history.any() or not present[start:stop].any():
            continue
        rows = start + np.flatnonzero(present[start:stop])
        current = vectors[offsets[rows]]
        history_vectors = vectors[positions[has_history]]
        similarity = current @ history_vectors.T
        weights = DECAY[: len(history)][has_history]
        recent_positions = np.flatnonzero(has_history)
        for column, limit in ((1, 10), (4, 50)):
            mask = recent_positions < limit
            if not mask.any():
                continue
            weight = weights[mask]
            scores = similarity[:, mask]
            profile = unit(weight @ history_vectors[mask])
            values[rows, column] = (scores @ weight) / weight.sum()
            values[rows, column + 1] = (scores * weight).max(axis=1)
            values[rows, column + 2] = current @ profile
        event_two = events[has_history] == 2
        if event_two.any():
            values[rows, 7] = (similarity[:, event_two] * weights[event_two]).max(axis=1)
    return candidates.select(*KEY, "item_id").with_columns(
        [pl.Series(name, values[:, i]) for i, name in enumerate(NAMES)]
    )


def build_features(
    prepared,
    output,
    *,
    train_parts=8,
    train_start=None,
    resume=False,
    memory_mb=2048,
    threads=2,
    rank=48,
    max_items=500_000,
    min_degree=5,
):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    source = json.loads((prepared / "manifest.json").read_bytes())
    selected = selected_parts(source, train_parts, train_start)
    excluded = {part["source"] for part in selected}
    donors = [part for part in source["splits"]["train"]["parts"] if part["source"] not in excluded]
    if not donors or min(threads, rank, max_items, min_degree) < 1 or memory_mb < 128:
        raise ValueError("invalid SVD parameters or absent donor shards")
    queries = selected + source["splits"]["val"]["parts"] + source["splits"]["test"]["parts"]
    options = {
        "rank": rank,
        "max_items": max_items,
        "min_degree": min_degree,
        "seed": 42,
        "shift": 2,
        "power_iterations": 2,
        "half_life_positions": 10,
        "target_policy": POLICY,
    }

    @contextmanager
    def prepare():
        ensure_index(prepared, output, donors, queries, options, memory_mb, threads)
        yield tuple(np.load(output / name, mmap_mode="r", allow_pickle=False) for name in ARRAYS)

    def part(split, entry, arrays):
        candidates = pl.read_parquet(prepared / entry["candidate_path"], columns=[*KEY, "item_id"])
        feeds = pl.read_parquet(
            prepared / entry["feed_path"],
            columns=[
                "feed_id",
                "history__item_id",
                "history__event_type",
            ],
        )
        return score_part(candidates, feeds, *arrays)

    build_cache(
        prepared,
        output,
        train_parts=train_parts,
        train_start=train_start,
        group="collab_svd",
        names=NAMES,
        domains={**dict.fromkeys(NAMES, (-1, 1)), NAMES[0]: (0, 1)},
        producer=__file__,
        extra=options,
        build_part=part,
        resume=resume,
        excluded=excluded,
        prepare=prepare,
        artifacts=(*ARRAYS, INDEX),
    )
    with cache_lock(output):
        path = output / "manifest.json"
        result = json.loads(path.read_bytes())
        result["target_policy"] = POLICY
        write_json(path, result)
    return result
