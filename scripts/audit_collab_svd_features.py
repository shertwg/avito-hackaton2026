"""Audit collaborative embeddings and independently recompute sampled feature rows."""

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.common.collab_svd import ARRAYS, INDEX, NAMES, POLICY
from src.common.feature_cache import KEY, selected_parts, validate_frame
from src.common.preparation import fingerprint, write_json


def check_index(cache, options, artifacts):
    for name in (*ARRAYS, INDEX):
        if fingerprint(cache / name) != artifacts[name]:
            raise ValueError(f"index artifact changed: {name}")
    items, vectors = (np.load(cache / name, mmap_mode="r", allow_pickle=False) for name in ARRAYS)
    marker = json.loads((cache / INDEX).read_bytes())
    if (
        items.dtype != np.int32
        or vectors.dtype != np.float32
        or items.shape != (len(vectors),)
        or vectors.shape != (marker["items_with_vectors"], len(marker["singular_values"]))
        or len(items) > options["max_items"]
        or vectors.shape[1] > options["rank"]
        or (len(items) and items[0] <= 0)
        or not np.all(items[1:] > items[:-1])
    ):
        raise ValueError("invalid item vectors or IDs")
    for start in range(0, len(items), 8192):
        block = vectors[start : start + 8192]
        np.testing.assert_allclose(np.linalg.norm(block, axis=1), 1.0, atol=2e-6, rtol=0)
    return items, vectors, marker


def donor_degree(prepared, donors, sampled):
    with duckdb.connect(config={"threads": "2", "memory_limit": "1024MB"}) as db:
        db.from_parquet([str(prepared / part["candidate_path"]) for part in donors]).create_view("donors")
        return dict(
            db.execute(
                """
            SELECT item_id,count(DISTINCT user_id) FROM donors
            WHERE target=1 AND user_id>0 AND item_id IN (SELECT unnest(?)) GROUP BY item_id
        """,
                [sampled],
            ).fetchall()
        )


def recompute(item, history, events, embeddings):
    vector = embeddings.get(item)
    if vector is None:
        return [0.0] * len(NAMES)
    records = [
        (i, event, 2.0 ** (-i / 10.0), embeddings[old])
        for i, (old, event) in enumerate(zip(history[:50], events[:50], strict=True))
        if old in embeddings
    ]
    result = [1.0]
    for limit in (10, 50):
        window = [(weight, target) for i, _, weight, target in records if i < limit]
        weighted = [weight * float(vector @ target) for weight, target in window]
        if not window:
            result.extend((0.0, 0.0, 0.0))
            continue
        profile = sum(weight * target for weight, target in window)
        norm = sqrt(float(profile @ profile))
        result.extend(
            (
                sum(weighted) / sum(weight for weight, _ in window),
                max(weighted),
                float(vector @ profile) / max(norm, 1e-12),
            )
        )
    result.append(
        max(
            (weight * float(vector @ target) for _, event, weight, target in records if event == 2),
            default=0.0,
        )
    )
    return result


def check_payloads(prepared, cache, partitions, meta, items, vectors):
    def small_lookup(ids):
        result = {}
        for item in set(ids):
            index = int(np.searchsorted(items, item))
            if index < len(items) and items[index] == item:
                result[item] = vectors[index].astype(np.float64)
        return result

    counts, sampled = {}, set()
    for split, entries in partitions.items():
        saved = meta["splits"][split]["parts"]
        if set(saved) != {part["source"] for part in entries}:
            raise ValueError(f"wrong {split} coverage")
        rows = has_vec = matched = histories_checked = 0
        for number, entry in enumerate(entries):
            record = saved[entry["source"]]
            path = cache / record["path"]
            if fingerprint(path) != record["fingerprint"]:
                raise ValueError(f"feature file changed: {path}")
            frame = pl.read_parquet(path)
            expected = pl.read_parquet(prepared / entry["candidate_path"], columns=[*KEY, "item_id"])
            validate_frame(frame, NAMES, entry["candidates"], meta["domains"])
            if not frame.select(*KEY, "item_id").equals(expected.sort("feed_id", "candidate_index")):
                raise ValueError("candidate occurrence keys differ")
            ids = frame["item_id"].to_numpy()
            positions = np.searchsorted(items, ids)
            valid = positions < len(items)
            valid[valid] = items[positions[valid]] == ids[valid]
            if not np.array_equal(frame["svd_has_vec"].to_numpy(), valid):
                raise ValueError("candidate vector availability differs from index")
            rows += frame.height
            has_vec += int(valid.sum())
            matched += int((frame["svd_cos_max_50"] != 0).sum())
            if number not in {0, len(entries) // 2, len(entries) - 1} or not rows or not frame.height:
                continue
            history = pl.read_parquet(
                prepared / entry["feed_path"], columns=["feed_id", "history__item_id", "history__event_type"]
            )
            probes = {
                0,
                frame.height - 1,
                frame["svd_cos_max_50"].arg_max(),
                frame["svd_evt2_cos_max_50"].arg_max(),
            }
            for idx in sorted(probes):
                candidate = frame.row(idx, named=True)
                _, past, events = history.filter(pl.col("feed_id") == candidate["feed_id"]).row(0)
                item = candidate["item_id"]
                embeddings = small_lookup([item, *past[:50]])
                sampled.update(embeddings)
                np.testing.assert_allclose(
                    [candidate[n] for n in NAMES],
                    recompute(item, past, events, embeddings),
                    rtol=2e-5,
                    atol=2e-6,
                    err_msg=f"{split} feed {candidate['feed_id']}",
                )
                histories_checked += 1
        counts[split] = {
            "rows": rows,
            "vector_rows": has_vec,
            "nonzero_max50_rows": matched,
            "scalar_candidates": histories_checked,
        }
    return counts, sorted(sampled)


def audit(prepared, cache, output):
    prepared, cache = Path(prepared), Path(cache)
    source_bytes = (prepared / "manifest.json").read_bytes()
    cache_bytes = (cache / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    meta = json.loads(cache_bytes)
    identity = meta["identity"]
    if (
        not meta["complete"]
        or meta["features"] != NAMES
        or meta["target_policy"] != POLICY
        or identity["prepared_manifest_sha256"] != hashlib.sha256(source_bytes).hexdigest()
    ):
        raise ValueError("incomplete or incompatible collaborative SVD cache")
    excluded = meta["excluded_train_sources"]
    chosen = selected_parts(source, len(excluded), identity.get("train_start"))
    if (
        sorted(part["source"] for part in chosen) != excluded
        or identity["excluded_train_sources"] != excluded
    ):
        raise ValueError("donor exclusions differ")
    for rel, old in identity["input_fingerprints"].items():
        if fingerprint(prepared / rel) != old:
            raise ValueError(f"prepared input changed: {rel}")
    splits = source["splits"]
    donors = [part for part in splits["train"]["parts"] if part["source"] not in excluded]
    options = identity["extra"]
    items, vectors, graph = check_index(cache, options, meta["artifacts"])
    partitions = {"train": chosen, "val": splits["val"]["parts"], "test": splits["test"]["parts"]}
    counts, sampled = check_payloads(prepared, cache, partitions, meta, items, vectors)
    degrees = donor_degree(prepared, donors, sampled) if sampled else {}
    if set(degrees) != set(sampled) or any(value < options["min_degree"] for value in degrees.values()):
        raise ValueError("sampled vector lacks sufficient clicks on excluded-cohort donors")
    if (cache / "manifest.json").read_bytes() != cache_bytes:
        raise ValueError("cache manifest changed during audit")
    result = {
        "complete": True,
        "manifest_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "splits": counts,
        "features": len(NAMES),
        "donor_shards": len(donors),
        "excluded_train_sources": excluded,
        "sampled_vectors_with_verified_donor_degree": len(degrees),
        "graph": graph,
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
