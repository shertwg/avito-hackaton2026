"""Attach donor features in the original candidate order."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.common import item2item, itemstats, time_features, weighted_item2item
from src.nn.dataset import collate_ranking_batch, find_parquet_parts


class DonorFeatureCollator:
    def __init__(self, root, data, split, train_parts, group, collate=collate_ranking_batch):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.group = "item2item" if group == "weighted_item2item" else group
        self.collate = collate
        self.features = {
            "itemstats": itemstats,
            "item2item": item2item,
            "time": time_features,
            "weighted_item2item": weighted_item2item,
        }[group].feature_names()
        if not manifest.get("complete") or manifest.get("feature_group") != group:
            raise ValueError("Expected a complete donor cache")
        if manifest.get("features") != self.features:
            raise ValueError("Unexpected donor columns")
        all_train = find_parquet_parts(data, "train")
        fit = {path.name for path in all_train[:train_parts]}
        identity = manifest["identity"]
        if group == "weighted_item2item":
            parent = identity["extra"]["parent_identity"]
            if set(parent["excluded_train_sources"]) != set(manifest["excluded_train_sources"]):
                raise ValueError("Weighted item2item parent cohort differs from the cache")
            identity = parent
        donors = set(identity.get("extra", identity)["donor_sources"])
        excluded = set(manifest["excluded_train_sources"])
        train_sources = {path.name for path in all_train}
        if not fit <= excluded <= train_sources or not donors or donors != train_sources - excluded:
            raise ValueError("Feature donor shards do not match the training split")
        parts = manifest["splits"][split]["parts"]
        expected = excluded if split == "train" else {p.name for p in find_parquet_parts(data, split)}
        if set(parts) != expected:
            raise ValueError(f"Feature shard coverage differs for {split}")

        self.raw_width = 6 if group == "time" else len(self.features)
        self.rows = {}
        keys = ["feed_id", "candidate_index", "item_id"]
        for name, part in parts.items():
            if split == "train" and name not in fit:
                continue
            table = pl.read_parquet(root / split / name, columns=keys + self.features).sort(keys[:2])
            if table.height != part["rows"]:
                raise ValueError(f"Wrong feature row count for {name}")
            feeds, starts = np.unique(table["feed_id"].to_numpy(), return_index=True)
            ends = np.r_[starts[1:], table.height]
            indices = table["candidate_index"].to_numpy()
            items = table["item_id"].to_numpy()
            raw = table.select(self.features).to_numpy(order="c")
            if group == "time":
                hour, weekday, age, seen = raw.T
                hour_phase = hour * (2 * np.pi / 24)
                week_phase = weekday * (2 * np.pi / 7)
                values = np.column_stack((
                    np.sin(hour_phase), np.cos(hour_phase),
                    np.sin(week_phase), np.cos(week_phase), np.log1p(age), seen,
                ))
            else:
                values = np.log1p(raw)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features for {name}")
            for feed, start, end in zip(feeds, starts, ends):
                if feed in self.rows or not np.array_equal(indices[start:end], np.arange(end - start)):
                    raise ValueError(f"Invalid candidate indices for feed {feed}")
                self.rows[feed] = (items[start:end], values[start:end])

    @property
    def width(self):
        return 2 * self.raw_width

    def __call__(self, samples):
        batch = self.collate(samples)
        width = self.raw_width
        result = np.zeros((*batch["candidate_mask"].shape, self.width), dtype=np.float32)
        for index, sample in enumerate(samples):
            items, values = self.rows[sample["feed_id"]]
            if not np.array_equal(items, sample["cand_item_id"]):
                raise ValueError(f"Feature candidate order differs for feed {sample['feed_id']}")
            size = len(items)
            result[index, :size, :width] = values
            std = values.std(axis=0)
            result[index, :size, width:] = (values - values.mean(axis=0)) / (std + 1e-5)
        batch[self.group] = torch.from_numpy(result)
        return batch
