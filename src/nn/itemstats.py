"""Attach donor item statistics in the original candidate order."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.common.itemstats import feature_names
from src.nn.dataset import collate_ranking_batch, find_parquet_parts


class ItemStatsCollator:
    def __init__(self, root, data, split, train_parts):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.features = feature_names()
        if not manifest.get("complete") or manifest.get("feature_group") != "itemstats":
            raise ValueError("Expected a complete itemstats cache")
        if manifest.get("features") != self.features:
            raise ValueError("Unexpected itemstats columns")
        all_train = find_parquet_parts(data, "train")
        fit = {path.name for path in all_train[:train_parts]}
        donors = set(manifest["identity"]["donor_sources"])
        if set(manifest["excluded_train_sources"]) != fit or donors != {p.name for p in all_train} - fit:
            raise ValueError("Itemstats donor shards do not match the training split")
        parts = manifest["splits"][split]["parts"]
        expected = fit if split == "train" else {p.name for p in find_parquet_parts(data, split)}
        if set(parts) != expected:
            raise ValueError(f"Itemstats shard coverage differs for {split}")

        self.rows = {}
        keys = ["feed_id", "candidate_index", "item_id"]
        for name, part in parts.items():
            table = pl.read_parquet(root / split / name, columns=keys + self.features).sort(keys[:2])
            if table.height != part["rows"]:
                raise ValueError(f"Wrong itemstats row count for {name}")
            feeds, starts = np.unique(table["feed_id"].to_numpy(), return_index=True)
            ends = np.r_[starts[1:], table.height]
            indices = table["candidate_index"].to_numpy()
            items = table["item_id"].to_numpy()
            values = np.log1p(table.select(self.features).to_numpy(order="c"))
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite itemstats for {name}")
            for feed, start, end in zip(feeds, starts, ends):
                if feed in self.rows or not np.array_equal(indices[start:end], np.arange(end - start)):
                    raise ValueError(f"Invalid candidate indices for feed {feed}")
                self.rows[feed] = (items[start:end], values[start:end])

    @property
    def width(self):
        return 2 * len(self.features)

    def __call__(self, samples):
        batch = collate_ranking_batch(samples)
        width = len(self.features)
        result = np.zeros((*batch["candidate_mask"].shape, self.width), dtype=np.float32)
        for index, sample in enumerate(samples):
            items, values = self.rows[sample["feed_id"]]
            if not np.array_equal(items, sample["cand_item_id"]):
                raise ValueError(f"Itemstats candidate order differs for feed {sample['feed_id']}")
            size = len(items)
            result[index, :size, :width] = values
            std = values.std(axis=0)
            result[index, :size, width:] = (values - values.mean(axis=0)) / (std + 1e-5)
        batch["itemstats"] = torch.from_numpy(result)
        return batch
