"""Evaluate fixed-schedule train + val adaptation with four temporal folds."""
from __future__ import annotations

import argparse
from itertools import chain
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import mean_ndcg
from src.nn.dataset import (
    FeedRankingDataset, ShardedFeedRankingDataset, collate_ranking_batch,
)
from src.nn.model import FeedRankingModel
from src.nn.predict import predict_single_model
from src.nn.train import listnet_loss, seed_everything


def batches(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_ranking_batch)


def train_epoch(model, optimizer, loader, device):
    start = perf_counter()
    model.train()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        scores, _ = model(batch, use_pos_bias=True)
        loss = listnet_loss(scores, batch["target"], batch["candidate_mask"], target_temperature=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    stats = {"seconds": perf_counter() - start, "steps": len(losses), "loss": float(np.mean(losses))}
    if not np.isfinite(stats["loss"]):
        raise RuntimeError("Non-finite training loss")
    print(stats, flush=True)
    return stats


def score_checkpoint(path, dataset, batch_size, device):
    feed, item, score, _ = predict_single_model(path, batches(dataset, batch_size), device)
    return pl.DataFrame({"feed_id": feed, "item_id": item, "score": score}).cast(
        {"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}
    )


def evaluate(data, predictions):
    keys = ["feed_id", "item_id"]
    truth = load_labels(data, "val").cast(dict.fromkeys(keys, pl.Int32))
    if truth.height != predictions.height:
        raise ValueError("Predictions must cover all validation candidates")
    joined = truth.join(predictions, on=keys, how="left", validate="1:1")
    scores = joined["score"]
    if scores.null_count() or not scores.is_finite().all():
        raise ValueError("Missing or non-finite validation predictions")
    feed, target, score = (joined[c].to_numpy() for c in ("feed_id", "target", "score"))
    public, private = pseudo_split(feed, joined["timestamp"].to_numpy())
    result = {}
    for name, feeds in (("public", public), ("private", private)):
        mask = mask_feeds(feed, feeds)
        result[name] = mean_ndcg(feed[mask], target[mask], score[mask], cutoff=30)
    return result


def run(args):
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    config = {"use_numeric_head": True}
    model = FeedRankingModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = ShardedFeedRankingDataset(args.data, "train", max_parts=args.train_parts, seed=args.seed)
    timing = {"base": [], "adaptation": {}}

    # The base model fits train only, on a schedule fixed before scoring validation.
    for epoch in range(1, args.base_epochs + 1):
        train.set_epoch(epoch)
        print(f"Base epoch {epoch}", flush=True)
        timing["base"].append(train_epoch(model, optimizer, batches(train, args.batch_size), device))
    base_path = out / "base.pt"
    torch.save({"model_config": config, "model_state": model.state_dict(), "epoch": args.base_epochs}, base_path)

    val = FeedRankingDataset(args.data, "val")
    parts = [pl.read_parquet(path, columns=["feed_id", "timestamp"]) for path in val.parts]
    rows = pl.concat(parts)
    row_feed = rows["feed_id"].to_numpy()
    if len(np.unique(row_feed)) != len(val):
        raise ValueError("Expected one validation row per feed")
    feeds, folds = time_folds(row_feed, rows["timestamp"].to_numpy(), 4)
    row_fold = folds[np.searchsorted(feeds, row_feed)]
    if len(np.unique(row_fold)) != 4:
        raise ValueError("Validation must contain four nonempty folds")

    base = torch.load(base_path, map_location="cpu")
    predictions = []
    for fold in range(5):
        final = fold == 4
        name = "final" if final else f"fold{fold}"
        kept = np.flatnonzero(row_fold != fold)
        held = np.flatnonzero(row_fold == fold)
        fit_val = Subset(val, kept.tolist())
        seed_everything(args.seed)
        model.load_state_dict(base["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"{name}: train={len(train)}, fit val={len(fit_val)}, held val={len(held)}", flush=True)
        train.set_epoch(args.base_epochs + 1)
        loader = chain(batches(train, args.batch_size), batches(fit_val, args.batch_size))
        timing["adaptation"][name] = train_epoch(model, optimizer, loader, device)
        path = out / f"{name}.pt"
        torch.save({"model_config": config, "model_state": model.state_dict(), "epoch": args.base_epochs + 1}, path)
        if not final:
            scored = score_checkpoint(path, Subset(val, held.tolist()), args.batch_size, device)
            if not np.array_equal(np.sort(scored["feed_id"].unique().to_numpy()), np.sort(row_feed[held])):
                raise ValueError(f"Wrong held-out feeds in {name}")
            predictions.append(scored)

    oof = pl.concat(predictions)
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "train_feeds": len(train), "args": vars(args), "timing": timing,
        "base": evaluate(args.data, score_checkpoint(base_path, val, args.batch_size, device)),
        "oof": evaluate(args.data, oof),
    }
    oof.write_parquet(out / "val_pred.parquet")
    rows.with_columns(pl.Series("fold", row_fold)).write_parquet(out / "folds.parquet")
    test = score_checkpoint(out / "final.pt", FeedRankingDataset(args.data, "test"), args.batch_size, device)
    test.write_parquet(out / "test_pred.parquet")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="competition")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-parts", type=int, default=2)
    parser.add_argument("--base-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.train_parts, args.base_epochs, args.batch_size) < 1 or not 0 < args.lr < float("inf"):
        parser.error("Expected positive training parameters and finite learning rate")
    run(args)


if __name__ == "__main__":
    main()
