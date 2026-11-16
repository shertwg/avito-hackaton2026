"""Evaluate fixed-schedule train + val adaptation with four temporal folds."""
from __future__ import annotations

import argparse
from itertools import chain, islice
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
from src.nn.donor_features import DonorFeatureCollator
from src.nn.model import FeedRankingModel
from src.nn.predict import predict_single_model
from src.nn.ranking_losses import boost_hard_negative_logits, lambda_ndcg30_loss
from src.nn.train import listnet_loss, seed_everything


def batches(dataset, batch_size, collate=collate_ranking_batch):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def adaptation_listnet(scores, target, mask):
    return listnet_loss(scores, target, mask, target_temperature=0.5)


def train_epoch(model, optimizer, loader, device, loss_fn=adaptation_listnet, reference_model=None):
    start = perf_counter()
    model.train()
    losses = []
    feed_visits = 0
    for batch in loader:
        feed_visits += batch["feed_id"].numel()
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        scores, _ = model(batch, use_pos_bias=True)
        target, mask = batch["target"], batch["candidate_mask"]
        if reference_model is not None:
            with torch.no_grad():
                _, reference_scores = reference_model(batch, use_pos_bias=False)
            scores = boost_hard_negative_logits(scores, target, mask, reference_scores)
        loss = loss_fn(scores, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    stats = {"seconds": perf_counter() - start, "steps": len(losses), "feed_visits": feed_visits, "loss": float(np.mean(losses))}
    if not np.isfinite(stats["loss"]):
        raise RuntimeError("Non-finite training loss")
    print(stats, flush=True)
    return stats


def score_checkpoint(path, dataset, batch_size, device, collate=collate_ranking_batch):
    feed, item, score, _ = predict_single_model(path, batches(dataset, batch_size, collate), device)
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
    for option in ("event_cosine", "history_match"):
        if getattr(args, option):
            config[f"use_{option}"] = True
    collators = dict.fromkeys(("train", "val", "test"), collate_ranking_batch)
    for group in ("itemstats", "item2item", "time"):
        root = getattr(args, group)
        if root:
            collators = {
                split: DonorFeatureCollator(root, args.data, split, args.train_parts, group, collate)
                for split, collate in collators.items()
            }
            config[f"num_{group}"] = collators["train"].width
    model = FeedRankingModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = ShardedFeedRankingDataset(args.data, "train", max_parts=args.train_parts, seed=args.seed)
    timing = {"base": [], "adaptation": {}}

    base_path = out / "base.pt"
    if args.base_run:
        previous = Path(args.base_run)
        existing = json.loads((previous / "report.json").read_text(encoding="utf-8"))
        keys = ("data", "seed", "train_parts", "base_epochs", "base_steps", "base_feeds_per_shard", "batch_size", "itemstats",
                "item2item", "time", "event_cosine", "history_match")
        for key in keys:
            default = False if key in ("event_cosine", "history_match") else None
            if existing["args"].get(key, default) != getattr(args, key):
                raise ValueError(f"Base run has different {key}")
        base = torch.load(previous / "base.pt", map_location="cpu")
        if base["model_config"] != config or not 1 <= base["epoch"] <= args.base_epochs:
            raise ValueError("Base checkpoint configuration does not match")
        torch.save(base, base_path)
    else:
        # Fit on train only, on a schedule fixed before scoring validation.
        base_train = train
        if args.base_feeds_per_shard:
            base_train = ShardedFeedRankingDataset(
                args.data, "train", max_parts=args.train_parts, seed=args.seed,
                max_feeds_per_part=args.base_feeds_per_shard,
            )
        remaining = args.base_steps
        for epoch in range(1, args.base_epochs + 1):
            base_train.set_epoch(epoch)
            print(f"Base epoch {epoch}", flush=True)
            loader = batches(base_train, args.batch_size, collators["train"])
            stats = train_epoch(model, optimizer, loader if remaining is None else islice(loader, remaining), device)
            timing["base"].append(stats)
            if remaining is not None:
                remaining -= stats["steps"]
                if remaining == 0:
                    break
        if remaining:
            raise ValueError("Base step budget exceeds the available epochs")
        torch.save({"model_config": config, "model_state": model.state_dict(), "epoch": epoch}, base_path)

    val = FeedRankingDataset(args.data, "val")
    parts = [
        pl.scan_parquet(path).select(
            "feed_id", "timestamp", pl.col("item_id").list.len().alias("candidate_count")
        ).collect()
        for path in val.parts
    ]
    rows = pl.concat(parts)
    eligible = rows["candidate_count"].to_numpy() >= args.adapt_min_candidates
    rows = rows.drop("candidate_count")
    row_feed = rows["feed_id"].to_numpy()
    if len(np.unique(row_feed)) != len(val):
        raise ValueError("Expected one validation row per feed")
    feeds, folds = time_folds(row_feed, rows["timestamp"].to_numpy(), 4)
    row_fold = folds[np.searchsorted(feeds, row_feed)]
    if len(np.unique(row_fold)) != 4:
        raise ValueError("Validation must contain four nonempty folds")

    base = torch.load(base_path, map_location="cpu")
    adaptation_epoch = base["epoch"] + 1
    predictions = []
    loss_fn = lambda_ndcg30_loss if args.adapt_loss == "lambda30" else adaptation_listnet
    reference_model = None
    if args.adapt_loss == "hard-listnet":
        reference_model = FeedRankingModel(**config).to(device)
        reference_model.load_state_dict(base["model_state"])
        reference_model.eval().requires_grad_(False)
    for fold in range(5):
        final = fold == 4
        name = "final" if final else f"fold{fold}"
        kept = np.flatnonzero((row_fold != fold) & eligible)
        if not len(kept) and args.adapt_val_only:
            raise ValueError(f"No eligible adaptation feeds in {name}")
        held = np.flatnonzero(row_fold == fold)
        fit_val = Subset(val, kept.tolist())
        seed_everything(args.seed)
        model.load_state_dict(base["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"{name}: train={0 if args.adapt_val_only else len(train)}, fit val={len(fit_val)}, held val={len(held)}", flush=True)
        train.set_epoch(adaptation_epoch)
        val_loader = batches(fit_val, args.batch_size, collators["val"])
        loader = val_loader if args.adapt_val_only else chain(
            batches(train, args.batch_size, collators["train"]), val_loader,
        )
        timing["adaptation"][name] = train_epoch(
            model, optimizer, loader, device, loss_fn, reference_model,
        )
        path = out / f"{name}.pt"
        torch.save({"model_config": config, "model_state": model.state_dict(), "epoch": adaptation_epoch}, path)
        if not final:
            scored = score_checkpoint(path, Subset(val, held.tolist()), args.batch_size, device, collators["val"])
            if not np.array_equal(np.sort(scored["feed_id"].unique().to_numpy()), np.sort(row_feed[held])):
                raise ValueError(f"Wrong held-out feeds in {name}")
            predictions.append(scored)

    oof = pl.concat(predictions)
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "train_feeds": len(train), "args": vars(args), "timing": timing,
        "base": evaluate(args.data, score_checkpoint(base_path, val, args.batch_size, device, collators["val"])),
        "oof": evaluate(args.data, oof),
    }
    oof.write_parquet(out / "val_pred.parquet")
    rows.with_columns(pl.Series("fold", row_fold)).write_parquet(out / "folds.parquet")
    test = score_checkpoint(out / "final.pt", FeedRankingDataset(args.data, "test"), args.batch_size, device, collators["test"])
    test.write_parquet(out / "test_pred.parquet")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="competition")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-parts", type=int, default=2)
    parser.add_argument("--base-epochs", type=int, default=3)
    parser.add_argument("--base-steps", type=int, help="Cap base training at this many optimizer steps")
    parser.add_argument("--base-feeds-per-shard", type=int, help="Sample this many feeds per shard in each base epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--itemstats", help="Path to an itemstats cache built for these train shards")
    parser.add_argument("--item2item", help="Path to an item2item cache built for these train shards")
    parser.add_argument("--time", help="Path to a donor time feature cache")
    parser.add_argument("--event-cosine", action="store_true", help="Add event-type cosine features over the latest 50 history events")
    parser.add_argument("--history-match", action="store_true", help="Learn candidate-to-event similarity over the latest 20 history events")
    parser.add_argument("--base-run", help="Reuse the train-only base checkpoint from a completed run")
    parser.add_argument("--adapt-val-only", action="store_true", help="Adapt only on the fit portion of val")
    parser.add_argument("--adapt-min-candidates", type=int, default=1, help="Minimum val feed size for adaptation")
    parser.add_argument("--adapt-loss", choices=("listnet", "lambda30", "hard-listnet"), default="listnet")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.train_parts, args.base_epochs, args.batch_size, args.adapt_min_candidates) < 1 or not 0 < args.lr < float("inf"):
        parser.error("Expected positive training parameters and finite learning rate")
    for value in (args.base_steps, args.base_feeds_per_shard):
        if value is not None and value < 1:
            parser.error("Base step and per-shard feed limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
