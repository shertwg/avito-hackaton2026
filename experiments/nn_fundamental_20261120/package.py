"""Export the selected two-seed NN with four-fold test predictions."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import polars as pl
import torch
from torch.utils.data import Subset

from scripts.validate_submission import validate
from src.common.scores import feed_zscore
from src.nn.dataset import FeedRankingDataset, collate_ranking_batch
from src.nn.donor_features import DonorFeatureCollator
from src.nn.train_cv import score_checkpoint
from .compare import KEY, RUNS, OUT, paths, attach


def standardize(frame):
    return feed_zscore(frame["feed_id"].to_numpy(),frame["score"].to_numpy())


def predict_folds(folder, subset, device, collate):
    metadata=json.loads((folder/"report.json").read_text())
    parts=[]
    for fold in range(4):
        pred=score_checkpoint(folder/f"fold{fold}.pt",subset,metadata["args"]["batch_size"],device,collate)
        if not parts:
            keys=pred.select(KEY)
        elif not pred.select(KEY).equals(keys):
            raise ValueError("Test keys differ between folds")
        parts.append(standardize(pred))
    return keys.with_columns(pl.Series("score",np.mean(parts,axis=0)))


def package(variant, out):
    if out.exists() or out.with_suffix(".zip").exists():
        raise FileExistsError(out)
    folders=paths(False)
    new43=RUNS/"nn-context-fusion-s43-001"
    data=Path("competition")
    dataset=FeedRankingDataset(data,"test")
    table=pl.read_parquet(data/"sample_submission.parquet",columns=KEY).cast(dict.fromkeys(KEY,pl.Int32))
    collate=collate_ranking_batch
    args=json.loads((new43/"report.json").read_text())["args"]
    for group in ("itemstats","item2item","time"):
        collate=DonorFeatureCollator(args[group],data,"test",args["train_parts"],group,collate)
    device=torch.device(args["device"])
    def add_component(table,prediction,name):
        p=prediction.select(*KEY,pl.Series(name,standardize(prediction)))
        if table.height!=p.height:
            raise ValueError("Incomplete predictions")
        result=table.join(p,on=KEY,how="left",validate="1:1")
        if result[name].null_count() or not result[name].is_finite().all():
            raise ValueError("Invalid test scores")
        return result
    components = ["fusion42", "fusion43"]
    cached = RUNS / "nn-avgust-context-fusion2-oof4"
    use_cached = variant == "balanced" and (cached / "metrics.json").exists()
    if use_cached:
        metadata = json.loads((cached / "metrics.json").read_text())
        if metadata["experiment"] != "fusion2" or metadata["validator"] != "passed":
            raise ValueError("Invalid reference export")
        table = attach(table, cached / "test_pred.parquet", "fusion2")
    else:
        for name, folder in (("fusion42", folders["fusion"]), ("fusion43", new43)):
            table = add_component(table, predict_folds(folder, dataset, device, collate), name)
    if variant=="balanced":
        short=pl.concat([pl.read_parquet(p,columns=["item_id"]).select(pl.col("item_id").list.len()<=150) for p in dataset.parts]).to_series().to_numpy()
        prior=pl.read_parquet(RUNS/"nn-v2-length-rule-s42-oof4-001/test_pred.parquet")
        table=add_component(table,prior,"rule42")
        branches=[predict_folds(folders[name],Subset(dataset,np.flatnonzero(mask).tolist()),device,collate)
                  for name,mask in (("periodic43",short),("control43",~short))]
        table=add_component(table,pl.concat(branches),"rule43")
        components+=["rule42","rule43"]
    mean_score = (2 * pl.col("fusion2") + pl.col("rule42") + pl.col("rule43")) / 4 if use_cached else pl.mean_horizontal(components)
    test = table.select(*KEY, mean_score.cast(pl.Float32).alias("score"))
    val=pl.read_parquet(OUT/"candidates.parquet").select(*KEY,pl.col(variant).alias("score"))
    if val.height!=1389971 or val.select(KEY).is_duplicated().any() or not val["score"].is_finite().all():
        raise ValueError("Invalid OOF scores")
    results=json.loads((OUT/"oof_replicate.json").read_text())
    forward=json.loads((OUT/"forward_replicate.json").read_text())
    report={"experiment":variant,"seeds":[42,43],"train_shards":2,"folds":4,"val_rows":val.height,
            "test_rows":test.height,"aggregation":"mean of per-feed z-normalized folds within branch, then z-normalized branches averaged",
            "test_components":components,"oof":{k:r[variant] for k,r in results.items()},
            "forward":{k:r[variant] for k,r in forward.items()}}
    out.mkdir()
    val.write_parquet(out/"val_pred.parquet")
    test.write_parquet(out/"test_pred.parquet")
    errors=validate(str(data/"sample_submission.parquet"),str(out/"test_pred.parquet"))
    if errors:
        raise ValueError(errors)
    report["validator"]="passed"
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    with ZipFile(out.with_suffix(".zip"),"x",ZIP_DEFLATED,compresslevel=3) as z:
        for filename in ("val_pred.parquet","test_pred.parquet","metrics.json"):
            z.write(out/filename,filename)
    with ZipFile(out.with_suffix(".zip")) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity failure")
    print(out.with_suffix(".zip"),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("variant",choices=("fusion2","balanced"))
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    package(args.variant,args.out)
