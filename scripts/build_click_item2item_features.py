import argparse
from pathlib import Path
from src.common.click_item2item import build_features

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build directed donor click and co-click features.")
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-parts", type=int, default=8)
    p.add_argument("--train-start", type=int)
    p.add_argument("--memory-mb", type=int, default=1536)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    build_features(
        a.prepared,
        a.out,
        train_parts=a.train_parts,
        train_start=a.train_start,
        memory_mb=a.memory_mb,
        threads=a.threads,
        resume=a.resume,
    )
