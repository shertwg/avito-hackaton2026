"""Build UTC clock features and donor-based item age."""
import argparse
from pathlib import Path
from src.common.time_features import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=1536)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.out, train_parts=args.train_parts, memory_mb=args.memory_mb, resume=args.resume)
