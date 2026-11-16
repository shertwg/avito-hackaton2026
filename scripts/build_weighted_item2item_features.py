"""Build event- and position-weighted item2item from an existing donor index."""
import argparse
from pathlib import Path
from src.common.weighted_item2item import build_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.parent, args.out, train_parts=args.train_parts, resume=args.resume)
