"""Cache within-feed ranks of strong features."""
import argparse
from pathlib import Path
from src.common.feature_ranks import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--itemstats', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.cosine, args.itemstats, args.out, train_parts=args.train_parts, resume=args.resume)
