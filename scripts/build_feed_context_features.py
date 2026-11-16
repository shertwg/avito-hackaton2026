"""Build within-feed context of cosine, item2item, CTR and f1-f5."""
import argparse
from pathlib import Path
from src.common.feed_context import build_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--itemstats', type=Path, required=True)
    parser.add_argument('--item2item', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.cosine, args.itemstats, args.item2item, args.out,
                   train_parts=args.train_parts, resume=args.resume)
