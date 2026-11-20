"""Build cosine features with exponential decay over original history positions."""
import argparse
from pathlib import Path
from src.common.recency_cosine import build_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, default=Path('data/features/recency-cosine'))
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--prefetch-parts', type=int, default=2)
    parser.add_argument('--train-start', type=int, help='zero-based index of a contiguous train block')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.raw, args.prepared, args.out, train_parts=args.train_parts, prefetch_parts=args.prefetch_parts, resume=args.resume, train_start=args.train_start)
