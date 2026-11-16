"""Build exact item2item co-history features from a bounded-memory inverted index."""
import argparse
from pathlib import Path
from src.common.item2item import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=1536)
    parser.add_argument('--candidate-batch', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--first-train-shards', action='store_true', help='exclude the first N train shards used by the neural loader')
    args = parser.parse_args()
    build_features(args.prepared, args.out, train_parts=args.train_parts, memory_mb=args.memory_mb,
                   candidate_batch=args.candidate_batch, resume=args.resume, first_train_shards=args.first_train_shards)
