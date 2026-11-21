"""Reproduce a CatBoost pipeline or replay a frozen ensemble from saved predictions."""
import argparse
import hashlib
from pathlib import Path

from scripts.reproduce_data import reproduce
from src.common.data import list_parts


def reproduce_ranking(raw=Path('competition'), output=Path('runs/reproduced-ranking'), *,
                      prepared=Path('data/prepared'), feature_dir=Path('data/features'),
                      catalogue=Path('data/embedding_catalogue'),
                      train_parts=8, infeed=False, history=False, itemstats=False, userstats=False,
                      category_cooc=False, event_cosine=False, feature_ranks=False, time_features=False, item2item=False,
                      weighted_item2item=False, recency_cosine=False, feed_context=False, bayesian_ctr=False,
                      iterations=100, depth=6, seed=42, threads=2, memory_mb=2048, resume=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a fresh output directory for the model run')
    if train_parts < 1 or iterations < 1 or depth < 1 or threads < 1 or memory_mb < 128:
        raise ValueError('invalid training or resource parameters')
    raw, prepared, feature_dir = (Path(path).resolve() for path in (raw, prepared, feature_dir))
    if not (raw / 'train/impressions').is_dir() and (raw / 'competition/train/impressions').is_dir():
        raw = raw / 'competition'
    available_parts = len(list_parts(raw, 'train'))
    if train_parts > available_parts:
        raise ValueError('requested training size exceeds the source shard count')
    needs_itemstats = itemstats or feature_ranks or feed_context
    needs_item2item = item2item or weighted_item2item or feed_context
    if (needs_itemstats or needs_item2item or userstats or category_cooc or time_features or bayesian_ctr) and train_parts == available_parts:
        raise ValueError('statistics require at least one donor shard outside model training')
    reproduce(raw, prepared, output.parent / 'eda', catalogue=Path(catalogue), features=feature_dir / 'cosine',
              feature_train_parts=train_parts, infeed=feature_dir / 'infeed' if infeed else None,
              resume=resume)
    roots = [feature_dir / 'cosine']
    if infeed:
        roots.append(feature_dir / 'infeed')
    if history:
        from src.common.history import build_features
        root = feature_dir / 'history'
        build_features(prepared, root, train_parts=train_parts, resume=resume)
        roots.append(root)
    for enabled, name in ((needs_itemstats, 'itemstats'), (userstats, 'userstats')):
        if enabled:
            if name == 'itemstats':
                from src.common.itemstats import build_features
            else:
                from src.common.userstats import build_features
            root = feature_dir / f'{name}-{train_parts:03d}'
            build_features(prepared, root, train_parts=train_parts, memory_mb=memory_mb,
                           threads=1, resume=resume)
            if name != 'itemstats' or itemstats:
                roots.append(root)
    if category_cooc:
        from src.common.category_cooc import build_features
        root = feature_dir / f'category-cooc-{train_parts:03d}'
        build_features(prepared, root, train_parts=train_parts, resume=resume)
        roots.append(root)
    if event_cosine:
        from src.common.event_cosine import build_features
        root = feature_dir / 'event-cosine'
        build_features(raw, prepared, root, train_parts=train_parts, resume=resume)
        roots.append(root)
    if feature_ranks:
        from src.common.feature_ranks import build_features
        root = feature_dir / f'feature-ranks-{train_parts:03d}'
        build_features(prepared, feature_dir / 'cosine', feature_dir / f'itemstats-{train_parts:03d}',
                       root, train_parts=train_parts, resume=resume)
        roots.append(root)
    if time_features:
        from src.common.time_features import build_features
        root = feature_dir / f'time-{train_parts:03d}'
        build_features(prepared, root, train_parts=train_parts, memory_mb=memory_mb, resume=resume)
        roots.append(root)
    if needs_item2item:
        from src.common.item2item import build_features
        root = feature_dir / f'item2item-{train_parts:03d}'
        build_features(prepared, root, train_parts=train_parts, memory_mb=memory_mb, resume=resume)
        if item2item:
            roots.append(root)
    if weighted_item2item:
        from src.common.weighted_item2item import build_features
        root = feature_dir / f'weighted-item2item-{train_parts:03d}'
        build_features(prepared, feature_dir / f'item2item-{train_parts:03d}', root,
                       train_parts=train_parts, resume=resume)
        roots.append(root)
    if recency_cosine:
        from src.common.recency_cosine import build_features
        root = feature_dir / 'recency-cosine'
        build_features(raw, prepared, root, train_parts=train_parts, resume=resume)
        roots.append(root)
    if feed_context:
        from src.common.feed_context import build_features
        root = feature_dir / f'feed-context-{train_parts:03d}'
        build_features(prepared, feature_dir / 'cosine', feature_dir / f'itemstats-{train_parts:03d}',
                       feature_dir / f'item2item-{train_parts:03d}', root, train_parts=train_parts, resume=resume)
        roots.append(root)
    if bayesian_ctr:
        from src.common.bayesian_ctr import build_features
        root = feature_dir / f'bayesian-ctr-{train_parts:03d}'
        build_features(prepared, root, train_parts=train_parts, memory_mb=memory_mb, threads=1, resume=resume)
        roots.append(root)
    from scripts.run_baseline_check import run
    report = run(prepared, raw, output, train_parts=train_parts, iterations=iterations,
                 depth=depth, seed=seed, threads=threads, feature_root=roots)
    import polars as pl
    from src.common.preparation import write_json, write_parquet_atomic
    handoff = {}
    for source, name, columns in (
        ('validation_predictions.parquet', 'val_pred.parquet', ['feed_id', 'item_id', 'score', 'target']),
        ('submission.parquet', 'test_pred.parquet', ['feed_id', 'item_id', 'score']),
    ):
        frame = pl.read_parquet(output / source, columns=columns)
        path = output / name
        write_parquet_atomic(frame, path)
        handoff[name] = {'rows': frame.height, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(output / 'pipeline.json', {
        'complete': True, 'pipeline_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_roots': [str(root) for root in roots], 'train_parts': train_parts,
        'model_git': report['git'], 'predictions': handoff,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features'))
    parser.add_argument('--catalogue', type=Path, default=Path('data/embedding_catalogue'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ensemble-config', type=Path, help='frozen recipe with paths, score transforms and hashes')
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--infeed', action='store_true')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--itemstats', action='store_true')
    parser.add_argument('--userstats', action='store_true')
    parser.add_argument('--category-cooc', action='store_true')
    parser.add_argument('--event-cosine', action='store_true')
    parser.add_argument('--feature-ranks', action='store_true')
    parser.add_argument('--time-features', action='store_true')
    parser.add_argument('--item2item', action='store_true')
    parser.add_argument('--weighted-item2item', action='store_true')
    parser.add_argument('--recency-cosine', action='store_true')
    parser.add_argument('--feed-context', action='store_true')
    parser.add_argument('--bayesian-ctr', action='store_true')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-mb', type=int, default=2048, help='DuckDB aggregation memory limit')
    parser.add_argument('--resume', action='store_true', help='reuse matching data and feature caches')
    args = parser.parse_args()
    if args.ensemble_config:
        from scripts.reproduce_ensemble import rebuild_ensemble
        rebuild_ensemble(args.raw, args.out, args.ensemble_config)
        return
    reproduce_ranking(args.raw, args.out, prepared=args.prepared, feature_dir=args.feature_dir,
                      catalogue=args.catalogue,
                      train_parts=args.train_parts, infeed=args.infeed, history=args.history,
                      itemstats=args.itemstats, userstats=args.userstats, iterations=args.iterations,
                      category_cooc=args.category_cooc, event_cosine=args.event_cosine,
                      feature_ranks=args.feature_ranks, time_features=args.time_features,
                      item2item=args.item2item,
                      weighted_item2item=args.weighted_item2item, recency_cosine=args.recency_cosine,
                      feed_context=args.feed_context, bayesian_ctr=args.bayesian_ctr,
                      depth=args.depth, seed=args.seed, threads=args.threads,
                      memory_mb=args.memory_mb, resume=args.resume)


if __name__ == '__main__':
    main()
