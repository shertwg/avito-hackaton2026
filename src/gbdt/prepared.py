"""Чтение подготовленных данных Димы (data/prepared + кеши признаков) для раннера GBDT.

Вместо сборки признаков из сырых лент (src/common/data.py — медленно и весь split в памяти)
берём готовые таблицы кандидатов и группы признаков:
    data/prepared/<split>/candidates/part-*.parquet   (scripts.reproduce_data / prepare_data)
    data/features/cosine, data/features/infeed         (build_similarity_features / build_infeed_features)
Ключ строки — (feed_id, candidate_index); группы присоединяются строго 1:1 с проверкой ключей.
Когда порядок ключей уже совпадает (стандартный случай для всех кешей), используется zero-copy
hstack за O(1) вместо тяжёлого hash-join с копированием всех предыдущих колонок (~10x быстрее, -15 ГБ RAM).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from src.common.train_selection import select_parts

BASE = ["f1", "f2", "f3", "f4", "f5", "category_id"]
JOIN_KEYS = ["feed_id", "candidate_index", "item_id"]


def _manifest(root: Path) -> dict:
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not m.get("complete"):
        raise ValueError(f"{root}: подготовка данных не завершена (manifest complete=false)")
    return m


def select_train_parts(prepared: Path, n_parts: int, train_start: int | None = None) -> list[dict]:
    return select_parts(_manifest(prepared), n_parts, train_start)


def _attach_fast(
    frame: pl.DataFrame,
    frame_keys: pl.DataFrame,
    feature_root: str | Path,
    split: str,
    source_parts: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Строгое 1:1 присоединение группы признаков с быстрым путём hstack при совпадении ключей."""
    root = Path(feature_root)
    manifest = _manifest(root)
    if split == "train" and "excluded_train_sources" in manifest:
        if not set(source_parts) <= set(manifest["excluded_train_sources"]):
            raise ValueError(
                f"{manifest.get('feature_group', root.name)} donor shards overlap requested model training shards"
            )
    split_parts = manifest.get("splits", {}).get(split, {}).get("parts", {})
    missing = set(source_parts) - set(split_parts)
    if missing:
        raise ValueError(f"{root}: cache does not cover {len(missing)} requested {split} shards")

    feat_names: list[str] = list(manifest["features"])
    paths = [root / split_parts[name]["path"] for name in source_parts]
    read_cols = ["split", *JOIN_KEYS, *feat_names]
    features = pl.concat([pl.read_parquet(p, columns=read_cols) for p in paths], how="vertical", rechunk=False)

    if features.height != frame.height:
        raise ValueError(f"{root}: feature rows ({features.height}) != candidate rows ({frame.height})")
    if features["split"].unique().to_list() != [split]:
        raise ValueError(f"{root}: wrong feature split")

    feat_only = features.select(feat_names)
    if feat_only.null_count().to_numpy().any():
        raise ValueError(f"{root}: null values in features")

    f_keys = features.select(JOIN_KEYS)
    if f_keys.equals(frame_keys):
        return frame.hstack(feat_only), feat_names

    # Если шарды были не отсортированы — сортируем только узкую таблицу новой группы и проверяем снова
    features_sorted = features.sort("feed_id", "candidate_index")
    if features_sorted.select(JOIN_KEYS).equals(frame_keys):
        return frame.hstack(features_sorted.select(feat_names)), feat_names

    # Резервный строгий 1:1 hash-join
    joined = frame.join(features.select(*JOIN_KEYS, *feat_names), on=JOIN_KEYS, how="left", validate="1:1")
    if joined.select(feat_names).null_count().to_numpy().any():
        raise ValueError(f"{root}: missing feature keys after 1:1 join")
    return joined, feat_names


def load_prepared(
    prepared: str | Path,
    split: str,
    feature_roots: list[str | Path],
    train_parts: int = 1,
    train_start: int | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Таблица кандидатов split с признаками групп. Возвращает (frame, имена признаков модели)."""
    prepared = Path(prepared)
    man = _manifest(prepared)
    parts = (
        select_train_parts(prepared, train_parts, train_start)
        if split == "train"
        else man["splits"][split]["parts"]
    )
    schema = pl.read_parquet_schema(prepared / parts[0]["candidate_path"])
    cols = ["feed_id", "timestamp", "candidate_index", "item_id", *BASE]
    cols += [c for c in ("target", "pos") if c in schema]
    df = pl.concat([pl.read_parquet(prepared / p["candidate_path"], columns=cols) for p in parts], how="vertical")
    df = df.sort("feed_id", "candidate_index")
    frame_keys = df.select(JOIN_KEYS)
    feats = list(BASE)
    sources = [p["source"] for p in parts]
    for root in feature_roots:
        df, names = _attach_fast(df, frame_keys, root, split, sources)
        clash = set(feats) & set(names)
        if clash:
            raise ValueError(f"повтор имён признаков между группами: {sorted(clash)}")
        feats += names
    return df, feats
