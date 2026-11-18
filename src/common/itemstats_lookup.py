"""Sorted Float32 lookup arrays for already aggregated item/category statistics."""
from pathlib import Path

import numpy as np
import polars as pl


def lookup_paths(output, entity):
    return [Path(output) / f'_{entity}_lookup_keys.npy', Path(output) / f'_{entity}_lookup_values.npy']


def export_lookup(connection, output, entity, fields):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = f'{entity}_id'
    names = [f'is_{entity}_{field}' for field in fields]
    count = connection.execute(f'SELECT count(*) FROM {entity}_lookup').fetchone()[0]
    key_path, value_path = lookup_paths(output, entity)
    temporary_keys = key_path.with_suffix('.npy.tmp')
    temporary_values = value_path.with_suffix('.npy.tmp')
    keys = np.lib.format.open_memmap(temporary_keys, mode='w+', dtype=np.int32, shape=(count,))
    values = None
    try:
        values = np.lib.format.open_memmap(temporary_values, mode='w+', dtype=np.float32, shape=(count, len(fields)))
        batches = connection.execute(f'SELECT {key},{",".join(names)} FROM {entity}_lookup ORDER BY {key}').fetch_record_batch(100_000)
        offset = 0
        for batch in batches:
            stop = offset + batch.num_rows
            keys[offset:stop] = batch[key].to_numpy(zero_copy_only=False)
            matrix = np.column_stack([batch[name].to_numpy(zero_copy_only=False) for name in names])
            if not np.isfinite(matrix).all():
                raise ValueError('nonfinite itemstats lookup value')
            values[offset:stop] = matrix
            offset = stop
        if offset != count:
            raise ValueError('lookup export row count differs')
        keys.flush()
        values.flush()
    finally:
        keys._mmap.close()
        if values is not None:
            values._mmap.close()
    temporary_keys.replace(key_path)
    temporary_values.replace(value_path)
    return [key_path, value_path]


class LookupReader:
    def __init__(self, output, fields):
        self.arrays = {}
        try:
            for entity in ('item', 'category'):
                keys = np.load(Path(output) / f'_{entity}_lookup_keys.npy', mmap_mode='r', allow_pickle=False)
                try:
                    values = np.load(Path(output) / f'_{entity}_lookup_values.npy', mmap_mode='r', allow_pickle=False)
                except BaseException:
                    keys._mmap.close()
                    raise
                self.arrays[entity] = keys, values
                if (keys.dtype != np.int32 or keys.ndim != 1 or values.dtype != np.float32
                        or values.shape != (len(keys), len(fields))):
                    raise ValueError('invalid itemstats lookup arrays')
                for start in range(0, len(keys), 1_000_000):
                    chunk = keys[max(0, start - 1):start + 1_000_000]
                    if (chunk[1:] <= chunk[:-1]).any():
                        raise ValueError('lookup keys are not strictly increasing')
        except BaseException:
            self.close()
            raise

    def close(self):
        for keys, values in self.arrays.values():
            keys._mmap.close()
            values._mmap.close()
        self.arrays.clear()

    def build_part(self, candidates, names):
        frame = candidates.sort('feed_id', 'candidate_index')
        chunks = []
        for entity in ('item', 'category'):
            keys, values = self.arrays[entity]
            query = frame[f'{entity}_id'].to_numpy()
            positions = np.searchsorted(keys, query)
            if (positions >= len(keys)).any() or not np.array_equal(keys[positions], query):
                raise ValueError('candidate key missing from itemstats query lookup')
            chunks.append(values[positions])
        matrix = np.concatenate(chunks, axis=1)
        return frame.select('split', 'feed_id', 'candidate_index', 'item_id').with_columns([
            pl.Series(name, matrix[:, index], dtype=pl.Float32) for index, name in enumerate(names)])
