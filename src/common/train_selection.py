"""Select train shards by manifest position or the existing linspace policy."""
from numbers import Integral

import numpy as np


def select_parts(source, count, start=None):
    parts = source['splits']['train']['parts']
    if isinstance(count, bool) or not isinstance(count, Integral) or not 1 <= count <= len(parts):
        raise ValueError('training shard count outside source range')
    if len({part['source'] for part in parts}) != len(parts):
        raise ValueError('duplicate training shard source')
    if start is None:
        return [parts[int(index)] for index in np.linspace(0, len(parts) - 1, count, dtype=int)]
    if isinstance(start, bool) or not isinstance(start, Integral) or start < 0 or start + count > len(parts):
        raise ValueError('training block outside source range')
    return parts[start:start + count]
