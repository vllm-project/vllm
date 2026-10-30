# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference helpers for speculative verify trees.

A request's tree is ``parent``: ``parent[i]`` is the row that row ``i``
extends, ``-1`` for the root (row 0), and a parent always precedes its child.
"""

import numpy as np


def random_parents(gen: np.random.Generator, num_rows: int) -> list[int]:
    """A random tree: each row extends a uniformly drawn earlier row."""
    return [-1] + [int(gen.integers(0, row)) for row in range(1, num_rows)]


def depths_and_ancestors(parent) -> tuple[list[int], list[int]]:
    """Each row's depth and its ancestor bitmask (bit ``j``: row ``j`` is on the
    row's path from the root, the row itself included)."""
    depth: list[int] = []
    ancestors: list[int] = []
    for row in range(len(parent)):
        up = int(parent[row])
        depth.append(0 if row == 0 else depth[up] + 1)
        ancestors.append((1 << row) | (0 if row == 0 else ancestors[up]))
    return depth, ancestors


def path_to(parent, row: int) -> list[int]:
    """The rows from the root down to ``row``."""
    path = [row]
    while int(parent[path[-1]]) >= 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]
