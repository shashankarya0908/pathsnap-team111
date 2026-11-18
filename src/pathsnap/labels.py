"""Per-observation reference labels derived from the .route file.

The .route file is a chronological list of arc ids, not one label per GPS
row. To get a reference arc for every observation we solve a monotonic
alignment: assign each GPS point i a route position j(i) with
j(0) <= j(1) <= ... (the vehicle never goes back along the route list),
minimising the sum of point-to-arc distances. This is an O(N*M) dynamic
programme with a running prefix minimum.

These labels are used for training and for server-side evaluation ONLY.
"""
from __future__ import annotations

import numpy as np

from .data import Trip


def align_route(trip: Trip, route: np.ndarray):
    """Return (label_arc[N], label_pos[N], label_dist[N])."""
    N, M = len(trip.xy), len(route)
    a = trip.node_xy[trip.arcs[route, 0]]
    b = trip.node_xy[trip.arcs[route, 1]]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)

    def row_cost(p):
        tt = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
        proj = a + tt[:, None] * ab
        return np.hypot(*(p - proj).T)

    back = np.zeros((N, M), dtype=np.int32)
    prev = row_cost(trip.xy[0])
    back[0] = np.arange(M)
    for i in range(1, N):
        # prefix minimum of prev and its argmin
        pm_idx = np.zeros(M, dtype=np.int32)
        best = np.minimum.accumulate(prev)
        # argmin position of the running minimum
        is_new = np.concatenate([[True], prev[1:] < best[:-1]])
        pm_idx = np.maximum.accumulate(np.where(is_new, np.arange(M), 0)).astype(np.int32)
        back[i] = pm_idx
        prev = best + row_cost(trip.xy[i])
    j = int(np.argmin(prev))
    pos = np.zeros(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    lab = route[pos]
    dist, _ = trip.point_arc_distance(trip.xy, lab)
    return lab, pos, dist
