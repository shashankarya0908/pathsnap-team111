"""Raw-data loading and geometry helpers for PathSnap (team 111).

File formats (verified during EDA, see scripts/01_eda.py):
  .track  lon  lat  t      (whitespace separated, no header)
  .nodes  lon  lat         (row index = node id)
  .arcs   from_node to_node (row index = arc id)
  .route  arc_id           (chronological ground-truth arcs; server-side only)

Coordinates are WGS84 degrees. Trips are spread across the world, so every
trip gets its own local equirectangular projection centred on the trip's GPS
mean: x = R * dlon * cos(lat0), y = R * dlat. Over the extent of one trip
(tens of km) the distortion of this projection is well below GPS noise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

EARTH_R = 6371008.8  # mean Earth radius in metres


def trip_files(trip_dir: str) -> dict:
    tid = os.path.basename(os.path.normpath(trip_dir))
    return {ext: os.path.join(trip_dir, f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}


def list_trips(partition_dir: str) -> list[str]:
    if not os.path.isdir(partition_dir):
        return []
    return sorted(d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d)))


def _load(path, ncols, dtype=float):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    return arr[:, :ncols]


def load_route(trip_dir: str) -> np.ndarray:
    """Ground-truth arc sequence. Only evaluation / label code may call this."""
    p = trip_files(trip_dir)["route"]
    if not os.path.exists(p):
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


@dataclass
class Trip:
    trip_id: str
    lonlat: np.ndarray      # (N,2) GPS lon/lat after cleaning
    t: np.ndarray           # (N,) seconds
    keep_idx: np.ndarray    # indices of kept rows in the raw .track file
    node_lonlat: np.ndarray # (V,2)
    arcs: np.ndarray        # (E,2) from,to
    origin: tuple = (0.0, 0.0)
    xy: np.ndarray = None
    node_xy: np.ndarray = None
    arc_len: np.ndarray = None
    arc_bearing: np.ndarray = None
    has_reverse: np.ndarray = None
    out_deg: np.ndarray = None
    in_deg: np.ndarray = None
    n_raw_points: int = 0
    _tree: cKDTree = field(default=None, repr=False)
    _sample_arc: np.ndarray = field(default=None, repr=False)

    # ------------------------------------------------------------------ geometry
    def project(self, lonlat: np.ndarray) -> np.ndarray:
        lon0, lat0 = self.origin
        x = np.radians(lonlat[:, 0] - lon0) * EARTH_R * np.cos(np.radians(lat0))
        y = np.radians(lonlat[:, 1] - lat0) * EARTH_R
        return np.column_stack([x, y])

    def build(self, sample_step: float = 10.0):
        self.origin = (float(np.mean(self.lonlat[:, 0])), float(np.mean(self.lonlat[:, 1])))
        self.xy = self.project(self.lonlat)
        self.node_xy = self.project(self.node_lonlat)
        a = self.node_xy[self.arcs[:, 0]]
        b = self.node_xy[self.arcs[:, 1]]
        d = b - a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        # bearing measured clockwise from north, degrees in [0,360)
        self.arc_bearing = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
        pairs = set(map(tuple, self.arcs.tolist()))
        self.has_reverse = np.array([(v, u) in pairs for u, v in self.arcs.tolist()], dtype=np.int8)
        V = len(self.node_xy)
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=V)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=V)
        # KD-tree over points sampled along every arc (arcs are straight segments)
        n_samp = np.maximum(2, np.ceil(self.arc_len / sample_step).astype(int) + 1)
        arc_ids = np.repeat(np.arange(len(self.arcs)), n_samp)
        frac = np.concatenate([np.linspace(0, 1, n) for n in n_samp])
        pts = self.node_xy[self.arcs[arc_ids, 0]] + frac[:, None] * d[arc_ids]
        self._tree = cKDTree(pts)
        self._sample_arc = arc_ids
        self._sample_step = sample_step
        return self

    def point_arc_distance(self, pxy: np.ndarray, arc_ids: np.ndarray):
        """Perpendicular distance + projection fraction of points to arcs (paired)."""
        a = self.node_xy[self.arcs[arc_ids, 0]]
        b = self.node_xy[self.arcs[arc_ids, 1]]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        tt = np.clip(((pxy - a) * ab).sum(1) / L2, 0.0, 1.0)
        proj = a + tt[:, None] * ab
        dist = np.hypot(*(pxy - proj).T)
        return dist, tt

    def arcs_near(self, pxy: np.ndarray, radius: float) -> list[np.ndarray]:
        r = radius + self._sample_step
        hits = self._tree.query_ball_point(pxy, r)
        return [np.unique(self._sample_arc[h]) if len(h) else np.zeros(0, int) for h in hits]

    def nearest_arcs(self, p: np.ndarray, k: int) -> np.ndarray:
        _, idx = self._tree.query(p, k=min(len(self._sample_arc), 40 * k))
        arcs = []
        for a in self._sample_arc[np.atleast_1d(idx)]:
            if a not in arcs:
                arcs.append(a)
            if len(arcs) >= k:
                break
        return np.array(arcs, dtype=int)


def load_trip(trip_dir: str) -> Trip:
    """Load track + road network (never the route) and clean the GPS track.

    Cleaning: drop rows with non-finite or out-of-range coordinates, drop
    duplicated / non-increasing timestamps (keep first occurrence).
    """
    f = trip_files(trip_dir)
    raw = _load(f["track"], 3)
    n_raw = len(raw)
    ok = np.isfinite(raw).all(1) & (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    idx = np.flatnonzero(ok)
    if len(idx):
        t = raw[idx, 2]
        mono = np.concatenate([[True], np.diff(np.maximum.accumulate(t)) > 0])
        idx = idx[mono]
    nodes = _load(f["nodes"], 2)
    arcs = _load(f["arcs"], 2, dtype=np.int64)
    trip = Trip(
        trip_id=os.path.basename(os.path.normpath(trip_dir)),
        lonlat=raw[idx, :2],
        t=raw[idx, 2],
        keep_idx=idx,
        node_lonlat=nodes,
        arcs=arcs,
        n_raw_points=n_raw,
    )
    return trip.build()


def gps_heading_speed(xy: np.ndarray, t: np.ndarray, win: int = 2):
    """Heading (deg from north) and speed (m/s) from a centred finite difference."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - win, 0, n - 1)
    i1 = np.clip(np.arange(n) + win, 0, n - 1)
    d = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    dist = np.hypot(d[:, 0], d[:, 1])
    heading = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
    speed = dist / dt
    return heading, speed, dist


def angle_diff(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d
