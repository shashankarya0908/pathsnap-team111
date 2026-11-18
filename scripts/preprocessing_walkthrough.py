"""PathSnap Team 111 - preprocessing in one file (for presentation / viva).

Standalone, readable copy of the preprocessing steps used in src/pathsnap/
(data.py, labels.py, features.py). It runs every step on one trip and prints what
each step produced. The real pipeline (training + API) imports src/pathsnap.

Usage:  python scripts/preprocessing_walkthrough.py [partition] [trip_id]
        python scripts/preprocessing_walkthrough.py student_test 00000076
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree

R = 6371008.8  # mean Earth radius (m)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
part = sys.argv[1] if len(sys.argv) > 1 else "student_test"
tid = sys.argv[2] if len(sys.argv) > 2 else "00000076"
d = os.path.join(ROOT, "data", "raw", part, tid)
f = lambda ext: os.path.join(d, f"{tid}.{ext}")

# ---------------------------------------------------------------- 1. load + clean
raw = np.loadtxt(f("track"), ndmin=2)                        # lon, lat, t
nodes = np.loadtxt(f("nodes"), ndmin=2)                      # lon, lat  (row = node id)
arcs = np.loadtxt(f("arcs"), dtype=int, ndmin=2)             # from, to  (row = arc id)
route = np.loadtxt(f("route"), dtype=int, ndmin=1)           # ordered arc ids (labels only)

ok = np.isfinite(raw).all(1) & (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
idx = np.flatnonzero(ok)
mono = np.concatenate([[True], np.diff(np.maximum.accumulate(raw[idx, 2])) > 0])
idx = idx[mono]                                              # drop repeated/backwards timestamps
lonlat, t = raw[idx, :2], raw[idx, 2]
print(f"1. LOAD+CLEAN  {len(raw)} raw rows -> {len(idx)} kept | {len(nodes)} nodes, {len(arcs)} arcs, route {len(route)} arcs")

# ---------------------------------------------------------------- 2. per-trip projection to metres
lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
def project(ll):
    return np.column_stack([np.radians(ll[:, 0] - lon0) * R * np.cos(np.radians(lat0)),
                            np.radians(ll[:, 1] - lat0) * R])
xy, node_xy = project(lonlat), project(nodes)
print(f"2. PROJECT     origin ({lon0:.4f}, {lat0:.4f}); GPS extent {np.ptp(xy[:,0])/1000:.1f} x {np.ptp(xy[:,1])/1000:.1f} km")

# ---------------------------------------------------------------- 3. road geometry + spatial index
A, B = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
AB = B - A
arc_len = np.hypot(AB[:, 0], AB[:, 1])
arc_bearing = (np.degrees(np.arctan2(AB[:, 0], AB[:, 1])) + 360) % 360   # clockwise from north
pairs = set(map(tuple, arcs.tolist()))
has_reverse = np.array([(v, u) in pairs for u, v in arcs.tolist()], dtype=int)
n_samp = np.maximum(2, np.ceil(arc_len / 10).astype(int) + 1)            # a point every ~10 m
samp_arc = np.repeat(np.arange(len(arcs)), n_samp)
frac = np.concatenate([np.linspace(0, 1, k) for k in n_samp])
tree = cKDTree(A[samp_arc] + frac[:, None] * AB[samp_arc])
print(f"3. NETWORK     median arc {np.median(arc_len):.1f} m, {100*has_reverse.mean():.0f}% two-way, KD-tree on {len(samp_arc):,} samples")

def point_arc(p, a_ids):
    """Perpendicular distance (m) and projection position (0..1) of points p onto arcs a_ids."""
    a, ab = A[a_ids], AB[a_ids]
    tt = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    return np.hypot(*(p - (a + tt[:, None] * ab)).T), tt

# ---------------------------------------------------------------- 4. labels: route -> arc per GPS point
# D[i, j] = dist(i, route[j]) + min_{k<=j} D[i-1, k]  (route position never goes backwards)
N, M = len(xy), len(route)
cost = lambda p: point_arc(np.repeat(p[None], M, 0), route)[0]
back = np.zeros((N, M), dtype=np.int32)
prev = cost(xy[0])
for i in range(1, N):
    best = np.minimum.accumulate(prev)
    new = np.concatenate([[True], prev[1:] < best[:-1]])
    back[i] = np.maximum.accumulate(np.where(new, np.arange(M), 0))
    prev = best + cost(xy[i])
pos = np.zeros(N, dtype=int); j = int(prev.argmin())
for i in range(N - 1, -1, -1):
    pos[i] = j; j = back[i, j]
label = route[pos]
label_dist, _ = point_arc(xy, label)
print(f"4. LABELS      median GPS-to-label {np.median(label_dist):.1f} m, p95 {np.percentile(label_dist,95):.1f} m, "
      f"{100*(label_dist>100).mean():.1f}% > 100 m")

# ---------------------------------------------------------------- 5. candidates (50 m, <=8, fallback 3)
cands, miss, fallback = [], 0, 0
for i, hits in enumerate(tree.query_ball_point(xy, 50 + 10)):
    a_ids = np.unique(samp_arc[hits]) if hits else np.zeros(0, int)
    if len(a_ids):
        dd, _ = point_arc(np.repeat(xy[i][None], len(a_ids), 0), a_ids)
        a_ids = a_ids[dd <= 50][np.argsort(dd[dd <= 50])][:8]
    if len(a_ids) == 0:                                      # nothing within 50 m -> 3 nearest arcs
        _, k = tree.query(xy[i], k=200)
        a_ids = np.array(list(dict.fromkeys(samp_arc[k]))[:3]); fallback += 1
    cands.append(a_ids); miss += int(label[i] not in a_ids)
print(f"5. CANDIDATES  {np.mean([len(c) for c in cands]):.2f} per point, miss rate {100*miss/N:.2f}%, fallback {fallback} points")

# ---------------------------------------------------------------- 6. a few key features for one point
w = 2
i0, i1 = np.clip(np.arange(N) - w, 0, N - 1), np.clip(np.arange(N) + w, 0, N - 1)
dxy = xy[i1] - xy[i0]
heading = (np.degrees(np.arctan2(dxy[:, 0], dxy[:, 1])) + 360) % 360       # +-2 point window
speed = np.hypot(dxy[:, 0], dxy[:, 1]) / np.maximum(t[i1] - t[i0], 1e-6)
nearest = np.array([point_arc(xy[i][None], c[:1])[0][0] for i, c in enumerate(cands)])
sigma = float(np.clip(1.4826 * np.median(nearest), 2, 80))                # per-trip noise scale
i = N // 2
dd, tt = point_arc(np.repeat(xy[i][None], len(cands[i]), 0), cands[i])
hd = np.abs((heading[i] - arc_bearing[cands[i]] + 180) % 360 - 180)
print(f"6. FEATURES    trip noise sigma {sigma:.1f} m; point {i}: speed {speed[i]:.1f} m/s, heading {heading[i]:.0f} deg")
print("   arc      dist_m  dist_z  head_diff  two_way  is_label")
for a, d_, h in zip(cands[i], dd, hd):
    print(f"   {a:<8d} {d_:6.1f}  {d_/sigma:6.2f}  {h:9.0f}  {has_reverse[a]:7d}  {'  <-- reference' if a == label[i] else ''}")
