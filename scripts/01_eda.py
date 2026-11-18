"""Exploratory data analysis on the train + student_test partitions.

Outputs: results/eda_inventory.csv, results/eda_summary.json, figures/eda_*.png
"""
import os, sys, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.data import trip_files, list_trips, load_trip, load_route, gps_heading_speed

RAW = os.path.join(ROOT, "data", "raw")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)

rows, all_dt, all_speed = [], [], []
for part in ["train", "student_test"]:
    for tid in list_trips(os.path.join(RAW, part)):
        f = trip_files(os.path.join(RAW, part, tid))
        raw = pd.read_csv(f["track"], sep=r"\s+", header=None, names=["lon", "lat", "t"])
        nodes = pd.read_csv(f["nodes"], sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(f["arcs"], sep=r"\s+", header=None, names=["u", "v"])
        route = load_route(os.path.join(RAW, part, tid))
        dt = np.diff(raw.t.values)
        trip = load_trip(os.path.join(RAW, part, tid))
        _, sp, _ = gps_heading_speed(trip.xy, trip.t, win=1)
        all_dt.append(dt); all_speed.append(sp)
        G = nx.DiGraph(); G.add_nodes_from(range(len(nodes))); G.add_edges_from(arcs.values.tolist())
        wcc = list(nx.weakly_connected_components(G))
        ext = trip.xy.max(0) - trip.xy.min(0)
        rows.append(dict(
            partition=part, trip=tid,
            files_present=sum(os.path.exists(p) for p in f.values()),
            track_kb=os.path.getsize(f["track"]) / 1024, nodes_kb=os.path.getsize(f["nodes"]) / 1024,
            n_gps=len(raw), n_missing=int(raw.isna().sum().sum()),
            n_bad_coord=int(((raw.lon.abs() > 180) | (raw.lat.abs() > 90)).sum()),
            n_nonincreasing_t=int((dt <= 0).sum()),
            dt_median=float(np.median(dt)), dt_max=float(dt.max()), frac_dt_eq_1=float((dt == 1).mean()),
            n_gaps_gt10s=int((dt > 10).sum()), duration_min=float((raw.t.iloc[-1] - raw.t.iloc[0]) / 60),
            lon_center=float(raw.lon.mean()), lat_center=float(raw.lat.mean()),
            extent_x_km=float(ext[0] / 1000), extent_y_km=float(ext[1] / 1000),
            n_nodes=len(nodes), n_arcs=len(arcs), n_route_arcs=len(route), n_unique_route_arcs=len(set(route)),
            n_isolated_nodes=int(sum(1 for n in G.nodes if G.degree(n) == 0)),
            n_weak_components=len(wcc), largest_wcc_frac=max(len(c) for c in wcc) / len(nodes),
            n_strong_components=nx.number_strongly_connected_components(G),
            frac_twoway_arcs=float(trip.has_reverse.mean()),
            arc_len_median_m=float(np.median(trip.arc_len)),
            route_consecutive_connected=float(np.mean(arcs.v.values[route[:-1]] == arcs.u.values[route[1:]])),
            speed_p95_ms=float(np.percentile(sp, 95)),
        ))
inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(RES, "eda_inventory.csv"), index=False)
summary = {
    "trips_per_partition": inv.partition.value_counts().to_dict(),
    "gps_points_per_partition": inv.groupby("partition").n_gps.sum().to_dict(),
    "total_missing_values": int(inv.n_missing.sum()),
    "total_bad_coords": int(inv.n_bad_coord.sum()),
    "total_nonincreasing_timestamps": int(inv.n_nonincreasing_t.sum()),
    "frac_dt_eq_1_overall": float((np.concatenate(all_dt) == 1).mean()),
    "trips_with_gaps_gt10s": int((inv.n_gaps_gt10s > 0).sum()),
    "max_gap_s": float(inv.dt_max.max()),
    "route_consecutive_connected_mean": float(inv.route_consecutive_connected.mean()),
    "median_nodes": float(inv.n_nodes.median()), "median_arcs": float(inv.n_arcs.median()),
    "trips_with_isolated_nodes": int((inv.n_isolated_nodes > 0).sum()),
    "median_weak_components": float(inv.n_weak_components.median()),
}
json.dump(summary, open(os.path.join(RES, "eda_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))

# ---------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(inv.n_gps, bins=30, color="#3b6ea5"); ax[0].set(title="GPS points per trip (train+student_test)", xlabel="points", ylabel="trips")
dt = np.concatenate(all_dt); ax[1].hist(np.clip(dt, 0, 10), bins=np.arange(0, 11.5, 0.5), color="#3b6ea5")
ax[1].set(title="Sampling interval (clipped at 10 s)", xlabel="seconds", ylabel="count", yscale="log")
sp = np.concatenate(all_speed); ax[2].hist(np.clip(sp, 0, 50), bins=50, color="#3b6ea5")
ax[2].set(title="Point speed (m/s, clipped at 50)", xlabel="m/s")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_distributions.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
for part, c in [("train", "#3b6ea5"), ("student_test", "#d1495b")]:
    s = inv[inv.partition == part]; ax.scatter(s.lon_center, s.lat_center, s=12, c=c, label=part)
ax.set(title="Trip centres (lon/lat)", xlabel="longitude", ylabel="latitude"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_trip_centres.png"), dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(inv.n_nodes, inv.n_arcs, s=10); ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
ax[1].hist(inv.n_weak_components, bins=30); ax[1].set(title="Weakly connected components per trip network", xlabel="components")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_network.png"), dpi=130); plt.close()

# example trajectories over road network
for tid, part in [("00000048", "train"), ("00000076", "student_test")]:
    tdir = os.path.join(RAW, part, tid); trip = load_trip(tdir); route = load_route(tdir)
    fig, ax = plt.subplots(figsize=(8, 8))
    lo, hi = trip.lonlat.min(0) - 0.003, trip.lonlat.max(0) + 0.003
    nl = trip.node_lonlat
    for u, v in trip.arcs:
        a, b = nl[u], nl[v]
        if (lo <= a).all() and (a <= hi).all():
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", lw=.5)
    for r in route:
        a, b = nl[trip.arcs[r, 0]], nl[trip.arcs[r, 1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=2.5)
    ax.scatter(trip.lonlat[:, 0], trip.lonlat[:, 1], s=2, c="blue", zorder=3)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), title=f"Trip {tid} ({part}): GPS (blue), reference route (green), network (grey)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"eda_trip_{tid}.png"), dpi=130); plt.close()

# label quality figure (needs 02_build_dataset to have run)
lq = os.path.join(RES, "label_quality.csv")
if os.path.exists(lq):
    q = pd.read_csv(lq)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(q)), q.candidate_miss_rate, color=np.where(q.partition == "train", "#3b6ea5", "#d1495b"))
    ax.set(title="Candidate miss rate per trip (50 m radius, k<=8)", xlabel="trip index", ylabel="miss rate")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_candidate_miss.png"), dpi=130); plt.close()
print(inv.describe().T[["mean", "50%", "min", "max"]].to_string())
