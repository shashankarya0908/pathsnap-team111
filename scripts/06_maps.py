"""Geographical visualisation of predictions on student_test trips.

Produces for each selected trip:
  maps/trip_<id>.html  interactive Folium map with layer control:
        road network (grey) | raw GPS (blue) | reference route (green) |
        classifier-only route (orange) | HMM route (red) | low-confidence points
  figures/map_<id>.png static matplotlib version of the same layers (for the PDF)
Usage: python scripts/06_maps.py [trip_id ...]   (default: three student_test trips)
Requires `pip install folium` for the HTML maps.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, MOD, FIG
from pathsnap.data import load_route
from pathsnap.hmm import stitch_route
from pathsnap.pipeline import PathSnapPredictor

try:
    import folium
except ImportError:
    folium = None
    print("folium not installed -> only static PNG maps will be written (pip install folium)")

MAPS = os.path.join(ROOT, "maps"); os.makedirs(MAPS, exist_ok=True)
trips = sys.argv[1:] or ["00000076", "00000009", "00000056"]
pred = PathSnapPredictor(MOD)


def arcs_to_lines(trip, arcs):
    nl = trip.node_lonlat
    return [[(nl[u, 1], nl[u, 0]), (nl[v, 1], nl[v, 0])] for u, v in trip.arcs[list(arcs)]]


for tid in trips:
    tdir = os.path.join(ROOT, "data", "raw", "student_test", tid)
    out = pred.predict(tdir); ev = pred.evaluate(tdir, out)
    trip, G = out["trip"], out["G"]
    ref = load_route(tdir)
    clf_route, _, _ = stitch_route(G, out["clf_arcs"])
    conf = np.array(out["response"]["confidence"])
    lo, hi = trip.lonlat.min(0) - 0.002, trip.lonlat.max(0) + 0.002
    a = trip.node_lonlat[trip.arcs[:, 0]]
    near = np.flatnonzero(((a >= lo) & (a <= hi)).all(1))
    title = (f"Trip {tid}: HMM acc {ev['point_accuracy']:.3f} vs classifier-only {ev['classifier_only_point_accuracy']:.3f}, "
             f"route F1 {ev['route_f1']:.3f}")
    print(title)

    # ---- static PNG
    fig, ax = plt.subplots(figsize=(9, 9))
    for seg in arcs_to_lines(trip, near):
        ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="#cccccc", lw=.5, zorder=1)
    for arcs, c, lw, lab in [(ref, "green", 5, "reference route"), (clf_route, "orange", 2.5, "classifier-only"),
                             (out["route"], "red", 1.6, "HMM (hybrid decoder)")]:
        for k, seg in enumerate(arcs_to_lines(trip, arcs)):
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color=c, lw=lw, zorder=2, label=lab if k == 0 else None)
    ax.scatter(trip.lonlat[:, 0], trip.lonlat[:, 1], s=3, c="blue", zorder=3, label="GPS observations")
    low = conf < 0.6
    ax.scatter(trip.lonlat[low, 0], trip.lonlat[low, 1], s=18, facecolors="none", edgecolors="magenta", zorder=4,
               label="HMM confidence < 0.6")
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="longitude", ylabel="latitude")
    ax.set_title(title, fontsize=10); ax.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"map_{tid}.png"), dpi=130); plt.close()

    # ---- interactive Folium
    if folium is None:
        continue
    m = folium.Map(location=[float(trip.lonlat[:, 1].mean()), float(trip.lonlat[:, 0].mean())], zoom_start=14,
                   tiles=None)
    # English place names worldwide (CARTO uses English names where available); others as alternatives
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
                     attr="&copy; OpenStreetMap &copy; CARTO", name="English (CARTO Voyager)", subdomains="abcd",
                     max_zoom=20).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Esri streets", max_zoom=19).add_to(m)
    folium.TileLayer("openstreetmap", name="OpenStreetMap (local language)").add_to(m)
    fg_net = folium.FeatureGroup("Road network", show=False)
    for seg in arcs_to_lines(trip, near[:20000]):
        folium.PolyLine(seg, color="#999999", weight=1, opacity=.6).add_to(fg_net)
    fg_ref = folium.FeatureGroup("Reference route (green)")
    for seg in arcs_to_lines(trip, ref):
        folium.PolyLine(seg, color="green", weight=7, opacity=.6).add_to(fg_ref)
    fg_clf = folium.FeatureGroup("Classifier-only route (orange)", show=False)
    for seg in arcs_to_lines(trip, clf_route):
        folium.PolyLine(seg, color="orange", weight=4).add_to(fg_clf)
    fg_hmm = folium.FeatureGroup("HMM predicted route (red)")
    folium.PolyLine(out["response"]["route_latlon"], color="red", weight=3).add_to(fg_hmm)
    fg_gps = folium.FeatureGroup("GPS observations (blue)")
    for i, (lon, lat) in enumerate(trip.lonlat):
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, weight=0, fill_opacity=.8,
                            tooltip=f"obs {i} conf {conf[i]:.2f}").add_to(fg_gps)
    fg_low = folium.FeatureGroup("Low-confidence points (magenta)")
    for lon, lat in trip.lonlat[low]:
        folium.CircleMarker([lat, lon], radius=5, color="magenta", fill=False, weight=2).add_to(fg_low)
    for fg in (fg_net, fg_ref, fg_clf, fg_hmm, fg_gps, fg_low):
        fg.add_to(m)
    folium.Marker([trip.lonlat[0, 1], trip.lonlat[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([trip.lonlat[-1, 1], trip.lonlat[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    legend = (f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;padding:8px 10px;"
              f"border:1px solid #999;font:12px sans-serif'><b>Team 111 · trip {tid}</b><br>"
              "<span style='color:blue'>&#9679;</span> GPS &nbsp; <span style='color:green'>&#9644;</span> reference &nbsp;"
              "<span style='color:orange'>&#9644;</span> classifier-only &nbsp; <span style='color:red'>&#9644;</span> HMM"
              f"<br>{title.split(': ')[1]}</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    m.save(os.path.join(MAPS, f"trip_{tid}.html"))
    print("saved", os.path.join(MAPS, f"trip_{tid}.html"))
