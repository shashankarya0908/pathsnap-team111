"""End-to-end inference pipeline used by the API (team 111).

predict()  : track + road network -> candidates -> features -> classifier ->
             learned-transition HMM (Viterbi + forward-backward) -> stitched route.
             It never opens the .route file.
evaluate() : server-side only; loads the reference route, derives per-point
             reference arcs and scores the prediction. Reference geometry is
             not returned to the client.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .data import load_trip, load_route
from .features import build_features
from .hmm import Decoder, RoadGraph, candidates_from_frame, stitch_route
from .labels import align_route
from .metrics import route_continuity, route_set_metrics


class PathSnapPredictor:
    def __init__(self, model_dir: str):
        self.cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
        self.hmm_cfg = json.load(open(os.path.join(model_dir, "hmm_config.json")))
        self.model = joblib.load(os.path.join(model_dir, "final_model.joblib"))
        self.trans_model = joblib.load(os.path.join(model_dir, "transition_model.joblib"))
        mode = self.hmm_cfg["selected_mode"]
        self.decoder = Decoder(mode=mode, trans_model=self.trans_model,
                               unreach_penalty=self.hmm_cfg["unreach_penalty"], min_p=self.hmm_cfg["min_p"],
                               **self.hmm_cfg["modes"][mode])

    def predict(self, trip_dir: str) -> dict:
        timings = {}
        t0 = time.time(); trip = load_trip(trip_dir); G = RoadGraph(trip); timings["load_s"] = time.time() - t0
        t0 = time.time(); df = build_features(trip, **self.cfg["candidates"]); timings["features_s"] = time.time() - t0
        t0 = time.time(); df["p"] = self.model.predict_proba(df[self.cfg["features"]].values)[:, 1]
        timings["classifier_s"] = time.time() - t0
        df = df.sort_values(["pt", "dist_rank"])
        cands = candidates_from_frame(df)
        clf_arcs = np.array([c[0][int(np.argmax(c[2]))] for c in cands])
        t0 = time.time(); _, arcs, conf = self.decoder.decode(trip, cands, G); timings["hmm_s"] = time.time() - t0
        t0 = time.time(); route, filled, breaks = stitch_route(G, arcs); timings["stitch_s"] = time.time() - t0
        nl = trip.node_lonlat
        route_latlon = []
        for a in route:  # polyline of the stitched route as [lat, lon]
            u, v = trip.arcs[a]
            if not route_latlon:
                route_latlon.append([float(nl[u, 1]), float(nl[u, 0])])
            route_latlon.append([float(nl[v, 1]), float(nl[v, 0])])
        return dict(
            trip=trip, G=G, point_arcs=arcs, clf_arcs=clf_arcs, route=route,
            response=dict(
                trip_id=trip.trip_id,
                n_points=int(len(arcs)), n_raw_points=int(trip.n_raw_points),
                gps=[[float(la), float(lo)] for lo, la in trip.lonlat],
                point_arc=arcs.astype(int).tolist(),
                confidence=np.round(conf, 4).tolist(),
                route_arcs=[int(a) for a in route],
                route_latlon=route_latlon,
                gaps_filled=int(filled), unbridgeable_breaks=int(breaks),
                mean_confidence=float(conf.mean()),
                low_confidence_share=float((conf < 0.6).mean()),
                timings=timings,
            ),
        )

    @staticmethod
    def evaluate(trip_dir: str, pred: dict) -> dict:
        """Server-side scoring against the reference route (never sent to the model)."""
        route = load_route(trip_dir)
        if route is None:
            return {"available": False}
        trip = pred["trip"]
        lab, _, ldist = align_route(trip, route)
        arcs, clf = pred["point_arcs"], pred["clf_arcs"]
        out = dict(available=True,
                   point_accuracy=float((arcs == lab).mean()),
                   classifier_only_point_accuracy=float((clf == lab).mean()),
                   route_continuity=route_continuity(arcs, trip.arcs),
                   classifier_only_route_continuity=route_continuity(clf, trip.arcs),
                   reference_label_median_dist_m=float(np.median(ldist)))
        out.update(route_set_metrics(pred["route"], route, trip.arc_len))
        return out
