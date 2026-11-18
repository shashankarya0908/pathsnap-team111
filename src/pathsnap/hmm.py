"""HMM / Viterbi route reconstruction (team 111).

Hidden state at observation t = one candidate road arc of that observation
(arc id + projection fraction along it), so the candidate -> arc mapping is
preserved by construction.

Emission  : log P(obs | arc) = w_e * log( p_clf(arc) / sum_c p_clf(c) )
            where p_clf is the selected classifier's probability.
Transition: two variants
  * "classic"  : Newson-Krumm style, log P = -|d_route - d_gps| / beta_eff,
                 beta_eff = beta * max(1, dt)  (more slack after time gaps)
  * "hybrid"   : classic term + w_learn * learned term (learned correction
                 for turn angle / U-turns / backward motion on top of the
                 physical distance-consistency prior).
  * "learned"  : team-111 variant. A logistic model trained on reference
                 transitions of the training trips scores each pair using
                 [|d_route-d_gps|, ratio, same arc, U-turn, backwards on arc,
                 turn angle, dt, unreachable]. Its log-probability is the
                 transition score.
Unreachable pairs (no path within the search cutoff) get a fixed penalty
instead of -inf, so decoding never dies: a disconnected step behaves like a
soft restart of the chain.
After Viterbi we also run forward-backward to give a posterior confidence per
observation, and "stitch" the decoded arcs into one continuous route by
inserting shortest paths between non-adjacent consecutive arcs.
"""
from __future__ import annotations

import heapq
import numpy as np

from .data import Trip, angle_diff

TRANS_FEATURES = ["log_diff", "ratio", "same", "uturn", "backward", "turn", "log_dt", "unreach"]


class RoadGraph:
    def __init__(self, trip: Trip):
        self.trip = trip
        V = len(trip.node_xy)
        self.adj = [[] for _ in range(V)]
        for e, (u, v) in enumerate(trip.arcs.tolist()):
            self.adj[u].append((v, float(trip.arc_len[e]), e))
        self.arc_of = {(int(u), int(v)): e for e, (u, v) in enumerate(trip.arcs.tolist())}
        self.cache = {}

    def dists_from(self, src: int, cutoff: float):
        key = (src, cutoff)
        if key in self.cache:
            return self.cache[key]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18) or d > cutoff:
                continue
            for v, w, _ in self.adj[u]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        if cutoff <= 1500:
            self.cache[key] = dist
        return dist

    def shortest_arc_path(self, src: int, dst: int, cutoff: float = 5000.0):
        """Arc ids of the shortest node path src -> dst (empty if none)."""
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, 1e18) or d > cutoff:
                continue
            for v, w, e in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = (u, e)
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None
        path, n = [], dst
        while n != src:
            u, e = prev[n]; path.append(e); n = u
        return path[::-1]


def cutoff_for(d_gc: float) -> float:
    return 1000.0 if d_gc < 300 else float(3 * d_gc + 500)


def transition_features(G: RoadGraph, A, TA, B, TB, d_gc, dt):
    """Feature tensor (K1, K2, F) for all candidate pairs between two observations."""
    trip = G.trip
    K1, K2 = len(A), len(B)
    F = np.zeros((K1, K2, len(TRANS_FEATURES)))
    cut = cutoff_for(d_gc)
    for i in range(K1):
        a, ta = A[i], TA[i]
        la = trip.arc_len[a]; to_a = trip.arcs[a, 1]
        dist = None
        for j in range(K2):
            b, tb = B[j], TB[j]
            backward = 0.0
            if a == b:
                dr = (tb - ta) * la
                if dr < 0:
                    backward = 1.0; dr = -dr
                unreach = 0.0
            else:
                if dist is None:
                    dist = G.dists_from(int(to_a), cut)
                sp = dist.get(int(trip.arcs[b, 0]))
                if sp is None:
                    dr, unreach = cut, 1.0
                else:
                    dr, unreach = (1 - ta) * la + sp + tb * trip.arc_len[b], 0.0
            diff = abs(dr - d_gc)
            F[i, j] = [np.log1p(diff), diff / (d_gc + 5.0), float(a == b),
                       float(trip.arcs[a, 0] == trip.arcs[b, 1] and trip.arcs[a, 1] == trip.arcs[b, 0]),
                       backward, angle_diff(trip.arc_bearing[a], trip.arc_bearing[b]) / 180.0,
                       np.log1p(dt), unreach]
    return F


class Decoder:
    def __init__(self, mode="learned", beta=5.0, w_emit=1.0, w_trans=1.0, w_learn=0.5, unreach_penalty=-15.0,
                 trans_model=None, min_p=1e-4):
        self.mode, self.beta, self.w_emit, self.w_trans, self.w_learn = mode, beta, w_emit, w_trans, w_learn
        self.unreach_penalty, self.trans_model, self.min_p = unreach_penalty, trans_model, min_p

    def trans_logp(self, F, d_gc, dt):
        classic = learned = 0.0
        if self.mode in ("classic", "hybrid"):
            diff = np.expm1(F[..., 0])
            classic = -diff / (self.beta * max(1.0, dt))
        if self.mode in ("learned", "hybrid"):
            K1, K2, nf = F.shape
            z = self.trans_model.decision_function(F.reshape(-1, nf)).reshape(K1, K2)
            learned = -np.logaddexp(0, -z)  # log sigmoid
        lp = classic + (self.w_learn if self.mode == "hybrid" else 1.0) * learned
        lp = np.where(F[..., 7] > 0, self.unreach_penalty, lp)
        return self.w_trans * lp

    @staticmethod
    def precompute(trip: Trip, cands, G: RoadGraph):
        """Transition feature tensors for every step (independent of decoder weights)."""
        out = [None]
        for t in range(1, len(cands)):
            A, TA, _ = cands[t - 1]; B, TB, _ = cands[t]
            d_gc = float(np.hypot(*(trip.xy[t] - trip.xy[t - 1])))
            dt = float(trip.t[t] - trip.t[t - 1])
            out.append((transition_features(G, A, TA, B, TB, d_gc, dt), d_gc, dt))
        return out

    def decode(self, trip: Trip, cands, G: RoadGraph = None, pre=None):
        """cands: list over observations of (arcs, t_proj, p_clf) arrays.
        Returns (state index per obs, arc per obs, posterior confidence per obs)."""
        G = G or RoadGraph(trip)
        pre = pre or self.precompute(trip, cands, G)
        N = len(cands)
        emis = []
        for arcs, tp, p in cands:
            p = np.maximum(p, self.min_p); emis.append(self.w_emit * np.log(p / p.sum()))
        delta = [emis[0]]; back = [None]; trans = [None]
        for t in range(1, N):
            F, d_gc, dt = pre[t]
            T = self.trans_logp(F, d_gc, dt)
            trans.append(T)
            s = delta[-1][:, None] + T
            back.append(s.argmax(0))
            d = s.max(0) + emis[t]
            delta.append(d - d.max())  # normalise to avoid drift
        states = np.zeros(N, dtype=int); states[-1] = int(delta[-1].argmax())
        for t in range(N - 1, 0, -1):
            states[t - 1] = back[t][states[t]]
        # forward-backward posteriors (log space)
        la = [emis[0] - np.logaddexp.reduce(emis[0])]
        for t in range(1, N):
            x = np.logaddexp.reduce(la[-1][:, None] + trans[t], axis=0) + emis[t]
            la.append(x - np.logaddexp.reduce(x))
        lb = [np.zeros(len(cands[-1][0]))]
        for t in range(N - 1, 0, -1):
            x = np.logaddexp.reduce(trans[t] + (emis[t] + lb[0])[None, :], axis=1)
            lb.insert(0, x - np.logaddexp.reduce(x))
        conf = np.zeros(N)
        for t in range(N):
            post = la[t] + lb[t]; post = np.exp(post - np.logaddexp.reduce(post))
            conf[t] = post[states[t]]
        arcs = np.array([cands[t][0][states[t]] for t in range(N)])
        return states, arcs, conf


def stitch_route(G: RoadGraph, point_arcs):
    """Collapse per-observation arcs and fill gaps with shortest paths.
    Returns (route arc list, number of gaps filled, number of unbridgeable breaks)."""
    trip = G.trip
    seq = []
    for a in point_arcs:
        a = int(a)
        if not seq or seq[-1] != a:
            seq.append(a)
    route, filled, breaks = [seq[0]] if seq else [], 0, 0
    for a, b in zip(seq[:-1], seq[1:]):
        if trip.arcs[a, 1] != trip.arcs[b, 0]:
            path = G.shortest_arc_path(int(trip.arcs[a, 1]), int(trip.arcs[b, 0]))
            if path is None:
                breaks += 1
            else:
                route.extend(path); filled += 1
        route.append(b)
    return route, filled, breaks


def candidates_from_frame(df, score_col="p"):
    """Group a candidate frame (sorted by pt) into the decoder's per-observation list."""
    out = []
    for _, g in df.groupby("pt", sort=True):
        out.append((g["arc"].values.astype(int), g["t_proj"].values, g[score_col].values))
    return out
