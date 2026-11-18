"""Command-line client for the PathSnap API (handy for testing from Laptop 2's terminal).
Usage: python scripts/api_client.py http://<laptop1-ip>:8000 [trip_id]
"""
import sys, json, requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
print("health:", requests.get(base + "/health", timeout=10).json())
trips = requests.get(base + "/trips", timeout=10).json()["trips"]
print("trips:", trips)
for tid in (sys.argv[2:] or trips):
    r = requests.post(base + "/predict", json={"trip_id": tid}, timeout=600).json()
    print(tid, json.dumps({"n_points": r["n_points"], "route_arcs": len(r["route_arcs"]),
                           "inference_time_s": round(r["inference_time_s"], 2),
                           **{k: round(v, 4) if isinstance(v, float) else v for k, v in r["metrics"].items()}}))
