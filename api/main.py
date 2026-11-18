"""PathSnap FastAPI backend (Laptop 1) - team 111.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder with the evaluator trips (default data/raw/evaluator_test)
      PATHSNAP_MODEL_DIR folder with model artifacts (default models)

Endpoints
  GET  /health            -> status + loaded model info
  GET  /trips             -> the evaluation trip ids available on this server
  POST /predict           -> body {"trip_id": "00000012"}; runs the full pipeline
  GET  /predict/{trip_id} -> same, convenient for a browser / curl
The reference route is read only inside PathSnapPredictor.evaluate() and only
aggregate metrics are returned; no reference geometry or labels leave the server.
"""
import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.data import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "raw", "evaluator_test"))
MODEL_DIR = os.environ.get("PATHSNAP_MODEL_DIR", os.path.join(ROOT, "models"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API - Team 111", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PathSnapPredictor(MODEL_DIR)
log.info("Model loaded from %s | eval dir %s | trips %s", MODEL_DIR, EVAL_DIR, list_trips(EVAL_DIR))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("REQUEST %s %s from %s -> %s in %.2fs", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, time.time() - t0)
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "team": "111", "model": predictor.cfg["model"],
            "decoder": predictor.hmm_cfg["selected_mode"], "eval_dir": EVAL_DIR}


@app.get("/trips")
def trips():
    return {"trips": list_trips(EVAL_DIR)}


def _run(trip_id: str):
    trip_id = os.path.basename(trip_id.strip())
    if trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {trip_id}")
    tdir = os.path.join(EVAL_DIR, trip_id)
    t0 = time.time()
    pred = predictor.predict(tdir)
    infer_s = time.time() - t0
    metrics = predictor.evaluate(tdir, pred)
    resp = pred["response"]
    resp["inference_time_s"] = infer_s
    resp["metrics"] = metrics
    log.info("PREDICT trip=%s points=%d route_arcs=%d acc=%.4f clf_only=%.4f route_f1=%.4f infer=%.2fs",
             trip_id, resp["n_points"], len(resp["route_arcs"]), metrics.get("point_accuracy", float("nan")),
             metrics.get("classifier_only_point_accuracy", float("nan")), metrics.get("route_f1", float("nan")), infer_s)
    return resp


@app.post("/predict")
def predict(req: PredictRequest):
    return _run(req.trip_id)


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str):
    return _run(trip_id)
