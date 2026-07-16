"""
app.py
======

A small FastAPI service that exposes the pKa prediction pipeline over HTTP so a
frontend (e.g. the Vercel dashboard) can request predictions.

Endpoints
---------
GET  /            -> health check / metadata
GET  /health      -> simple liveness probe
POST /predict     -> predict pKa for a batch of SMILES

Run locally:
    uvicorn app:app --reload --port 8000

Environment variables (optional):
    ACIDIC_MODEL_PATH  path to a joblib-pickled acidic-pKa regressor
    BASIC_MODEL_PATH   path to a joblib-pickled basic-pKa regressor
    ALLOWED_ORIGINS    comma-separated CORS origins (default "*")
If the model paths are unset, the service uses the rule-based fallback so it
runs out of the box.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pka_predictor import FeatureExtractor, PkaPipeline, pKaPredictor

# --------------------------------------------------------------------------- #
#  Build the pipeline once at startup (models loaded here if provided).
# --------------------------------------------------------------------------- #
_extractor = FeatureExtractor()
_acidic_path = os.getenv("ACIDIC_MODEL_PATH")
_basic_path = os.getenv("BASIC_MODEL_PATH")

if _acidic_path or _basic_path:
    _predictor = pKaPredictor.load(_extractor, _acidic_path, _basic_path)
    _model_mode = "trained"
else:
    _predictor = pKaPredictor(_extractor)  # rule-based fallback
    _model_mode = "rule_based_fallback"

_pipeline = PkaPipeline(extractor=_extractor, predictor=_predictor)

# --------------------------------------------------------------------------- #
#  App + CORS
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="pKa Predictor API",
    version="1.0.0",
    description="Estimate acidic/basic pKa values for small molecules from SMILES.",
)

_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else
    [o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Request / response schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Accepts either a list of SMILES (with optional ids) or raw text.

    Examples
    --------
    {"smiles": ["CC(=O)Oc1ccccc1C(=O)O"], "ids": ["aspirin"]}
    {"text": "aspirin, CC(=O)Oc1ccccc1C(=O)O\\nlidocaine, CCN(CC)CC(=O)Nc1c(C)cccc1C"}
    """

    smiles: Optional[List[str]] = Field(default=None, description="List of SMILES strings")
    ids: Optional[List[str]] = Field(default=None, description="Optional molecule IDs")
    text: Optional[str] = Field(default=None, description="Newline-separated 'id,SMILES' or bare SMILES lines")


import math

import numpy as np


def _json_safe(record: dict) -> dict:
    """Convert NaN and numpy scalars in a record to plain JSON-safe values."""
    clean = {}
    for key, value in record.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        clean[key] = value
    return clean


def _parse_text(text: str):
    """Parse newline-separated 'id,SMILES' or bare-SMILES lines."""
    ids: List[str] = []
    smiles: List[str] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        if "," in line:
            mol_id, smi = line.split(",", 1)
            ids.append(mol_id.strip() or f"mol_{i}")
            smiles.append(smi.strip())
        else:
            ids.append(f"mol_{i}")
            smiles.append(line)
    return smiles, ids


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "service": "pka-predictor",
        "version": "1.0.0",
        "model_mode": _model_mode,
        "endpoints": {"predict": "POST /predict", "health": "GET /health"},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict pKa for a batch of molecules; returns one row per ionisable centre."""
    if req.text:
        smiles, ids = _parse_text(req.text)
    elif req.smiles:
        smiles = req.smiles
        ids = req.ids
        if ids is not None and len(ids) != len(smiles):
            raise HTTPException(400, "ids and smiles must be the same length")
    else:
        raise HTTPException(400, "Provide either 'smiles' (list) or 'text' (string)")

    if not smiles:
        raise HTTPException(400, "No SMILES found in request")
    if len(smiles) > 1000:
        raise HTTPException(413, "Batch too large; limit is 1000 molecules per request")

    df = _pipeline.run(smiles, ids)
    # DataFrame -> JSON-safe records. Float columns keep NaN even after
    # .where(..., None), so sanitize each value explicitly.
    records = [_json_safe(rec) for rec in df.to_dict(orient="records")]
    return {
        "model_mode": _model_mode,
        "count": len(records),
        "results": records,
    }
