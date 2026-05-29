"""FastAPI app for CPU-capable pIC50 prediction and compound assessment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.demo_cpu import (
    CPUDemoEndpointModel,
    CPUDemoEndpointPredictorAdapter,
    CPUDemoPIC50Model,
    CPUDemoPredictorAdapter,
)
from src.pipeline.compound_assessment import CompoundAssessmentPipeline

DEFAULT_MODEL_PATH = Path("models/demo_cpu_pic50_model.json")


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    target: str = "CHEMBL238"
    endpoint: str | None = None
    endpoints: list[str] | None = None


class AssessRequest(PredictRequest):
    include_3d: bool = True
    include_reactions: bool = True
    include_image: bool = False


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create an API app backed by the CPU demo model."""
    resolved_model_path = Path(
        model_path or os.environ.get("PIC50_MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    model_payload = json.loads(resolved_model_path.read_text(encoding="utf-8"))
    is_endpoint_model = "endpoints" in model_payload
    model = (
        CPUDemoEndpointModel(model_payload)
        if is_endpoint_model
        else CPUDemoPIC50Model(model_payload)
    )

    app = FastAPI(
        title="Multi-Target pIC50 Predictor",
        version="2.1.0",
        description=(
            "CPU-capable research triage API. Not for clinical, regulatory, "
            "manufacturing, or patient-care decisions."
        ),
    )
    app.state.model = model
    app.state.model_path = resolved_model_path

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_summary: dict[str, Any] = {
            "path": str(app.state.model_path),
            "model_version": model.model_version,
            "model_kind": model.model_kind,
            "device": model.device,
        }
        if is_endpoint_model:
            model_summary["endpoints"] = sorted(model.endpoints.keys())
            model_summary["targets_by_endpoint"] = {
                endpoint: sorted(payload["targets"].keys())
                for endpoint, payload in model.endpoints.items()
            }
        else:
            model_summary["targets"] = sorted(model.targets.keys())
        return {
            "status": "healthy",
            "model": model_summary,
            "context_of_use": model.context_of_use,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            if is_endpoint_model:
                endpoints = request.endpoints or [request.endpoint or "pIC50"]
                predictions = {
                    endpoint: model.predict(request.smiles, request.target, endpoint).to_dict()
                    for endpoint in endpoints
                }
                return {
                    "smiles": request.smiles,
                    "target": request.target,
                    "predictions": predictions,
                    "model_version": model.model_version,
                    "model_kind": model.model_kind,
                    "device": model.device,
                }
            if request.endpoint and request.endpoint != "pIC50":
                raise ValueError("Legacy pIC50 model only supports endpoint pIC50")
            return model.predict(request.smiles, request.target).to_dict()
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/assess")
    def assess(request: AssessRequest) -> dict[str, Any]:
        adapter = (
            CPUDemoEndpointPredictorAdapter(model, request.target)
            if is_endpoint_model
            else CPUDemoPredictorAdapter(model, request.target)
        )
        pipeline = CompoundAssessmentPipeline(
            predictor=adapter,
            target=request.target,
            include_coordinates=False,
        )
        try:
            result = pipeline.assess(
                request.smiles,
                include_3d=request.include_3d,
                include_reactions=request.include_reactions,
                include_image=request.include_image,
            )
        except ValueError as exc:
            raise _http_error(exc) from exc

        payload = result.to_dict()
        if adapter.last_result is not None:
            payload["applicability_domain"] = adapter.last_result.applicability_domain
            payload["model"] = {
                "model_version": adapter.last_result.model_version,
                "model_kind": adapter.last_result.model_kind,
                "device": adapter.last_result.device,
            }
            if is_endpoint_model:
                payload["model"]["endpoint"] = adapter.last_result.endpoint
                payload["endpoint_prediction"] = adapter.last_result.endpoint_prediction
        return payload

    return app


def _http_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if message.startswith("Unsupported target"):
        return HTTPException(status_code=400, detail=message)
    return HTTPException(status_code=422, detail=message)


app = create_app()
