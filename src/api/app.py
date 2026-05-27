"""FastAPI app for CPU-capable pIC50 prediction and compound assessment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.demo_cpu import CPUDemoPIC50Model, CPUDemoPredictorAdapter
from src.pipeline.compound_assessment import CompoundAssessmentPipeline

DEFAULT_MODEL_PATH = Path("models/demo_cpu_pic50_model.json")


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    target: str = "CHEMBL238"


class AssessRequest(PredictRequest):
    include_3d: bool = True
    include_reactions: bool = True
    include_image: bool = False


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create an API app backed by the CPU demo model."""
    resolved_model_path = Path(
        model_path or os.environ.get("PIC50_MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    model = CPUDemoPIC50Model.from_file(resolved_model_path)

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
        return {
            "status": "healthy",
            "model": {
                "path": str(app.state.model_path),
                "model_version": model.model_version,
                "model_kind": model.model_kind,
                "device": model.device,
                "targets": sorted(model.targets.keys()),
            },
            "context_of_use": model.context_of_use,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            return model.predict(request.smiles, request.target).to_dict()
        except ValueError as exc:
            raise _http_error(exc) from exc

    @app.post("/assess")
    def assess(request: AssessRequest) -> dict[str, Any]:
        adapter = CPUDemoPredictorAdapter(model, request.target)
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
        return payload

    return app


def _http_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if message.startswith("Unsupported target"):
        return HTTPException(status_code=400, detail=message)
    return HTTPException(status_code=422, detail=message)


app = create_app()
