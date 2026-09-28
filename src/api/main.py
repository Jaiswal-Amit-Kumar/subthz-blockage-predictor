"""
FastAPI microservice exposing the blockage predictor over HTTP.

Designed for integration with FlexRIC via a lightweight telemetry bridge.
The caller POSTs one KPM frame per request; the predictor maintains its own
sliding window state internally.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.model.predict import BlockagePredictor
from src.api.schemas import KPMFrame, PredictionResponse, HealthResponse

predictor = BlockagePredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    predictor.load_model()
    yield


app = FastAPI(
    title="Sub-THz Blockage Predictor",
    version="1.0.0",
    description=(
        "Lightweight microservice for predictive sub-THz line-of-sight "
        "blockage detection. Designed for O-RAN / FlexRIC testbeds."
    ),
    lifespan=lifespan,
)

@app.get("/")
def read_root():
    return {"message": "Sub-THz Blockage Predictor API is running!"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness probe for orchestration and monitoring."""
    return HealthResponse(
        status="ok",
        model_ready=predictor.is_ready,
        buffer_fill=predictor.buffer_fill,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(frame: KPMFrame) -> PredictionResponse:
    """
    Ingest one KPM frame and return a blockage prediction.

    The caller should invoke this endpoint for every KPM indication received
    from the FlexRIC E42 stream.
    """
    result = predictor.update(frame.model_dump())
    return PredictionResponse(
        timestamp_ms=result.timestamp_ms,
        blockage_probability=result.blockage_probability,
        blockage_predicted=result.blockage_predicted,
        inference_latency_ms=result.inference_latency_ms,
        trigger_action=result.trigger_action,
    )