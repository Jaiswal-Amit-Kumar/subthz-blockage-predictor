"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KPMFrame(BaseModel):
    """One KPM indication frame from the E2SM-KPM stream."""

    timestamp_ms: float = Field(..., description="Monotonic timestamp in ms")

    # RLC / MAC metrics
    rlc_delay_dl: float = Field(0.0, description="RLC SDU delay, DL (ms)")
    rlc_drop_rate: float = Field(0.0, description="RLC packet drop rate")
    harq_retx_ratio: float = Field(0.0, description="HARQ retransmission ratio")
    prb_utilization: float = Field(0.0, description="PRB utilization [0,1]")
    ue_buffer_occupancy: float = Field(0.0, description="UE buffer occupancy (bytes)")

    # PHY-layer metrics (available in OAI KPM or injected by bridge)
    rssi_dbm: float = Field(-80.0, description="Received signal strength (dBm)")
    sinr_db: float = Field(22.0, description="SINR (dB)")
    doppler_hz: float = Field(0.0, description="Doppler shift (Hz)")


class PredictionResponse(BaseModel):
    """Prediction result returned to the caller."""

    timestamp_ms: float
    blockage_probability: float
    blockage_predicted: bool
    inference_latency_ms: float
    trigger_action: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_ready: bool
    buffer_fill: int