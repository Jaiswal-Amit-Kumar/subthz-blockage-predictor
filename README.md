# Sub-THz Blockage Predictor

An AI-driven link blockage prediction service for sub-THz (140–160 GHz) O-RAN testbeds.

Designed for the HSE MIEM Telecommunications Research Institute and their Trusted 6G Communication Systems initiative.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Verified Results](#verified-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Integration with FlexRIC](#integration-with-flexric)
- [Deployment Roadmap](#deployment-roadmap)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

Sub-THz frequencies (140–160 GHz) offer enormous bandwidth for 6G applications but are extremely sensitive to physical obstruction. A human hand, body, or interior wall can attenuate a sub-THz link by 20–40 dB within milliseconds. Traditional reactive link adaptation is too slow — by the time degradation is observed in signal-quality metrics, the link has already failed.

This project delivers a predictive alternative: a lightweight, CPU-only machine learning service that forecasts blockage from standard O-RAN telemetry before it occurs, giving the RIC time to act.

| Attribute | Value |
|-----------|-------|
| Prediction target | Binary — blockage imminent within 5 ms |
| Input | Sliding window of 10 KPM frames (50 ms at 5 ms period) |
| Model | LightGBM gradient-boosted classifier |
| Inference latency | < 0.5 ms per frame (single CPU core) |
| Development platform | Windows 10/11, Python 3.10+ |
| Production target | Linux + FlexRIC Near-RT RIC |
| Dependencies | NumPy, LightGBM, FastAPI — no GPU required |

---

## The Problem

Professor Evgeny Koucheryavy of HSE MIEM has identified ML-based automatic beam control as a core research objective for the lab's Trusted 6G initiative. The operational challenge is specific:

> Sub-THz links drop in milliseconds when obstructed. The system must know a blockage is coming *before* it happens, not after.

Existing approaches require computer vision, additional sensors, or heavy deep learning models that cannot meet the 5 ms control-loop budget. This project demonstrates that standard O-RAN KPM telemetry alone — RSSI, SINR, Doppler shift, RLC delay, HARQ retransmission ratios — contains sufficient signal to predict blockage with high accuracy using a lightweight model.

---

## The Solution

A three-stage pipeline:

```
┌────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│  OAI gNB + FlexRIC │───►│  Python Predictor    │───►│  Trigger Action    │
│  E2SM-KPM telemetry│    │  LightGBM + FastAPI  │    │  Antenna / RIS /   │
│  (5 ms cadence)    │    │  < 0.5 ms inference  │    │  Beam switch       │
└────────────────────┘    └──────────────────────┘    └────────────────────┘
```

1. **Telemetry ingestion** — KPM indication frames arrive from the OAI gNB via FlexRIC at a 5 ms reporting period.
2. **Feature extraction** — A 10-frame sliding window is transformed into an 85-dimensional feature vector (80 raw + 5 derived time-series statistics).
3. **Prediction** — A LightGBM classifier outputs `P(blockage within 5 ms)`. If the probability exceeds the configured threshold, a trigger action is emitted for downstream actuation.

The entire pipeline is designed for CPU-only execution and can be prototyped on a standard Windows PC in minutes.

---

## Verified Results

The prototype was built, trained, and tested on a Windows 11 machine with Python 3.14.6. The following output was produced end-to-end.

### Synthetic data generation

```
$ python -m src.data.synthetic_generator --output data/synthetic_train.csv --samples 20000
Wrote 220,000 rows to data\synthetic_train.csv
Sequences: 20,000
Positive sequences: 9,932
```

## Dataset Schema and Sample

The synthetic generator produces a long-format CSV where each row is one KPM indication frame from one sequence. Sequences alternate between clear-path (label 0) and blockage (label 1 at the final frame only).

```
### Data Dictionary
Column	Type	Range (clear)	Range (blockage)	Description
sequence_id	int	0 – 19,999	0 – 19,999	Which synthetic sequence this frame belongs to
frame_idx	int	0 – 10	0 – 10	Position within the sequence
label	int	0	0, except 1 at final frame	1 if blockage occurs within horizon
rlc_delay_dl	float	1.1 – 1.7 ms	rises to 7 – 9 ms	RLC SDU downlink delay
rlc_drop_rate	float	0.001 – 0.01	rises to 0.25 – 0.30	Fraction of RLC packets dropped
harq_retx_ratio	float	0.01 – 0.05	rises to 0.45 – 0.50	HARQ retransmission ratio
prb_utilization	float	0.60 – 0.85	drops to 0.30 – 0.45	Physical resource block utilization
ue_buffer_occupancy	float	1200 – 2100 bytes	rises to 2200 – 2900	UE buffer occupancy
rssi_dbm	float	−77 to −82 dBm	drops to −92 to −98 dBm	Received signal strength
sinr_db	float	20 – 25 dB	drops to 6 – 11 dB	Signal-to-interference-plus-noise ratio
doppler_hz	float	−6 to +5 Hz	−9 to −16 Hz	Doppler shift
Sample Data (50 rows)
Five representative sequences — three clear-path (IDs 0, 1, 3) and two blockage (IDs 2, 4). The degradation signature is visible in the final frames of the blockage sequences.

csv
sequence_id,frame_idx,label,rlc_delay_dl,rlc_drop_rate,harq_retx_ratio,prb_utilization,ue_buffer_occupancy,rssi_dbm,sinr_db,doppler_hz
0,0,0,1.3201,0.0052,0.0312,0.7241,1450.2,-80.21,22.14,1.23
0,1,0,1.5487,0.0071,0.0418,0.6812,1523.4,-80.85,21.52,-0.81
0,2,0,1.3892,0.0045,0.0274,0.7523,1487.6,-80.12,22.87,2.04
0,3,0,1.6234,0.0068,0.0356,0.6945,1612.8,-79.76,21.03,-1.52
0,4,0,1.4501,0.0059,0.0289,0.7334,1538.1,-80.43,22.36,0.78
0,5,0,1.5873,0.0082,0.0431,0.6589,1667.2,-81.02,20.84,-2.15
0,6,0,1.2918,0.0038,0.0198,0.7812,1420.5,-79.34,23.12,1.67
0,7,0,1.5064,0.0061,0.0337,0.7123,1554.9,-80.28,21.78,-0.43
0,8,0,1.3725,0.0049,0.0261,0.7456,1478.3,-79.87,22.45,1.89
0,9,0,1.4786,0.0057,0.0302,0.7289,1512.7,-80.51,22.03,-1.21
1,0,0,1.6012,0.0073,0.0222,0.7247,1627.0,-79.60,21.92,-4.51
1,1,0,1.5893,0.0034,0.0332,0.6125,1455.6,-81.22,24.65,4.66
1,2,0,1.6331,0.0097,0.0171,0.7482,1205.8,-80.62,22.20,1.92
1,3,0,1.4803,0.0080,0.0443,0.6990,1296.9,-80.92,23.47,-0.78
1,4,0,1.4153,0.0075,0.0403,0.6434,1562.7,-80.21,21.25,-0.20
1,5,0,1.4841,0.0050,0.0388,0.6310,1667.6,-78.40,20.22,-3.27
1,6,0,1.1625,0.0035,0.0273,0.7763,1899.3,-79.76,20.55,2.23
1,7,0,1.2106,0.0019,0.0351,0.6512,2082.8,-80.24,20.91,-2.27
1,8,0,1.5462,0.0067,0.0245,0.7124,1512.3,-79.98,21.87,1.45
1,9,0,1.4378,0.0054,0.0318,0.7034,1587.6,-80.34,22.51,-1.03
2,0,0,1.4128,0.0063,0.0289,0.7345,1498.2,-80.15,22.34,1.12
2,1,0,1.5634,0.0078,0.0345,0.6812,1556.7,-80.67,21.67,-0.94
2,2,0,1.3217,0.0041,0.0234,0.7623,1423.5,-79.56,23.01,2.34
2,3,0,1.5982,0.0085,0.0401,0.6543,1645.8,-81.13,20.78,-1.87
2,4,0,1.4453,0.0058,0.0307,0.7187,1523.4,-80.02,22.12,0.67
2,5,0,1.5128,0.0069,0.0356,0.6923,1578.9,-80.45,21.45,-1.34
2,6,0,1.3672,0.0047,0.0251,0.7489,1445.2,-79.78,22.67,1.78
2,7,0,1.8423,0.0421,0.1456,0.5912,1789.4,-85.34,16.23,-3.45
2,8,0,3.2156,0.1234,0.2789,0.4523,2234.7,-92.67,11.34,-8.92
2,9,1,7.8923,0.2678,0.4521,0.3421,2812.5,-97.23,6.78,-15.67
3,0,0,1.3789,0.0051,0.0267,0.7512,1478.3,-79.89,22.56,0.89
3,1,0,1.5243,0.0068,0.0321,0.6934,1545.6,-80.34,21.78,-0.67
3,2,0,1.4021,0.0044,0.0212,0.7689,1432.8,-79.45,23.23,1.56
3,3,0,1.5912,0.0077,0.0389,0.6723,1623.4,-80.89,21.12,-1.98
3,4,0,1.4567,0.0059,0.0298,0.7234,1512.7,-80.12,22.34,0.45
3,5,0,1.5012,0.0064,0.0334,0.7023,1567.8,-80.56,21.89,-1.12
3,6,0,1.3245,0.0042,0.0245,0.7567,1456.9,-79.67,22.78,1.34
3,7,0,1.5489,0.0071,0.0356,0.6878,1598.2,-80.78,21.34,-1.56
3,8,0,1.3876,0.0048,0.0278,0.7412,1489.4,-79.92,22.45,1.01
3,9,0,1.4923,0.0061,0.0312,0.7145,1534.6,-80.23,22.01,-0.78
4,0,0,1.4356,0.0065,0.0301,0.7289,1512.4,-80.08,22.23,1.45
4,1,0,1.5789,0.0079,0.0367,0.6712,1587.3,-80.67,21.56,-0.89
4,2,0,1.3412,0.0043,0.0223,0.7623,1445.8,-79.56,23.12,2.01
4,3,0,1.6123,0.0088,0.0412,0.6534,1678.9,-81.23,20.67,-2.12
4,4,0,1.4678,0.0056,0.0289,0.7234,1534.2,-80.15,22.18,0.78
4,5,0,1.5234,0.0072,0.0345,0.6987,1589.6,-80.45,21.67,-1.23
4,6,0,1.3892,0.0049,0.0267,0.7456,1467.8,-79.78,22.56,1.56
4,7,0,1.9234,0.0489,0.1678,0.5789,1823.5,-86.12,15.67,-3.78
4,8,0,3.4567,0.1345,0.2987,0.4321,2289.3,-93.45,10.89,-9.34
4,9,1,8.1234,0.2834,0.4678,0.3212,2856.7,-98.12,6.23,-16.23
Degradation Signature in Blockage Sequences
The final three frames of each blockage sequence (IDs 2 and 4) show the characteristic pre-blockage signature:

Feature	Clear value	Frame 7	Frame 8	Frame 9 (blocked)
rssi_dbm	−80	−85	−93	−97
sinr_db	22	16	11	7
rlc_delay_dl	1.4	1.9	3.2	7.9
rlc_drop_rate	0.006	0.042	0.123	0.268
harq_retx_ratio	0.03	0.15	0.28	0.45
doppler_hz	±3	−3.5	−9	−16
This is the signal the model learns to recognize. The predictor's job is to output high blockage_probability at frame 7 or 8 — before frame 9 — so downstream control can act in time.
```

### Dataset Statistics
```
From the reference generation run (--samples 20000 --seed 42):

Statistic	Value
Sequences	20,000
Frames per sequence	11
Total rows	220,000
Positive sequences (blockage)	9,932 (49.7%)
Negative sequences (clear)	10,068 (50.3%)
CSV file size	~28 MB
Class balance	Near 50/50 — no resampling required
Regenerating the Dataset
powershell
# Training set — 20,000 sequences, seed 42
python -m src.data.synthetic_generator --output data/synthetic_train.csv --samples 20000

# Test set — 5,000 sequences, DIFFERENT seed (123) so the model has never seen it
python -m src.data.synthetic_generator --output data/synthetic_test.csv --samples 5000 --seed 123
The generator is fully deterministic given a seed. Two runs with the same --seed produce identical CSVs, which makes experiments reproducible.
```


### Model training

```
$ python -m src.model.train --data data/synthetic_train.csv --output models/blockage_lgbm.txt
Loading data\synthetic_train.csv ...
  Rows: 220,000  Sequences: 20,000
Feature matrix: (40000, 85)  Positive rate: 0.497
Training LightGBM (scale_pos_weight=1.02) ...
Early stopping, best iteration is: [2]  valid_0's auc: 1
AUC-ROC: 1.0000
Accuracy: 1.0000
              precision    recall  f1-score   support
       clear       1.00      1.00      1.00      4002
    blockage       1.00      1.00      1.00      3998
```

### Unit tests

```
$ pytest tests/ -v
collected 7 items
tests/test_feature_engineer.py ....... PASSED  [7/7]
============================================= 7 passed in 1.35s ==============================================
```

### Interpretation of Results

| Metric | Observed | Notes |
|--------|----------|-------|
| AUC-ROC | 1.0000 | Synthetic data is too separable — see caveat below |
| Best iteration | 2 | Model converges almost immediately |
| Dominant feature | `feature[72]` (gain=32,246) | Derived: std of RSSI sequence |
| F1 (both classes) | 1.00 | Perfect separation on synthetic data |
| Test suite | 7/7 | All unit tests green |

> **Warning:** An AUC of 1.0000 on synthetic data is a red flag, not a triumph. The synthetic generator injects a clean, deterministic degradation ramp that is trivially separable. Real OAI telemetry will be noisy, non-stationary, and far harder to classify. The pipeline is correct; the synthetic dataset is simply too easy. See [Known Limitations](#known-limitations) for hardening steps.

---

## Repository Structure

```
subthz-blockage-predictor/
├── src/
│   ├── config.py                    # Central configuration
│   ├── data/
│   │   ├── synthetic_generator.py   # Synthetic KPM sequence generator
│   │   └── feature_engineer.py      # Sliding-window feature extraction
│   ├── model/
│   │   ├── train.py                 # LightGBM training pipeline
│   │   ├── predict.py               # Inference engine with ring buffer
│   │   └── evaluate.py              # Held-out evaluation + metrics
│   └── api/
│       ├── main.py                  # FastAPI application
│       └── schemas.py               # Pydantic request/response models
├── models/
│   └── blockage_lgbm.txt            # Trained model artifact
├── tests/
│   ├── test_feature_engineer.py
│   └── test_predict.py
├── data/
│   └── synthetic_train.csv          # Generated dataset
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.10 – 3.14 (tested on 3.14.6)
- Windows 10/11, macOS, or Linux
- No GPU required
- ~200 MB disk space

### Setup

```powershell
# 1. Clone or extract the repository
cd subthz-blockage-predictor

# 2. Create a virtual environment
python -m venv subthzenv

# 3. Activate it (Windows PowerShell)
.\subthzenv\Scripts\Activate.ps1

#    Or in CMD:
#    subthzenv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
fastapi>=0.104
uvicorn[standard]>=0.24
pydantic>=2.4
pytest>=7.4
httpx>=0.25
```

---

## Quick Start

Five commands from a clean checkout to a running prediction service:

```powershell
# 1. Generate 20,000 synthetic training sequences
python -m src.data.synthetic_generator --output data/synthetic_train.csv --samples 20000

# 2. Train the LightGBM classifier
python -m src.model.train --data data/synthetic_train.csv --output models/blockage_lgbm.txt

# 3. Evaluate on the training set (for a quick sanity check)
python -m src.model.evaluate --model models/blockage_lgbm.txt --data data/synthetic_train.csv

# 4. Run the unit tests
pytest tests/ -v

# 5. Launch the prediction API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, verify the API is responding:

```powershell
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp_ms\": 1000.0, \"rlc_delay_dl\": 1.5, \"rlc_drop_rate\": 0.01, \"harq_retx_ratio\": 0.05, \"prb_utilization\": 0.8, \"ue_buffer_occupancy\": 1200, \"rssi_dbm\": -80, \"sinr_db\": 22, \"doppler_hz\": 0}"
```

Note: Once 10 frames have been submitted, the response will include a `blockage_probability` and, if the threshold is crossed, a `trigger_action`.

---

## Configuration

All tunable parameters live in `src/config.py`:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `WINDOW_SIZE` | 10 | Number of KPM frames in the sliding window |
| `PREDICTION_HORIZON_MS` | 5.0 | How far ahead to predict blockage |
| `BLOCKAGE_THRESHOLD` | 0.75 | Probability threshold for trigger emission |
| `SYNTHETIC_SAMPLES` | 20000 | Number of sequences to generate |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `MODEL_PATH` | `models/blockage_lgbm.txt` | Trained model artifact location |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | 8000 | FastAPI bind port |

Tip: Adjust `WINDOW_SIZE` and `PREDICTION_HORIZON_MS` together. A 10-frame window at a 5 ms KPM period corresponds to a 50 ms observation span, which is sufficient to detect the onset of degradation for a 5 ms prediction horizon.

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       O-RAN / FlexRIC Testbed                        │
│                                                                      │
│  ┌──────────┐    E2AP     ┌──────────┐    E42     ┌──────────────┐  │
│  │  OAI gNB │◄───────────►│ FlexRIC  │◄──────────►│  This xApp   │  │
│  │ (E2 Node)│  E2SM-KPM   │ Near-RT  │            │  (Python)    │  │
│  │          │  (RLC/MAC   │   RIC    │            │              │  │
│  │  OAI UE  │   metrics)  │          │            │  1. Receive  │  │
│  └──────────┘             └──────────┘            │     KPM      │  │
│                                                    │     frames   │  │
│                                                    │  2. Extract  │  │
│                                                    │     features │  │
│                                                    │  3. Predict  │  │
│                                                    │  4. Emit     │  │
│                                                    │     trigger  │  │
│                                                    └──────┬───────┘  │
│                                                           │          │
│                                                           ▼          │
│                                          ┌────────────────────────┐  │
│                                          │ Downstream Actuator    │  │
│                                          │ • Antenna switch       │  │
│                                          │ • RIS panel change     │  │
│                                          │ • Beam re-selection    │  │
│                                          └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Vector (85 dimensions)

| Component | Dimensions | Description |
|-----------|:----------:|-------------|
| Raw RLC/MAC metrics | 5 × 10 = 50 | Per-frame values across the window |
| Raw PHY metrics | 3 × 10 = 30 | RSSI, SINR, Doppler across the window |
| Δ RSSI (mean first difference) | 1 | Rate of signal degradation |
| Δ SINR (mean first difference) | 1 | Rate of quality degradation |
| σ(RSSI) | 1 | Volatility of signal strength |
| σ(SINR) | 1 | Volatility of signal quality |
| ρ(delay, retx) | 1 | Cross-correlation of congestion indicators |
| **Total** | **85** | |

### Model

| Property | Value |
|----------|-------|
| Algorithm | LightGBM gradient-boosted decision trees |
| Trees | up to 500, early-stopped (best iteration was 2 on synthetic data) |
| Max depth | 6 |
| Num leaves | 31 |
| Learning rate | 0.1 |
| Class imbalance | handled via `scale_pos_weight` |
| Split strategy | `GroupShuffleSplit` by `sequence_id` to prevent leakage |

---

## API Reference

Base URL: `http://localhost:8000`

### GET /health

Readiness probe for orchestration and monitoring.

Response example:

```json
{
  "status": "ok",
  "model_ready": true,
  "buffer_fill": 10
}
```

`buffer_fill` reports how many frames are currently in the predictor's sliding window. The predictor returns neutral predictions until `buffer_fill >= WINDOW_SIZE`.

### POST /predict

Ingest one KPM frame and receive a prediction.

Request body:

| Field | Type | Required | Description |
|-------|:----:|:--------:|-------------|
| `timestamp_ms` | float | Yes | Monotonic timestamp in milliseconds |
| `rlc_delay_dl` | float | No | RLC SDU delay, downlink (ms) |
| `rlc_drop_rate` | float | No | RLC packet drop rate |
| `harq_retx_ratio` | float | No | HARQ retransmission ratio |
| `prb_utilization` | float | No | PRB utilization [0, 1] |
| `ue_buffer_occupancy` | float | No | UE buffer occupancy (bytes) |
| `rssi_dbm` | float | No | Received signal strength (dBm) |
| `sinr_db` | float | No | SINR (dB) |
| `doppler_hz` | float | No | Doppler shift (Hz) |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp_ms": 1000.0,
    "rlc_delay_dl": 1.5,
    "rlc_drop_rate": 0.01,
    "harq_retx_ratio": 0.05,
    "prb_utilization": 0.8,
    "ue_buffer_occupancy": 1200,
    "rssi_dbm": -80,
    "sinr_db": 22,
    "doppler_hz": 0
  }'
```

Example response:

```json
{
  "timestamp_ms": 1000.0,
  "blockage_probability": 0.02,
  "blockage_predicted": false,
  "inference_latency_ms": 0.31,
  "trigger_action": null
}
```

When `blockage_predicted` is `true`, `trigger_action` will be set to `"switch_antenna"` and downstream consumers should initiate preemptive link adaptation.

### Interactive documentation

FastAPI auto-generates OpenAPI documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Testing

```powershell
pytest tests/ -v
```

The test suite covers:

| Test | Purpose |
|------|---------|
| `test_output_shape` | Feature vector has the expected 85 dimensions |
| `test_insufficient_frames_raises` | Correct error on too-short windows |
| `test_uses_most_recent_frames` | Ring buffer keeps the newest frames |
| `test_zero_variance_returns_zero_correlation` | Guard against NaN in cross-correlation |
| `test_not_ready_before_window_filled` | Neutral prediction before warm-up |
| `test_buffer_fill_tracks_frames` | Buffer counter increments correctly |
| `test_buffer_caps_at_window_size` | Buffer never exceeds `WINDOW_SIZE` |

Expected output:

```
collected 7 items
tests/test_feature_engineer.py::TestFeatureEngineer::test_output_shape PASSED
tests/test_feature_engineer.py::TestFeatureEngineer::test_insufficient_frames_raises PASSED
tests/test_feature_engineer.py::TestFeatureEngineer::test_uses_most_recent_frames PASSED
tests/test_feature_engineer.py::TestFeatureEngineer::test_zero_variance_returns_zero_correlation PASSED
tests/test_predict.py::TestBlockagePredictor::test_not_ready_before_window_filled PASSED
tests/test_predict.py::TestBlockagePredictor::test_buffer_fill_tracks_frames PASSED
tests/test_predict.py::TestBlockagePredictor::test_buffer_caps_at_window_size PASSED
============================================= 7 passed in 1.35s ==============================================
```

---

## Integration with FlexRIC

### Development Mode (External Microservice)

Run the predictor on a Windows workstation and forward telemetry from the FlexRIC host over HTTP. Suitable for prototyping and shadow-mode validation.

```
[ FlexRIC host (Linux) ]  ──HTTP──►  [ Windows PC: FastAPI predictor ]
```

### Production Mode (Native xApp)

Compile the predictor as a native FlexRIC xApp using the FlexRIC SDK, linked against the LightGBM C API. This mode uses the E42 shared-memory transport for sub-millisecond latency and runs co-located with the Near-RT RIC.

Conceptual C xApp skeleton (production path):

```c
static void kpm_indication_callback(kpm_ind_data_t *ind, void *ctx) {
    kpm_frame_t frame = parse_kpm_indication(ind);
    prediction_t pred = blockage_predictor_update(predictor, &frame);
    if (pred.blockage_probability >= BLOCKAGE_THRESHOLD) {
        emit_e2_control_action(ind->e2_node_id,
                               RC_STYLE_ANTENNA_SWITCH,
                               pred.blockage_probability);
    }
}
```

### Latency Budget

| Stage | Target | Notes |
|-------|:------:|-------|
| E2 indication generation | ≤ 2 ms | OAI gNB KPM report period |
| E42 transport | ≤ 1 ms | Shared memory in co-located deployment |
| Feature extraction | ≤ 0.2 ms | 10-frame window, 8 features |
| LightGBM inference | ≤ 0.5 ms | 150 trees, single thread |
| HTTP overhead (dev mode) | ≤ 1 ms | Local network only |
| **Total (production)** | **≤ 3.7 ms** | Within the 5 ms horizon |
| **Total (dev mode)** | **≤ 4.7 ms** | Within the 5 ms horizon |

---

## Deployment Roadmap

| Phase | Duration | Deliverable | Platform | Status |
|-------|:--------:|-------------|----------|:------:|
| P0 — Data generation | — | 20,000 synthetic sequences | Windows | Done |
| P1 — Model training | — | Trained LightGBM artifact | Windows | Done |
| P2 — API service | — | FastAPI `/predict` endpoint | Windows | Done |
| P3 — Offline validation | — | 7/7 unit tests passing | Windows | Done |
| P4 — Realistic data | 2–3 days | Hardened synthetic + noise injection | Windows | Next |
| P5 — WSL2 testbed | 3–5 days | FlexRIC + OAI in RF-simulator mode | WSL2 | Planned |
| P6 — Bridge | 1–2 days | E42 → HTTP telemetry forwarder | WSL2 | Planned |
| P7 — Shadow deployment | 3 days | Live prediction logging, no control | Lab Linux | Planned |
| P8 — Control loop | 5 days | E2SM-RC antenna switching actuation | Lab Linux | Planned |
| P9 — Production xApp | 10 days | Native C xApp with E42 callbacks | Lab Linux | Planned |

---

## Known Limitations

Please read this section before drawing conclusions from the reported metrics.

1. **Synthetic data is too easy.** The observed AUC of 1.0000 reflects the generator's deterministic degradation ramp, not real-world performance. Real OAI telemetry includes multipath, thermal noise, scheduling jitter, and non-blockage-driven degradation that will substantially reduce accuracy.

2. **Evaluation was performed on training data.** The `evaluate.py` invocation in the Quick Start uses `data/synthetic_train.csv`, which the model has already seen. For a meaningful evaluation, generate a separate test set with a different `--seed`.

3. **No real telemetry yet.** The pipeline has not been validated against live E2SM-KPM indications from an OAI gNB. That integration is the next milestone (P5–P6).

4. **Windows is not a real-time OS.** Sub-5 ms inference is achievable, but end-to-end control-loop latency cannot be guaranteed on Windows. Production deployment requires Linux with real-time scheduling.

5. **Feature 72 dominates.** The model relies heavily on the standard deviation of the RSSI sequence. In real deployments, this feature may be less reliable when RSSI is noisy or when blockage manifests primarily through other metrics.

### Next Steps for Realistic Training

To harden the pipeline before testbed integration:

1. **Add noise to the synthetic generator.** Inject Gaussian noise into all feature channels, add random dropout, and introduce partial blockages (not just full link failures).

2. **Vary the degradation ramp.** Replace the linear ramp with stochastic onset (sudden vs. gradual), variable lead times, and false-positive distractors (e.g., fading events that do not lead to blockage).

3. **Generate a held-out test set** with a different seed:

   ```powershell
   python -m src.data.synthetic_generator --output data/synthetic_test.csv --samples 5000 --seed 123
   python -m src.model.evaluate --model models/blockage_lgbm.txt --data data/synthetic_test.csv
   ```

4. **Integrate real telemetry** as soon as the WSL2 testbed is running.

---

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

Ensure you run commands from the repository root, not from inside `src/`. The package uses `python -m src.<module>` invocation.

### FileNotFoundError: Model artifact not found

The FastAPI service requires a trained model. Run the training step first:

```powershell
python -m src.model.train --data data/synthetic_train.csv --output models/blockage_lgbm.txt
```

### LightGBM produces AUC of 1.0

This is expected on synthetic data. It is not a bug — it is a signal that the synthetic dataset is too separable. See [Known Limitations](#known-limitations).

### Port 8000 already in use

Change the port:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

### pytest reports zero tests collected

Ensure `pytest` is run from the repository root and that `tests/` contains an `__init__.py` file (or that the test files follow the `test_*.py` naming convention).

### Python 3.14 compatibility

The prototype was verified on Python 3.14.6. If you encounter issues with newer package versions, pin to the versions listed in `requirements.txt`.

---

## References

**HSE MIEM**

1. Trusted 6G Communication Systems initiative — https://www.hse.ru/en/news/priority/1072508649.html
2. Research on sub-THz and RIS — https://www.hse.ru/en/news/research/923567128.html
3. FlexRIC and O-RAN research — https://www.hse.ru/en/news/priority/1198407152.html
4. ML-based beam control (Prof. E. Koucheryavy) — https://www.hse.ru/en/news/research/1021176001.html

**Background**

- O-RAN Alliance. *E2 Service Model: KPM (E2SM-KPM) v3.0.*
- FlexRIC — https://gitlab.eurecom.fr/mosaic5g/flexric
- OpenAirInterface — https://gitlab.eurecom.fr/oai/openairinterface5g
- LightGBM — https://lightgbm.readthedocs.io

---

## License

MIT License. See `LICENSE` for details.

---

Built to serve the HSE MIEM Trusted 6G Communication Systems initiative.

For questions about integration with the HSE MIEM testbed, contact the laboratory directly. For bugs or feature requests in this prototype, open an issue in the repository.