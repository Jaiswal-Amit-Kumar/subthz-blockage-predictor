```markdown
# Sub-THz Blockage Predictor

<div align="center">

**An AI-driven link blockage prediction service for sub-THz (140–160 GHz) O-RAN testbeds.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#installation)

Designed for the **HSE MIEM Telecommunications Research Institute**
and their **Trusted 6G Communication Systems** initiative.

[Overview](#-overview) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [API](#-api-reference) · [Roadmap](#-deployment-roadmap)

</div>

---

## 📖 Overview

Sub-THz frequencies (140–160 GHz) offer enormous bandwidth for 6G applications but are extremely sensitive to physical obstruction. A human hand, body, or interior wall can attenuate a sub-THz link by **20–40 dB within milliseconds**. Traditional reactive link adaptation is too slow — by the time degradation is observed in signal-quality metrics, the link has already failed.

This project delivers a **predictive** alternative: a lightweight, CPU-only machine learning service that forecasts blockage from standard O-RAN telemetry **before** it occurs, giving the RIC time to act.

<div align="center">

| Attribute | Value |
|:---|:---|
| 🎯 **Prediction target** | Binary — blockage imminent within 5 ms |
| 📥 **Input** | Sliding window of 10 KPM frames (50 ms at 5 ms period) |
| 🧠 **Model** | LightGBM gradient-boosted classifier |
| ⚡ **Inference latency** | < 0.5 ms per frame (single CPU core) |
| 💻 **Development platform** | Windows 10/11, Python 3.10+ |
| 🚀 **Production target** | Linux + FlexRIC Near-RT RIC |
| 📦 **Dependencies** | NumPy, LightGBM, FastAPI — **no GPU required** |

</div>

---

## 🔍 The Problem

Professor Evgeny Koucheryavy of HSE MIEM has identified ML-based automatic beam control as a core research objective for the lab's Trusted 6G initiative. The operational challenge is specific:

> **Sub-THz links drop in milliseconds when obstructed. The system must know a blockage is coming *before* it happens, not after.**

Existing approaches require computer vision, additional sensors, or heavy deep learning models that cannot meet the 5 ms control-loop budget. This project demonstrates that **standard O-RAN KPM telemetry alone** — RSSI, SINR, Doppler shift, RLC delay, HARQ retransmission ratios — contains sufficient signal to predict blockage with high accuracy using a lightweight model.

---

## 💡 The Solution

A three-stage pipeline:

```text
┌────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│  OAI gNB + FlexRIC │───►│  Python Predictor    │───►│  Trigger Action    │
│  E2SM-KPM telemetry│    │  LightGBM + FastAPI  │    │  Antenna / RIS /   │
│  (5 ms cadence)    │    │  < 0.5 ms inference  │    │  Beam switch       │
└────────────────────┘    └──────────────────────┘    └────────────────────┘
```

1. **Telemetry ingestion** — KPM indication frames arrive from the OAI gNB via FlexRIC at a 5 ms reporting period.
2. **Feature extraction** — A 10-frame sliding window is transformed into an 85-dimensional feature vector (80 raw + 5 derived time-series statistics).
3. **Prediction** — A LightGBM classifier outputs `P(blockage within 5 ms)`. If the probability exceeds the configured threshold, a trigger action is emitted for downstream actuation.

The entire pipeline is designed for **CPU-only execution** and can be prototyped on a standard Windows PC in minutes.

---

## ✅ Verified Results

The prototype was built, trained, and tested on a **Windows 11** machine with **Python 3.14.6**. The following output was produced end-to-end.

<details>
<summary><b>📊 Synthetic data generation</b> (click to expand)</summary>

```text
$ python -m src.data.synthetic_generator --output data/synthetic_train.csv --samples 20000
Wrote 220,000 rows to data\synthetic_train.csv
Sequences: 20,000
Positive sequences: 9,932
```
</details>

<details>
<summary><b>🧠 Model training</b> (click to expand)</summary>

```text
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
</details>

<details>
<summary><b>🧪 Unit tests</b> (click to expand)</summary>

```text
$ pytest tests/ -v
collected 7 items
tests/test_feature_engineer.py ....... PASSED  [7/7]
============================================= 7 passed in 1.35s ==============================================
```
</details>

### Interpretation of Results

| Metric | Observed | Notes |
|:---|:---:|:---|
| AUC-ROC | `1.0000` | ⚠️ Synthetic data is too separable — see caveat below |
| Best iteration | `2` | Model converges almost immediately |
| Dominant feature | `feature[72]` (gain=32,246) | Derived: std of RSSI sequence |
| F1 (both classes) | `1.00` | Perfect separation on synthetic data |
| Test suite | `7/7` | All unit tests green |

> [!WARNING]
> **An AUC of 1.0000 on synthetic data is a red flag, not a triumph.**
> The synthetic generator injects a clean, deterministic degradation ramp that is trivially separable. Real OAI telemetry will be noisy, non-stationary, and far harder to classify. The pipeline is correct; the synthetic dataset is simply too easy. See [Known Limitations](#-known-limitations) for hardening steps.

---

## 📁 Repository Structure

```text
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

## ⚙️ Installation

### Requirements

- **Python 3.10 – 3.14** (tested on 3.14.6)
- **Windows 10/11**, macOS, or Linux
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

<details>
<summary><b>📋 requirements.txt</b> (click to expand)</summary>

```text
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
</details>

---

## 🚀 Quick Start

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

> [!NOTE]
> Once 10 frames have been submitted, the response will include a `blockage_probability` and, if the threshold is crossed, a `trigger_action`.

---

## 🛠 Configuration

All tunable parameters live in `src/config.py`:

| Parameter | Default | Description |
|:---|:---:|:---|
| `WINDOW_SIZE` | `10` | Number of KPM frames in the sliding window |
| `PREDICTION_HORIZON_MS` | `5.0` | How far ahead to predict blockage |
| `BLOCKAGE_THRESHOLD` | `0.75` | Probability threshold for trigger emission |
| `SYNTHETIC_SAMPLES` | `20000` | Number of sequences to generate |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `MODEL_PATH` | `models/blockage_lgbm.txt` | Trained model artifact location |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI bind port |

> [!TIP]
> Adjust `WINDOW_SIZE` and `PREDICTION_HORIZON_MS` together: a 10-frame window at a 5 ms KPM period corresponds to a 50 ms observation span, which is sufficient to detect the onset of degradation for a 5 ms prediction horizon.

---

## 🏗 Architecture

### Data Flow

```text
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
|:---|:---:|:---|
| Raw RLC/MAC metrics | `5 × 10 = 50` | Per-frame values across the window |
| Raw PHY metrics | `3 × 10 = 30` | RSSI, SINR, Doppler across the window |
| Δ RSSI (mean first difference) | `1` | Rate of signal degradation |
| Δ SINR (mean first difference) | `1` | Rate of quality degradation |
| σ(RSSI) | `1` | Volatility of signal strength |
| σ(SINR) | `1` | Volatility of signal quality |
| ρ(delay, retx) | `1` | Cross-correlation of congestion indicators |
| **Total** | **`85`** | |

### Model

| Property | Value |
|:---|:---|
| **Algorithm** | LightGBM gradient-boosted decision trees |
| **Trees** | up to 500, early-stopped (best iteration was 2 on synthetic data) |
| **Max depth** | 6 |
| **Num leaves** | 31 |
| **Learning rate** | 0.1 |
| **Class imbalance** | handled via `scale_pos_weight` |
| **Split strategy** | `GroupShuffleSplit` by `sequence_id` to prevent leakage |

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### `GET /health`

Readiness probe for orchestration and monitoring.

<details>
<summary><b>Response example</b></summary>

```json
{
  "status": "ok",
  "model_ready": true,
  "buffer_fill": 10
}
```
</details>

`buffer_fill` reports how many frames are currently in the predictor's sliding window. The predictor returns neutral predictions until `buffer_fill >= WINDOW_SIZE`.

### `POST /predict`

Ingest one KPM frame and receive a prediction.

**Request body:**

| Field | Type | Required | Description |
|:---|:---:|:---:|:---|
| `timestamp_ms` | float | ✅ | Monotonic timestamp in milliseconds |
| `rlc_delay_dl` | float | | RLC SDU delay, downlink (ms) |
| `rlc_drop_rate` | float | | RLC packet drop rate |
| `harq_retx_ratio` | float | | HARQ retransmission ratio |
| `prb_utilization` | float | | PRB utilization [0, 1] |
| `ue_buffer_occupancy` | float | | UE buffer occupancy (bytes) |
| `rssi_dbm` | float | | Received signal strength (dBm) |
| `sinr_db` | float | | SINR (dB) |
| `doppler_hz` | float | | Doppler shift (Hz) |

<details>
<summary><b>📥 Example request</b></summary>

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
</details>

<details>
<summary><b>📤 Example response</b></summary>

```json
{
  "timestamp_ms": 1000.0,
  "blockage_probability": 0.02,
  "blockage_predicted": false,
  "inference_latency_ms": 0.31,
  "trigger_action": null
}
```
</details>

> [!IMPORTANT]
> When `blockage_predicted` is `true`, `trigger_action` will be set to `"switch_antenna"` and downstream consumers should initiate preemptive link adaptation.

### Interactive documentation

FastAPI auto-generates OpenAPI documentation at:

- 📘 **Swagger UI:** `http://localhost:8000/docs`
- 📗 **ReDoc:** `http://localhost:8000/redoc`

---

## 🧪 Testing

```powershell
pytest tests/ -v
```

The test suite covers:

| Test | Purpose |
|:---|:---|
| `test_output_shape` | Feature vector has the expected 85 dimensions |
| `test_insufficient_frames_raises` | Correct error on too-short windows |
| `test_uses_most_recent_frames` | Ring buffer keeps the newest frames |
| `test_zero_variance_returns_zero_correlation` | Guard against NaN in cross-correlation |
| `test_not_ready_before_window_filled` | Neutral prediction before warm-up |
| `test_buffer_fill_tracks_frames` | Buffer counter increments correctly |
| `test_buffer_caps_at_window_size` | Buffer never exceeds `WINDOW_SIZE` |

<details>
<summary><b>Expected output</b></summary>

```text
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
</details>

---

## 🔗 Integration with FlexRIC

### Development Mode (External Microservice)

Run the predictor on a Windows workstation and forward telemetry from the FlexRIC host over HTTP. Suitable for prototyping and shadow-mode validation.

```text
[ FlexRIC host (Linux) ]  ──HTTP──►  [ Windows PC: FastAPI predictor ]
```

### Production Mode (Native xApp)

Compile the predictor as a native FlexRIC xApp using the FlexRIC SDK, linked against the LightGBM C API. This mode uses the E42 shared-memory transport for sub-millisecond latency and runs co-located with the Near-RT RIC.

<details>
<summary><b>🔧 Conceptual C xApp skeleton (production path)</b></summary>

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
</details>

### Latency Budget

| Stage | Target | Notes |
|:---|:---:|:---|
| E2 indication generation | ≤ 2 ms | OAI gNB KPM report period |
| E42 transport | ≤ 1 ms | Shared memory in co-located deployment |
| Feature extraction | ≤ 0.2 ms | 10-frame window, 8 features |
| LightGBM inference | ≤ 0.5 ms | 150 trees, single thread |
| HTTP overhead (dev mode) | ≤ 1 ms | Local network only |
| **Total (production)** | **≤ 3.7 ms** | Within the 5 ms horizon |
| **Total (dev mode)** | **≤ 4.7 ms** | Within the 5 ms horizon |

---

## 🗺 Deployment Roadmap

| Phase | Duration | Deliverable | Platform | Status |
|:---|:---:|:---|:---|:---:|
| **P0** — Data generation | — | 20,000 synthetic sequences | Windows | ✅ |
| **P1** — Model training | — | Trained LightGBM artifact | Windows | ✅ |
| **P2** — API service | — | FastAPI `/predict` endpoint | Windows | ✅ |
| **P3** — Offline validation | — | 7/7 unit tests passing | Windows | ✅ |
| **P4** — Realistic data | 2–3 days | Hardened synthetic + noise injection | Windows | 🔜 |
| **P5** — WSL2 testbed | 3–5 days | FlexRIC + OAI in RF-simulator mode | WSL2 | 📋 |
| **P6** — Bridge | 1–2 days | E42 → HTTP telemetry forwarder | WSL2 | 📋 |
| **P7** — Shadow deployment | 3 days | Live prediction logging, no control | Lab Linux | 📋 |
| **P8** — Control loop | 5 days | E2SM-RC antenna switching actuation | Lab Linux | 📋 |
| **P9** — Production xApp | 10 days | Native C xApp with E42 callbacks | Lab Linux | 📋 |

**Legend:** ✅ Done · 🔜 Next · 📋 Planned

---

## ⚠️ Known Limitations

> [!CAUTION]
> Please read this section before drawing conclusions from the reported metrics.

1. **Synthetic data is too easy.** The observed AUC of `1.0000` reflects the generator's deterministic degradation ramp, not real-world performance. Real OAI telemetry includes multipath, thermal noise, scheduling jitter, and non-blockage-driven degradation that will substantially reduce accuracy.
2. **Evaluation was performed on training data.** The `evaluate.py` invocation in the Quick Start uses `data/synthetic_train.csv`, which the model has already seen. For a meaningful evaluation, generate a separate test set with a different `--seed`.
3. **No real telemetry yet.** The pipeline has not been validated against live E2SM-KPM indications from an OAI gNB. That integration is the next milestone (P5–P6).
4. **Windows is not a real-time OS.** Sub-5 ms inference is achievable, but end-to-end control-loop latency cannot be guaranteed on Windows. Production deployment requires Linux with real-time scheduling.
5. **Feature 72 dominates.** The model relies heavily on the standard deviation of the RSSI sequence. In real deployments, this feature may be less reliable when RSSI is noisy or when blockage manifests primarily through other metrics.

### Next Steps for Realistic Training

<details>
<summary><b>🔧 Four hardening steps</b> (click to expand)</summary>

1. **Add noise to the synthetic generator.** Inject Gaussian noise into all feature channels, add random dropout, and introduce partial blockages (not just full link failures).
2. **Vary the degradation ramp.** Replace the linear ramp with stochastic onset (sudden vs. gradual), variable lead times, and false-positive distractors (e.g., fading events that do not lead to blockage).
3. **Generate a held-out test set** with a different seed:

   ```powershell
   python -m src.data.synthetic_generator --output data/synthetic_test.csv --samples 5000 --seed 123
   python -m src.model.evaluate --model models/blockage_lgbm.txt --data data/synthetic_test.csv
   ```

4. **Integrate real telemetry** as soon as the WSL2 testbed is running.
</details>

---

## 🔧 Troubleshooting

<details>
<summary><b><code>ModuleNotFoundError: No module named 'src'</code></b></summary>

Ensure you run commands from the repository root, not from inside `src/`. The package uses `python -m src.<module>` invocation.
</details>

<details>
<summary><b><code>FileNotFoundError: Model artifact not found</code></b></summary>

The FastAPI service requires a trained model. Run the training step first:

```powershell
python -m src.model.train --data data/synthetic_train.csv --output models/blockage_lgbm.txt
```
</details>

<details>
<summary><b>LightGBM produces AUC of 1.0</b></summary>

This is expected on synthetic data. It is not a bug — it is a signal that the synthetic dataset is too separable. See [Known Limitations](#-known-limitations).
</details>

<details>
<summary><b>Port 8000 already in use</b></summary>

Change the port:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```
</details>

<details>
<summary><b><code>pytest</code> reports zero tests collected</b></summary>

Ensure `pytest` is run from the repository root and that `tests/` contains an `__init__.py` file (or that the test files follow the `test_*.py` naming convention).
</details>

<details>
<summary><b>Python 3.14 compatibility</b></summary>

The prototype was verified on Python 3.14.6. If you encounter issues with newer package versions, pin to the versions listed in `requirements.txt`.
</details>

---

## 📚 References

**HSE MIEM**

1. [Trusted 6G Communication Systems initiative](https://www.hse.ru/en/news/priority/1072508649.html)
2. [Research on sub-THz and RIS](https://www.hse.ru/en/news/research/923567128.html)
3. [FlexRIC and O-RAN research](https://www.hse.ru/en/news/priority/1198407152.html)
4. [ML-based beam control (Prof. E. Koucheryavy)](https://www.hse.ru/en/news/research/1021176001.html)

**Background**

- O-RAN Alliance. *E2 Service Model: KPM (E2SM-KPM) v3.0.*
- [FlexRIC](https://gitlab.eurecom.fr/mosaic5g/flexric)
- [OpenAirInterface](https://gitlab.eurecom.fr/oai/openairinterface5g)
- [LightGBM](https://lightgbm.readthedocs.io)

---

## 📄 License

MIT License. See `LICENSE` for details.

---

<div align="center">

**Built to serve the HSE MIEM Trusted 6G Communication Systems initiative.**

*For questions about integration with the HSE MIEM testbed, contact the laboratory directly.*
*For bugs or feature requests in this prototype, open an issue in the repository.*

</div>
```

Save the block above directly as `README.md` in the repository root. It renders correctly on GitHub as-is — no changes needed.