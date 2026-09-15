# AI-Powered Predictive Maintenance — Dashboard (Modified)

This is a modified version of the dashboard from
[enesuslu15/AI-Powered-Predictive-Maintenance-System](https://github.com/enesuslu15/AI-Powered-Predictive-Maintenance-System),
restyled per request:

- **Left side:** a big red/yellow/green status panel (colored block + a large gauge on top of it) showing SAFE / WARNING / CRITICAL and the failure probability.
- **Right side:** the real-time telemetry line charts (Torque, Process Temperature).
- **Bottom:** a **"🔧 Recommended Action"** tab with a rule-based recommendation that names which specific sensor(s) are abnormal (e.g. elevated torque, dropped rotational speed, high tool wear) and what to check.

Only the files needed to run the dashboard are included here (the original repo's CMAPSS dataset and notebooks were left out since they aren't used by the dashboard).

## Setup

```bash
pip install -r requirements.txt
```

## Run

Open two terminals from this folder:

**Terminal 1 — start the dashboard:**
```bash
streamlit run src/app.py
```

**Terminal 2 — start the simulator (feeds fake sensor data):**
```bash
python src/simulator.py
```

The dashboard auto-refreshes every second and will start showing live data as soon as the simulator is running. The simulator runs machines normally for the first ~25 seconds, then simulates a developing fault (rising temperature/torque, dropping speed) for about 35 seconds before resetting — a good way to watch the status panel move from SAFE → WARNING → CRITICAL and to see the Recommended Action tab update.

## Files

- `src/app.py` — Streamlit dashboard (modified layout)
- `src/simulator.py` — generates simulated sensor telemetry (unmodified)
- `src/train_model.py` — script used to train `models/rf_model.joblib` (unmodified)
- `models/rf_model.joblib` — pretrained Random Forest classifier
- `requirements.txt` — Python dependencies
