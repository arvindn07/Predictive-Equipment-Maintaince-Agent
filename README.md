# AI-Powered Predictive Maintenance System

A real-time dashboard that predicts machine failure risk from live sensor telemetry using a machine learning model, and recommends what to do about it.

The idea behind this project is simple: industrial machines don't usually fail out of nowhere. In the lead-up to a breakdown, sensor readings like temperature, torque, and rotational speed start drifting away from their normal range. If you can catch that drift early, you can schedule maintenance before the machine actually fails, instead of finding out the hard way.

This project simulates that scenario end-to-end — a sensor stream, a trained ML model reading that stream, and a live dashboard that turns the model's output into something a technician can actually act on.

## What it does

- Simulates a running industrial machine, streaming telemetry (air temperature, process temperature, rotational speed, torque, and tool wear) once per second, including a scripted "gradual failure" scenario so the system has something to detect.
- Feeds each new reading into a trained Random Forest classifier, which outputs a failure probability.
- Displays that probability on a live dashboard with a big color-coded status indicator (SAFE / WARNING / CRITICAL), a gauge, and rolling graphs of the key readings.
- Generates a plain-language recommended action, naming which specific sensor reading looks abnormal (e.g. torque running high, rotational speed dropping) rather than just showing a number.

## How it works

1. **Simulator** (`src/simulator.py`) generates telemetry values every second and writes the latest reading to a shared JSON file. It starts machines in a healthy state, then after a set number of cycles gradually pushes temperature and torque up and rotational speed down to imitate a developing mechanical fault, before eventually resetting back to normal.
2. **Model** (`models/rf_model.joblib`) is a Random Forest classifier trained on machine telemetry (see `src/train_model.py`) to output the probability that a given reading corresponds to an at-risk machine.
3. **Dashboard** (`src/app.py`) is a Streamlit app that polls the shared telemetry file, runs each new reading through the model, and renders the result: a status panel and gauge, live line charts of torque and temperature, and a recommended-action panel.

The dashboard and the simulator run as two independent processes and communicate through a shared data file — this mirrors how a real system might have a sensor/PLC layer publishing readings that a separate monitoring application consumes.

## Project structure

```
.
├── src/
│   ├── app.py            
│   ├── simulator.py       
│   └── train_model.py    
├── models/
│   └── rf_model.joblib    
├── data/
│   └── shared_data.json   
├── requirements.txt
└── README.md
```

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run src/app.py
```

### 3. Start the simulator (in a second terminal)

```bash
python src/simulator.py
```

Once the simulator is running, the dashboard will start updating automatically — no need to refresh the page. Leave it running for a minute or so to watch the status move from SAFE, into WARNING, and up to CRITICAL as the simulated fault develops, then back down again once the simulator resets.

## Possible next steps

- Swap the Random Forest for the LSTM-based approach explored in the notebooks, for sequence-aware failure prediction.
- Persist history to a database instead of an in-memory session so trends survive a dashboard restart.
- Add per-machine tracking for a fleet of multiple simulated machines instead of just one.

## Author

Built as a predictive maintenance demo project combining a simulated sensor pipeline, a trained ML model, and a real-time monitoring dashboard.
