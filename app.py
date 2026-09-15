import streamlit as st
import pandas as pd
import json
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Trim default padding so everything fits without scrolling
st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# app.py sits at the repo root, alongside the models/ and data/ folders.
# If you move app.py into a subfolder (e.g. src/app.py), change the line below
# to: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA_FILE = os.path.join(BASE_DIR, "data", "shared_data.json")
# rf_model.joblib sits directly at the repo root (no models/ subfolder)
MODEL_PATH       = os.path.join(BASE_DIR, "rf_model.joblib")

# ── Baseline ("healthy") sensor values, taken from the simulator's normal state ─
BASELINES = {
    "Air temperature":     298.0,
    "Process temperature": 308.0,
    "Rotational speed":    1500,
    "Torque":              40.0,
    "Tool wear":           0,
}

# ── Load Model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ── Session-state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_ts" not in st.session_state:
    st.session_state.last_ts = None

# ── Helper: read latest JSON written by simulator ─────────────────────────────
def read_sensor_file():
    try:
        if os.path.exists(SHARED_DATA_FILE):
            with open(SHARED_DATA_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# ── Fallback: generate telemetry internally when no external simulator.py ─────
# process is writing to the shared file (e.g. when deployed on Streamlit
# Community Cloud, which only runs this single app.py — it can't also run a
# second simulator.py process). Mirrors the same "gradual fault" pattern as
# simulator.py, but keeps its state in st.session_state instead of a file.
def get_internal_reading():
    import random

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = {
            "machine_type": random.choice(["L", "M", "H"]),
            "air_temp": 298.0,
            "process_temp": 308.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 0,
            "fault_mode_active": False,
            "cycle_count": 0,
        }

    s = st.session_state.sim_state

    s["air_temp"] += random.uniform(-0.1, 0.1)
    s["process_temp"] += random.uniform(-0.2, 0.2)
    s["rotational_speed"] += random.randint(-15, 15)
    s["torque"] += random.uniform(-0.5, 0.5)
    s["tool_wear"] += random.randint(1, 3)

    s["cycle_count"] += 1
    if s["cycle_count"] > 25 and not s["fault_mode_active"]:
        s["fault_mode_active"] = True

    if s["fault_mode_active"]:
        s["process_temp"] += random.uniform(0.5, 1.5)
        s["rotational_speed"] -= random.randint(20, 50)
        s["torque"] += random.uniform(2.0, 5.0)
        s["tool_wear"] += random.randint(5, 15)

    if s["rotational_speed"] < 0:
        s["rotational_speed"] = 0
    if s["process_temp"] > 400:
        s["process_temp"] = 400.0
    if s["torque"] > 100:
        s["torque"] = 100.0

    if s["cycle_count"] > 60:
        s["fault_mode_active"] = False
        s["cycle_count"] = 0
        s["air_temp"] = 298.0
        s["process_temp"] = 308.0
        s["rotational_speed"] = 1500
        s["torque"] = 40.0
        s["tool_wear"] = 0

    return {
        "Timestamp_raw":        time.time(),
        "Type":                 s["machine_type"],
        "Air temperature":      round(s["air_temp"], 1),
        "Process temperature":  round(s["process_temp"], 1),
        "Rotational speed":     int(s["rotational_speed"]),
        "Torque":               round(s["torque"], 1),
        "Tool wear":            int(s["tool_wear"]),
    }

# ── Helper: risk level from failure probability ───────────────────────────────
def risk_level(prob_val):
    if prob_val < 30:
        return "SAFE", "#2ecc71", "✅"
    elif prob_val < 70:
        return "WARNING", "#f1c40f", "⚠️"
    else:
        return "CRITICAL", "#e74c3c", "🚨"

# ── Helper: build a detailed, rule-based recommended action ───────────────────
def build_recommendation(level, last):
    findings = []

    if last["Process temperature"] > BASELINES["Process temperature"] + 5:
        findings.append(
            f"**Process temperature** elevated at {last['Process temperature']} K "
            f"(baseline ~{BASELINES['Process temperature']} K) — check cooling/lubrication."
        )
    if last["Air temperature"] > BASELINES["Air temperature"] + 3:
        findings.append(
            f"**Air temperature** above normal at {last['Air temperature']} K "
            f"(baseline ~{BASELINES['Air temperature']} K) — check ambient/ventilation."
        )
    if last["Rotational speed"] < BASELINES["Rotational speed"] - 100:
        findings.append(
            f"**Rotational speed** dropped to {last['Rotational speed']} RPM "
            f"(baseline ~{BASELINES['Rotational speed']} RPM) — check belt/bearing wear."
        )
    if last["Torque"] > BASELINES["Torque"] + 8:
        findings.append(
            f"**Torque** elevated at {last['Torque']} Nm "
            f"(baseline ~{BASELINES['Torque']} Nm) — check for binding/overload."
        )
    if last["Tool wear"] > 150:
        findings.append(
            f"**Tool wear** high at {last['Tool wear']} min — schedule a tool replacement."
        )

    if level == "SAFE":
        headline = "No action required."
        body = "All telemetry is within its normal operating range."
    elif level == "WARNING":
        headline = "Schedule an inspection soon."
        body = "Telemetry is drifting outside normal range — plan a maintenance check."
    else:
        headline = "Stop the machine and inspect immediately."
        body = "Failure risk is high — halt production and inspect now."

    if not findings:
        findings.append("No individual sensor has crossed its abnormal threshold yet.")

    return headline, body, findings

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 🏭 Predictive Machine Maintenance System")

raw = read_sensor_file()

# Check freshness: data must be written in the last 3 seconds
external_sim_live = (
    raw is not None
    and time.time() - raw.get("Timestamp_raw", 0) < 3.0
)

if external_sim_live:
    data_source_caption = "📡 Live data source: external `simulator.py` process"
else:
    data_source_caption = "🧪 Live data source: built-in simulator (no external `simulator.py` detected)"
    raw = get_internal_reading()

model_status = "✅ Random Forest model loaded" if model is not None else "❌ Model not found"
st.caption(f"{data_source_caption}  ·  {model_status}")

if model is None:
    try:
        root_files = sorted(os.listdir(BASE_DIR))
    except Exception:
        root_files = ["(could not list directory)"]
    st.warning(
        f"Looking for the model file at `{MODEL_PATH}` but it isn't there.\n\n"
        f"Files Streamlit actually finds in this app's root folder:\n"
    )
    st.code("\n".join(root_files))

# ── Ingest new sample only if timestamp changed ────────────────────────────────
ts_raw = raw["Timestamp_raw"]
if ts_raw != st.session_state.last_ts:
    st.session_state.last_ts = ts_raw

    prob = 0.0
    if model is not None:
        df = pd.DataFrame([raw])
        df = df.drop(columns=["Timestamp_raw"], errors="ignore")
        prob = model.predict_proba(df)[0][1] * 100

    entry = {
        "Timestamp":          datetime.now().strftime("%H:%M:%S"),
        "Type":               raw["Type"],
        "Air temperature":    raw["Air temperature"],
        "Process temperature":raw["Process temperature"],
        "Rotational speed":   raw["Rotational speed"],
        "Torque":             raw["Torque"],
        "Tool wear":          raw["Tool wear"],
        "Failure_Prob":       prob,
    }
    st.session_state.history.append(entry)
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

history = st.session_state.history
if not history:
    st.info("⏳ First data point arriving...")
else:
    df_hist  = pd.DataFrame(history)
    last     = history[-1]
    prob_val = last["Failure_Prob"]
    level, color, icon = risk_level(prob_val)

    # ── Metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Air Temp",        f"{last['Air temperature']} K")
    c2.metric("Process Temp",    f"{last['Process temperature']} K")
    c3.metric("Rotational Speed",f"{last['Rotational speed']} RPM")
    c4.metric("Torque",          f"{last['Torque']} Nm")
    c5.metric("Tool Wear",       f"{last['Tool wear']} Min")

    # ── Left: big status panel + gauge  |  Right: telemetry graph ───────────
    left, right = st.columns([2, 3])

    with left:
        st.markdown(
            f"""
            <div style="
                background-color:{color}25;
                border: 3px solid {color};
                border-radius: 14px;
                padding: 10px 12px;
                text-align: center;
                margin: 6px 0 4px 0;
            ">
                <div style="font-size: 34px; line-height: 1;">{icon}</div>
                <div style="font-size: 26px; font-weight: 800; color:{color}; margin-top: 4px;">
                    {level}
                </div>
                <div style="font-size: 14px; color: #555;">
                    Failure Probability: <b>{prob_val:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_g = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = prob_val,
            number = {"suffix": "%", "font": {"size": 34}},
            title  = {"text": "Risk Score", "font": {"size": 14}},
            domain = {"x": [0, 1], "y": [0, 1]},
            gauge  = {
                "axis":      {"range": [0, 100]},
                "bar":       {"color": color, "thickness": 0.3},
                "bgcolor":   "rgba(0,0,0,0)",
                "steps":     [
                    {"range": [0,  30], "color": "#2ecc71"},
                    {"range": [30, 70], "color": "#f4d03f"},
                    {"range": [70,100], "color": "#e74c3c"},
                ],
                "threshold": {"line": {"color": "white", "width": 4},
                              "thickness": 0.75, "value": 90},
            },
        ))
        fig_g.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10),
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

    with right:
        # Combine both series into one dual-axis chart to save vertical space
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["Timestamp"], y=df_hist["Torque"],
            mode="lines", name="Torque (Nm)", line=dict(color="orange"),
        ))
        fig.add_trace(go.Scatter(
            x=df_hist["Timestamp"], y=df_hist["Process temperature"],
            mode="lines", name="Process Temp (K)", line=dict(color="red"),
            yaxis="y2",
        ))
        fig.update_layout(
            title="Real-Time Telemetry — Torque & Process Temperature",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Torque (Nm)"),
            yaxis2=dict(title="Process Temp (K)", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Bottom: Recommended Action tab ───────────────────────────────────────
    tab_action, = st.tabs(["🔧 Recommended Action"])
    with tab_action:
        headline, body, findings = build_recommendation(level, last)

        if level == "SAFE":
            st.success(f"**{headline}** {body}")
        elif level == "WARNING":
            st.warning(f"**{headline}** {body}")
        else:
            st.error(f"**{headline}** {body}")

        for f in findings:
            st.markdown(f"- {f}")

# ── Auto-refresh every 1 second ───────────────────────────────────────────────
time.sleep(1)
st.rerun()
