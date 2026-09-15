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
    page_title="Machine-health-monitor",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA_FILE = os.path.join(BASE_DIR, "data", "shared_data.json")
MODEL_PATH       = os.path.join(BASE_DIR, "models", "rf_model.joblib")

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
            f"**Process temperature** is elevated at {last['Process temperature']} K "
            f"(baseline ~{BASELINES['Process temperature']} K) — check the cooling/"
            f"lubrication system for a possible overheating condition."
        )
    if last["Air temperature"] > BASELINES["Air temperature"] + 3:
        findings.append(
            f"**Air temperature** is above normal at {last['Air temperature']} K "
            f"(baseline ~{BASELINES['Air temperature']} K) — check ambient/ventilation "
            f"conditions around the machine."
        )
    if last["Rotational speed"] < BASELINES["Rotational speed"] - 100:
        findings.append(
            f"**Rotational speed** has dropped to {last['Rotational speed']} RPM "
            f"(baseline ~{BASELINES['Rotational speed']} RPM) — inspect for belt "
            f"slippage, bearing wear, or motor load issues."
        )
    if last["Torque"] > BASELINES["Torque"] + 8:
        findings.append(
            f"**Torque** is elevated at {last['Torque']} Nm "
            f"(baseline ~{BASELINES['Torque']} Nm) — inspect for mechanical binding, "
            f"misalignment, or an overload condition."
        )
    if last["Tool wear"] > 150:
        findings.append(
            f"**Tool wear** is high at {last['Tool wear']} min — schedule a tool "
            f"replacement soon to avoid degraded part quality."
        )

    if level == "SAFE":
        headline = "No action required."
        body = (
            "All telemetry is within its normal operating range. Continue routine "
            "monitoring — no maintenance action is needed right now."
        )
    elif level == "WARNING":
        headline = "Schedule an inspection soon."
        body = (
            "Telemetry is drifting outside normal range. The machine can keep "
            "running, but plan an inspection at the next convenient maintenance "
            "window before the condition worsens."
        )
    else:
        headline = "Stop the machine and inspect immediately."
        body = (
            "Failure risk is high. Halt production on this machine and perform an "
            "immediate manual inspection before resuming operation."
        )

    if not findings:
        findings.append("No individual sensor has crossed its abnormal threshold yet.")

    return headline, body, findings

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Machine-Health-Monitor 🏭")
st.markdown("Real-time telemetry from the *(Simulator)* is analyzed via a **Random Forest** AI Model.")

sidebar, main = st.columns([1, 4])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with sidebar:
    st.header("Control Panel")
    st.markdown("---")
    st.markdown("**ML Model**")
    if model:
        st.success("✅ Loaded (Random Forest)")
    else:
        st.error("❌ Not Found")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.info(
        "1. Run this dashboard\n"
        "2. Open **localhost:8505**\n"
        "3. Start `simulator.py` in a second terminal\n"
        "4. Data appears automatically ✅"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
with main:
    raw = read_sensor_file()

    # Check freshness: data must be written in the last 3 seconds
    external_sim_live = (
        raw is not None
        and time.time() - raw.get("Timestamp_raw", 0) < 3.0
    )

    if external_sim_live:
        # A real simulator.py process is running and feeding shared_data.json
        st.caption("📡 Live data source: external `simulator.py` process")
    else:
        # No external simulator detected (e.g. running on Streamlit Cloud) —
        # generate telemetry internally instead, so the dashboard still works.
        st.caption("🧪 Live data source: built-in simulator (no external `simulator.py` detected)")
        raw = get_internal_reading()

    # ── Ingest new sample only if timestamp changed ────────────────────────
    ts_raw = raw["Timestamp_raw"]
    if ts_raw != st.session_state.last_ts:
        st.session_state.last_ts = ts_raw

        # Prediction
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

        # ── Metrics ────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Air Temp",        f"{last['Air temperature']} K")
        c2.metric("Process Temp",    f"{last['Process temperature']} K")
        c3.metric("Rotational Speed",f"{last['Rotational speed']} RPM")
        c4.metric("Torque",          f"{last['Torque']} Nm")
        c5.metric("Tool Wear",       f"{last['Tool wear']} Min")

        st.markdown("---")

        # ── Left: big status panel + gauge  |  Right: telemetry graphs ──────
        left, right = st.columns([2, 3])

        with left:
            st.markdown(
                f"""
                <div style="
                    background-color:{color}25;
                    border: 3px solid {color};
                    border-radius: 16px;
                    padding: 24px 16px;
                    text-align: center;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 52px; line-height: 1;">{icon}</div>
                    <div style="font-size: 34px; font-weight: 800; color:{color}; margin-top: 8px;">
                        {level}
                    </div>
                    <div style="font-size: 16px; color: #555; margin-top: 4px;">
                        Failure Probability: <b>{prob_val:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_g = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = prob_val,
                title  = {"text": "Failure Probability (%)"},
                domain = {"x": [0, 1], "y": [0, 1]},
                gauge  = {
                    "axis":      {"range": [0, 100]},
                    "bar":       {"color": color},
                    "bgcolor":   f"{color}15",
                    "steps":     [
                        {"range": [0,  30], "color": "#2ecc7133"},
                        {"range": [30, 70], "color": "#f1c40f33"},
                        {"range": [70,100], "color": "#e74c3c33"},
                    ],
                    "threshold": {"line": {"color": "#e74c3c", "width": 4},
                                  "thickness": 0.75, "value": 90},
                },
            ))
            fig_g.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)

        with right:
            st.markdown("#### Real-Time Telemetry")

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Torque"],
                mode="lines", name="Torque (Nm)", line=dict(color="orange")
            ))
            fig_t.update_layout(title="Torque Over Time",
                                 height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_t, use_container_width=True)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Process temperature"],
                mode="lines", name="Process Temp (K)", line=dict(color="red")
            ))
            fig_p.update_layout(title="Temperature Over Time",
                                 height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_p, use_container_width=True)

        # ── Bottom: Recommended Action tab ──────────────────────────────────
        st.markdown("---")
        tab_action, = st.tabs(["🔧 Recommended Action"])
        with tab_action:
            headline, body, findings = build_recommendation(level, last)

            if level == "SAFE":
                st.success(f"**{headline}**  \n{body}")
            elif level == "WARNING":
                st.warning(f"**{headline}**  \n{body}")
            else:
                st.error(f"**{headline}**  \n{body}")

            st.markdown("**Findings:**")
            for f in findings:
                st.markdown(f"- {f}")

# ── Auto-refresh every 1 second ───────────────────────────────────────────────
time.sleep(1)
st.rerun()
