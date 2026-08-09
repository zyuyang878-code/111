"""
HKU Smart Dining: Live Queue Analytics & Nudge System
=====================================================
Multi-window dining hall queue simulator with real-time behavioural
nudge. Renders 13 real HKU dining halls as mobile-style cards.

Run locally:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="HKU Smart Dining",
    page_icon="\U0001F37D\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Constants — 13 HKU Dining Halls
# ============================================================
# (Chinese, English, service_mean_min, peak_share_of_lambda)
# Ordered slowest → fastest. Shares sum to 1.0.
DINING_HALLS = [
    ("庄月明食堂",     "Meng Wah Complex",          3.5, 0.22, "\U0001F35C"),
    ("学生会食堂",     "Student Union Canteen",     3.2, 0.18, "\U0001F371"),
    ("方树泉食堂",     "Fong Shu Chuen Hall",       2.8, 0.12, "\U0001F961"),
    ("亚洲滋味餐厅",   "Asian Flavours",            2.6, 0.08, "\U0001F372"),
    ("一念素食",       "Yi Nian Vegetarian",        2.5, 0.06, "\U0001F957"),
    ("cafe330",        "Cafe 330",                  2.5, 0.06, "\u2615"),
    ("Coffee Academics","Coffee Academics",         2.2, 0.05, "\u2615"),
    ("U Deli",         "U Deli",                    2.0, 0.05, "\U0001F96A"),
    ("alfafa cafe",    "Alfafa Cafe",               2.0, 0.04, "\U0001F96A"),
    ("Sandwich Club",  "Sandwich Club",             2.0, 0.04, "\U0001F96A"),
    ("Super Sandwiches","Super Sandwiches",         1.8, 0.03, "\U0001F96A"),
    ("Subway",         "Subway",                    2.0, 0.04, "\U0001F956"),
    ("星巴克",         "Starbucks",                 1.5, 0.03, "\u2615"),
]

SERVICE_STD = 0.5

# Status thresholds (minutes)
TH_CONGESTED = 15.0
TH_BUSY      = 5.0

STATUS = {
    "congested": {
        "bg": "#FFEBEE", "border": "#E74C3C", "accent": "#C62828",
        "icon": "\U0001F534", "label": "Congested",
    },
    "busy": {
        "bg": "#FFF8E1", "border": "#F39C12", "accent": "#E65100",
        "icon": "\U0001F7E0", "label": "Busy",
    },
    "smooth": {
        "bg": "#E8F5E9", "border": "#16A085", "accent": "#2E7D32",
        "icon": "\U0001F7E2", "label": "Smooth Flow",
    },
}


# ============================================================
# Core Simulation Engine
# ============================================================
def mg1_mean_wait(lam, mean_s, std_s):
    """Pollaczek-Khinchine: E[W] for M/G/1 queue."""
    rho = lam * mean_s
    if rho >= 1:
        return 999.0
    e_s2 = std_s ** 2 + mean_s ** 2
    return lam * e_s2 / (2 * (1 - rho))


def estimate_queue_length(lam, mean_s):
    """Little's law: L = lambda * W."""
    if lam * mean_s >= 1:
        return 999
    return lam * mg1_mean_wait(lam, mean_s, SERVICE_STD)


def classify(wait_min):
    if wait_min >= TH_CONGESTED:
        return "congested"
    elif wait_min >= TH_BUSY:
        return "busy"
    else:
        return "smooth"


def _simulate_mgc(arrivals, services, n_windows=1):
    n = len(arrivals)
    if n == 0:
        return np.array([])
    window_free = np.zeros(n_windows)
    waits = np.zeros(n)
    for i in range(n):
        w = np.argmin(window_free)
        start = max(arrivals[i], window_free[w])
        waits[i] = start - arrivals[i]
        window_free[w] = start + services[i]
    return waits


def simulate_dynamics(lam_total, hall_idx, seed=42):
    """Run detailed sim for a single hall. Returns (time_labels, queue_len)."""
    chinese, english, mean_s, share, _icon = DINING_HALLS[hall_idx]
    lam = lam_total * share
    rng = np.random.default_rng(seed)
    mu, sigma = np.log(1 + (SERVICE_STD / mean_s) ** 2) / 2, np.sqrt(np.log(1 + (SERVICE_STD / mean_s) ** 2))
    mu = np.log(mean_s) - sigma ** 2 / 2
    SIM_MIN = 90

    n_arr = rng.poisson(lam * SIM_MIN)
    if n_arr == 0:
        return [f"12:{t:02d}" if t < 60 else f"13:{t-60:02d}" for t in range(SIM_MIN)], np.zeros(SIM_MIN, dtype=int)

    inter = rng.exponential(1.0 / max(lam, 1e-6), n_arr)
    arr = np.cumsum(inter)
    arr = arr[arr < SIM_MIN]
    serv = rng.lognormal(mu, sigma, len(arr))
    n = len(arr)

    events = []
    window_free = 0.0
    for i in range(n):
        a = arr[i]
        start = max(a, window_free)
        end = start + serv[i]
        events.append((a, 1))
        events.append((end, -1))
        window_free = end
    events.sort()

    q = np.zeros(SIM_MIN, dtype=int)
    cur = 0
    ei = 0
    for t in range(SIM_MIN):
        while ei < len(events) and events[ei][0] <= t:
            cur += events[ei][1]
            ei += 1
        q[t] = max(cur, 0)

    time_labels = [f"12:{t:02d}" if t < 60 else f"13:{t-60:02d}" for t in range(SIM_MIN)]
    return time_labels, q


@st.cache_data(show_spinner=False)
def compute_all_metrics(lam_peak, nudge):
    """Compute per-hall wait times & recommend the best hall."""
    results = []
    for i, (cn, en, mean_s, share, icon) in enumerate(DINING_HALLS):
        lam_i = lam_peak * share
        wait = mg1_mean_wait(lam_i, mean_s, SERVICE_STD)
        q_len = estimate_queue_length(lam_i, mean_s)
        rho = lam_i * mean_s
        results.append({
            "idx": i, "cn": cn, "en": en, "mean_s": mean_s,
            "share": share, "icon": icon, "lam": lam_i,
            "wait": wait, "q_len": q_len, "rho": rho,
            "status": classify(wait),
        })
    results.sort(key=lambda r: r["wait"])
    fastest_idx = results[0]["idx"]
    slowest_idx = results[-1]["idx"]
    avg_wait = np.mean([r["wait"] for r in results])
    return {
        "halls": results,
        "fastest": fastest_idx,
        "slowest": slowest_idx,
        "avg_wait": avg_wait,
    }


# ============================================================
# Custom CSS — Mobile App Look
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f0f2f5 0%, #e8eaf0 100%);
    }
    .block-container {
        max-width: 760px;
        padding-top: 1.2rem;
    }

    /* Top bar */
    .top-bar {
        background: linear-gradient(135deg, #003366, #1a4789);
        color: white;
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0,51,102,0.18);
    }
    .top-bar h1 {
        font-size: 1.4rem;
        margin: 0;
        font-weight: 700;
    }
    .top-bar .sub {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 3px;
    }
    .live-dot {
        display: inline-block;
        width: 9px; height: 9px;
        background: #2ecc71;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 1.6s infinite;
        vertical-align: middle;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    /* Summary strip */
    .summary-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    .summary-card {
        background: white;
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .summary-card .label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .summary-card .value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #003366;
        margin-top: 2px;
    }

    /* Hall card */
    .hall-card {
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.07);
        border-left: 5px solid;
        position: relative;
    }
    .hall-card.smooth    { background: #E8F5E9; border-color: #16A085; }
    .hall-card.busy      { background: #FFF8E1; border-color: #F39C12; }
    .hall-card.congested { background: #FFEBEE; border-color: #E74C3C; }

    .hall-card .name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 2px;
    }
    .hall-card .name-en {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 6px;
    }
    .hall-card .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .hall-card .status-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 10px;
        letter-spacing: 0.3px;
    }
    .hall-card.smooth .status-badge    { background: #16A085; color: white; }
    .hall-card.busy .status-badge      { background: #F39C12; color: white; }
    .hall-card.congested .status-badge { background: #E74C3C; color: white; }

    .hall-card .reco-badge {
        background: #FFD700;
        color: #5d4037;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 10px;
    }
    .hall-card .metrics {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 6px;
    }
    .hall-card .metric {
        font-size: 0.85rem;
        color: #444;
    }
    .hall-card .metric strong {
        font-size: 1rem;
        color: #1a1a1a;
    }
    .hall-card .congestion-bar {
        height: 6px;
        background: rgba(0,0,0,0.08);
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
    }
    .hall-card .congestion-fill {
        height: 100%;
        border-radius: 3px;
    }
    .hall-card.smooth .congestion-fill    { background: #16A085; }
    .hall-card.busy .congestion-fill      { background: #F39C12; }
    .hall-card.congested .congestion-fill { background: #E74C3C; }
    .hall-card .cong-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #666;
        margin-top: 3px;
    }

    /* Nudge banner */
    .nudge-banner {
        background: linear-gradient(135deg, #fff8e1, #fff3cd);
        border-left: 5px solid #F39C12;
        padding: 14px 18px;
        border-radius: 12px;
        font-size: 0.95rem;
        color: #5d4037;
        margin-top: 14px;
        box-shadow: 0 3px 8px rgba(243,156,18,0.18);
    }
    .nudge-banner strong { color: #E65100; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f7f8fa;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.markdown("### \u2699\ufe0f Simulation Controls")
st.sidebar.markdown("---")

lam_peak = st.sidebar.slider(
    "Peak Arrival Rate \u03bb\n(students / min)",
    min_value=0.3, max_value=1.8, value=1.0, step=0.1,
    help="Total lunch-peak arrival rate across all halls.",
)

nudge = st.sidebar.checkbox(
    "Enable Smart Nudge",
    value=True,
    help="Highlight the fastest hall in real time.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Live Parameters**

| Parameter | Value |
|---|---|
| \u03bb (total) | {lam_peak:.1f} /min |
| Halls tracked | 13 |
| Smart Nudge | {'ON \u2705' if nudge else 'OFF \u274c'} |
| Updated | {datetime.now().strftime('%H:%M:%S')} |
""")

st.sidebar.markdown("---")
with st.sidebar.expander("\U0001F4DA Model Details"):
    st.markdown("""
Each hall is modelled as an **M/G/1** queue. Real-time wait times are
computed via the **Pollaczek-Khinchine** formula:

$$E[W] = \\frac{\\lambda_i \\cdot E[S^2]}{2(1-\\rho_i)}, \\quad \\rho_i = \\lambda_i \\cdot \\mu_i$$

where \u03bb_i = \u03bb \u00d7 (hall's peak share) and \u03bc_i is the
hall-specific mean service time.

**Hall shares** reflect typical lunchtime popularity (slower halls
attract more arrivals).
""")

# Optional detailed simulation (collapsed by default)
with st.sidebar.expander("\U0001F4CA Run Detailed Simulation"):
    sim_hall_idx = st.selectbox(
        "Hall to simulate",
        options=range(len(DINING_HALLS)),
        format_func=lambda i: f"{DINING_HALLS[i][0]} ({DINING_HALLS[i][1]})",
    )
    run_sim = st.button("\u25B6 Simulate", use_container_width=True)


# ============================================================
# Main Page
# ============================================================
metrics = compute_all_metrics(lam_peak, nudge)
halls = metrics["halls"]
fastest_idx = metrics["fastest"]
slowest_idx = metrics["slowest"]
fastest = halls[0]  # already sorted ascending by wait
slowest = halls[-1]

# --- Top bar (phone header style) ---
st.markdown(f"""
<div class="top-bar">
    <h1>\U0001F37D\uFE0F HKU Smart Dining</h1>
    <div class="sub">
        <span class="live-dot"></span>
        LIVE &nbsp;\u2022&nbsp; Live Queue Status &nbsp;\u2022&nbsp; Updated {datetime.now().strftime('%H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Summary strip ---
fastest_name = fastest["cn"]
slowest_name = slowest["cn"]
fastest_wait = fastest["wait"]
slowest_wait = slowest["wait"]

st.markdown(f"""
<div class="summary-strip">
    <div class="summary-card">
        <div class="label">Fastest Hall</div>
        <div class="value" style="color:#16A085;">{fastest_name}</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px;">
            \u2705 ~{fastest_wait:.1f} min wait
        </div>
    </div>
    <div class="summary-card">
        <div class="label">Avg Campus Wait</div>
        <div class="value">{metrics["avg_wait"]:.1f} min</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px;">all 13 halls</div>
    </div>
    <div class="summary-card">
        <div class="label">Avoid</div>
        <div class="value" style="color:#E74C3C;">{slowest_name}</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px;">
            \u26a0 ~{slowest_wait:.1f} min wait
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Section header ---
st.markdown("### \U0001F4CD Live Hall Status  &nbsp;<span style='font-size:0.8rem;color:#888;font-weight:400;'>(sorted by current wait time)</span>", unsafe_allow_html=True)

# --- Render hall cards in 2-column grid ---
def render_card(hall, is_recommended=False):
    """Render a single hall card as HTML."""
    s = STATUS[hall["status"]]
    cn = hall["cn"]
    en = hall["en"]
    icon = hall["icon"]
    q = hall["q_len"]
    w = hall["wait"]
    rho_pct = min(hall["rho"] * 100, 100)
    reco_html = '<span class="reco-badge">\u2605 Recommended (Fastest)</span>' if is_recommended else ""

    wait_display = f"{w:.1f} min" if w < 900 else "Overloaded"
    q_display = f"~{int(round(q))}" if q < 900 else ">100"

    return f"""
    <div class="hall-card {hall['status']}">
        <div class="status-row">
            <div>
                <div class="name">{icon} {cn}</div>
                <div class="name-en">{en} &nbsp;\u2022&nbsp; \u03bc = {hall['mean_s']:.1f} min/student</div>
            </div>
            {reco_html if is_recommended else f'<span class="status-badge">{s["icon"]} {s["label"]}</span>'}
        </div>
        <div class="metrics">
            <span class="metric">\U0001F465 <strong>{q_display}</strong> ppl in queue</span>
            <span class="metric">\u23F1\uFE0F <strong>{wait_display}</strong> est. wait</span>
        </div>
        <div class="congestion-bar">
            <div class="congestion-fill" style="width:{rho_pct:.0f}%;"></div>
        </div>
        <div class="cong-label">
            <span>Congestion Level</span>
            <span>{rho_pct:.0f}%</span>
        </div>
    </div>
    """


# Layout: 2 columns, fast halls on top
# First render the recommended hall as a wide card
recommended_html = render_card(fastest, is_recommended=True)
st.markdown(recommended_html, unsafe_allow_html=True)

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# Remaining 12 halls in 2-column grid
remaining = [h for h in halls if h["idx"] != fastest_idx]
cols_per_row = 2
for row_start in range(0, len(remaining), cols_per_row):
    row = remaining[row_start:row_start + cols_per_row]
    cols = st.columns(len(row), gap="small")
    for col, hall in zip(cols, row):
        with col:
            st.markdown(render_card(hall, is_recommended=False), unsafe_allow_html=True)


# --- Nudge banner at bottom ---
if nudge:
    if fastest_wait < 5:
        nudge_text = (
            f"\U0001F4A1 <strong>Smart Nudge:</strong> All halls are flowing smoothly "
            f"({fastest_wait:.1f} min at {fastest_name}). No redirection needed."
        )
    else:
        time_saved = slowest_wait - fastest_wait
        nudge_text = (
            f"\U0001F4A1 <strong>Smart Nudge:</strong> Choosing <strong>{fastest_name}</strong> "
            f"saves you ~<strong>{time_saved:.0f} minutes</strong> vs. <strong>{slowest_name}</strong>. "
            f"Smart routing optimises campus flow \u2014 try the faster option!"
        )
else:
    nudge_text = (
        "\U0001F4A1 <strong>Smart Nudge:</strong> Enable the toggle in the sidebar to see "
        "real-time routing recommendations."
    )

st.markdown(f'<div class="nudge-banner">{nudge_text}</div>', unsafe_allow_html=True)


# --- Optional detailed simulation output ---
if run_sim:
    st.markdown("---")
    st.markdown(f"### \U0001F4C8 Detailed Simulation: {DINING_HALLS[sim_hall_idx][0]}")
    time_labels, q = simulate_dynamics(lam_peak, sim_hall_idx)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    cn, en, mean_s, share, _ = DINING_HALLS[sim_hall_idx]
    ax.plot(range(len(q)), q, color="#3498DB", linewidth=2)
    ax.fill_between(range(len(q)), q, alpha=0.15, color="#3498DB")
    ax.set_title(
        f"{cn} ({en}) \u2014 \u03bc={mean_s:.1f} min, \u03bb={lam_peak*share:.2f}/min, \u03c1={lam_peak*share*mean_s:.2f}",
        fontsize=12, fontweight="bold", color="#003366",
    )
    ax.set_xlabel("Time (min from 12:00)")
    ax.set_ylabel("Queue length (students)")
    ax.set_xticks([0, 30, 60, 89])
    ax.set_xticklabels(["12:00", "12:30", "13:00", "13:30"])
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.75rem;'>"
    "HKU Smart Dining &mdash; Decision Analytics Mini-Project &nbsp;\u2022&nbsp; "
    "M/G/1 Queue Simulation + Behavioural Nudge"
    "</div>",
    unsafe_allow_html=True,
)
