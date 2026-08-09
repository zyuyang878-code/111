"""
HKU Smart Dining: Live Queue Analytics & Nudge System
=====================================================
A Streamlit interactive web application that wraps the M/G/c queue
simulation engine and demonstrates a real-time behavioural-nudge
strategy for HKU dining-hall optimisation.

Run locally:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

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
# Constants & Model Parameters
# ============================================================
SERVICE_MEAN_B = 2.5      # Window B: minutes per student (default)
SERVICE_STD    = 0.5      # service-time std (lognormal)
SIM_MINUTES    = 90       # lunch peak: 12:00 - 13:30

C_RED   = "#E74C3C"
C_GREEN = "#16A085"
C_BLUE  = "#3498DB"
C_AMBER = "#F39C12"


# ============================================================
# Core Simulation Engine (extracted from hku_dining_simulation.py)
# ============================================================
def _lognormal_params(mean, std):
    """Convert (mean, std) to lognormal (mu, sigma)."""
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu    = np.log(mean) - sigma ** 2 / 2
    return mu, sigma


def mg1_mean_wait(lam, mean_s, std_s):
    """Pollaczek-Khinchine formula: E[W] for an M/G/1 queue.

    E[W] = lam * E[S^2] / (2 * (1 - rho)),  rho = lam * E[S]
    Returns 999.0 if the system is unstable (rho >= 1).
    """
    rho = lam * mean_s
    if rho >= 1:
        return 999.0
    e_s2 = std_s ** 2 + mean_s ** 2
    return lam * e_s2 / (2 * (1 - rho))


def _simulate_mgc(arrivals, services, n_windows=1):
    """Single M/G/c discrete-event run. Returns per-student wait times."""
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


def _quick_mc(lam, mean_s, std_s, n_sims=300, seed=42):
    """Quick Monte Carlo: return (mean, p95, all_waits) for an M/G/1 queue."""
    rng = np.random.default_rng(seed)
    mu, sigma = _lognormal_params(mean_s, std_s)
    all_waits = []
    for _ in range(n_sims):
        n_exp = rng.poisson(lam * SIM_MINUTES)
        if n_exp == 0:
            continue
        inter = rng.exponential(1.0 / lam, n_exp)
        arr = np.cumsum(inter)
        arr = arr[arr < SIM_MINUTES]
        serv = rng.lognormal(mu, sigma, len(arr))
        w = _simulate_mgc(arr, serv, 1)
        if len(w) > 0:
            all_waits.extend(w)
    if not all_waits:
        return 0.0, 0.0, np.array([])
    arr_w = np.array(all_waits)
    return float(np.mean(arr_w)), float(np.percentile(arr_w, 95)), arr_w


@st.cache_data(show_spinner=False)
def compute_metrics(lam_peak, mean_a, nudge):
    """Compute all metrics for both nudge-on and nudge-off scenarios.

    Returns a dict with wait times and P95 for both scenarios.
    """
    mean_b = SERVICE_MEAN_B

    # --- Current scenario (depends on nudge toggle) ---
    split_a_cur = 0.3 if nudge else 0.5
    lam_a_cur = lam_peak * split_a_cur
    lam_b_cur = lam_peak * (1 - split_a_cur)

    wait_a_cur = mg1_mean_wait(lam_a_cur, mean_a, SERVICE_STD)
    wait_b_cur = mg1_mean_wait(lam_b_cur, mean_b, SERVICE_STD)

    # Quick MC for P95 (combined across both windows)
    _, p95_a_cur, w_a_cur = _quick_mc(lam_a_cur, mean_a, SERVICE_STD)
    _, p95_b_cur, w_b_cur = _quick_mc(lam_b_cur, mean_b, SERVICE_STD)
    combined_cur = np.concatenate([w_a_cur, w_b_cur]) if len(w_a_cur) + len(w_b_cur) > 0 else np.array([0])
    p95_combined_cur = float(np.percentile(combined_cur, 95))

    # --- Baseline scenario (always no nudge, for comparison) ---
    split_a_base = 0.5
    lam_a_base = lam_peak * split_a_base
    lam_b_base = lam_peak * (1 - split_a_base)

    _, p95_a_base, w_a_base = _quick_mc(lam_a_base, mean_a, SERVICE_STD)
    _, p95_b_base, w_b_base = _quick_mc(lam_b_base, mean_b, SERVICE_STD)
    combined_base = np.concatenate([w_a_base, w_b_base]) if len(w_a_base) + len(w_b_base) > 0 else np.array([0])
    p95_combined_base = float(np.percentile(combined_base, 95))

    p95_reduction = (1 - p95_combined_cur / p95_combined_base) * 100 if p95_combined_base > 0 else 0

    return {
        "wait_a": wait_a_cur,
        "wait_b": wait_b_cur,
        "p95_combined": p95_combined_cur,
        "p95_base": p95_combined_base,
        "p95_reduction": p95_reduction,
        "lam_a": lam_a_cur,
        "lam_b": lam_b_cur,
    }


def simulate_dynamics(lam_total, mean_a, mean_b, nudge, seed=42):
    """Run a single detailed discrete-event simulation.

    Returns (DataFrame of per-minute queue lengths, waits_a, waits_b, arrivals_a, arrivals_b).
    """
    rng = np.random.default_rng(seed)

    split_a = 0.3 if nudge else 0.5
    lam_a = lam_total * split_a
    lam_b = lam_total * (1 - split_a)

    # --- Generate arrivals ---
    def gen_arrivals(lam):
        n = rng.poisson(lam * SIM_MINUTES)
        if n == 0:
            return np.array([])
        inter = rng.exponential(1.0 / max(lam, 1e-6), n)
        arr = np.cumsum(inter)
        return arr[arr < SIM_MINUTES]

    arr_a = gen_arrivals(lam_a)
    arr_b = gen_arrivals(lam_b)

    # --- Service times ---
    mu_a, sig_a = _lognormal_params(mean_a, SERVICE_STD)
    mu_b, sig_b = _lognormal_params(mean_b, SERVICE_STD)
    serv_a = rng.lognormal(mu_a, sig_a, len(arr_a)) if len(arr_a) > 0 else np.array([])
    serv_b = rng.lognormal(mu_b, sig_b, len(arr_b)) if len(arr_b) > 0 else np.array([])

    # --- Wait times ---
    waits_a = _simulate_mgc(arr_a, serv_a, 1)
    waits_b = _simulate_mgc(arr_b, serv_b, 1)

    # --- Track queue length at each minute ---
    def track_queue_len(arrivals, services):
        n = len(arrivals)
        if n == 0:
            return np.zeros(SIM_MINUTES, dtype=int)
        events = []
        window_free = 0.0
        for i in range(n):
            arr = arrivals[i]
            start = max(arr, window_free)
            end = start + services[i]
            events.append((arr, 1))    # arrival
            events.append((end, -1))   # departure
            window_free = end
        events.sort()
        q_len = np.zeros(SIM_MINUTES, dtype=int)
        cur_q = 0
        ei = 0
        for t in range(SIM_MINUTES):
            while ei < len(events) and events[ei][0] <= t:
                cur_q += events[ei][1]
                ei += 1
            q_len[t] = max(cur_q, 0)
        return q_len

    qa = track_queue_len(arr_a, serv_a)
    qb = track_queue_len(arr_b, serv_b)

    time_labels = []
    for t in range(SIM_MINUTES):
        if t < 60:
            time_labels.append(f"12:{t:02d}")
        else:
            time_labels.append(f"13:{t - 60:02d}")

    df = pd.DataFrame({
        "Time": time_labels,
        "Window A (Queue Length)": qa,
        "Window B (Queue Length)": qb,
    })
    return df, waits_a, waits_b, arr_a, arr_b


# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #003366;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem;
        color: #555;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Nudge banner */
    .nudge-banner {
        background: linear-gradient(135deg, #fff8e1, #fff3cd);
        border-left: 4px solid #F39C12;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 1.05rem;
        color: #5d4037;
        margin-top: 1rem;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #003366;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #003366;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #003366;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar — Interactive Controls
# ============================================================
st.sidebar.markdown('<p class="sidebar-title">Simulation Controls</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

lam_peak = st.sidebar.slider(
    "Peak Arrival Rate \u03bb\n(students / min)",
    min_value=0.3, max_value=1.5, value=1.0, step=0.1,
    help="Lunch-peak arrival rate. 1.0 = 60 students/hr (baseline)."
)

mean_a = st.sidebar.slider(
    "Counter A Service Time\n(min / student)",
    min_value=1.5, max_value=4.0, value=3.0, step=0.1,
    help="Average time Window A takes per student. Higher = slower service."
)

nudge = st.sidebar.checkbox(
    "Enable Smart Nudge",
    value=True,
    help="When ON, 70% of students are guided to Window B (faster). When OFF, 50/50 split."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Current Configuration**

| Parameter | Value |
|---|---|
| \u03bb (total) | {lam_peak:.1f} /min |
| Window A \u03bc | {mean_a:.1f} min |
| Window B \u03bc | {SERVICE_MEAN_B:.1f} min |
| Nudge | {'ON \u2705' if nudge else 'OFF \u274c'} |
| Split (A:B) | {'30:70' if nudge else '50:50'} |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
*Model: M/G/1 queue per window*
*Poisson arrivals, lognormal service*
*90-min peak window (12:00-13:30)*
""")


# ============================================================
# Main Page
# ============================================================
st.markdown('<p class="main-title">\U0001F37D\uFE0F HKU Smart Dining: Live Queue Analytics & Nudge System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time M/G/c queue simulation with behavioural-nudge decision support &mdash; Centennial Campus Dining Hall</p>', unsafe_allow_html=True)

# --- Compute metrics ---
metrics = compute_metrics(lam_peak, mean_a, nudge)

# --- Metric Cards ---
st.markdown('<p class="section-header">Real-Time Metrics Dashboard</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    wait_a_display = f"{metrics['wait_a']:.1f} min" if metrics['wait_a'] < 900 else "OVERLOADED"
    delta_a = f"\u03bc_A = {mean_a:.1f} min" if metrics['wait_a'] < 900 else "\u26a0\ufe0f \u03c1 \u2265 1"
    st.metric(
        label="\U0001F534 Window A Wait Time",
        value=wait_a_display,
        delta=delta_a,
    )

with col2:
    wait_b_display = f"{metrics['wait_b']:.1f} min" if metrics['wait_b'] < 900 else "OVERLOADED"
    delta_b = f"\u03bc_B = {SERVICE_MEAN_B:.1f} min" if metrics['wait_b'] < 900 else "\u26a0\ufe0f \u03c1 \u2265 1"
    st.metric(
        label="\U0001F7E2 Window B Wait Time",
        value=wait_b_display,
        delta=delta_b,
    )

with col3:
    reduction = metrics["p95_reduction"]
    if nudge:
        p95_delta = f"P95: {metrics['p95_combined']:.1f} \u2193 from {metrics['p95_base']:.1f} min"
    else:
        p95_delta = f"P95: {metrics['p95_combined']:.1f} min (baseline)"
    st.metric(
        label="\U0001F4C8 P95 Wait Time Reduction",
        value=f"-{reduction:.1f}%" if nudge else "0.0%",
        delta=p95_delta,
        delta_color="inverse" if nudge else "off",
    )

st.markdown("---")

# --- Simulation Run Button ---
st.markdown('<p class="section-header">Queue Dynamics Simulation</p>', unsafe_allow_html=True)

st.markdown("""
Click the button below to run a real-time discrete-event simulation of the
lunch peak (12:00 - 13:30). The chart shows per-minute **queue length** at
each service window.
""")

if st.button("\u25B6 Run Real-Time Queue Simulation", type="primary", use_container_width=False):
    with st.spinner("Running M/G/c discrete-event simulation (90 min peak)..."):
        df, w_a, w_b, arr_a, arr_b = simulate_dynamics(lam_peak, mean_a, SERVICE_MEAN_B, nudge)

    # --- Queue dynamics line chart ---
    st.markdown("##### Queue Length Over Peak Hours")

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Shade peak zone
    ax.axvspan(0, SIM_MINUTES, alpha=0.03, color="orange")

    ax.plot(range(SIM_MINUTES), df["Window A (Queue Length)"],
            color=C_RED, linewidth=2.2, label=f"Window A (\u03bc={mean_a:.1f} min)")
    ax.plot(range(SIM_MINUTES), df["Window B (Queue Length)"],
            color=C_GREEN, linewidth=2.2, label=f"Window B (\u03bc={SERVICE_MEAN_B:.1f} min)")

    # Fill
    ax.fill_between(range(SIM_MINUTES), df["Window A (Queue Length)"], alpha=0.1, color=C_RED)
    ax.fill_between(range(SIM_MINUTES), df["Window B (Queue Length)"], alpha=0.1, color=C_GREEN)

    # X-axis labels
    tick_positions = [0, 15, 30, 45, 60, 75, 89]
    tick_labels = ["12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Queue Length (students)", fontsize=11)
    title_suffix = "with Smart Nudge (30:70 split)" if nudge else "no nudge (50:50 split)"
    ax.set_title(f"Queue Dynamics Over Lunch Peak \u2014 {title_suffix}",
                 fontsize=13, fontweight="bold", color="#003366")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(-1, SIM_MINUTES)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Summary stats ---
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Students Served (A)", f"{len(w_a)}")
    with col_s2:
        st.metric("Students Served (B)", f"{len(w_b)}")
    with col_s3:
        avg_wait = np.mean(np.concatenate([w_a, w_b])) if len(w_a) + len(w_b) > 0 else 0
        st.metric("Avg Wait (combined)", f"{avg_wait:.1f} min")
    with col_s4:
        max_wait = np.max(np.concatenate([w_a, w_b])) if len(w_a) + len(w_b) > 0 else 0
        st.metric("Max Wait", f"{max_wait:.1f} min")

    # --- Wait-time distribution ---
    st.markdown("##### Wait-Time Distribution")
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))

    all_w = np.concatenate([w_a, w_b]) if len(w_a) + len(w_b) > 0 else np.array([0])
    bins = np.linspace(0, max(np.max(all_w), 10), 30)
    ax2.hist(w_a, bins=bins, alpha=0.6, color=C_RED, label=f"Window A (mean={np.mean(w_a):.1f} min)" if len(w_a) > 0 else "Window A")
    ax2.hist(w_b, bins=bins, alpha=0.6, color=C_GREEN, label=f"Window B (mean={np.mean(w_b):.1f} min)" if len(w_b) > 0 else "Window B")
    ax2.set_xlabel("Wait Time (min)", fontsize=11)
    ax2.set_ylabel("Number of Students", fontsize=11)
    ax2.set_title("Distribution of Student Wait Times", fontsize=12, fontweight="bold", color="#003366")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

else:
    st.info("\u23f1\ufe0f Click **\u25b6 Run Real-Time Queue Simulation** to generate the queue dynamics chart.")


# --- Nudge Banner ---
st.markdown("")
if nudge and metrics["wait_a"] < 900 and metrics["wait_b"] < 900:
    diff = abs(metrics["wait_a"] - metrics["wait_b"])
    if diff > 0.5:
        faster = "A" if metrics["wait_a"] < metrics["wait_b"] else "B"
        slower = "B" if faster == "A" else "A"
        nudge_text = (
            f"\U0001F4A1 <strong>Smart Nudge:</strong> Window {faster} is "
            f"<strong>{diff:.0f} min faster</strong> than Window {slower}. "
            f"Choosing Window {faster} optimises campus flow!"
        )
    else:
        nudge_text = (
            "\U0001F4A1 <strong>Smart Nudge:</strong> Both windows have similar wait times. "
            "Flow is balanced \u2014 no redirection needed."
        )
else:
    nudge_text = (
        "\U0001F4A1 <strong>Smart Nudge:</strong> Enable the nudge toggle in the sidebar "
        "to see real-time wait-time optimisation recommendations."
    )

st.markdown(f'<div class="nudge-banner">{nudge_text}</div>', unsafe_allow_html=True)


# --- Model Info (expandable) ---
with st.expander("\U0001F4DA Model Details & Methodology"):
    st.markdown("""
**Queue Model: M/G/1 per window**
- **M** (Markovian/Poisson arrivals): inter-arrival times ~ Exponential(\u03bb)
- **G** (General service): service times ~ Lognormal(\u03bc, \u03c3) \u2014 right-skewed, realistic
- **1** server per window

**Pollaczek-Khinchine Formula** (real-time estimate):

$$E[W] = \\frac{\\lambda \\cdot E[S^2]}{2(1 - \\rho)}, \\quad \\rho = \\lambda \\cdot E[S]$$

**Smart Nudge Mechanism:**
- **OFF**: students split 50/50 between windows (blind choice)
- **ON**: the mini-program displays real-time queue lengths, guiding 70% of students
  to the faster window \u2014 effectively reshaping \u03bb(t) from a sharp peak to a flattened plateau

**Monte Carlo Validation:**
- 300 replications per scenario for P95 estimation
- 90-minute simulation window (12:00 \u2013 13:30 lunch peak)
- Lognormal service: mean = window-specific, std = 0.5 min

| Parameter | Symbol | Default | Adjustable |
|---|---|---|---|
| Peak arrival rate | \u03bb | 1.0 /min (60/hr) | \u2705 Sidebar |
| Window A service mean | \u03bc_A | 3.0 min | \u2705 Sidebar |
| Window B service mean | \u03bc_B | 2.5 min | fixed |
| Service std | \u03c3 | 0.5 min | fixed |
| Sim duration | T | 90 min | fixed |
""")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem;">
HKU Smart Dining &mdash; Decision Analytics Mini-Project &nbsp;|&nbsp;
M/G/c Queue Simulation + Behavioural Nudge &nbsp;|&nbsp;
Built with Streamlit
</div>
""", unsafe_allow_html=True)
