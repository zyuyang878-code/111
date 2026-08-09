"""
HKU Smart Dining: Live Queue Analytics & Nudge System
=====================================================
Multi-window dining hall queue simulator with real-time behavioural
nudge. Renders 13 real HKU dining halls as mobile-style cards.

Features:
  - Halls ordered by popularity Tier (1=最火爆 → 3=较少人), shuffled within tier.
  - Student Hub sidebar with "我的位置" picker + walking-time estimates.
  - Real-time M/G/1 wait estimates via Pollaczek-Khinchine formula.
  - Smart Nudge banner recommending the fastest hall.

Run locally:
    streamlit run hku_canteen_app.py
"""

import numpy as np
import streamlit as st
from datetime import datetime

# matplotlib is intentionally NOT imported at module level. Streamlit Cloud
# runs matplotlib 3.10+/3.11, which removed/broke `rcParams["font.family"]`.
# We lazy-import matplotlib ONLY when the user clicks the detailed-simulation
# button, so the home page never touches the broken API.


# ============================================================
# Constants — 13 HKU Dining Halls
# ============================================================
# (Chinese, English, service_mean_min, peak_share, icon, campus_zone)
DINING_HALLS = [
    ("庄月明食堂",       "Meng Wah Complex",          3.5, 0.22, "\U0001F35C", "main"),
    ("学生会食堂",       "Student Union Canteen",     3.2, 0.18, "\U0001F371", "main"),
    ("方树泉食堂",       "Fong Shu Chuen Hall",       2.8, 0.12, "\U0001F961", "main"),
    ("亚洲滋味餐厅",     "Asian Flavours",            2.6, 0.08, "\U0001F372", "main"),
    ("一念素食",         "Yi Nian Vegetarian",        2.5, 0.06, "\U0001F957", "main"),
    ("cafe330",          "Cafe 330",                  2.5, 0.06, "\u2615",     "centenary"),
    ("Coffee Academics", "Coffee Academics",          2.2, 0.05, "\u2615",     "centenary"),
    ("U Deli",           "U Deli",                    2.0, 0.05, "\U0001F96A", "main"),
    ("alfafa cafe",      "Alfafa Cafe",               2.0, 0.04, "\U0001F96A", "centenary"),
    ("Sandwich Club",    "Sandwich Club",             2.0, 0.04, "\U0001F96A", "main"),
    ("Super Sandwiches", "Super Sandwiches",          1.8, 0.03, "\U0001F96A", "mtr"),
    ("Subway",           "Subway",                    2.0, 0.04, "\U0001F956", "mtr"),
    ("星巴克",           "Starbucks",                 1.5, 0.03, "\u2615",     "centenary"),
]

# Popularity tiers (small周 spec):
#   Tier 1 = 最火爆,  Tier 2 = 受欢迎,  Tier 3 = 较少人
TIERS = {
    1: ["庄月明食堂", "方树泉食堂"],
    2: ["一念素食", "alfafa cafe"],
    3: [
        "星巴克", "cafe330", "Super Sandwiches", "学生会食堂",
        "U Deli", "Sandwich Club", "Coffee Academics",
        "亚洲滋味餐厅", "Subway",
    ],
}

TIER_LABELS = {
    1: "\U0001F525 Tier 1 — 最火爆",
    2: "\U0001F4C8 Tier 2 — 受欢迎",
    3: "\u2705 Tier 3 — 较少人",
}

# HKU location picker — (Chinese, English, campus_zone)
HKU_LOCATIONS = [
    ("百周年校园",      "Centenary Campus",                "centenary"),
    ("本部大楼",        "Main Building",                   "main"),
    ("医学院",          "Li Ka Shing Faculty of Medicine", "medical"),
    ("港大地铁站",      "HKU MTR Station",                 "mtr"),
    ("庄月明楼",        "Meng Wah Complex",                "main"),
    ("邵逸夫楼",        "Run Run Shaw Tower",              "main"),
    ("学生会大楼",      "Student Union Building",          "main"),
    ("百周年体育中心",  "Centennial Sports Centre",        "centenary"),
]

# Walking minutes between campus zones (rough HKU hillside estimate)
WALKING_MIN = {
    ("centenary", "centenary"):  2,
    ("centenary", "main"):       7,
    ("centenary", "medical"):   12,
    ("centenary", "mtr"):        5,
    ("main",      "centenary"):  7,
    ("main",      "main"):       3,
    ("main",      "medical"):   10,
    ("main",      "mtr"):        8,
    ("medical",   "centenary"): 12,
    ("medical",   "main"):      10,
    ("medical",   "medical"):    3,
    ("medical",   "mtr"):       10,
    ("mtr",       "centenary"):  5,
    ("mtr",       "main"):       8,
    ("mtr",       "medical"):   10,
    ("mtr",       "mtr"):        1,
}


def walking_minutes(from_zone, to_zone):
    return WALKING_MIN.get((from_zone, to_zone), 8)


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


def hall_tier(cn_name):
    for tier, names in TIERS.items():
        if cn_name in names:
            return tier
    return 3


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
    chinese, english, mean_s, share, _icon, _zone = DINING_HALLS[hall_idx]
    lam = lam_total * share
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + (SERVICE_STD / mean_s) ** 2))
    mu = np.log(mean_s) - sigma ** 2 / 2
    SIM_MIN = 90

    n_arr = rng.poisson(lam * SIM_MIN)
    if n_arr == 0:
        return ([f"12:{t:02d}" if t < 60 else f"13:{t-60:02d}" for t in range(SIM_MIN)],
                np.zeros(SIM_MIN, dtype=int))

    inter = rng.exponential(1.0 / max(lam, 1e-6), n_arr)
    arr = np.cumsum(inter)
    arr = arr[arr < SIM_MIN]
    serv = rng.lognormal(mu, sigma, len(arr))

    events = []
    window_free = 0.0
    for i in range(len(arr)):
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
def compute_all_metrics(lam_peak, nudge, shuffle_seed=42):
    """Compute per-hall wait times. Return halls in Tier order
    (Tier 1 → 2 → 3), randomized within tier. Fastest/slowest by
    wait time for the Nudge banner.
    """
    rng = np.random.default_rng(shuffle_seed)

    results = []
    for i, (cn, en, mean_s, share, icon, zone) in enumerate(DINING_HALLS):
        lam_i = lam_peak * share
        wait = mg1_mean_wait(lam_i, mean_s, SERVICE_STD)
        q_len = estimate_queue_length(lam_i, mean_s)
        rho = lam_i * mean_s
        results.append({
            "idx": i, "cn": cn, "en": en, "mean_s": mean_s,
            "share": share, "icon": icon, "zone": zone, "lam": lam_i,
            "wait": wait, "q_len": q_len, "rho": rho,
            "status": classify(wait),
            "tier": hall_tier(cn),
        })

    sorted_by_wait = sorted(results, key=lambda r: r["wait"])
    fastest = sorted_by_wait[0]
    slowest = sorted_by_wait[-1]
    avg_wait = float(np.mean([r["wait"] for r in results]))

    ordered = []
    for tier in [1, 2, 3]:
        tier_halls = [r for r in results if r["tier"] == tier]
        rng.shuffle(tier_halls)
        ordered.extend(tier_halls)

    return {
        "halls": ordered,
        "fastest": fastest,
        "slowest": slowest,
        "avg_wait": avg_wait,
    }


# ============================================================
# Custom CSS — Mobile App Look + Sidebar Fix
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f0f2f5 0%, #e8eaf0 100%);
    }
    .block-container {
        max-width: 780px;
        padding-top: 1.2rem;
    }

    /* ---------- SIDEBAR FIX ---------- */
    /* Force the sidebar to a clean light theme so all text is readable. */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox > div > label,
    section[data-testid="stSidebar"] .stSlider > div > label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox span {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #003366 !important;
    }
    /* Sidebar expander: light background, dark text */
    section[data-testid="stSidebar"] details,
    section[data-testid="stSidebar"] .stExpander {
        background: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] details summary,
    section[data-testid="stSidebar"] details summary span,
    section[data-testid="stSidebar"] details summary p {
        color: #003366 !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] details[open] summary {
        border-bottom: 1px solid #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    /* Sidebar tables: ensure cells readable */
    section[data-testid="stSidebar"] table {
        color: #1a1a1a !important;
        background: #ffffff !important;
    }
    section[data-testid="stSidebar"] table th,
    section[data-testid="stSidebar"] table td {
        color: #1a1a1a !important;
        background: #ffffff !important;
        border-color: #e0e0e0 !important;
    }
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: #003366 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #1a4789 !important;
        color: #ffffff !important;
    }
    /* The "expand sidebar" arrow icon */
    button[kind="header"] {
        color: #003366 !important;
    }

    /* ---------- Top bar ---------- */
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

    /* ---------- Summary strip ---------- */
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
        font-size: 1.05rem;
        font-weight: 700;
        color: #003366;
        margin-top: 2px;
    }

    /* ---------- Tier header ---------- */
    .tier-header {
        margin: 14px 0 8px 0;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 700;
        color: white;
    }
    .tier-1 { background: linear-gradient(135deg, #C62828, #E74C3C); }
    .tier-2 { background: linear-gradient(135deg, #E65100, #F39C12); }
    .tier-3 { background: linear-gradient(135deg, #2E7D32, #16A085); }

    /* ---------- Hall card ---------- */
    .hall-card {
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.07);
        border-left: 5px solid;
        position: relative;
    }
    .hall-card.smooth    { background: #E8F5E9; border-color: #16A085; }
    .hall-card.busy      { background: #FFF8E1; border-color: #F39C12; }
    .hall-card.congested { background: #FFEBEE; border-color: #E74C3C; }

    .hall-card .name {
        font-size: 1.0rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 2px;
    }
    .hall-card .name .tier-tag {
        font-size: 0.65rem;
        background: rgba(0,0,0,0.08);
        padding: 1px 6px;
        border-radius: 8px;
        margin-left: 4px;
        color: #555;
        font-weight: 600;
    }
    .hall-card .name-en {
        font-size: 0.72rem;
        color: #666;
        margin-bottom: 6px;
    }
    .hall-card .name-en .walk {
        color: #1976D2;
        font-weight: 600;
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

    /* ---------- Nudge banner ---------- */
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
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar — Student Hub
# ============================================================
st.sidebar.markdown("### \U0001F393 学生页面 / Student Hub")
st.sidebar.markdown("---")

# --- My Location picker ---
st.sidebar.markdown("#### \U0001F4CD 我的位置 / My Location")
loc_options = [f"{cn} ({en})" for cn, en, _ in HKU_LOCATIONS]
selected_loc_str = st.sidebar.selectbox(
    "选择你当前所在区域",
    options=loc_options,
    index=0,
    label_visibility="collapsed",
)
selected_idx = loc_options.index(selected_loc_str)
user_zone = HKU_LOCATIONS[selected_idx][2]
user_loc_cn = HKU_LOCATIONS[selected_idx][0]

st.sidebar.markdown("---")

# --- Simulation Controls ---
st.sidebar.markdown("#### \u2699\ufe0f 模拟参数 / Simulation Controls")

lam_peak = st.sidebar.slider(
    "高峰到达率 \u03bb (students / min)",
    min_value=0.3, max_value=1.8, value=1.0, step=0.1,
    help="Total lunch-peak arrival rate across all halls.",
)

nudge = st.sidebar.checkbox(
    "开启智能推荐 (Smart Nudge)",
    value=True,
    help="Highlight the fastest hall in real time.",
)

# --- Reshuffle button ---
if st.sidebar.button("\U0001F500 重新打乱同梯队顺序", use_container_width=True):
    st.session_state["shuffle_seed"] = int(datetime.now().timestamp() * 1000) % 100000

if "shuffle_seed" not in st.session_state:
    st.session_state["shuffle_seed"] = 42
shuffle_seed = st.session_state["shuffle_seed"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**实时参数 / Live Parameters**

| 参数 | 值 |
|---|---|
| 我的位置 | {user_loc_cn} |
| \u03bb (总到达率) | {lam_peak:.1f} /min |
| 食堂数量 | 13 |
| Smart Nudge | {'ON \u2705' if nudge else 'OFF \u274c'} |
| 排序种子 | #{shuffle_seed} |
| 更新时间 | {datetime.now().strftime('%H:%M:%S')} |
""")

st.sidebar.markdown("---")
with st.sidebar.expander("\U0001F4DA 模型说明 / Model Details"):
    st.markdown("""
每个食堂建模为 **M/G/1** 队列。实时等待时间通过 **Pollaczek-Khinchine** 公式计算:

$$E[W] = \\frac{\\lambda_i \\cdot E[S^2]}{2(1-\\rho_i)}, \\quad \\rho_i = \\lambda_i \\cdot \\mu_i$$

其中 $\\lambda_i = \\lambda \\times$ (食堂高峰份额), $\\mu_i$ 是食堂特定的平均服务时间。

**食堂梯队 / Hall Tiers**:
- **Tier 1** (最火爆): 庄月明食堂、方树泉食堂
- **Tier 2** (受欢迎): 一念素食、Alfafa cafe
- **Tier 3** (较少人): 其余 9 个

排序规则: 按 Tier 1 → 2 → 3 顺序排列，同一梯队内**每次刷新随机打乱**。
""")

with st.sidebar.expander("\U0001F4CA 详细仿真 / Run Detailed Simulation"):
    sim_hall_idx = st.selectbox(
        "选择食堂仿真 / Hall to simulate",
        options=range(len(DINING_HALLS)),
        format_func=lambda i: f"{DINING_HALLS[i][0]} ({DINING_HALLS[i][1]})",
    )
    run_sim = st.button("\u25B6 运行仿真 / Simulate", use_container_width=True)


# ============================================================
# Main Page
# ============================================================
metrics = compute_all_metrics(lam_peak, nudge, shuffle_seed=shuffle_seed)
halls = metrics["halls"]
fastest = metrics["fastest"]
slowest = metrics["slowest"]

# --- Top bar ---
st.markdown(f"""
<div class="top-bar">
    <h1>\U0001F37D\uFE0F HKU Smart Dining</h1>
    <div class="sub">
        <span class="live-dot"></span>
        LIVE &nbsp;\u2022&nbsp; 实时排队状态 &nbsp;\u2022&nbsp;
        你的位置: {user_loc_cn} &nbsp;\u2022&nbsp; {datetime.now().strftime('%H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Summary strip ---
fastest_name = fastest["cn"]
slowest_name = slowest["cn"]
fastest_wait = fastest["wait"]
slowest_wait = slowest["wait"]
fastest_walk = walking_minutes(user_zone, fastest["zone"])
slowest_walk = walking_minutes(user_zone, slowest["zone"])

st.markdown(f"""
<div class="summary-strip">
    <div class="summary-card">
        <div class="label">推荐 / Recommended</div>
        <div class="value" style="color:#16A085;">{fastest_name}</div>
        <div style="font-size:0.72rem;color:#666;margin-top:2px;">
            \u2705 ~{fastest_wait:.1f} min wait &nbsp;\u2022&nbsp; \U0001F6B6 {fastest_walk} min walk
        </div>
    </div>
    <div class="summary-card">
        <div class="label">平均等待 / Avg Wait</div>
        <div class="value">{metrics["avg_wait"]:.1f} min</div>
        <div style="font-size:0.72rem;color:#666;margin-top:2px;">all 13 halls</div>
    </div>
    <div class="summary-card">
        <div class="label">\u26a0\ufe0f 警惕 / Avoid</div>
        <div class="value" style="color:#E74C3C;">{slowest_name}</div>
        <div style="font-size:0.72rem;color:#666;margin-top:2px;">
            \u26a0 ~{slowest_wait:.1f} min wait &nbsp;\u2022&nbsp; \U0001F6B6 {slowest_walk} min walk
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Section header ---
st.markdown("### \U0001F4CD 实时食堂状态 / Live Hall Status")
st.markdown(
    "<div style='font-size:0.8rem;color:#888;margin-top:-8px;margin-bottom:8px;'>"
    "按人气梯队排列 (Tier 1 → 2 → 3)，同梯队内随机排序 · "
    "点击侧边栏「\U0001F500 重新打乱」刷新顺序"
    "</div>",
    unsafe_allow_html=True,
)


# --- Render hall cards grouped by Tier ---
def render_card(hall, is_recommended=False, user_zone_val=None):
    """Render a single hall card as HTML."""
    s = STATUS[hall["status"]]
    cn = hall["cn"]
    en = hall["en"]
    icon = hall["icon"]
    q = hall["q_len"]
    w = hall["wait"]
    rho_pct = min(hall["rho"] * 100, 100)
    tier = hall["tier"]
    walk = walking_minutes(user_zone_val, hall["zone"]) if user_zone_val else None

    reco_html = ('<span class="reco-badge">\u2605 Recommended (Fastest)</span>'
                 if is_recommended else
                 f'<span class="status-badge">{s["icon"]} {s["label"]}</span>')

    wait_display = f"{w:.1f} min" if w < 900 else "Overloaded"
    q_display = f"~{int(round(q))}" if q < 900 else ">100"
    walk_display = f"\U0001F6B6 {walk} min" if walk is not None else ""
    tier_tag = f'<span class="tier-tag">T{tier}</span>'

    return f"""
    <div class="hall-card {hall['status']}">
        <div class="status-row">
            <div>
                <div class="name">{icon} {cn} {tier_tag}</div>
                <div class="name-en">{en} &nbsp;\u2022&nbsp; \u03bc = {hall['mean_s']:.1f} min/student
                    &nbsp;\u2022&nbsp; <span class="walk">{walk_display}</span></div>
            </div>
            {reco_html}
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


def flush_row(cards_html):
    """Render a buffered row of cards in a 2-column grid."""
    if not cards_html:
        return
    cols = st.columns(len(cards_html), gap="small")
    for col, html in zip(cols, cards_html):
        with col:
            st.markdown(html, unsafe_allow_html=True)


current_tier = None
buffer = []

for hall in halls:
    if hall["tier"] != current_tier:
        # flush previous tier
        flush_row(buffer)
        buffer = []
        current_tier = hall["tier"]
        st.markdown(
            f'<div class="tier-header tier-{current_tier}">{TIER_LABELS[current_tier]}</div>',
            unsafe_allow_html=True,
        )

    is_reco = nudge and hall["idx"] == fastest["idx"]
    buffer.append(render_card(hall, is_recommended=is_reco, user_zone_val=user_zone))

# flush last tier
flush_row(buffer)


# --- Nudge banner ---
if nudge:
    if fastest_wait < 5:
        nudge_text = (
            f"\U0001F4A1 <strong>Smart Nudge:</strong> 所有食堂都很流畅 "
            f"(最快 {fastest_wait:.1f} min @ {fastest_name}). 无需重新路由。"
        )
    else:
        time_saved = slowest_wait - fastest_wait
        nudge_text = (
            f"\U0001F4A1 <strong>Smart Nudge:</strong> 选择 <strong>{fastest_name}</strong> "
            f"可比 <strong>{slowest_name}</strong> 节省 ~<strong>{time_saved:.0f} 分钟</strong>排队时间 "
            f"(步行 {fastest_walk} min vs {slowest_walk} min). "
            f"智能路由优化校园人流 \u2014 跟着推荐走!"
        )
else:
    nudge_text = (
        "\U0001F4A1 <strong>Smart Nudge:</strong> 在侧边栏开启 Smart Nudge "
        "查看实时路由推荐。"
    )

st.markdown(f'<div class="nudge-banner">{nudge_text}</div>', unsafe_allow_html=True)


# --- Optional detailed simulation output ---
if run_sim:
    # Lazy-import matplotlib here so the home page never touches the
    # matplotlib 3.10+ font.rcParams API that breaks on Streamlit Cloud.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("matplotlib is not installed. Add `matplotlib` to requirements.txt.")
        st.stop()
    plt.rcParams.setdefault("font.sans-serif",
                            ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"])
    plt.rcParams.setdefault("font.family", "sans-serif")

    st.markdown("---")
    st.markdown(f"### \U0001F4C8 详细仿真: {DINING_HALLS[sim_hall_idx][0]}")
    time_labels, q = simulate_dynamics(lam_peak, sim_hall_idx)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    cn, en, mean_s, share, _icon, _zone = DINING_HALLS[sim_hall_idx]
    ax.plot(range(len(q)), q, color="#3498DB", linewidth=2)
    ax.fill_between(range(len(q)), q, alpha=0.15, color="#3498DB")
    ax.set_title(
        f"{cn} ({en}) \u2014 \u03bc={mean_s:.1f} min, "
        f"\u03bb={lam_peak*share:.2f}/min, \u03c1={lam_peak*share*mean_s:.2f}",
        fontsize=12, fontweight="bold", color="#003366",
    )
    ax.set_xlabel("Time (min from 12:00)")
    ax.set_ylabel("Queue length (students)")
    ax.set_xticks([0, 30, 60, 89])
    ax.set_xticklabels(["12:00", "12:30", "13:00", "13:30"])
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.75rem;'>"
    "HKU Smart Dining &mdash; Decision Analytics Mini-Project &nbsp;\u2022&nbsp; "
    "M/G/1 Queue Simulation + Behavioural Nudge + Tiered Popularity Ranking"
    "</div>",
    unsafe_allow_html=True,
)
