"""
DDSF Demo – Data-Driven Decision Support Framework for Corporate KYC
Transaction Monitoring at Foreign Banks in Vietnam
Author: Hoang Yen Chi | Thesis Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDSF | AML Decision Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",   "role": "Admin",            "name": "System Admin"},
    "officer": {"password": "officer123", "role": "Compliance Officer","name": "Nguyen Thi Lan"},
    "senior":  {"password": "senior123",  "role": "Senior Officer",   "name": "Tran Van Minh"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION  (seed-based → deterministic)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_accounts(n=250, seed=42):
    rng = np.random.default_rng(seed)

    # ── L1: Volume / Velocity ─────────────────────────────────────────────
    transaction_count  = rng.integers(5, 120, n)
    total_amount       = rng.lognormal(11, 1.5, n)
    avg_amount         = total_amount / transaction_count
    amount_cv          = rng.beta(2, 5, n) * 3           # 0–3, right-skewed
    max_amount         = avg_amount * rng.uniform(2, 8, n)
    min_amount         = avg_amount * rng.uniform(0.05, 0.5, n)
    daily_velocity     = transaction_count / rng.uniform(15, 90, n)

    # ── L2: Behavioural ──────────────────────────────────────────────────
    unique_to_banks    = rng.integers(1, 18, n)
    unique_from_banks  = rng.integers(1, 10, n)
    weekend_ratio      = rng.beta(2, 8, n)
    night_ratio        = rng.beta(1, 9, n)
    cross_border_ratio = rng.beta(2, 6, n)
    amount_entropy     = rng.uniform(0, 3, n)
    currency_diversity = rng.integers(1, 6, n)
    fx_mismatch_flag   = (rng.random(n) < 0.25).astype(int)
    cycle_flag         = (rng.random(n) < 0.10).astype(int)

    # ── L3: VN Regulatory Context ─────────────────────────────────────────
    jurisdiction_risk  = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.15, 0.05])
    sector_risk        = rng.choice([1, 2, 3],    n, p=[0.50, 0.35, 0.15])
    offshore_flag      = (rng.random(n) < 0.12).astype(int)
    fdi_linked_flag    = (rng.random(n) < 0.30).astype(int)
    vrfs               = (
        0.35 * jurisdiction_risk / 3
        + 0.30 * (sector_risk - 1) / 2
        + 0.20 * offshore_flag
        + 0.15 * fdi_linked_flag
    )

    # ── Pseudo-model score (approximates XGBoost behaviour) ───────────────
    raw_score = (
        0.25 * np.clip(amount_cv / 3, 0, 1)
        + 0.20 * np.clip(unique_to_banks / 17, 0, 1)
        + 0.15 * np.clip(transaction_count / 120, 0, 1)
        + 0.10 * weekend_ratio * 2
        + 0.10 * cross_border_ratio * 2
        + 0.08 * cycle_flag
        + 0.07 * fx_mismatch_flag
        + 0.05 * rng.random(n) * 0.5        # noise
    )
    # Sigmoid squash → P̂(SAR)
    phat_sar = 1 / (1 + np.exp(-6 * (raw_score - 0.45)))

    # CRS formula from thesis
    crs = 0.40 * vrfs + 0.60 * phat_sar

    # Hard override: jurisdiction_risk == 3 → Senior Escalation
    def assign_tier(c, jr):
        if jr == 3:
            return "Senior Escalation"
        elif c < 0.25:
            return "Auto-Clear"
        elif c < 0.60:
            return "Officer Review"
        else:
            return "Senior Escalation"

    tiers = np.array([assign_tier(crs[i], jurisdiction_risk[i]) for i in range(n)])

    # Priority score (for Officer Review queue sorting)
    priority = 0.25 * jurisdiction_risk / 3 + 0.60 * phat_sar + 0.15 * np.clip(daily_velocity / 3, 0, 1)

    # SAR label (ground truth – for demo only, not shown to officers)
    is_sar = (crs > 0.55) & (rng.random(n) < 0.55)
    is_sar |= (jurisdiction_risk == 3) & (rng.random(n) < 0.40)

    account_ids = [f"ACC-{10000 + i}" for i in range(n)]

    df = pd.DataFrame({
        "account_id":       account_ids,
        "transaction_count": transaction_count,
        "total_amount":     total_amount.round(0),
        "avg_amount":       avg_amount.round(0),
        "amount_cv":        amount_cv.round(3),
        "max_amount":       max_amount.round(0),
        "min_amount":       min_amount.round(0),
        "daily_velocity":   daily_velocity.round(3),
        "unique_to_banks":  unique_to_banks,
        "unique_from_banks": unique_from_banks,
        "weekend_ratio":    weekend_ratio.round(3),
        "night_ratio":      night_ratio.round(3),
        "cross_border_ratio": cross_border_ratio.round(3),
        "amount_entropy":   amount_entropy.round(3),
        "currency_diversity": currency_diversity,
        "fx_mismatch_flag": fx_mismatch_flag,
        "cycle_flag":       cycle_flag,
        "jurisdiction_risk": jurisdiction_risk,
        "sector_risk":      sector_risk,
        "offshore_flag":    offshore_flag,
        "fdi_linked_flag":  fdi_linked_flag,
        "vrfs":             vrfs.round(4),
        "phat_sar":         phat_sar.round(4),
        "crs":              crs.round(4),
        "priority_score":   priority.round(4),
        "tier":             tiers,
        "is_sar":           is_sar,
    })
    return df


def compute_shap_values(row):
    """Approximate SHAP-like feature attributions for one account."""
    baseline = 0.28   # approximate mean prediction
    pred = float(row["phat_sar"])

    features = {
        "amount_cv":          float(row["amount_cv"]),
        "unique_to_banks":    float(row["unique_to_banks"]),
        "transaction_count":  float(row["transaction_count"]),
        "weekend_ratio":      float(row["weekend_ratio"]),
        "cross_border_ratio": float(row["cross_border_ratio"]),
        "cycle_flag":         float(row["cycle_flag"]),
        "fx_mismatch_flag":   float(row["fx_mismatch_flag"]),
        "jurisdiction_risk":  float(row["jurisdiction_risk"]),
        "sector_risk":        float(row["sector_risk"]),
        "offshore_flag":      float(row["offshore_flag"]),
    }
    weights = {
        "amount_cv":          0.25,
        "unique_to_banks":    0.20,
        "transaction_count":  0.15,
        "weekend_ratio":      0.10,
        "cross_border_ratio": 0.10,
        "cycle_flag":         0.08,
        "fx_mismatch_flag":   0.07,
        "jurisdiction_risk":  0.025,
        "sector_risk":        0.025,
        "offshore_flag":      0.05,
    }
    norms = {
        "amount_cv": 3, "unique_to_banks": 17, "transaction_count": 120,
        "weekend_ratio": 1, "cross_border_ratio": 1, "cycle_flag": 1,
        "fx_mismatch_flag": 1, "jurisdiction_risk": 3, "sector_risk": 3,
        "offshore_flag": 1,
    }

    raw = {k: weights[k] * np.clip(features[k] / norms[k], 0, 1) for k in features}
    total_raw = sum(raw.values())
    scale = (pred - baseline) / total_raw if total_raw != 0 else 1.0
    shap_vals = {k: raw[k] * scale for k in raw}
    return baseline, shap_vals


def waterfall_chart(baseline, shap_vals, pred):
    """Draw SHAP waterfall plot."""
    items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [k.replace("_", "\n") for k, _ in items]
    vals   = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    running = baseline
    y_pos = list(range(len(vals)))

    for i, v in enumerate(vals):
        color = "#e05252" if v > 0 else "#52b0e0"
        ax.barh(i, v, left=running, color=color, height=0.55, edgecolor="none")
        running += v
        ax.text(running + 0.002 * (1 if v > 0 else -1),
                i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right",
                fontsize=8, color="white")

    ax.axvline(baseline, color="#888", linestyle="--", linewidth=0.8, label=f"Baseline = {baseline:.3f}")
    ax.axvline(pred,     color="#f0c040", linestyle="-", linewidth=1.2, label=f"Prediction = {pred:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("SHAP contribution", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="lower right")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """<div style='text-align:center;'>
            <h1 style='color:#4fa3e0;'>🏦 DDSF</h1>
            <p style='color:#aaa;font-size:14px;'>Data-Driven Decision Support Framework<br>
            AML Transaction Monitoring · Foreign Banks in Vietnam</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user = {**USERS[username], "username": username}
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """<div style='text-align:center;color:#666;font-size:12px;'>
            <b>Demo accounts:</b><br>
            officer / officer123 &nbsp;|&nbsp; senior / senior123 &nbsp;|&nbsp; admin / admin123
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        u = st.session_state.user
        st.markdown(f"### 👤 {u['name']}")
        st.caption(f"Role: **{u['role']}**")
        st.divider()

        pages = ["📊 Dashboard", "📋 Review Queue", "🔍 Account Inspector"]
        selected = st.radio("Navigation", pages, label_visibility="collapsed")
        st.session_state.page = selected

        st.divider()
        # Mini stats
        tier_counts = df["tier"].value_counts()
        st.metric("Auto-Clear",        tier_counts.get("Auto-Clear", 0))
        st.metric("Officer Review",    tier_counts.get("Officer Review", 0))
        st.metric("Senior Escalation", tier_counts.get("Senior Escalation", 0))
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard(df):
    st.title("📊 Compliance Dashboard")
    st.caption("DDSF — Monitoring cycle overview · Sample data (250 accounts)")

    # KPI row
    total = len(df)
    ac   = (df["tier"] == "Auto-Clear").sum()
    orv  = (df["tier"] == "Officer Review").sum()
    se   = (df["tier"] == "Senior Escalation").sum()
    avg_crs = df["crs"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Accounts",       f"{total:,}")
    c2.metric("Auto-Clear",           f"{ac}",  f"{ac/total*100:.1f}%")
    c3.metric("Officer Review",       f"{orv}", f"{orv/total*100:.1f}%")
    c4.metric("Senior Escalation",    f"{se}",  f"{se/total*100:.1f}%")
    c5.metric("Avg CRS",              f"{avg_crs:.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tier Distribution")
        tier_colors = {
            "Auto-Clear": "#52b0e0",
            "Officer Review": "#f0c040",
            "Senior Escalation": "#e05252",
        }
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        counts  = [ac, orv, se]
        labels  = ["Auto-Clear", "Officer Review", "Senior Escalation"]
        colors  = [tier_colors[l] for l in labels]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white", "fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("CRS Score Distribution by Tier")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#0f1117")
        for tier, color in tier_colors.items():
            subset = df[df["tier"] == tier]["crs"]
            ax2.hist(subset, bins=20, alpha=0.7, color=color, label=tier, edgecolor="none")
        ax2.axvline(0.25, color="white", linestyle="--", linewidth=0.8)
        ax2.axvline(0.60, color="white", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Composite Risk Score (CRS)", color="white", fontsize=9)
        ax2.set_ylabel("Account Count", color="white", fontsize=9)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    st.subheader("SAR Rate by Tier (Monotonicity Validation)")
    tier_order = ["Auto-Clear", "Officer Review", "Senior Escalation"]
    sar_rates = {
        t: df[df["tier"] == t]["is_sar"].mean() * 100
        for t in tier_order
    }
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    colors_mono = ["#52b0e0", "#f0c040", "#e05252"]
    for i, (t, col) in enumerate(zip(tier_order, cols)):
        rate = sar_rates[t]
        col.metric(t, f"{rate:.2f}%", help="SAR rate within this tier")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REVIEW QUEUE
# ─────────────────────────────────────────────────────────────────────────────
def page_review_queue(df):
    st.title("📋 Review Queue")
    role = st.session_state.user["role"]

    if role == "Compliance Officer":
        queue_df = df[df["tier"] == "Officer Review"].copy()
        st.caption("Showing: Officer Review accounts · Sorted by priority score ↓")
    elif role == "Senior Officer":
        queue_df = df[df["tier"] == "Senior Escalation"].copy()
        st.caption("Showing: Senior Escalation accounts · Sorted by priority score ↓")
    else:
        tier_filter = st.selectbox("Filter by tier",
            ["All", "Officer Review", "Senior Escalation", "Auto-Clear"])
        queue_df = df if tier_filter == "All" else df[df["tier"] == tier_filter].copy()

    queue_df = queue_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    display_cols = ["account_id", "crs", "phat_sar", "vrfs", "tier",
                    "jurisdiction_risk", "amount_cv", "unique_to_banks",
                    "transaction_count", "priority_score"]

    def color_tier(val):
        colors = {
            "Auto-Clear": "color: #52b0e0",
            "Officer Review": "color: #f0c040",
            "Senior Escalation": "color: #e05252",
        }
        return colors.get(val, "")

    def color_crs(val):
        if val >= 0.60:
            return "color: #e05252; font-weight: bold"
        elif val >= 0.25:
            return "color: #f0c040"
        return "color: #52b0e0"

    styled = (
        queue_df[display_cols]
        .style
        .applymap(color_tier, subset=["tier"])
        .applymap(color_crs,  subset=["crs"])
        .format({
            "crs": "{:.4f}", "phat_sar": "{:.4f}",
            "vrfs": "{:.4f}", "amount_cv": "{:.3f}",
            "priority_score": "{:.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=420)
    st.caption(f"{len(queue_df)} accounts in queue")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ACCOUNT INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────
def page_account_inspector(df):
    st.title("🔍 Account Inspector")

    col_sel, col_tier = st.columns([2, 1])
    with col_tier:
        tier_filter = st.selectbox("Filter by tier",
            ["All", "Officer Review", "Senior Escalation", "Auto-Clear"])

    filtered = df if tier_filter == "All" else df[df["tier"] == tier_filter]
    with col_sel:
        account_id = st.selectbox("Select account", filtered["account_id"].tolist())

    row = df[df["account_id"] == account_id].iloc[0]

    st.divider()

    # ── Header ────────────────────────────────────────────────────────────
    tier = row["tier"]
    tier_color = {"Auto-Clear": "#52b0e0", "Officer Review": "#f0c040",
                  "Senior Escalation": "#e05252"}.get(tier, "#888")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Risk Score (CRS)", f"{row['crs']:.4f}")
    c2.metric("P̂(SAR) — ML Score",          f"{row['phat_sar']:.4f}")
    c3.metric("VRFS — Regulatory Score",    f"{row['vrfs']:.4f}")
    c4.markdown(
        f"<div style='margin-top:8px;'><b>Routing Decision</b><br>"
        f"<span style='color:{tier_color};font-size:22px;font-weight:bold;'>{tier}</span></div>",
        unsafe_allow_html=True,
    )

    if row["jurisdiction_risk"] == 3:
        st.warning("⚠️ Hard Override Active: jurisdiction_risk = 3 (FATF Black List) → Mandatory Senior Escalation regardless of CRS.")

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("SHAP Explanation — Feature Contributions")
        baseline, shap_vals = compute_shap_values(row)
        fig = waterfall_chart(baseline, shap_vals, float(row["phat_sar"]))
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "SHAP waterfall: red bars increase risk score, blue bars decrease it. "
            "Computed using TreeSHAP approximation (Lundberg & Lee, 2017)."
        )

    with right:
        st.subheader("Feature Values")

        feature_groups = {
            "L1 – Volume / Velocity": [
                "transaction_count", "total_amount", "avg_amount",
                "amount_cv", "max_amount", "daily_velocity",
            ],
            "L2 – Behavioural": [
                "unique_to_banks", "unique_from_banks", "weekend_ratio",
                "night_ratio", "cross_border_ratio", "currency_diversity",
                "fx_mismatch_flag", "cycle_flag",
            ],
            "L3 – VN Regulatory Context": [
                "jurisdiction_risk", "sector_risk",
                "offshore_flag", "fdi_linked_flag", "vrfs",
            ],
        }

        for group_name, features in feature_groups.items():
            with st.expander(group_name, expanded=(group_name == "L1 – Volume / Velocity")):
                for feat in features:
                    if feat in row.index:
                        val = row[feat]
                        fmt = f"{val:,.3f}" if isinstance(val, float) else str(val)
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:2px 0;border-bottom:1px solid #333;'>"
                            f"<span style='color:#aaa;font-size:13px;'>{feat}</span>"
                            f"<span style='color:#fff;font-size:13px;font-weight:bold;'>{fmt}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    st.divider()
    st.subheader("Routing Logic (CRS Formula)")
    st.code(
        f"CRS = 0.40 × VRFS  +  0.60 × P̂(SAR)\n"
        f"    = 0.40 × {row['vrfs']:.4f}  +  0.60 × {row['phat_sar']:.4f}\n"
        f"    = {row['crs']:.4f}\n\n"
        f"Tier thresholds:\n"
        f"  CRS < 0.25            → Auto-Clear\n"
        f"  0.25 ≤ CRS < 0.60    → Officer Review\n"
        f"  CRS ≥ 0.60            → Senior Escalation\n"
        f"  jurisdiction_risk = 3 → Senior Escalation (hard override)",
        language="text",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    df = generate_accounts()
    sidebar(df)

    page = st.session_state.page
    if "Dashboard" in page:
        page_dashboard(df)
    elif "Queue" in page:
        page_review_queue(df)
    elif "Inspector" in page:
        page_account_inspector(df)


if __name__ == "__main__":
    main()
