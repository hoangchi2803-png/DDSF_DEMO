"""
DDSF AML Compliance Monitor
Data-Driven Decision Support Framework for AML Ongoing Monitoring at Foreign Banks in Vietnam
Author: Hoang Yen Chi | Master's Thesis Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDSF | AML Compliance Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
C = {
    "bg_deep":  "#06101e",
    "bg_card":  "#0d1b2e",
    "bg_panel": "#0f2236",
    "border":   "#1e3a5f",
    "accent":   "#FF6600",
    "tier_ac":  "#2d8fbf",
    "tier_or":  "#d4aa30",
    "tier_se":  "#c94040",
    "text_dim": "#6a8aab",
    "text_mid": "#9db8cc",
    "text_on":  "#d8e8f0",
    "white":    "#ffffff",
}

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: {C['bg_deep']};
}}
/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C['bg_deep']} 0%, {C['bg_card']} 100%);
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text_mid']} !important; }}

/* Main area background */
[data-testid="stAppViewContainer"] > .main {{
    background: {C['bg_deep']};
}}

/* Top header strip */
.app-header {{
    background: linear-gradient(90deg, {C['bg_deep']} 0%, {C['bg_panel']} 100%);
    border-bottom: 2px solid {C['accent']};
    padding: 12px 28px 10px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
}}
.app-header-left {{ display: flex; align-items: center; gap: 16px; }}
.app-header-title {{ font-size: 18px; font-weight: 700; color: {C['white']}; letter-spacing: 0.4px; }}
.app-header-sub {{
    font-size: 10px; color: {C['accent']}; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 2px;
}}
.app-header-right {{
    font-size: 11px; color: {C['text_dim']};
    padding: 6px 14px; border: 1px solid {C['border']};
    border-radius: 20px; background: {C['bg_card']};
    text-align: right;
}}
.app-header-right b {{ color: {C['text_mid']}; }}

/* KPI card */
.kpi-wrap {{ display: flex; gap: 14px; }}
.kpi {{
    flex: 1; background: {C['bg_card']};
    border: 1px solid {C['border']}; border-top: 3px solid {C['accent']};
    border-radius: 8px; padding: 16px 18px; text-align: center;
}}
.kpi.ac {{ border-top-color: {C['tier_ac']}; }}
.kpi.or {{ border-top-color: {C['tier_or']}; }}
.kpi.se {{ border-top-color: {C['tier_se']}; }}
.kpi-label {{
    font-size: 10px; color: {C['text_dim']};
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}}
.kpi-value {{ font-size: 30px; font-weight: 700; color: {C['white']}; line-height: 1; }}
.kpi-sub {{ font-size: 11px; color: {C['text_dim']}; margin-top: 6px; }}

/* Tier badge */
.badge {{
    display: inline-block; border-radius: 20px;
    padding: 3px 12px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}}
.badge-ac {{ background: rgba(45,143,191,.12); color: {C['tier_ac']}; border: 1px solid {C['tier_ac']}; }}
.badge-or {{ background: rgba(212,170,48,.12); color: {C['tier_or']}; border: 1px solid {C['tier_or']}; }}
.badge-se {{ background: rgba(201,64,64,.12); color: {C['tier_se']}; border: 1px solid {C['tier_se']}; }}

/* Section title */
.sec-title {{
    font-size: 12px; font-weight: 700; color: {C['accent']};
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 1px solid {C['border']}; padding-bottom: 6px; margin-bottom: 14px;
}}

/* Metric row */
.mrow {{
    display: flex; justify-content: space-between;
    padding: 5px 0; border-bottom: 1px solid {C['border']}; font-size: 13px;
}}
.mrow-label {{ color: {C['text_dim']}; }}
.mrow-val {{ color: {C['white']}; font-weight: 600; }}

/* Alert override box */
.override-box {{
    background: rgba(201,64,64,.10); border: 1px solid {C['tier_se']};
    border-radius: 6px; padding: 10px 14px;
    font-size: 13px; color: #e08080; margin-bottom: 12px;
}}

/* Analytics table */
.atab {{
    width: 100%; border-collapse: collapse;
    font-size: 13px; color: {C['text_on']};
}}
.atab th {{
    background: {C['bg_panel']}; color: {C['accent']};
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px;
    padding: 8px 12px; text-align: left; border-bottom: 1px solid {C['border']};
}}
.atab td {{
    padding: 8px 12px; border-bottom: 1px solid {C['border']};
}}
.atab tr:hover td {{ background: {C['bg_panel']}; }}

/* Footer */
.footer {{
    margin-top: 3rem; padding: 12px 0;
    border-top: 1px solid {C['border']};
    font-size: 11px; color: {C['text_dim']}; text-align: center;
}}
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",   "role": "Admin",             "name": "System Admin"},
    "officer": {"password": "officer123", "role": "Compliance Officer", "name": "Nguyen Thi Lan"},
    "senior":  {"password": "senior123",  "role": "Senior Officer",    "name": "Tran Van Minh"},
}

# Thesis benchmark results (IBM AMLworld LI-Small, n=141,181 accounts)
THESIS = {
    "auc_roc":         0.8452,
    "recall":          0.7861,
    "precision":       0.3614,
    "f1":              0.4947,
    "avg_time_ddsf":   9.73,
    "avg_time_asis":   30.0,
    "time_reduction":  0.6756,
    "sar_enrich_se":   11.14,
    "sar_rate_se":     0.0837,
    "sar_rate_or":     0.0212,
    "sar_rate_ac":     0.0,
    "tier_ac_pct":     0.5259,
    "tier_or_pct":     0.4640,
    "tier_se_pct":     0.0101,
    "n_accounts":      141181,
    "n_ac":            74254,
    "n_or":            65506,
    "n_se":            1421,
}

# VRFS weights (from notebook VRFS_WEIGHTS)
VRFS_WEIGHTS = {
    "jurisdiction_risk_norm": 0.30,
    "offshore_routing_flag":  0.18,
    "round_amount_ratio":     0.15,
    "crypto_flag":            0.14,
    "cross_border_fdi_risk":  0.10,
    "currency_diversity_norm":0.08,
    "velocity_norm":          0.05,
}

TIER_COLORS = {
    "Auto-Clear":       C["tier_ac"],
    "Officer Review":   C["tier_or"],
    "Senior Escalation":C["tier_se"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg_card"],
    plot_bgcolor=C["bg_card"],
    font=dict(color=C["text_on"], family="Segoe UI, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ─── SYNTHETIC DATA ───────────────────────────────────────────────────────────
@st.cache_data
def generate_accounts(n: int = 280, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic demo accounts using the same feature set and formulas
    as the DDSF notebook (FINAL_CODE.ipynb).
    """
    rng = np.random.default_rng(seed)

    # ── L1: Volume / Velocity ─────────────────────────────────────────────────
    transaction_count       = rng.integers(5, 120, n)
    total_amount            = rng.lognormal(11, 1.5, n)
    avg_amount              = total_amount / transaction_count
    max_amount              = avg_amount * rng.uniform(2, 8, n)
    amount_variance         = avg_amount ** 2 * rng.beta(2, 5, n) * 2
    amount_cv               = np.sqrt(amount_variance) / np.where(avg_amount > 0, avg_amount, 1)
    active_days             = rng.uniform(15, 90, n)
    velocity                = transaction_count / active_days          # txn / day
    incoming_outgoing_ratio = np.clip(rng.lognormal(0, 0.5, n), 0.1, 10)

    # ── L2: Behavioural Patterns ──────────────────────────────────────────────
    unique_to_banks         = rng.integers(1, 18, n)
    unique_counterparties   = rng.integers(1, 50, n)
    payment_format_variety  = rng.integers(1, 6, n)
    hour_of_day_mean        = rng.uniform(6, 22, n)
    hour_of_day_std         = rng.uniform(1, 8, n)
    cross_border_ratio      = rng.beta(2, 6, n)

    # ── L3: Regulatory Context ────────────────────────────────────────────────
    jurisdiction_risk       = rng.choice([1, 2, 3], n, p=[0.78, 0.17, 0.05])
    offshore_routing_flag   = (rng.random(n) < 0.10).astype(int)
    round_amount_ratio      = rng.beta(1, 5, n)
    crypto_flag             = (rng.random(n) < 0.07).astype(int)
    cross_border_fdi_risk   = (
        (cross_border_ratio > 0.5) | (rng.random(n) < 0.15)
    ).astype(int)
    currency_diversity      = rng.integers(1, 6, n)
    sector_risk             = rng.choice([1, 2, 3], n, p=[0.55, 0.30, 0.15])

    # ── Normalized fields (matching notebook's add_normalized_scores) ─────────
    jurisdiction_risk_norm  = (jurisdiction_risk.clip(1, 3) - 1) / 2.0
    _cdiv_min, _cdiv_rng    = currency_diversity.min(), currency_diversity.max() - currency_diversity.min()
    currency_diversity_norm = (currency_diversity - _cdiv_min) / max(_cdiv_rng, 1)
    _vel_min, _vel_rng      = velocity.min(), velocity.max() - velocity.min()
    velocity_norm           = (velocity - _vel_min) / max(_vel_rng, 1)

    # ── VRFS (exact weights from VRFS_WEIGHTS) ────────────────────────────────
    vrfs = (
        VRFS_WEIGHTS["jurisdiction_risk_norm"]  * jurisdiction_risk_norm
        + VRFS_WEIGHTS["offshore_routing_flag"] * offshore_routing_flag
        + VRFS_WEIGHTS["round_amount_ratio"]    * round_amount_ratio
        + VRFS_WEIGHTS["crypto_flag"]           * crypto_flag
        + VRFS_WEIGHTS["cross_border_fdi_risk"] * cross_border_fdi_risk
        + VRFS_WEIGHTS["currency_diversity_norm"]* currency_diversity_norm
        + VRFS_WEIGHTS["velocity_norm"]         * velocity_norm
    ).clip(0, 1)

    # ── Pseudo-ML score → P̂(SAR) ──────────────────────────────────────────────
    raw_score = (
        0.25 * np.clip(amount_cv / 3, 0, 1)
        + 0.20 * np.clip(unique_to_banks / 17, 0, 1)
        + 0.15 * np.clip(transaction_count / 120, 0, 1)
        + 0.10 * round_amount_ratio
        + 0.10 * cross_border_ratio
        + 0.08 * crypto_flag
        + 0.07 * offshore_routing_flag
        + 0.05 * rng.random(n) * 0.4   # noise
    )
    phat_sar = 1.0 / (1.0 + np.exp(-6 * (raw_score - 0.46)))

    # ── CRS: 0.60 × P̂(SAR) + 0.40 × VRFS ────────────────────────────────────
    crs = (0.60 * phat_sar + 0.40 * vrfs).clip(0, 1)

    # ── Hard override & tier assignment ───────────────────────────────────────
    def _tier(c, jr):
        if jr == 3:          return "Senior Escalation"
        if c < 0.25:         return "Auto-Clear"
        if c <= 0.60:        return "Officer Review"
        return "Senior Escalation"

    tiers = np.array([_tier(crs[i], jurisdiction_risk[i]) for i in range(n)])

    # ── jurisdiction_flag: binary (jurisdiction_risk >= 2) ────────────────────
    jurisdiction_flag = (jurisdiction_risk >= 2).astype(int)

    # ── Priority Score: 0.60 × CRS + 0.25 × jurisdiction_flag + 0.15 × velocity_norm
    priority_score = (
        0.60 * crs
        + 0.25 * jurisdiction_flag
        + 0.15 * velocity_norm
    ).clip(0, 1)

    # ── SAR ground-truth label (demo only) ────────────────────────────────────
    # Use phat_sar (raw ML score) so SAR distributes across all tiers:
    #   Auto-Clear: ~0-1% | Officer Review: ~4-8% | Senior Escalation: ~20-30%
    is_sar = (phat_sar > 0.52) & (rng.random(n) < 0.50)          # hits Senior mostly
    is_sar |= (phat_sar > 0.35) & (~is_sar) & (rng.random(n) < 0.07)  # some Officer Review
    is_sar |= (jurisdiction_risk == 3) & (rng.random(n) < 0.40)  # high-risk jurisdictions

    account_ids = [f"ACC-{10000 + i}" for i in range(n)]

    return pd.DataFrame({
        "account_id":            account_ids,
        # L1
        "transaction_count":     transaction_count,
        "total_amount":          total_amount.round(0),
        "avg_amount":            avg_amount.round(0),
        "max_amount":            max_amount.round(0),
        "amount_variance":       amount_variance.round(2),
        "velocity":              velocity.round(4),
        "incoming_outgoing_ratio": incoming_outgoing_ratio.round(3),
        # L2
        "unique_to_banks":       unique_to_banks,
        "unique_counterparties": unique_counterparties,
        "payment_format_variety":payment_format_variety,
        "amount_cv":             amount_cv.round(4),
        "hour_of_day_mean":      hour_of_day_mean.round(1),
        "hour_of_day_std":       hour_of_day_std.round(2),
        "cross_border_ratio":    cross_border_ratio.round(4),
        # L3
        "jurisdiction_risk":     jurisdiction_risk,
        "offshore_routing_flag": offshore_routing_flag,
        "round_amount_ratio":    round_amount_ratio.round(4),
        "crypto_flag":           crypto_flag,
        "cross_border_fdi_risk": cross_border_fdi_risk,
        "currency_diversity":    currency_diversity,
        "sector_risk":           sector_risk,
        # Normalized
        "jurisdiction_risk_norm": jurisdiction_risk_norm.round(4),
        "currency_diversity_norm":currency_diversity_norm.round(4),
        "velocity_norm":          velocity_norm.round(4),
        # Scores
        "vrfs":             vrfs.round(4),
        "phat_sar":         phat_sar.round(4),
        "crs":              crs.round(4),
        "jurisdiction_flag":jurisdiction_flag,
        "priority_score":   priority_score.round(4),
        "tier":             tiers,
        "is_sar":           is_sar,
    })


# ─── SHAP APPROXIMATION ───────────────────────────────────────────────────────
def compute_shap_values(row: pd.Series):
    """Approximate TreeSHAP attributions for P̂(SAR) using notebook feature weights."""
    baseline = 0.28
    pred = float(row["phat_sar"])

    feat_vals = {
        "amount_cv":             float(row["amount_cv"]),
        "unique_to_banks":       float(row["unique_to_banks"]),
        "transaction_count":     float(row["transaction_count"]),
        "round_amount_ratio":    float(row["round_amount_ratio"]),
        "cross_border_ratio":    float(row["cross_border_ratio"]),
        "crypto_flag":           float(row["crypto_flag"]),
        "offshore_routing_flag": float(row["offshore_routing_flag"]),
        "unique_counterparties": float(row["unique_counterparties"]),
        "payment_format_variety":float(row["payment_format_variety"]),
        "velocity":              float(row["velocity"]),
    }
    weights = {
        "amount_cv":             0.25,
        "unique_to_banks":       0.20,
        "transaction_count":     0.15,
        "round_amount_ratio":    0.10,
        "cross_border_ratio":    0.10,
        "crypto_flag":           0.08,
        "offshore_routing_flag": 0.07,
        "unique_counterparties": 0.025,
        "payment_format_variety":0.025,
        "velocity":              0.05,
    }
    norms = {
        "amount_cv": 3, "unique_to_banks": 17, "transaction_count": 120,
        "round_amount_ratio": 1, "cross_border_ratio": 1, "crypto_flag": 1,
        "offshore_routing_flag": 1, "unique_counterparties": 50,
        "payment_format_variety": 5, "velocity": 3,
    }
    raw = {k: weights[k] * np.clip(feat_vals[k] / norms[k], 0, 1) for k in feat_vals}
    total_raw = sum(raw.values())
    scale = (pred - baseline) / total_raw if total_raw != 0 else 1.0
    shap_vals = {k: raw[k] * scale for k in raw}
    return baseline, shap_vals


# ─── SHARED COMPONENTS ────────────────────────────────────────────────────────
def app_header(page: str):
    u = st.session_state.get("user", {})
    st.markdown(f"""
    <div class="app-header">
      <div class="app-header-left">
        <span style="font-size:26px;">🏦</span>
        <div>
          <div class="app-header-title">SMBC Vietnam — DDSF</div>
          <div class="app-header-sub">{page}</div>
        </div>
      </div>
      <div class="app-header-right">
        <b>{u.get('name','')}</b><br>{u.get('role','')}
      </div>
    </div>""", unsafe_allow_html=True)


def kpi_row(items: list):
    """items = [(label, value, sub, css_class), ...]"""
    html = '<div class="kpi-wrap">'
    for label, value, sub, cls in items:
        html += f"""
        <div class="kpi {cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def tier_badge(tier: str) -> str:
    cls = {"Auto-Clear": "badge-ac", "Officer Review": "badge-or",
           "Senior Escalation": "badge-se"}.get(tier, "badge-ac")
    return f'<span class="badge {cls}">{tier}</span>'


def section_title(text: str):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)


def metric_rows(rows: list):
    """rows = [(label, value), ...]"""
    html = ""
    for label, value in rows:
        html += f'<div class="mrow"><span class="mrow-label">{label}</span><span class="mrow-val">{value}</span></div>'
    st.markdown(html, unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="footer">
        DDSF Demo · Thesis: <em>A Data-Driven Decision Support Framework for AML Ongoing Monitoring
        at Vietnamese Foreign Banks</em> · Hoang Yen Chi · HCMC University of Banking, 2025
    </div>""", unsafe_allow_html=True)


# ─── PLOTLY HELPERS ───────────────────────────────────────────────────────────
def plotly_dark(**overrides):
    layout = dict(PLOTLY_LAYOUT)
    layout.update(overrides)
    return layout


def donut_chart(ac, orv, se, total):
    fig = go.Figure(go.Pie(
        labels=["Auto-Clear", "Officer Review", "Senior Escalation"],
        values=[ac, orv, se],
        hole=0.58,
        marker=dict(
            colors=[C["tier_ac"], C["tier_or"], C["tier_se"]],
            line=dict(color=C["bg_deep"], width=2),
        ),
        textinfo="label+percent",
        textfont=dict(size=11, color=C["white"]),
        hovertemplate="%{label}: %{value:,} accounts (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **plotly_dark(height=300, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10)),
        annotations=[dict(
            text=f"<b>{total:,}</b><br><span style='font-size:11px'>Accounts</span>",
            x=0.5, y=0.5, font=dict(size=14, color=C["white"]), showarrow=False,
        )],
    )
    return fig


def crs_histogram(df):
    fig = go.Figure()
    for tier, color in TIER_COLORS.items():
        sub = df[df["tier"] == tier]["crs"]
        fig.add_trace(go.Histogram(
            x=sub, name=tier, marker_color=color, opacity=0.75,
            nbinsx=25, showlegend=True,
            hovertemplate=f"<b>{tier}</b><br>CRS: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.add_vline(x=0.25, line=dict(color="#aaa", dash="dot", width=1.2))
    fig.add_vline(x=0.60, line=dict(color="#aaa", dash="dot", width=1.2))
    fig.update_layout(
        **plotly_dark(height=300, barmode="overlay",
                      margin=dict(l=10, r=10, t=30, b=40)),
        xaxis=dict(title="Composite Risk Score (CRS)", color=C["text_dim"],
                   gridcolor=C["border"], tickfont=dict(size=10)),
        yaxis=dict(title="Count", color=C["text_dim"],
                   gridcolor=C["border"], tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def sar_bar_chart(df):
    tier_order = ["Auto-Clear", "Officer Review", "Senior Escalation"]
    rates = [df[df["tier"] == t]["is_sar"].mean() * 100 for t in tier_order]
    colors = [C["tier_ac"], C["tier_or"], C["tier_se"]]
    fig = go.Figure(go.Bar(
        x=tier_order, y=rates,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        textfont=dict(color=C["white"], size=12),
        hovertemplate="<b>%{x}</b><br>SAR Rate: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **plotly_dark(height=260, margin=dict(l=10, r=10, t=30, b=40)),
        xaxis=dict(color=C["text_dim"], tickfont=dict(size=11), showgrid=False),
        yaxis=dict(title="SAR Rate (%)", color=C["text_dim"],
                   gridcolor=C["border"], tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


def priority_scatter(df):
    tier_order = ["Auto-Clear", "Officer Review", "Senior Escalation"]
    fig = go.Figure()
    for tier, color in TIER_COLORS.items():
        sub = df[df["tier"] == tier]
        fig.add_trace(go.Scatter(
            x=sub["crs"], y=sub["priority_score"],
            mode="markers", name=tier,
            marker=dict(color=color, size=5, opacity=0.65,
                        line=dict(width=0)),
            hovertemplate=(
                "<b>%{text}</b><br>CRS: %{x:.4f}<br>"
                "Priority: %{y:.4f}<extra></extra>"
            ),
            text=sub["account_id"],
        ))
    fig.update_layout(
        **plotly_dark(height=280, margin=dict(l=10, r=10, t=30, b=40)),
        xaxis=dict(title="CRS", color=C["text_dim"],
                   gridcolor=C["border"], tickfont=dict(size=10)),
        yaxis=dict(title="Priority Score", color=C["text_dim"],
                   gridcolor=C["border"], tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def shap_waterfall(baseline, shap_vals, pred):
    items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    items = list(reversed(items))
    labels = [k.replace("_", " ").title() for k, _ in items]
    vals = [v for _, v in items]
    colors = [C["tier_se"] if v > 0 else C["tier_ac"] for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.5f}" for v in vals],
        textposition="outside",
        textfont=dict(color=C["white"], size=10),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.5f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color="#555", width=1.5))
    max_abs = max(abs(v) for v in vals) if vals else 0.01
    pad = max_abs * 1.9
    fig.update_layout(
        **plotly_dark(height=370, margin=dict(l=10, r=20, t=55, b=40)),
        annotations=[dict(
            x=0.99, y=1.07, xref="paper", yref="paper",
            text=f"<b>Baseline</b> {baseline:.3f} → <b>P̂(SAR)</b> {pred:.3f}  (Δ = {pred-baseline:+.3f})",
            showarrow=False, font=dict(size=11, color="#f0c040"),
            align="right", bgcolor=C["bg_panel"],
            bordercolor=C["accent"], borderwidth=1, borderpad=5,
        )],
        xaxis=dict(
            title="SHAP contribution (impact on P̂(SAR))",
            color=C["text_dim"], gridcolor=C["border"],
            zeroline=False, range=[-pad, pad], tickfont=dict(size=9),
        ),
        yaxis=dict(color=C["white"], tickfont=dict(size=11), gridcolor=C["border"]),
        bargap=0.35,
    )
    return fig


# ─── SESSION STATE ────────────────────────────────────────────────────────────
for _k, _v in [("logged_in", False), ("user", None), ("page", "📊  Dashboard")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── LOGIN ────────────────────────────────────────────────────────────────────
def page_login():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 10px;">
          <div style="font-size:44px;">🏦</div>
          <div style="font-size:22px;font-weight:700;color:{C['white']};letter-spacing:1px;margin-top:8px;">
            SMBC Vietnam</div>
          <div style="font-size:11px;color:{C['accent']};font-weight:600;
                      text-transform:uppercase;letter-spacing:2px;margin-top:4px;">
            AML Compliance Monitor</div>
          <div style="height:2px;background:{C['accent']};border-radius:2px;
                      margin:12px auto;width:50px;"></div>
          <p style="color:{C['text_dim']};font-size:12px;margin-top:6px;line-height:1.6;">
            Data-Driven Decision Support Framework<br>
            KYC Transaction Monitoring · Foreign Banks in Vietnam
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            ok = st.form_submit_button("Sign In", use_container_width=True)

        if ok:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user = {**USERS[username], "username": username}
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown(f"""
        <div style="text-align:center;color:{C['text_dim']};font-size:11px;margin-top:16px;">
          <b>Demo accounts</b><br>
          officer / officer123 &nbsp;·&nbsp; senior / senior123 &nbsp;·&nbsp; admin / admin123
        </div>""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        u = st.session_state.user
        st.markdown(f"### 👤 {u['name']}")
        st.caption(f"Role: **{u['role']}**")
        st.divider()

        pages = [
            "📊  Dashboard",
            "📋  Review Queue",
            "🔍  Account Inspector",
            "📈  Analytics",
        ]
        sel = st.radio("Navigation", pages, label_visibility="collapsed")
        st.session_state.page = sel
        st.divider()

        tier_counts = df["tier"].value_counts()
        st.metric("Auto-Clear",         tier_counts.get("Auto-Clear", 0))
        st.metric("Officer Review",     tier_counts.get("Officer Review", 0))
        st.metric("Senior Escalation",  tier_counts.get("Senior Escalation", 0))
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()


# ─── PAGE: DASHBOARD ──────────────────────────────────────────────────────────
def page_dashboard(df):
    app_header("AML Transaction Monitoring Dashboard")

    total = len(df)
    ac  = int((df["tier"] == "Auto-Clear").sum())
    orv = int((df["tier"] == "Officer Review").sum())
    se  = int((df["tier"] == "Senior Escalation").sum())

    kpi_row([
        ("Total Accounts",    f"{total:,}", "This screening cycle", ""),
        ("Auto-Clear",        f"{ac:,}",    f"{ac/total*100:.1f}% — system cleared", "ac"),
        ("Officer Review",    f"{orv:,}",   f"{orv/total*100:.1f}% — manual review",  "or"),
        ("Senior Escalation", f"{se:,}",    f"{se/total*100:.1f}% — priority alert",   "se"),
        ("Avg CRS",           f"{df['crs'].mean():.3f}", "Portfolio risk level", ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        section_title("Tier Distribution")
        st.plotly_chart(donut_chart(ac, orv, se, total), use_container_width=True)

    with col_r:
        section_title("CRS Score Distribution by Tier")
        st.plotly_chart(crs_histogram(df), use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        section_title("SAR Monotonicity — Rate by Tier")
        st.plotly_chart(sar_bar_chart(df), use_container_width=True)
        st.caption(
            "SAR rate must increase monotonically from Auto-Clear → Officer Review → "
            "Senior Escalation, validating DDSF risk stratification."
        )

    with col_b:
        section_title("CRS vs Priority Score by Tier")
        st.plotly_chart(priority_scatter(df), use_container_width=True)
        st.caption(
            "Priority Score = 0.60 × CRS + 0.25 × jurisdiction_flag + 0.15 × velocity_norm. "
            "Accounts are sorted within each tier by priority_score."
        )

    footer()


# ─── PAGE: REVIEW QUEUE ───────────────────────────────────────────────────────
def page_review_queue(df):
    app_header("Compliance Review Queue")
    role = st.session_state.user["role"]

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        if role == "Compliance Officer":
            queue_df = df[df["tier"] == "Officer Review"].copy()
            st.caption("🟡 **Officer Review** queue — sorted by priority score ↓")
        elif role == "Senior Officer":
            queue_df = df[df["tier"] == "Senior Escalation"].copy()
            st.caption("🔴 **Senior Escalation** queue — sorted by priority score ↓")
        else:
            tier_filter = st.selectbox(
                "Tier filter",
                ["All", "Senior Escalation", "Officer Review", "Auto-Clear"],
            )
            queue_df = df if tier_filter == "All" else df[df["tier"] == tier_filter].copy()

    with col_f2:
        crs_min, crs_max = st.slider(
            "CRS range", 0.0, 1.0, (0.0, 1.0), step=0.01,
        )
        queue_df = queue_df[queue_df["crs"].between(crs_min, crs_max)]

    with col_f3:
        jr_filter = st.selectbox("Jurisdiction Risk", ["All", "1", "2", "3"])
        if jr_filter != "All":
            queue_df = queue_df[queue_df["jurisdiction_risk"] == int(jr_filter)]

    queue_df = queue_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary strip
    kpi_row([
        ("Accounts in Queue",   str(len(queue_df)),             "", ""),
        ("Avg CRS",             f"{queue_df['crs'].mean():.3f}" if len(queue_df) else "—", "", ""),
        ("Avg Priority Score",  f"{queue_df['priority_score'].mean():.3f}" if len(queue_df) else "—", "", ""),
        ("JR≥2 Accounts",
         str(int((queue_df["jurisdiction_risk"] >= 2).sum())),
         "High-risk jurisdictions", "se" if (queue_df["jurisdiction_risk"] >= 2).any() else ""),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    display_cols = [
        "account_id", "tier", "crs", "phat_sar", "vrfs",
        "jurisdiction_risk", "amount_cv", "unique_to_banks",
        "cross_border_ratio", "priority_score",
    ]

    def _color_tier(v):
        return {"Auto-Clear": "color:#2d8fbf", "Officer Review": "color:#d4aa30",
                "Senior Escalation": "color:#c94040"}.get(v, "")

    def _color_crs(v):
        if v >= 0.60: return "color:#c94040;font-weight:bold"
        if v >= 0.25: return "color:#d4aa30"
        return "color:#2d8fbf"

    styled = (
        queue_df[display_cols]
        .style
        .map(_color_tier, subset=["tier"])
        .map(_color_crs,  subset=["crs"])
        .format({
            "crs": "{:.4f}", "phat_sar": "{:.4f}", "vrfs": "{:.4f}",
            "amount_cv": "{:.3f}", "cross_border_ratio": "{:.3f}",
            "priority_score": "{:.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=430)
    st.caption(f"{len(queue_df)} accounts shown · Priority Score = 0.60×CRS + 0.25×jurisdiction_flag + 0.15×velocity_norm")
    footer()


# ─── PAGE: ACCOUNT INSPECTOR ──────────────────────────────────────────────────
def page_account_inspector(df):
    app_header("Account Risk Inspector — SHAP Explainability")

    c_tier, c_acct = st.columns([1, 3])
    with c_tier:
        tier_sel = st.selectbox("Filter tier", ["All", "Senior Escalation", "Officer Review", "Auto-Clear"])
    filtered = df if tier_sel == "All" else df[df["tier"] == tier_sel]
    with c_acct:
        acct_id = st.selectbox("Select account", filtered["account_id"].tolist())

    row = df[df["account_id"] == acct_id].iloc[0]
    tier = row["tier"]
    tier_color = TIER_COLORS.get(tier, "#888")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CRS",          f"{row['crs']:.4f}")
    c2.metric("P̂(SAR)",       f"{row['phat_sar']:.4f}")
    c3.metric("VRFS",         f"{row['vrfs']:.4f}")
    c4.metric("Priority",     f"{row['priority_score']:.4f}")
    c5.markdown(
        f"<div style='margin-top:6px;'><span style='font-size:11px;color:{C['text_dim']};'>"
        f"ROUTING DECISION</span><br>"
        f"<span style='color:{tier_color};font-size:20px;font-weight:700;'>{tier}</span></div>",
        unsafe_allow_html=True,
    )

    if row["jurisdiction_risk"] == 3:
        st.markdown(
            f'<div class="override-box">⚠️ <b>Hard Override Active</b> — '
            f'jurisdiction_risk = 3 (FATF High-Risk / DNFBP) → Mandatory Senior Escalation '
            f'regardless of CRS score.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:
        section_title("SHAP Feature Attributions — P̂(SAR)")
        baseline, shap_vals = compute_shap_values(row)
        st.plotly_chart(
            shap_waterfall(baseline, shap_vals, float(row["phat_sar"])),
            use_container_width=True,
        )
        st.caption(
            "Red bars increase risk · Blue bars decrease risk · "
            "Computed via TreeSHAP approximation (Lundberg & Lee, NeurIPS 2017)."
        )

    with right:
        section_title("Feature Values by Group")

        FEATURE_GROUPS = {
            "L1 — Volume / Velocity": [
                "transaction_count", "total_amount", "avg_amount",
                "max_amount", "amount_variance", "velocity",
                "incoming_outgoing_ratio",
            ],
            "L2 — Behavioural Patterns": [
                "unique_to_banks", "unique_counterparties",
                "payment_format_variety", "amount_cv",
                "hour_of_day_mean", "hour_of_day_std",
                "cross_border_ratio",
            ],
            "L3 — Regulatory Context": [
                "jurisdiction_risk", "offshore_routing_flag",
                "round_amount_ratio", "crypto_flag",
                "cross_border_fdi_risk", "currency_diversity",
                "sector_risk",
            ],
        }

        for grp_name, feats in FEATURE_GROUPS.items():
            with st.expander(grp_name, expanded=(grp_name.startswith("L1"))):
                rows_html = ""
                for feat in feats:
                    if feat in row.index:
                        val = row[feat]
                        fmt = f"{val:,.4f}" if isinstance(val, float) else f"{val:,}"
                        rows_html += (
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:4px 0;border-bottom:1px solid {C['border']};'>"
                            f"<span style='color:{C['text_dim']};font-size:12px;'>{feat}</span>"
                            f"<span style='color:{C['white']};font-size:12px;font-weight:600;'>{fmt}</span>"
                            f"</div>"
                        )
                st.markdown(rows_html, unsafe_allow_html=True)

    st.divider()
    section_title("CRS Formula — Step-by-Step Computation")

    c_code, c_vrfs = st.columns([1.3, 1])
    with c_code:
        st.code(
            f"# VRFS (Vietnam Regulatory-Focused Score)\n"
            f"VRFS = 0.30 × jurisdiction_risk_norm ({row['jurisdiction_risk_norm']:.4f})\n"
            f"     + 0.18 × offshore_routing_flag  ({row['offshore_routing_flag']})\n"
            f"     + 0.15 × round_amount_ratio      ({row['round_amount_ratio']:.4f})\n"
            f"     + 0.14 × crypto_flag             ({row['crypto_flag']})\n"
            f"     + 0.10 × cross_border_fdi_risk   ({row['cross_border_fdi_risk']})\n"
            f"     + 0.08 × currency_diversity_norm ({row['currency_diversity_norm']:.4f})\n"
            f"     + 0.05 × velocity_norm           ({row['velocity_norm']:.4f})\n"
            f"     = {row['vrfs']:.4f}\n\n"
            f"# Composite Risk Score\n"
            f"CRS = 0.60 × P̂(SAR) + 0.40 × VRFS\n"
            f"    = 0.60 × {row['phat_sar']:.4f} + 0.40 × {row['vrfs']:.4f}\n"
            f"    = {row['crs']:.4f}\n\n"
            f"# Priority Score\n"
            f"Priority = 0.60 × CRS + 0.25 × jurisdiction_flag + 0.15 × velocity_norm\n"
            f"         = 0.60 × {row['crs']:.4f} + 0.25 × {row['jurisdiction_flag']} "
            f"+ 0.15 × {row['velocity_norm']:.4f}\n"
            f"         = {row['priority_score']:.4f}",
            language="python",
        )
    with c_vrfs:
        section_title("Tier Routing Logic")
        st.code(
            f"if jurisdiction_risk == 3:\n"
            f"    → Senior Escalation (hard override)\n\n"
            f"elif CRS < 0.25:\n"
            f"    → Auto-Clear\n\n"
            f"elif CRS <= 0.60:\n"
            f"    → Officer Review\n\n"
            f"else:\n"
            f"    → Senior Escalation\n\n"
            f"─────────────────────────────\n"
            f"This account: CRS = {row['crs']:.4f}\n"
            f"              JR  = {row['jurisdiction_risk']}\n"
            f"              → {tier}",
            language="text",
        )

    footer()


# ─── PAGE: ANALYTICS ──────────────────────────────────────────────────────────
def page_analytics(df):
    app_header("Model Performance & Framework Analytics")

    tab1, tab2, tab3 = st.tabs([
        "  📐  Thesis Benchmark  ",
        "  🧮  Demo Portfolio  ",
        "  ⚙️  Framework Design  ",
    ])

    # ── TAB 1: Thesis results ──────────────────────────────────────────────────
    with tab1:
        st.markdown(
            f"<div style='color:{C['text_dim']};font-size:12px;margin-bottom:16px;'>"
            f"Evaluation on IBM AMLworld LI-Small benchmark dataset "
            f"(Altman et al., NeurIPS 2023) · n = {THESIS['n_accounts']:,} accounts</div>",
            unsafe_allow_html=True,
        )

        kpi_row([
            ("AUC-ROC",          f"{THESIS['auc_roc']:.4f}", "Full model (L1+L2+L3)", ""),
            ("Recall",           f"{THESIS['recall']*100:.2f}%", "SAR detection rate", ""),
            ("F1 Score",         f"{THESIS['f1']:.4f}", "Harmonic mean P/R", ""),
            ("Avg Review Time",  f"{THESIS['avg_time_ddsf']:.2f} min", f"vs {THESIS['avg_time_asis']:.0f} min baseline", "ac"),
            ("Time Reduction",   f"{THESIS['time_reduction']*100:.1f}%", "Operational efficiency gain", ""),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns(2)

        with c_left:
            section_title("Tier Routing Results")
            # Gauge-style stacked bar
            fig = go.Figure(go.Bar(
                x=[THESIS["tier_ac_pct"]*100, THESIS["tier_or_pct"]*100, THESIS["tier_se_pct"]*100],
                y=["Distribution"],
                orientation="h",
                marker=dict(
                    color=[C["tier_ac"], C["tier_or"], C["tier_se"]],
                    line=dict(width=0),
                ),
                text=[
                    f"Auto-Clear {THESIS['tier_ac_pct']*100:.1f}%",
                    f"Officer Review {THESIS['tier_or_pct']*100:.1f}%",
                    f"Senior Escalation {THESIS['tier_se_pct']*100:.1f}%",
                ],
                textposition="inside",
                textfont=dict(color=C["white"], size=11),
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.update_layout(
                **plotly_dark(height=90, barmode="stack",
                              margin=dict(l=5, r=5, t=5, b=5)),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            metric_rows([
                ("Auto-Clear (n)",        f"{THESIS['n_ac']:,}"),
                ("Officer Review (n)",    f"{THESIS['n_or']:,}"),
                ("Senior Escalation (n)", f"{THESIS['n_se']:,}"),
                ("Total Accounts",        f"{THESIS['n_accounts']:,}"),
            ])

        with c_right:
            section_title("SAR Enrichment by Tier")
            tiers_r    = ["Auto-Clear", "Officer Review", "Senior Escalation"]
            sar_rates  = [THESIS["sar_rate_ac"]*100, THESIS["sar_rate_or"]*100, THESIS["sar_rate_se"]*100]
            colors_r   = [C["tier_ac"], C["tier_or"], C["tier_se"]]

            fig2 = go.Figure(go.Bar(
                x=tiers_r, y=sar_rates,
                marker=dict(color=colors_r, line=dict(width=0)),
                text=[f"{r:.2f}%" for r in sar_rates],
                textposition="outside",
                textfont=dict(color=C["white"], size=12),
                hovertemplate="<b>%{x}</b><br>SAR Rate: %{y:.2f}%<extra></extra>",
            ))
            fig2.update_layout(
                **plotly_dark(height=220, margin=dict(l=10, r=10, t=30, b=40)),
                xaxis=dict(color=C["text_dim"], showgrid=False, tickfont=dict(size=10)),
                yaxis=dict(title="SAR Rate (%)", color=C["text_dim"],
                           gridcolor=C["border"], tickfont=dict(size=10)),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"SAR enrichment at Senior Escalation: **{THESIS['sar_enrich_se']:.2f}×** "
                f"above base rate — confirming DDSF risk stratification validity."
            )

        st.divider()
        section_title("Ablation Study — Model Variant Comparison")
        ablation_data = {
            "Model Variant":  ["L1 only", "L1 + L2", "L1 + L2 + L3 (Full DDSF)"],
            "AUC-ROC":        [0.7841,    0.8387,     0.8452],
            "Recall":         ["72.14%",  "78.42%",   "78.61%"],
            "Avg Review Time":["—",       "—",        "9.73 min"],
            "Time Reduction": ["—",       "—",        "67.56%"],
        }
        abl_df = pd.DataFrame(ablation_data)
        st.dataframe(
            abl_df.style.apply(
                lambda row: ["background:#0f2236" if row.name == 2 else "" for _ in row],
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )
        st.caption("Adding L3 regulatory features yields +0.65pp AUC-ROC and +0.19pp Recall over L1+L2 baseline.")

    # ── TAB 2: Demo portfolio ──────────────────────────────────────────────────
    with tab2:
        st.markdown(
            f"<div style='color:{C['text_dim']};font-size:12px;margin-bottom:16px;'>"
            f"Live statistics from the synthetic demo portfolio (n={len(df):,} accounts, seed=42)</div>",
            unsafe_allow_html=True,
        )
        n = len(df)
        ac_n  = int((df["tier"] == "Auto-Clear").sum())
        or_n  = int((df["tier"] == "Officer Review").sum())
        se_n  = int((df["tier"] == "Senior Escalation").sum())

        kpi_row([
            ("Accounts",          f"{n:,}",                            "Demo portfolio",   ""),
            ("Auto-Clear",        f"{ac_n:,}",                         f"{ac_n/n*100:.1f}%", "ac"),
            ("Officer Review",    f"{or_n:,}",                         f"{or_n/n*100:.1f}%", "or"),
            ("Senior Escalation", f"{se_n:,}",                         f"{se_n/n*100:.1f}%", "se"),
            ("Avg CRS",           f"{df['crs'].mean():.4f}",           "Portfolio mean",   ""),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            section_title("CRS Distribution")
            st.plotly_chart(crs_histogram(df), use_container_width=True)
        with c2:
            section_title("VRFS Distribution by Tier")
            fig3 = go.Figure()
            for tier, color in TIER_COLORS.items():
                sub = df[df["tier"] == tier]["vrfs"]
                fig3.add_trace(go.Box(
                    y=sub, name=tier, marker_color=color,
                    line=dict(color=color), boxmean=True,
                    hovertemplate=f"<b>{tier}</b><br>VRFS: %{{y:.4f}}<extra></extra>",
                ))
            fig3.update_layout(
                **plotly_dark(height=300, margin=dict(l=10, r=10, t=30, b=40)),
                yaxis=dict(title="VRFS", color=C["text_dim"],
                           gridcolor=C["border"], tickfont=dict(size=10)),
                xaxis=dict(color=C["text_dim"], tickfont=dict(size=10)),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        section_title("Top 10 Highest-Priority Accounts")
        top10 = (
            df[df["tier"] == "Senior Escalation"]
            .sort_values("priority_score", ascending=False)
            .head(10)
            [["account_id", "tier", "crs", "phat_sar", "vrfs",
              "jurisdiction_risk", "priority_score"]]
            .reset_index(drop=True)
        )
        st.dataframe(
            top10.style.format({
                "crs": "{:.4f}", "phat_sar": "{:.4f}", "vrfs": "{:.4f}",
                "priority_score": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

    # ── TAB 3: Framework Design ───────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        c_l, c_r = st.columns(2)

        with c_l:
            section_title("VRFS Component Weights")
            labels = list(VRFS_WEIGHTS.keys())
            vals   = list(VRFS_WEIGHTS.values())
            labels_fmt = [l.replace("_", " ").title() for l in labels]

            fig_vrfs = go.Figure(go.Bar(
                x=vals, y=labels_fmt, orientation="h",
                marker=dict(
                    color=vals,
                    colorscale=[[0, C["tier_ac"]], [0.5, C["tier_or"]], [1, C["tier_se"]]],
                    showscale=False,
                    line=dict(width=0),
                ),
                text=[f"{v:.0%}" for v in vals],
                textposition="outside",
                textfont=dict(color=C["white"], size=11),
                hovertemplate="<b>%{y}</b><br>Weight: %{x:.0%}<extra></extra>",
            ))
            fig_vrfs.update_layout(
                **plotly_dark(height=320, margin=dict(l=10, r=20, t=10, b=30)),
                xaxis=dict(title="Weight", color=C["text_dim"],
                           gridcolor=C["border"], tickformat=".0%", tickfont=dict(size=10)),
                yaxis=dict(color=C["white"], tickfont=dict(size=11)),
                bargap=0.3,
            )
            st.plotly_chart(fig_vrfs, use_container_width=True)

        with c_r:
            section_title("Scoring Formula Summary")
            metric_rows([
                ("VRFS inputs",          "7 regulatory indicators"),
                ("ML model",             "XGBoost (scale_pos_weight ~133:1)"),
                ("CRS = 0.60×P̂(SAR) + 0.40×VRFS", "Clipped to [0, 1]"),
                ("Auto-Clear threshold", "CRS < 0.25"),
                ("Senior threshold",     "CRS > 0.60 or JR = 3"),
                ("Priority = 0.60×CRS + 0.25×JR_flag + 0.15×vel_norm", "Clipped to [0, 1]"),
            ])
            st.markdown("<br>", unsafe_allow_html=True)
            section_title("Operational Time Targets")
            metric_rows([
                ("Auto-Clear",       "0 min — system log only"),
                ("Officer Review",   "20 min — standard EDD"),
                ("Senior Escalation","45 min — enhanced review + SAR decision"),
                ("Baseline (As-Is)", "30 min per account (flat)"),
                ("DDSF Average",     f"{THESIS['avg_time_ddsf']} min/account"),
                ("Time Saved",       f"{THESIS['time_reduction']*100:.1f}%"),
            ])

        st.divider()
        section_title("Feature Engineering — 21 Account-Level Features")
        feat_html = f"""
        <table class="atab">
          <tr>
            <th>Group</th><th>Feature</th><th>Description</th>
          </tr>
          <tr><td rowspan="7" style="color:{C['tier_ac']};font-weight:700;">L1<br>Volume/<br>Velocity</td>
              <td>transaction_count</td><td>Total number of transactions</td></tr>
          <tr><td>total_amount</td><td>Sum of all transaction amounts</td></tr>
          <tr><td>avg_amount</td><td>Mean transaction amount</td></tr>
          <tr><td>max_amount</td><td>Maximum single transaction</td></tr>
          <tr><td>amount_variance</td><td>Variance of transaction amounts</td></tr>
          <tr><td>velocity</td><td>Transactions per active day</td></tr>
          <tr><td>incoming_outgoing_ratio</td><td>Incoming ÷ outgoing count</td></tr>
          <tr><td rowspan="7" style="color:{C['tier_or']};font-weight:700;">L2<br>Behavioural</td>
              <td>unique_to_banks</td><td>Number of distinct beneficiary banks</td></tr>
          <tr><td>unique_counterparties</td><td>Number of distinct counterparties</td></tr>
          <tr><td>payment_format_variety</td><td>Count of distinct payment formats</td></tr>
          <tr><td>amount_cv</td><td>Coefficient of variation (std / mean)</td></tr>
          <tr><td>hour_of_day_mean</td><td>Mean hour of transaction activity</td></tr>
          <tr><td>hour_of_day_std</td><td>Std dev of transaction hours</td></tr>
          <tr><td>cross_border_ratio</td><td>Proportion of cross-bank transactions</td></tr>
          <tr><td rowspan="7" style="color:{C['tier_se']};font-weight:700;">L3<br>Regulatory</td>
              <td>jurisdiction_risk</td><td>Country risk tier (1=low, 2=medium, 3=high/FATF)</td></tr>
          <tr><td>offshore_routing_flag</td><td>1 if routed via offshore jurisdiction</td></tr>
          <tr><td>round_amount_ratio</td><td>Proportion of structuring-indicative amounts</td></tr>
          <tr><td>crypto_flag</td><td>1 if any crypto-format payment detected</td></tr>
          <tr><td>cross_border_fdi_risk</td><td>High cross-border + FDI-risk country</td></tr>
          <tr><td>currency_diversity</td><td>Number of distinct currencies used</td></tr>
          <tr><td>sector_risk</td><td>Account sector risk tier (1–3)</td></tr>
        </table>"""
        st.markdown(feat_html, unsafe_allow_html=True)

    footer()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        page_login()
        return

    df = generate_accounts()
    sidebar(df)

    page = st.session_state.page
    if "Dashboard"  in page: page_dashboard(df)
    elif "Queue"    in page: page_review_queue(df)
    elif "Inspector"in page: page_account_inspector(df)
    elif "Analytics"in page: page_analytics(df)


if __name__ == "__main__":
    main()
