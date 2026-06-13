# DDSF — Data-Driven Decision Support Framework
## AML Transaction Monitoring for Foreign Banks in Vietnam

This is the interactive demo application for the undergraduate thesis:
> **"Designing a Data-Driven Decision Support Framework for Corporate KYC Ongoing Transaction Monitoring: A Case Study of Foreign Banks in Vietnam"**  
> Author: Hoang Yen Chi | Foreign Trade University

---

## Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hoangchi2803-png-ddsf-demo-app-mp8axy.streamlit.app)

### Demo Accounts
| Username | Password   | Role             | Access                  |
|----------|------------|------------------|-------------------------|
| officer  | officer123 | Compliance Officer | Officer Review queue  |
| senior   | senior123  | Senior Officer   | Senior Escalation queue |
| admin    | admin123   | Admin            | All tiers               |

---

## Framework Overview

The DDSF routes corporate accounts into three tiers based on a **Composite Risk Score (CRS)**:
CRS = 0.60 × P̂(SAR) + 0.40 × VRFS

| Tier | CRS Range | Action |
|------|-----------|--------|
| Auto-Clear | CRS < 0.25 | Automated compliance log; periodic QA sampling |
| Officer Review | 0.25 ≤ CRS < 0.60 | Compliance officer case review |
| Senior Escalation | CRS ≥ 0.60 | Senior officer EDD review |

**Hard override**: `jurisdiction_risk = 3` (FATF Black List) → mandatory Senior Escalation regardless of CRS.

**Priority Score** (queue ordering):
Priority = 0.60 × CRS + 0.25 × jurisdiction_flag + 0.15 × velocity_norm

---

## App Features

- **Dashboard**: Tier distribution, CRS histogram, SAR monotonicity validation chart
- **Review Queue**: Role-filtered account queue sorted by priority score, with VRFS breakdown
- **Account Inspector**: CRS breakdown, SHAP waterfall explanation, all 21 features across 3 groups
- **Analytics**: Model performance metrics, ablation study, thesis benchmark numbers

---

## Feature Groups (21 features)

| Group | Features |
|-------|----------|
| L1 — Volume / Velocity | transaction_count, total_amount, avg_amount, max_amount, amount_variance, velocity, incoming_outgoing_ratio |
| L2 — Behavioural Patterns | unique_to_banks, unique_counterparties, payment_format_variety, amount_cv, hour_of_day_mean, hour_of_day_std, cross_border_ratio |
| L3 — Regulatory Context | jurisdiction_risk, offshore_routing_flag, round_amount_ratio, crypto_flag, cross_border_fdi_risk, currency_diversity, sector_risk |

---

## Thesis Benchmark Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8452 |
| Recall | 78.61% |
| Avg review time | 9.73 min/account |
| Time reduction | 67.56% vs. manual baseline |
| SAR enrichment | 11.14× |

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Regulatory Basis

- Circular 09/2023/TT-NHNN (SBV) — AML obligations for credit institutions
- FATF Risk-Based Approach (2012, updated 2023)
- APG Mutual Evaluation Report Vietnam (2022)

---

## Thesis Methodology

Design Science Research (Hevner et al., 2004) — three-cycle model:

- **Relevance Cycle**: AML compliance gap at foreign banks operating in Vietnam
- **Design Cycle**: DDSF architecture (XGBoost + SHAP + 3-tier routing)
- **Rigour Cycle**: IBM AMLworld benchmark evaluation (LI-Small, 141,181 accounts)
