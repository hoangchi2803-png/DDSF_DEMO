# DDSF — Data-Driven Decision Support Framework
## AML Transaction Monitoring for Foreign Banks in Vietnam

This is the interactive demo application for the undergraduate thesis:

> **"Designing a Data-Driven Decision Support Framework for Corporate KYC Ongoing Transaction Monitoring: A Case Study of Foreign Banks in Vietnam"**  
> Author: Hoang Yen Chi | Foreign Trade University

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

### Demo Accounts

| Username | Password    | Role              | Access |
|----------|-------------|-------------------|--------|
| officer  | officer123  | Compliance Officer | Officer Review queue |
| senior   | senior123   | Senior Officer    | Senior Escalation queue |
| admin    | admin123    | Admin             | All tiers |

---

## Framework Overview

The DDSF routes corporate accounts into three tiers based on a **Composite Risk Score (CRS)**:

```
CRS = 0.40 × VRFS + 0.60 × P̂(SAR)
```

| Tier | CRS Range | Action |
|------|-----------|--------|
| Auto-Clear | CRS < 0.25 | Automated compliance log; periodic QA sampling |
| Officer Review | 0.25 ≤ CRS < 0.60 | Compliance officer case review (20 min avg) |
| Senior Escalation | CRS ≥ 0.60 | Senior officer EDD review (45 min avg) |

**Hard override**: `jurisdiction_risk = 3` (FATF Black List) → mandatory Senior Escalation.

---

## App Features

- **Dashboard**: Tier distribution, CRS histogram, SAR rate by tier (monotonicity validation)
- **Review Queue**: Role-filtered account queue sorted by priority score
- **Account Inspector**: CRS breakdown, SHAP waterfall explanation, all 21 features across 3 groups

---

## Feature Groups (21 features)

| Group | Features |
|-------|----------|
| L1 – Volume/Velocity | transaction_count, total_amount, avg_amount, amount_cv, max_amount, min_amount, daily_velocity |
| L2 – Behavioural | unique_to_banks, unique_from_banks, weekend_ratio, night_ratio, cross_border_ratio, amount_entropy, currency_diversity, fx_mismatch_flag, cycle_flag |
| L3 – VN Regulatory | jurisdiction_risk, sector_risk, offshore_flag, fdi_linked_flag, vrfs |

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Regulatory Basis

- Circular 09/2023/TT-NHNN (SBV) — Articles 9, 12, 13
- FATF Risk-Based Approach (2012, updated 2023)
- APG Mutual Evaluation Report Vietnam (2022)

---

## Thesis Methodology

Design Science Research (Hevner et al., 2004) — three-cycle model:
- **Relevance Cycle**: AML compliance challenge at foreign banks in Vietnam
- **Design Cycle**: DDSF architecture (XGBoost + SHAP + tier routing)
- **Rigour Cycle**: IBM IT-AML benchmark evaluation (LI-Small, 681,281 accounts)
