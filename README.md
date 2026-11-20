# Polish Macro Regime Detector & Mortgage Stress Forecaster

An end-to-end macroeconomic data pipeline, SQL Server feature store, and credit stress forecasting application for the Polish mortgage market. The system ingests official monetary, inflation, FX, and real-economy indicators, engineers point-in-time window features in T-SQL, classifies macroeconomic regimes using both rule-based thresholds and unsupervised Gaussian Mixture Models, and forecasts 3-month forward borrower repayment stress.

---

## Architecture Overview

```text
  NBP API (FX) & Finwire API (Macro / GUS)
                    |
                    v
      [ Python ETL Ingestion Layer ]
        (src/ingest_nbp.py, src/ingest_finwire.py)
                    |
                    v
      [ SQL Server 2025 Feature Store ]
        - dbo.macro_series          (Raw monthly series)
        - dbo.vw_monthly_features   (Rolling 36m Z-scores & 3m deltas)
        - dbo.features              (Materialized feature matrix & targets)
        - dbo.regimes               (Rule-based & GMM regime labels)
        - dbo.predictions           (Calibrated 3m forward stress probabilities)
                    |
                    v
      [ Machine Learning Pipeline ]
        (src/train_and_predict.py)
        - Unsupervised 3-State GMM Regime Discovery
        - Supervised 3M Stress Forecaster (TimeSeriesSplit CV)
                    |
                    v
      [ Interactive Analytics Dashboard ]
        (app/app.py - Streamlit + Plotly)
        - Macro Regime Timeline & Stress Probability Chart
        - PLN Fixed-Annuity Borrower DTI Stress Simulator

Key Features

    Automated Data Ingestion: Extracts 10 Polish macroeconomic and FX indicators (NBP_REF_RATE, CPI_YOY, BOND_10Y, EURPLN, USDPLN, UNEMP_RATE, REAL_WAGE_YOY, HOUSE_PRICE_YOY, RETAIL_SALES_YOY, INDUSTRIAL_PROD_YOY) from public NBP and Finwire REST APIs with domain unit validation.

    SQL Server Feature Store: Transforms raw monthly series into 3-month rate/FX shock deltas, rolling 36-month Z-scores with flat-regime variance guards (STDEV >= 0.0001), and right-censored 3-month forward stress targets (target_stress_3m).

    Dual Regime Classification: Compares transparent rule-based macroeconomic regimes (Expansion, Neutral, Stress) against an unsupervised 3-component Gaussian Mixture Model (ml_gmm_v1).

    Calibrated Stress Forecasting: Benchmarks L2-regularized Logistic Regression against HistGradientBoosting using chronological 5-fold TimeSeriesSplit validation and selects the champion model by lowest out-of-sample Brier score.

    PLN Mortgage DTI Simulator: Translates NBP policy rate scenarios and custom basis-point shocks into exact monthly fixed-annuity installments and household Debt-to-Income (DTI) ratios.

Model Performance (5-Fold Chronological TimeSeriesSplit)

Evaluated strictly out-of-sample across 120 monthly observations (2016-11-30 to 2026-10-31):

## Project structure
Polish-Macro-Regime-Detector-and-Mortgage-Stress-Forecaster/
├── app/
│   └── app.py                          # Streamlit dashboard & PLN DTI simulator
├── models/                             # Serialized champion model artifacts (.joblib)
├── sql/
│   ├── 01_create_tables.sql            # Core relational schema (DDL)
│   ├── 02_views_monthly_features.sql   # Rolling Z-score & lagged delta view
│   └── 03_rules_based_regimes.sql      # Transactional feature & regime refresh SP
├── src/
│   ├── db.py                           # SQLAlchemy + pymssql connection factory
│   ├── ingest_nbp.py                   # Historical EUR/PLN & USD/PLN ingestion
│   ├── ingest_finwire.py               # Macro & GUS series ingestion
│   ├── train_and_predict.py            # GMM clustering, TimeSeriesSplit CV, scoring
│   └── run_pipeline.py                 # Single-command end-to-end orchestrator
├── docker-compose.yml                  # Local SQL Server 2025 container
├── requirements.txt
└── .env.example


## Quick start
### 1. Environment Configuration

Copy .env.example to .env and set your SQL Server password:

```
bash
   cp .env.example .env
   # Edit .env and set MSSQL_SA_PASSWORD
```

### 2. Start SQL Server Container
```
```