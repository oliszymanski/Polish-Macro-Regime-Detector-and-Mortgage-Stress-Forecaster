from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import text
from src.db import get_engine

st.set_page_config(
    page_title="Polish Macro Regime & Mortgage Stress Forecaster",
    page_icon="🏦",
    layout="wide",
)

REGIME_COLORS = {
    "Expansion": "rgba(34, 197, 94, 0.16)",
    "Neutral": "rgba(234, 179, 8, 0.16)",
    "Stress": "rgba(239, 68, 68, 0.20)",
}

BADGE_COLORS = {
    "Expansion": "#16a34a",
    "Neutral": "#ca8a04",
    "Stress": "#dc2626",
}


@st.cache_data(ttl=300)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = get_engine()
    with engine.connect() as conn:
        features = pd.read_sql(
            text("SELECT * FROM dbo.features ORDER BY trade_date ASC;"), conn
        )
        regimes = pd.read_sql(
            text("SELECT * FROM dbo.regimes ORDER BY trade_date ASC;"), conn
        )
        predictions = pd.read_sql(
            text(
                "SELECT * FROM dbo.predictions WHERE horizon_months = 3 ORDER BY trade_date ASC;"
            ),
            conn,
        )

    for df in (features, regimes, predictions):
        df["trade_date"] = pd.to_datetime(df["trade_date"])

    numeric_cols = [
        "nbp_ref_rate",
        "nbp_ref_rate_chg_3m",
        "cpi_yoy",
        "cpi_yoy_chg_3m",
        "real_wage_yoy",
        "unemployment_rate",
        "eurpln",
        "eurpln_chg_3m",
        "bond_10y",
        "house_price_yoy",
        "retail_sales_yoy",
        "industrial_prod_yoy",
        "stress_score_rule",
    ]
    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    predictions["stress_probability"] = pd.to_numeric(
        predictions["stress_probability"], errors="coerce"
    )
    regimes["confidence"] = pd.to_numeric(regimes["confidence"], errors="coerce")
    return features, regimes, predictions


def compute_monthly_annuity(
    principal_pln: float, annual_rate_pct: float, tenor_years: int
) -> float:
    """Computes exact monthly PLN mortgage installment (equal principal + interest)."""
    n_months = max(int(tenor_years * 12), 1)
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate <= 1e-9:
        return principal_pln / n_months
    factor = (1.0 + monthly_rate) ** n_months
    return principal_pln * (monthly_rate * factor) / (factor - 1.0)


def add_regime_shading(fig: go.Figure, df_reg: pd.DataFrame, row: int = 1) -> None:
    """Adds contiguous background regime bands without excessive per-month shapes."""
    if df_reg.empty:
        return
    ordered = df_reg.sort_values("trade_date").reset_index(drop=True)
    ordered["block"] = (
        ordered["regime_label"] != ordered["regime_label"].shift(1)
    ).cumsum()

    for _, grp in ordered.groupby("block"):
        label = grp["regime_label"].iloc[0]
        start_dt = grp["trade_date"].iloc[0] - pd.Timedelta(days=15)
        end_dt = grp["trade_date"].iloc[-1] + pd.Timedelta(days=15)
        fig.add_vrect(
            x0=start_dt,
            x1=end_dt,
            fillcolor=REGIME_COLORS.get(label, "rgba(148, 163, 184, 0.1)"),
            opacity=1.0,
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )


def main() -> None:
    st.title("Polish Macro Regime Detector & Mortgage Stress Forecaster")
    st.caption(
        "SQL Server 2025 Feature Store • NBP & GUS Macro Pipeline • Calibrated 3-Month Borrower Stress Model"
    )

    try:
        features, regimes, predictions = load_dashboard_data()
    except Exception as exc:
        st.error(f"Database connection error: {exc}")
        st.stop()

    if features.empty:
        st.warning("No rows found in dbo.features. Run python -m src.run_pipeline first.")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Model & Regime Controls")
    available_models = sorted(predictions["model_name"].unique().tolist())
    default_model_idx = (
        available_models.index("logistic_l2_baseline")
        if "logistic_l2_baseline" in available_models
        else 0
    )
    selected_model = st.sidebar.selectbox(
        "Forecasting Model", available_models, index=default_model_idx
    )

    available_methods = sorted(regimes["regime_method"].unique().tolist())
    default_method_idx = (
        available_methods.index("rule_based")
        if "rule_based" in available_methods
        else 0
    )
    selected_regime_method = st.sidebar.selectbox(
        "Historical Regime Shading", available_methods, index=default_method_idx
    )

    st.sidebar.divider()
    st.sidebar.header("PLN Mortgage Stress Parameters")
    loan_principal = st.sidebar.number_input(
        "Remaining Loan Principal (PLN)",
        min_value=50_000,
        max_value=5_000_000,
        value=450_000,
        step=25_000,
    )
    tenor_years = st.sidebar.slider("Remaining Tenor (Years)", 5, 35, 22)
    bank_margin_pct = st.sidebar.slider(
        "Bank Margin + WIBOR Spread (%)", 1.00, 4.50, 2.15, step=0.05
    )
    net_income_pln = st.sidebar.number_input(
        "Household Net Monthly Income (PLN)",
        min_value=3_000,
        max_value=100_000,
        value=12_500,
        step=500,
    )
    custom_shock_bps = st.sidebar.slider(
        "Custom Rate Shock Scenario (+bps)", -150, 400, 150, step=25
    )

    # Latest snapshot metrics
    latest_feat = features.iloc[-1]
    latest_dt = latest_feat["trade_date"].strftime("%Y-%m-%d")

    model_preds = predictions[predictions["model_name"] == selected_model].copy()
    latest_pred = model_preds.iloc[-1]
    latest_prob = float(latest_pred["stress_probability"])

    gmm_latest = regimes[
        (regimes["regime_method"] == "ml_gmm_v1")
        & (regimes["trade_date"] == latest_feat["trade_date"])
    ]
    gmm_label = gmm_latest["regime_label"].iloc[0] if not gmm_latest.empty else "N/A"

    # Top KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Latest Observation",
        latest_dt,
        f"Score: {latest_feat['stress_score_rule']:.2f}",
    )
    k2.metric(
        "NBP Reference Rate",
        f"{latest_feat['nbp_ref_rate']:.2f}%",
        f"{latest_feat['nbp_ref_rate_chg_3m']:+.2f} pp (3m)",
    )
    k3.metric(
        "CPI Inflation (YoY)",
        f"{latest_feat['cpi_yoy']:.2f}%",
        f"{latest_feat['cpi_yoy_chg_3m']:+.2f} pp (3m)",
    )
    k4.metric(
        "Current Regime (Rule / GMM)",
        f"{latest_feat['regime_label_rule']} / {gmm_label}",
    )
    k5.metric(
        "3M Forward Stress Prob.",
        f"{latest_prob * 100:.2f}%",
        f"Signal: {latest_pred['predicted_regime']}",
        delta_color="inverse",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Macro Regimes & 3M Stress Forecast",
            "🏠 PLN Borrower Annuity & DTI Stress Test",
            "🔬 Feature Store & Regime Comparison",
        ]
    )

    # TAB 1: Macro & Forecast Time Series
    with tab1:
        shading_df = regimes[regimes["regime_method"] == selected_regime_method]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=(
                f"Polish Macro Indicators & Regimes ({selected_regime_method})",
                "3-Month Forward Mortgage Stress Probability (Out-of-Sample & Live)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=features["trade_date"],
                y=features["nbp_ref_rate"],
                name="NBP Reference Rate (%)",
                line=dict(color="#1d4ed8", width=2.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=features["trade_date"],
                y=features["cpi_yoy"],
                name="CPI YoY (%)",
                line=dict(color="#dc2626", width=2.0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=features["trade_date"],
                y=features["bond_10y"],
                name="10Y Gov Bond Yield (%)",
                line=dict(color="#0f766e", width=1.6, dash="dot"),
            ),
            row=1,
            col=1,
        )

        for m_name, grp in predictions.groupby("model_name"):
            is_selected = m_name == selected_model
            fig.add_trace(
                go.Scatter(
                    x=grp["trade_date"],
                    y=grp["stress_probability"] * 100.0,
                    name=f"P(Stress 3M) — {m_name}",
                    line=dict(
                        width=2.6 if is_selected else 1.5,
                        dash="solid" if is_selected else "dash",
                    ),
                ),
                row=2,
                col=1,
            )

        fig.add_hline(
            y=60.0,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="High Stress Threshold (60%)",
            row=2,
            col=1,
        )
        fig.add_hline(
            y=25.0,
            line_dash="dot",
            line_color="#ca8a04",
            annotation_text="Watchlist Threshold (25%)",
            row=2,
            col=1,
        )

        add_regime_shading(fig, shading_df, row=1)
        add_regime_shading(fig, shading_df, row=2)

        fig.update_yaxes(title_text="Rate / YoY (%)", row=1, col=1)
        fig.update_yaxes(title_text="Probability (%)", range=[0, 105], row=2, col=1)
        fig.update_layout(height=680, hovermode="x unified", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Household Annuity & DTI Stress Simulator
    with tab2:
        current_nbp = float(latest_feat["nbp_ref_rate"])
        peak_2022_nbp = float(features["nbp_ref_rate"].max())
        shocked_nbp = max(0.0, current_nbp + (custom_shock_bps / 100.0))

        scenarios = [
            ("Current Macro Baseline", current_nbp),
            (f"Custom Shock ({custom_shock_bps:+d} bps)", shocked_nbp),
            ("2022–2023 Peak Tightening Cycle", peak_2022_nbp),
        ]

        sim_rows = []
        base_pmt = compute_monthly_annuity(
            loan_principal, current_nbp + bank_margin_pct, tenor_years
        )
        for scen_name, nbp_val in scenarios:
            all_in_rate = nbp_val + bank_margin_pct
            pmt = compute_monthly_annuity(loan_principal, all_in_rate, tenor_years)
            dti_pct = (pmt / net_income_pln) * 100.0
            delta_pln = pmt - base_pmt
            status = (
                "Breach (>50% DTI)"
                if dti_pct > 50.0
                else ("Elevated (40–50% DTI)" if dti_pct > 40.0 else "Healthy (<=40% DTI)")
            )
            sim_rows.append(
                {
                    "Scenario": scen_name,
                    "NBP Rate (%)": round(nbp_val, 2),
                    "All-In Mortgage Rate (%)": round(all_in_rate, 2),
                    "Monthly Installment (PLN)": round(pmt, 2),
                    "Monthly Delta vs Baseline (PLN)": round(delta_pln, 2),
                    "Borrower DTI (%)": round(dti_pct, 2),
                    "KNF Stress Status": status,
                }
            )

        sim_df = pd.DataFrame(sim_rows)
        st.subheader("Borrower Debt-Service Sensitivity (PLN Fixed-Annuity Schedule)")
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        # Historical counterfactual monthly payment series
        hist_pmt = features[["trade_date", "nbp_ref_rate", "regime_label_rule"]].copy()
        hist_pmt["Monthly Installment (PLN)"] = hist_pmt["nbp_ref_rate"].apply(
            lambda r: compute_monthly_annuity(
                loan_principal, float(r) + bank_margin_pct, tenor_years
            )
        )
        hist_pmt["DTI (%)"] = (
            hist_pmt["Monthly Installment (PLN)"] / net_income_pln
        ) * 100.0

        fig_dti = px.line(
            hist_pmt,
            x="trade_date",
            y="DTI (%)",
            title="Historical Counterfactual Borrower DTI (%) Across Polish Rate Cycles",
        )
        fig_dti.add_hline(
            y=40.0, line_dash="dash", line_color="#ca8a04", annotation_text="40% DTI Caution"
        )
        fig_dti.add_hline(
            y=50.0, line_dash="dash", line_color="#dc2626", annotation_text="50% DTI Severe Stress"
        )
        fig_dti.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_dti, use_container_width=True)

    # TAB 3: Regime Comparison & SQL Feature Store Audit
    with tab3:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Rule-Based vs. Unsupervised GMM Confusion Matrix")
            piv_reg = regimes.pivot(
                index="trade_date", columns="regime_method", values="regime_label"
            ).dropna()
            if {"rule_based", "ml_gmm_v1"}.issubset(piv_reg.columns):
                ct = pd.crosstab(
                    piv_reg["rule_based"],
                    piv_reg["ml_gmm_v1"],
                    rownames=["Rule-Based Regime"],
                    colnames=["GMM Unsupervised Regime"],
                )
                st.dataframe(ct, use_container_width=True)

        with col_right:
            st.subheader("Out-of-Sample Model Benchmark (5-Fold TimeSeriesSplit)")
            bench_df = pd.DataFrame(
                [
                    {
                        "Model": "logistic_l2_baseline (Champion)",
                        "OOF ROC-AUC": 0.9937,
                        "OOF PR-AUC": 0.9928,
                        "Brier Score (Lower = Better)": 0.0342,
                    },
                    {
                        "Model": "gb_stress_3m_v1 (Challenger)",
                        "OOF ROC-AUC": 0.9457,
                        "OOF PR-AUC": 0.8504,
                        "Brier Score (Lower = Better)": 0.2734,
                    },
                ]
            )
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.subheader("SQL Server Feature Store (`dbo.features`) — Latest 24 Months")
        merged_audit = features.merge(
            model_preds[["trade_date", "stress_probability", "predicted_regime"]],
            on="trade_date",
            how="left",
        )
        merged_audit["trade_date"] = merged_audit["trade_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            merged_audit.tail(24).sort_values("trade_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
