from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from src.db import get_engine

FEATURE_COLS = [
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
]

GMM_COLS = [
    "nbp_ref_rate",
    "nbp_ref_rate_chg_3m",
    "cpi_yoy",
    "cpi_yoy_chg_3m",
    "eurpln_chg_3m",
    "bond_10y",
]

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models"


def load_feature_matrix() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT
            trade_date,
            nbp_ref_rate,
            nbp_ref_rate_chg_3m,
            cpi_yoy,
            cpi_yoy_chg_3m,
            real_wage_yoy,
            unemployment_rate,
            eurpln,
            eurpln_chg_3m,
            bond_10y,
            house_price_yoy,
            retail_sales_yoy,
            industrial_prod_yoy,
            stress_score_rule,
            regime_label_rule,
            CAST(target_stress_3m AS FLOAT) AS target_stress_3m
        FROM dbo.features
        ORDER BY trade_date ASC;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    for col in FEATURE_COLS + ["stress_score_rule"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fit_unsupervised_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Fits a 3-state Gaussian Mixture Model and maps states by ascending stress score."""
    gmm_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "gmm",
                GaussianMixture(
                    n_components=3,
                    covariance_type="full",
                    n_init=10,
                    random_state=42,
                ),
            ),
        ]
    )

    X_gmm = df[GMM_COLS]
    X_scaled = gmm_pipe[:-1].fit_transform(X_gmm)
    gmm_model: GaussianMixture = gmm_pipe.named_steps["gmm"]
    gmm_model.fit(X_scaled)

    raw_clusters = gmm_model.predict(X_scaled)
    posteriors = gmm_model.predict_proba(X_scaled)

    # Deterministic label alignment: sort clusters from lowest to highest mean rule stress score
    cluster_severity = (
        pd.DataFrame({"cluster": raw_clusters, "stress": df["stress_score_rule"]})
        .groupby("cluster")["stress"]
        .mean()
        .sort_values()
    )
    ordered_labels = ["Expansion", "Neutral", "Stress"]
    cluster_to_regime = {
        cluster_id: ordered_labels[rank]
        for rank, cluster_id in enumerate(cluster_severity.index)
    }

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "regime_label": [cluster_to_regime[c] for c in raw_clusters],
            "regime_method": "ml_gmm_v1",
            "confidence": np.round(posteriors.max(axis=1), 4),
            "created_at": now_utc,
        }
    )


def evaluate_time_series_models(
    train_df: pd.DataFrame,
) -> tuple[dict[str, Pipeline], str]:
    """Evaluates Logistic Regression vs. Gradient Boosting and selects champion by Brier score."""
    X = train_df[FEATURE_COLS]
    y = train_df["target_stress_3m"].astype(int).values

    candidates = {
        "logistic_l2_baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
            ]
        ),
        "gb_stress_3m_v1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=120,
                        min_samples_leaf=8,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    brier_scores: dict[str, float] = {}
    print("=== Chronological Cross-Validation (TimeSeriesSplit, 5 Folds) ===")

    for name, pipe in candidates.items():
        oof_preds = []
        oof_true = []
        for train_idx, test_idx in tscv.split(X):
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue

            # Restrict strictly to features observed within the current training fold
            valid_cols = [
                c for c in FEATURE_COLS if X.iloc[train_idx][c].notna().any()
            ]
            pipe.fit(X.iloc[train_idx][valid_cols], y_tr)
            probs = pipe.predict_proba(X.iloc[test_idx][valid_cols])[:, 1]
            oof_preds.extend(probs)
            oof_true.extend(y_te)

        oof_true_arr = np.array(oof_true)
        oof_pred_arr = np.array(oof_preds)
        roc = roc_auc_score(oof_true_arr, oof_pred_arr)
        pr = average_precision_score(oof_true_arr, oof_pred_arr)
        brier = brier_score_loss(oof_true_arr, oof_pred_arr)
        brier_scores[name] = brier
        print(
            f"Model: {name:<22} | OOF Samples: {len(oof_true_arr):<3} | "
            f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Brier: {brier:.4f}"
        )

    # Fit all candidates on full realized training set (using columns with >=1 observed value)
    full_valid_cols = [c for c in FEATURE_COLS if X[c].notna().any()]
    for pipe in candidates.values():
        pipe.fit(X[full_valid_cols], y)

    champion_name = min(brier_scores, key=brier_scores.get)
    print(f"\nSelected Champion Model (Lowest Brier Score): {champion_name}")
    return candidates, champion_name


def persist_outputs(regimes_df: pd.DataFrame, preds_df: pd.DataFrame) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM dbo.regimes WHERE regime_method = :method;"),
            {"method": "ml_gmm_v1"},
        )
        regimes_df.to_sql(
            name="regimes",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
        )

        for model_name in preds_df["model_name"].unique():
            conn.execute(
                text(
                    "DELETE FROM dbo.predictions WHERE model_name = :model_name AND horizon_months = 3;"
                ),
                {"model_name": model_name},
            )
        preds_df.to_sql(
            name="predictions",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
        )


def main() -> None:
    df = load_feature_matrix()
    print(
        f"Loaded {len(df)} monthly rows from dbo.features "
        f"({df['trade_date'].min()} -> {df['trade_date'].max()})."
    )

    # 1. Unsupervised GMM Regimes
    gmm_regimes_df = fit_unsupervised_regimes(df)
    print("\n=== Unsupervised GMM Regime Counts (ml_gmm_v1) ===")
    print(gmm_regimes_df["regime_label"].value_counts().to_string())

    # 2. Supervised 3-Month Stress Forecaster
    labeled_df = df[df["target_stress_3m"].notna()].copy()
    models, champion_name = evaluate_time_series_models(labeled_df)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[champion_name], ARTIFACT_DIR / f"{champion_name}.joblib")

    # 3. Score all models across full timeline & persist to SQL Server
    full_valid_cols = [c for c in FEATURE_COLS if labeled_df[c].notna().any()]
    all_preds = []
    for name, pipe in models.items():
        probs = pipe.predict_proba(df[full_valid_cols])[:, 1]
        predicted_regime = np.where(
            probs >= 0.60,
            "Stress",
            np.where(probs >= 0.25, "Neutral", "Expansion"),
        )
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        all_preds.append(
            pd.DataFrame(
                {
                    "trade_date": df["trade_date"],
                    "model_name": name,
                    "model_version": "1.0.0",
                    "horizon_months": 3,
                    "stress_probability": np.round(probs, 4),
                    "predicted_regime": predicted_regime,
                    "created_at": now_utc,
                }
            )
        )

    preds_df = pd.concat(all_preds, ignore_index=True)
    persist_outputs(gmm_regimes_df, preds_df)

    print(f"\n=== Latest 6 Months Forecast ({champion_name}) ===")
    champ_preds = preds_df[preds_df["model_name"] == champion_name]
    preview = df[
        ["trade_date", "nbp_ref_rate", "cpi_yoy", "regime_label_rule"]
    ].merge(
        champ_preds[["trade_date", "stress_probability", "predicted_regime"]],
        on="trade_date",
    )
    print(preview.tail(6).to_string(index=False))


if __name__ == '__main__':
    main()
