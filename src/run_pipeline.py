from sqlalchemy import text
from src.db import get_engine
from src.ingest_finwire import main as run_finwire
from src.ingest_nbp import main as run_nbp
from src.train_and_predict import main as run_ml


def run_all() -> None:
    print("=== [1/4] Ingesting NBP FX Series ===")
    run_nbp()

    print("\n=== [2/4] Ingesting Finwire Macro Series ===")
    run_finwire()

    print("\n=== [3/4] Refreshing SQL Server Feature Store & Rule-Based Regimes ===")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("EXEC dbo.sp_refresh_features_and_regimes;"))

    print("\n=== [4/4] Training Regime & Stress Models + Persisting Forecasts ===")
    run_ml()


if __name__ == "__main__":
    run_all()
