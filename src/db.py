#===============================================================================
#   IMPORTS
#===============================================================================

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv()

#===============================================================================
#   FUNCTIONS
#===============================================================================

def get_engine():
    db_pass = os.getenv("MSSQL_SA_PASSWORD")
    if not db_pass:
        raise ValueError("Environment variable MSSQL_SA_PASSWORD is not set.")

    url = URL.create(
        "mssql+pymssql",
        username=os.getenv("DB_USER", "sa"),
        password=db_pass,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 1433)),
        database=os.getenv("DB_NAME", "PolandMacroAnalytics"),
    )
    return create_engine(url)


def load_macro_series(df: pd.DataFrame) -> None:
    """
    Atomically replaces rows in dbo.macro_series per indicator_code.
    """
    if df.empty:
        print("Warning: Received empty DataFrame; skipping database load.")
        return

    required_cols = [
        "indicator_code",
        "indicator_name",
        "trade_date",
        "value",
        "unit",
        "frequency",
        "source",
    ]
    clean_df = df[required_cols].copy()
    clean_df["is_revised"] = 0
    clean_df = clean_df.drop_duplicates(
        subset=["indicator_code", "trade_date"], keep="last"
    )

    codes = clean_df["indicator_code"].unique().tolist()
    engine = get_engine()

    with engine.begin() as conn:
        for code in codes:
            conn.execute(
                text("DELETE FROM dbo.macro_series WHERE indicator_code = :code;"),
                {"code": code},
            )

        clean_df.to_sql(
            name="macro_series",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
            chunksize=1000,
        )

    print(
        f"Loaded {len(clean_df)} monthly rows across {len(codes)} indicator(s): {', '.join(codes)}"
    )
