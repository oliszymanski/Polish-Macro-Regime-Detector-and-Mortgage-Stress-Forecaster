#===============================================================================
#   IMPORTS
#===============================================================================

from datetime import date, timedelta
from dotenv import load_dotenv
import pandas as pd
import requests
from .db import load_macro_series

NBP_BASE_URL = "https://api.nbp.pl/api/exchangerates/rates/a"



#===============================================================================
#   FUNCTIONS
#===============================================================================

def fetch_nbp_fx_history(currency: str, years: int = 12) -> pd.DataFrame:
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=years * 365)

    all_rates = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=364), end_dt)
        url = (
            f"{NBP_BASE_URL}/{currency.lower()}/"
            f"{chunk_start.isoformat()}/{chunk_end.isoformat()}/?format=json"
        )
        resp = requests.get(url, timeout=15)

        if resp.status_code == 200:
            payload = resp.json()
            all_rates.extend(payload.get("rates", []))
        elif resp.status_code != 404:
            resp.raise_for_status()

        chunk_start = chunk_end + timedelta(days=1)

    if not all_rates:
        return pd.DataFrame()

    df = pd.DataFrame(all_rates)
    df["effectiveDate"] = pd.to_datetime(df["effectiveDate"])
    df = df.sort_values("effectiveDate").set_index("effectiveDate")

    # Resample to month-end taking the final fixing of each month
    monthly = df["mid"].resample("ME").last().dropna().reset_index()
    monthly["trade_date"] = monthly["effectiveDate"].dt.date

    code = f"{currency.upper()}PLN"
    return pd.DataFrame(
        {
            "indicator_code": code,
            "indicator_name": f"{currency.upper()}/PLN Mid Exchange Rate (Month-End)",
            "trade_date": monthly["trade_date"],
            "value": monthly["mid"].round(6),
            "unit": "PLN",
            "frequency": "M",
            "source": "NBP",
        }
    )


def main():
    print("Fetching historical EUR/PLN and USD/PLN from api.nbp.pl...")
    df_eur = fetch_nbp_fx_history("EUR", years=12)
    df_usd = fetch_nbp_fx_history("USD", years=12)
    combined = pd.concat([df_eur, df_usd], ignore_index=True)
    load_macro_series(combined)



#===============================================================================
#   EXECUTION
#===============================================================================

if __name__ == "__main__":
    load_dotenv()
    main()
