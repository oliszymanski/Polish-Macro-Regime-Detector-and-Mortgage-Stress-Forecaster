import re
import numpy as np
import pandas as pd
import requests
from src.db import load_macro_series

FINWIRE_BASE = "https://public-api.finwire.pl"

DATE_REGEX = re.compile(
    r"^\d{4}[-./](?:0?[1-9]|1[0-2]|Q[1-4])(?:[-./]\d{1,2})?$", re.IGNORECASE
)

COMMON_DATE_KEYS = (
    "date",
    "ym",
    "m",
    "effectiveDate",
    "effective_date",
    "validFrom",
    "valid_from",
    "period",
    "month",
    "quarter",
    "yearMonth",
    "year_month",
    "dt",
    "timestamp",
)

COMMON_VALUE_KEYS = (
    "yoy",
    "cpiYoy",
    "cpi_yoy",
    "value",
    "v",
    "rate",
    "unemploymentRate",
    "registeredRate",
    "interestRate",
    "interest_rate",
    "yield",
    "yieldPct",
    "yield_pct",
    "primaryM2",
    "secondaryM2",
    "pricePerSqm",
    "price_per_sqm",
    "medianPrice",
    "median_price",
    "price",
    "val",
)

EXCLUDED_NUMERIC_KEYS = {"id", "unitid", "teryt", "year", "month", "quarter"}


def _unwrap_data(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _find_date_and_value(record: dict, preferred_value_keys: tuple[str, ...]):
    raw_date = None
    for k in COMMON_DATE_KEYS:
        if record.get(k) is not None:
            raw_date = record[k]
            break

    if raw_date is None:
        for k, v in record.items():
            if isinstance(v, str) and DATE_REGEX.match(v.strip()):
                raw_date = v.strip()
                break

    if raw_date is None:
        return None, None

    raw_val = None
    for k in preferred_value_keys:
        val = record.get(k)
        if val is not None and not isinstance(val, bool):
            try:
                raw_val = float(val)
                break
            except (ValueError, TypeError):
                continue

    if raw_val is None:
        for k, val in record.items():
            if k.lower() in EXCLUDED_NUMERIC_KEYS or isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                raw_val = float(val)
                break

    return raw_date, raw_val


def _is_dated_dict(record: dict) -> bool:
    # Exclude catalog metadata dicts that describe multiple different indicators
    if "key" in record and "group" in record and "unit" in record:
        return False
    d, v = _find_date_and_value(record, COMMON_VALUE_KEYS)
    return d is not None and v is not None


def _extract_records(payload, depth: int = 0) -> list[dict]:
    data = _unwrap_data(payload) if depth == 0 else payload

    if isinstance(data, list):
        dated_items = [
            x for x in data if isinstance(x, dict) and _is_dated_dict(x)
        ]
        if dated_items:
            return dated_items
        return []

    if isinstance(data, dict) and depth < 5:
        priority_keys = (
            "history",
            "points",
            "series",
            "observations",
            "items",
            "rates",
            "values",
            "entries",
            "records",
            "results",
            "poland",
            "PL",
        )
        for key in priority_keys:
            if key in data:
                found = _extract_records(data[key], depth + 1)
                if found:
                    return found

        candidates = []
        for val in data.values():
            if isinstance(val, (list, dict)):
                found = _extract_records(val, depth + 1)
                if found:
                    candidates.append(found)
        if candidates:
            return max(candidates, key=len)

    return []


def _parse_flexible_date(raw_date: str) -> pd.Timestamp:
    s = str(raw_date).strip()
    quarter_match = re.match(r"^(\d{4})-?Q([1-4])$", s, re.IGNORECASE)
    if quarter_match:
        year, q = quarter_match.groups()
        return pd.Period(f"{year}Q{q}", freq="Q").to_timestamp(how="end").normalize()
    return pd.to_datetime(s, errors="coerce")


def _normalize_series(
    payload,
    value_keys: tuple[str, ...] = COMMON_VALUE_KEYS,
    indicator_code: str = "",
    indicator_name: str = "",
    unit: str = "%",
    yoy_lag_periods: int = 0,
    forward_fill_monthly: bool = False,
    min_valid: float = -100.0,
    max_valid: float = 250.0,
) -> pd.DataFrame:
    records = _extract_records(payload)
    rows = []

    for item in records:
        raw_date, raw_val = _find_date_and_value(item, value_keys)
        if raw_date is not None and raw_val is not None:
            rows.append({"raw_date": str(raw_date), "value": raw_val})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["dt"] = df["raw_date"].apply(_parse_flexible_date)
    df = df.dropna(subset=["dt"]).sort_values("dt")

    df["month_end"] = df["dt"] + pd.offsets.MonthEnd(0)
    monthly = (
        df.groupby("month_end", as_index=False)["value"]
        .last()
        .sort_values("month_end")
    )

    if yoy_lag_periods > 0:
        if len(monthly) > yoy_lag_periods:
            monthly["value"] = monthly["value"].pct_change(periods=yoy_lag_periods) * 100.0
        elif len(monthly) > 1:
            monthly["value"] = monthly["value"].pct_change(periods=1) * 400.0
        else:
            return pd.DataFrame()

    monthly["value"] = monthly["value"].replace([np.inf, -np.inf], np.nan)
    monthly = monthly.dropna(subset=["value"])

    # Domain range guard: reject non-percentage levels (e.g., PLN salaries > 250)
    monthly = monthly[
        (monthly["value"] >= min_valid) & (monthly["value"] <= max_valid)
    ]
    if monthly.empty:
        return pd.DataFrame()

    if forward_fill_monthly:
        end_month = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        full_idx = pd.date_range(
            start=monthly["month_end"].min(),
            end=max(monthly["month_end"].max(), end_month),
            freq="ME",
        )
        monthly = (
            monthly.set_index("month_end")
            .reindex(full_idx)
            .ffill()
            .reset_index()
            .rename(columns={"index": "month_end"})
        )

    return pd.DataFrame(
        {
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "trade_date": monthly["month_end"].dt.date,
            "value": monthly["value"].round(6),
            "unit": unit,
            "frequency": "M",
            "source": "finwire",
        }
    )


def fetch_endpoint(path: str, params: dict | None = None):
    url = f"{FINWIRE_BASE}{path}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def discover_openapi_series_paths() -> list[str]:
    """Discovers available /v1/series/... endpoints from finwire's OpenAPI spec."""
    try:
        spec = fetch_endpoint("/openapi.json")
        paths = list(spec.get("paths", {}).keys())
        series_paths = [p for p in paths if "/v1/series/" in p or "/v1/macro/" in p]
        print(f"Discovered finwire API routes: {series_paths}")
        return series_paths
    except Exception:
        return []


def extract_strict_catalog_points(
    indicators_list: list[dict], target_key: str, code: str, label: str
) -> pd.DataFrame:
    """Extracts current and previous observations strictly for target_key."""
    rows = []
    for item in indicators_list:
        if not isinstance(item, dict) or item.get("key") != target_key:
            continue

        for p_key, v_key in (
            ("period", "value"),
            ("previousPeriod", "previousValue"),
        ):
            raw_p = item.get(p_key)
            raw_v = item.get(v_key)
            if raw_p is not None and raw_v is not None:
                dt = _parse_flexible_date(str(raw_p))
                val = float(raw_v)
                if pd.notna(dt) and -100.0 <= val <= 250.0:
                    rows.append(
                        {
                            "indicator_code": code,
                            "indicator_name": label,
                            "trade_date": (dt + pd.offsets.MonthEnd(0)).date(),
                            "value": round(val, 6),
                            "unit": "%",
                            "frequency": "M",
                            "source": "finwire",
                        }
                    )
    return pd.DataFrame(rows)


def main():
    frames = []
    api_paths = discover_openapi_series_paths()

    # 1. NBP Reference Rate
    print("Fetching NBP reference rate...")
    raw = fetch_endpoint(
        "/v1/series/interest-rate/stopa-referencyjna", {"days": 4500}
    )
    frames.append(
        _normalize_series(
            raw,
            indicator_code="NBP_REF_RATE",
            indicator_name="NBP Reference Interest Rate",
            unit="%",
            forward_fill_monthly=True,
        )
    )

    # 2. CPI YoY
    print("Fetching CPI YoY...")
    raw = fetch_endpoint("/v1/series/cpi", {"type": "monthly", "limit": 200})
    frames.append(
        _normalize_series(
            raw,
            indicator_code="CPI_YOY",
            indicator_name="Consumer Price Index (YoY)",
            unit="%",
        )
    )

    # 3. 10-Year Polish Government Bond Yield
    print("Fetching 10Y Polish government bond yield...")
    raw = fetch_endpoint("/v1/series/bond-yields/PL", {"since": "2012-01"})
    frames.append(
        _normalize_series(
            raw,
            indicator_code="BOND_10Y",
            indicator_name="Poland 10-Year Government Bond Yield",
            unit="%",
        )
    )

    # 4. Try dedicated historical endpoints if present in OpenAPI spec
    candidate_routes = [
        (["/v1/series/transaction-prices", "/v1/series/housing/transaction-prices"], "HOUSE_PRICE_YOY", "Residential Housing Price Growth YoY", 4, True),
        (["/v1/series/wages", "/v1/series/wage"], "REAL_WAGE_YOY", "Enterprise Sector Real Wage Growth YoY", 0, False),
        (["/v1/series/retail-sales"], "RETAIL_SALES_YOY", "Retail Sales Constant Prices YoY", 0, False),
        (["/v1/series/industrial-production", "/v1/series/industry"], "INDUSTRIAL_PROD_YOY", "Sold Industrial Production YoY", 0, False),
        (["/v1/series/unemployment"], "UNEMP_RATE", "Registered Unemployment Rate (GUS)", 0, False),
    ]

    loaded_codes = {"NBP_REF_RATE", "CPI_YOY", "BOND_10Y"}
    for paths_to_try, code, label, yoy_lag, ffill in candidate_routes:
        for path in paths_to_try:
            if api_paths and path not in api_paths:
                continue
            try:
                raw = fetch_endpoint(path)
                sample_recs = _extract_records(raw)
                has_yoy = bool(
                    sample_recs
                    and any(k in sample_recs[0] for k in ("yoy", "yoyChange", "yoy_change"))
                )
                df_s = _normalize_series(
                    raw,
                    value_keys=("yoy", "yoyChange", "yoy_change") + COMMON_VALUE_KEYS,
                    indicator_code=code,
                    indicator_name=label,
                    unit="%",
                    yoy_lag_periods=0 if has_yoy else yoy_lag,
                    forward_fill_monthly=ffill,
                )
                if not df_s.empty:
                    frames.append(df_s)
                    loaded_codes.add(code)
                    break
            except requests.HTTPError:
                continue

    # 5. Strict 1-to-1 fallback from /v1/macro/indicators for any remaining indicators
    print("Fetching macro indicators catalog...")
    try:
        catalog_raw = _unwrap_data(fetch_endpoint("/v1/macro/indicators"))
        indicators_list = (
            catalog_raw.get("indicators", [])
            if isinstance(catalog_raw, dict)
            else catalog_raw
        )
        catalog_map = [
            ("bezrobocie-rejestrowane", "UNEMP_RATE", "Registered Unemployment Rate (GUS)"),
            ("sprzedaz-detaliczna-rr", "RETAIL_SALES_YOY", "Retail Sales Constant Prices YoY"),
            ("produkcja-przemyslowa-rr", "INDUSTRIAL_PROD_YOY", "Sold Industrial Production YoY"),
            ("wynagrodzenie-przedsiebiorstw-realnie-rr", "REAL_WAGE_YOY", "Enterprise Sector Real Wage Growth YoY"),
            ("ceny-mieszkan-rr", "HOUSE_PRICE_YOY", "Residential Housing Price Growth YoY"),
        ]
        for target_key, code, label in catalog_map:
            if code in loaded_codes:
                continue
            df_cat = extract_strict_catalog_points(
                indicators_list, target_key, code, label
            )
            if not df_cat.empty:
                frames.append(df_cat)
    except requests.HTTPError as exc:
        print(f"Skipping macro snapshot endpoint ({exc}).")

    valid_frames = [f for f in frames if not f.empty]
    if valid_frames:
        load_macro_series(pd.concat(valid_frames, ignore_index=True))


if __name__ == "__main__":
    main()
