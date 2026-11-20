USE PolandMacroAnalytics;
GO

CREATE OR ALTER VIEW dbo.vw_monthly_features
AS
WITH pivoted AS (
    SELECT
        trade_date,
        MAX(CASE WHEN indicator_code = 'NBP_REF_RATE'        THEN value END) AS nbp_ref_rate,
        MAX(CASE WHEN indicator_code = 'CPI_YOY'             THEN value END) AS cpi_yoy,
        MAX(CASE WHEN indicator_code = 'REAL_WAGE_YOY'       THEN value END) AS real_wage_yoy,
        MAX(CASE WHEN indicator_code = 'UNEMP_RATE'          THEN value END) AS unemployment_rate,
        MAX(CASE WHEN indicator_code = 'EURPLN'              THEN value END) AS eurpln,
        MAX(CASE WHEN indicator_code = 'BOND_10Y'            THEN value END) AS bond_10y,
        MAX(CASE WHEN indicator_code = 'HOUSE_PRICE_YOY'     THEN value END) AS house_price_yoy,
        MAX(CASE WHEN indicator_code = 'RETAIL_SALES_YOY'    THEN value END) AS retail_sales_yoy,
        MAX(CASE WHEN indicator_code = 'INDUSTRIAL_PROD_YOY' THEN value END) AS industrial_prod_yoy
    FROM dbo.macro_series
    GROUP BY trade_date
),
core_months AS (
    SELECT *
    FROM pivoted
    WHERE nbp_ref_rate IS NOT NULL
      AND cpi_yoy IS NOT NULL
),
with_lags_and_moments AS (
    SELECT
        trade_date,
        CAST(nbp_ref_rate AS DECIMAL(8,4))        AS nbp_ref_rate,
        CAST(nbp_ref_rate - LAG(nbp_ref_rate, 3) OVER (ORDER BY trade_date) AS DECIMAL(8,4)) AS nbp_ref_rate_chg_3m,
        CAST(cpi_yoy AS DECIMAL(8,4))             AS cpi_yoy,
        CAST(cpi_yoy - LAG(cpi_yoy, 3) OVER (ORDER BY trade_date) AS DECIMAL(8,4))           AS cpi_yoy_chg_3m,
        CAST(real_wage_yoy AS DECIMAL(8,4))       AS real_wage_yoy,
        CAST(unemployment_rate AS DECIMAL(8,4))   AS unemployment_rate,
        CAST(unemployment_rate - LAG(unemployment_rate, 3) OVER (ORDER BY trade_date) AS DECIMAL(8,4)) AS unemp_chg_3m,
        CAST(eurpln AS DECIMAL(10,4))             AS eurpln,
        CAST(((eurpln / NULLIF(LAG(eurpln, 3) OVER (ORDER BY trade_date), 0)) - 1.0) * 100.0 AS DECIMAL(8,4)) AS eurpln_chg_3m,
        CAST(bond_10y AS DECIMAL(8,4))            AS bond_10y,
        CAST(house_price_yoy AS DECIMAL(8,4))     AS house_price_yoy,
        CAST(retail_sales_yoy AS DECIMAL(8,4))    AS retail_sales_yoy,
        CAST(industrial_prod_yoy AS DECIMAL(8,4)) AS industrial_prod_yoy,

        -- Rolling 36-month statistics (minimum 12 observations required)
        CASE WHEN COUNT(nbp_ref_rate) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) >= 12
             THEN (nbp_ref_rate - AVG(nbp_ref_rate) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW))
                  / NULLIF(STDEV(nbp_ref_rate) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW), 0)
        END AS z_nbp_ref_rate_36m,

        CASE WHEN COUNT(cpi_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) >= 12
             THEN (cpi_yoy - AVG(cpi_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW))
                  / NULLIF(STDEV(cpi_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW), 0)
        END AS z_cpi_yoy_36m,

        CASE WHEN COUNT(real_wage_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) >= 12
             THEN (real_wage_yoy - AVG(real_wage_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW))
                  / NULLIF(STDEV(real_wage_yoy) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW), 0)
        END AS z_real_wage_yoy_36m,

        CASE WHEN COUNT(eurpln) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW) >= 12
             THEN (eurpln - AVG(eurpln) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW))
                  / NULLIF(STDEV(eurpln) OVER (ORDER BY trade_date ROWS BETWEEN 35 PRECEDING AND CURRENT ROW), 0)
        END AS z_eurpln_36m
    FROM core_months
)
SELECT
    trade_date,
    nbp_ref_rate,
    nbp_ref_rate_chg_3m,
    cpi_yoy,
    cpi_yoy_chg_3m,
    real_wage_yoy,
    unemployment_rate,
    unemp_chg_3m,
    eurpln,
    eurpln_chg_3m,
    bond_10y,
    house_price_yoy,
    retail_sales_yoy,
    industrial_prod_yoy,
    CAST(ISNULL(z_nbp_ref_rate_36m, 0) AS DECIMAL(8,4))   AS z_nbp_ref_rate_36m,
    CAST(ISNULL(z_cpi_yoy_36m, 0) AS DECIMAL(8,4))        AS z_cpi_yoy_36m,
    CAST(ISNULL(z_real_wage_yoy_36m, 0) AS DECIMAL(8,4))  AS z_real_wage_yoy_36m,
    CAST(ISNULL(z_eurpln_36m, 0) AS DECIMAL(8,4))         AS z_eurpln_36m
FROM with_lags_and_moments;
GO
