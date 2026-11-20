USE PolandMacroAnalytics;
GO

IF @@TRANCOUNT > 0
    ROLLBACK TRANSACTION;
GO

CREATE OR ALTER PROCEDURE dbo.sp_refresh_features_and_regimes
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    BEGIN TRY
        BEGIN TRANSACTION;

        DROP TABLE IF EXISTS #staged_features;

        ;WITH scored AS (
            SELECT
                f.trade_date,
                f.nbp_ref_rate,
                f.nbp_ref_rate_chg_3m,
                f.cpi_yoy,
                f.cpi_yoy_chg_3m,
                f.real_wage_yoy,
                f.unemployment_rate,
                f.eurpln,
                f.eurpln_chg_3m,
                f.bond_10y,
                f.house_price_yoy,
                f.retail_sales_yoy,
                f.industrial_prod_yoy,
                CAST(
                    (0.35 * ISNULL(f.nbp_ref_rate, 0))
                  + (0.25 * ISNULL(f.cpi_yoy, 0))
                  + (0.80 * CASE WHEN ISNULL(f.nbp_ref_rate_chg_3m, 0) > 0 THEN f.nbp_ref_rate_chg_3m ELSE 0 END)
                  - (0.30 * ISNULL(f.real_wage_yoy, 0))
                  + (0.10 * ISNULL(f.eurpln_chg_3m, 0))
                AS DECIMAL(8,4)) AS stress_score_rule,
                CASE
                  WHEN (ISNULL(f.nbp_ref_rate_chg_3m, 0) > 0.25 AND f.cpi_yoy > 5.0 AND ISNULL(f.real_wage_yoy, 0) < 1.0)
                     OR (f.nbp_ref_rate >= 5.50 AND f.cpi_yoy > 6.0)
                        THEN 'Stress'
                  WHEN f.nbp_ref_rate <= 4.00
                     AND ISNULL(f.unemp_chg_3m, 0) <= 0.0
                     AND f.cpi_yoy <= 4.5
                     AND (f.real_wage_yoy > 2.0 OR (f.real_wage_yoy IS NULL AND ISNULL(f.bond_10y, 3.0) <= 3.75))
                        THEN 'Expansion'
                  ELSE 'Neutral'
               END AS regime_label_rule
            FROM dbo.vw_monthly_features f
        ),
        with_targets AS (
            SELECT
                s.*,
                LEAD(s.regime_label_rule, 1) OVER (ORDER BY s.trade_date) AS next_1m_regime,
                LEAD(s.regime_label_rule, 2) OVER (ORDER BY s.trade_date) AS next_2m_regime,
                LEAD(s.regime_label_rule, 3) OVER (ORDER BY s.trade_date) AS next_3m_regime
            FROM scored s
        )
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
            CASE
                WHEN next_3m_regime IS NULL THEN NULL
                WHEN (
                    (CASE WHEN next_1m_regime = 'Stress' THEN 1 ELSE 0 END) +
                    (CASE WHEN next_2m_regime = 'Stress' THEN 1 ELSE 0 END) +
                    (CASE WHEN next_3m_regime = 'Stress' THEN 1 ELSE 0 END)
                ) >= 2 THEN CAST(1 AS BIT)
                ELSE CAST(0 AS BIT)
            END AS target_stress_3m
        INTO #staged_features
        FROM with_targets;

        DELETE FROM dbo.features;

        INSERT INTO dbo.features (
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
            target_stress_3m
        )
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
            target_stress_3m
        FROM #staged_features
        ORDER BY trade_date;

        DELETE FROM dbo.regimes WHERE regime_method = 'rule_based';
   
         INSERT INTO dbo.regimes (
            trade_date,
            regime_label,
            regime_method,
            confidence,
            created_at
         )
         SELECT
            trade_date,
            regime_label_rule,
            'rule_based',
            CAST(1.0000 AS DECIMAL(5,4)),
            SYSUTCDATETIME()
         FROM #staged_features
         ORDER BY trade_date;

        DROP TABLE IF EXISTS #staged_features;

        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;
        THROW;
    END CATCH
END;
GO
