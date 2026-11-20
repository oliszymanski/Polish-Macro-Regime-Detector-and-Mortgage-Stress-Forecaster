IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'PolandMacroAnalytics')
BEGIN
    CREATE DATABASE PolandMacroAnalytics;
END;
GO

USE PolandMacroAnalytics;
GO

IF OBJECT_ID('dbo.macro_series', 'U') IS NOT NULL DROP TABLE dbo.macro_series;
IF OBJECT_ID('dbo.regimes', 'U') IS NOT NULL DROP TABLE dbo.regimes;
IF OBJECT_ID('dbo.features', 'U') IS NOT NULL DROP TABLE dbo.features;
IF OBJECT_ID('dbo.predictions', 'U') IS NOT NULL DROP TABLE dbo.predictions;
GO

CREATE TABLE dbo.macro_series (
    series_id INT IDENTITY(1,1) NOT NULL,
    indicator_code VARCHAR(50) NOT NULL,
    indicator_name NVARCHAR(150) NOT NULL,
    trade_date DATE NOT NULL,
    value DECIMAL(18,6) NOT NULL,
    unit VARCHAR(30) NULL,
    frequency VARCHAR(10) NOT NULL,
    source VARCHAR(50) NOT NULL,
    is_revised BIT NOT NULL CONSTRAINT DF_macro_series_is_revised DEFAULT (0),
    loaded_at  DATETIME2 NOT NULL CONSTRAINT DF_macro_series_loaded_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_macro_series PRIMARY KEY CLUSTERED (indicator_code, trade_date)
);

CREATE NONCLUSTERED INDEX IX_macro_series_trade_date
ON dbo.macro_series (trade_date)
INCLUDE (indicator_code, value);

CREATE TABLE dbo.regimes (
   regime_id INT IDENTITY(1, 1) NOT NULL,
   trade_date DATE NOT NULL,
   regime_label VARCHAR(30) NOT NULL,
   regime_method VARCHAR(20) NOT NULL,
   confidence DECIMAL(5, 4) NULL,
   created_at DATETIME2 NOT NULL CONSTRAINT DF_regimes_created_at DEFAULT (SYSUTCDATETIME()),
   CONSTRAINT PK_regimes PRIMARY KEY CLUSTERED (trade_date, regime_method)
);

CREATE TABLE dbo.features (
    feature_id          INT IDENTITY(1,1)   NOT NULL,
    trade_date          DATE                NOT NULL,
    nbp_ref_rate        DECIMAL(8,4)        NULL,
    nbp_ref_rate_chg_3m DECIMAL(8,4)        NULL,
    cpi_yoy             DECIMAL(8,4)        NULL,
    cpi_yoy_chg_3m      DECIMAL(8,4)        NULL,
    real_wage_yoy       DECIMAL(8,4)        NULL,
    unemployment_rate   DECIMAL(8,4)        NULL,
    eurpln              DECIMAL(10,4)       NULL,
    eurpln_chg_3m       DECIMAL(8,4)        NULL,
    bond_10y            DECIMAL(8,4)        NULL,
    house_price_yoy     DECIMAL(8,4)        NULL,
    retail_sales_yoy    DECIMAL(8,4)        NULL,
    industrial_prod_yoy DECIMAL(8,4)        NULL,
    stress_score_rule   DECIMAL(8,4)        NULL,
    regime_label_rule   VARCHAR(30)         NULL,
    target_stress_3m    BIT                 NULL,
    created_at          DATETIME2           NOT NULL CONSTRAINT DF_features_created_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_features PRIMARY KEY CLUSTERED (trade_date)
);
GO

CREATE TABLE dbo.predictions (
    prediction_id      INT IDENTITY(1,1) NOT NULL,
    trade_date         DATE              NOT NULL,
    model_name         VARCHAR(50)       NOT NULL,
    model_version      VARCHAR(20)       NOT NULL,
    horizon_months     INT               NOT NULL,
    stress_probability DECIMAL(6,4)      NOT NULL,
    predicted_regime   VARCHAR(30)       NOT NULL,
    created_at         DATETIME2         NOT NULL CONSTRAINT DF_predictions_created_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_predictions PRIMARY KEY CLUSTERED (trade_date, model_name, horizon_months)
);

GO
