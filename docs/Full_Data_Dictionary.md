# Complete Data Dictionary - Scenario Engine

This document provides a comprehensive overview of all datasets used in the scenario engine project.

---

## Table of Contents
1. [RatesRegimes.parquet](#ratesregimesparquet)
2. [econ_jobs.parquet](#econ_jobsparquet)
3. [econ_reports.parquet](#econ_reportsparquet)

---

## RatesRegimes.parquet

**Location:** `data/raw/RatesRegimes.parquet`  
**Date Range:** 2010-01-01 to 2025-09-30  
**Total Columns:** 36 (including Date)  
**Primary Use:** Interest rate modeling, yield curve analysis, and regime detection

### Yield Curve (6 indicators)
- USGG2YR Index → US Treasury 2-Year Yield
- USGG3YR Index → US Treasury 3-Year Yield
- USGG5YR Index → US Treasury 5-Year Yield
- USGG7YR Index → US Treasury 7-Year Yield
- USGG10YR Index → US Treasury 10-Year Yield
- USGG30YR Index → US Treasury 30-Year Yield

### Volatility Regime (2 indicators)
- MOVE Index → Bond Market Volatility Index
- VIX Index → Equity Market Volatility Index

### Policy Regime (4 indicators)
- FEDL01 Index → Fed Funds Rate (Overnight)
- USOSFR1 Curncy → 1-Year OIS Rate
- USOSFR2 Curncy → 2-Year OIS Rate
- USOSFR5 Curncy → 5-Year OIS Rate

### BVMB Muni Tickers (6 indicators)
- BVMB2Y Index → Bloomberg Valuation Muni 2Y Benchmark
- BVMB3Y Index → Bloomberg Valuation Muni 3Y Benchmark
- BVMB5Y Index → Bloomberg Valuation Muni 5Y Benchmark
- BVMB7Y Index → Bloomberg Valuation Muni 7Y Benchmark
- BVMB10Y Index → Bloomberg Valuation Muni 10Y Benchmark
- BVMB30Y Index → Bloomberg Valuation Muni 30Y Benchmark

### BVCSUG US Agency Bond Valuation Curves (6 indicators)
- BVCSUG01 BVLI Index → Bloomberg Val Curve US Govt 1Y (Agency)
- BVCSUG02 BVLI Index → Bloomberg Val Curve US Govt 2Y (Agency)
- BVCSUG05 BVLI Index → Bloomberg Val Curve US Govt 5Y (Agency)
- BVCSUG07 BVLI Index → Bloomberg Val Curve US Govt 7Y (Agency)
- BVCSUG10 BVLI Index → Bloomberg Val Curve US Govt 10Y (Agency)
- BVCSUG20 BVLI Index → Bloomberg Val Curve US Govt 20Y (Agency)

### IGUUAC Corporate Bond Valuation Curves (IG) (4 indicators)
- IGUUAC01 BVLI Index → IG US Agg Corporate 1Y BVLI
- IGUUAC05 BVLI Index → IG US Agg Corporate 5Y BVLI
- IGUUAC07 BVLI Index → IG US Agg Corporate 7Y BVLI
- IGUUAC10 BVLI Index → IG US Agg Corporate 10Y BVLI

### Corporate Bond ETFs (2 indicators)
- LQD US Equity → iShares iBoxx $ Investment Grade Corporate Bond ETF
- HYG US Equity → iShares iBoxx $ High Yield Corporate Bond ETF

### OAS Indices (3 indicators)
- LUACOAS Index → Bloomberg US Aggregate Corporate OAS
- LF98OAS Index → Bloomberg US Corporate IG OAS (F98 Series)
- LUMSOAS Index → Bloomberg US Mortgage-Backed Securities OAS

### Total Return Indices (2 indicators)
- LUCMTRUU Index → Bloomberg US Corporate Total Return Unhedged USD
- LUMSTRUU Index → Bloomberg US Mortgage Securities Total Return Unhedged USD

**Total: 35 Data Columns + 1 Date Column = 36 Columns**

---

## econ_jobs.parquet

**Location:** `data/raw/econ_jobs.parquet`  
**Date Range:** 2010-10-06 to October 13, 2025  
**Total Records:** 4,266 rows  
**Total Columns:** 8  
**Primary Use:** Employment and labor market economic indicators

### Column Structure

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| Date | object | Release date and time of economic data (format: YYYY-MM-DD HH:MM:SS) |
| Event | object | Name of the economic indicator/event |
| Survey | object | Consensus forecast/survey expectation |
| Actual | object | Actual reported value |
| Prior | object | Previous period's value |
| Revised | object | Revised value from previous period (if applicable) |
| Relevance | float64 | Bloomberg relevance score (0-100) indicating market impact |
| Ticker | object | Bloomberg ticker for the economic indicator |

### Sample Data Snapshot

```
Date                  Event                          Survey    Actual    Prior     Revised   Relevance  Ticker
2010-10-06 08:15:00   ADP Employment Change         20000     -39000    -10000    99000     92.6174    ADP CHNG Index
2010-10-07 08:30:05   Initial Jobless Claims        455000    445000    453000    459000    98.6577    INJCJC Index
2010-10-07 08:30:06   Continuing Claims             4450000   4462000   4457000   4452000   69.0604    INJCSP Index
```

### Coverage Statistics
- **Unique Events:** 31 different labor market indicators
- **Update Frequency:** Weekly to monthly depending on indicator
- **Key Indicators Include:**
  - Nonfarm Payrolls (NFP)
  - ADP Employment Change
  - Initial & Continuing Jobless Claims
  - Unemployment Rate
  - Labor Force Participation Rate
  - Average Hourly Earnings
  - Job Openings (JOLTS)

---

## econ_reports.parquet

**Location:** `data/raw/econ_reports.parquet`  
**Date Range:** 2010-10-04 to October 13, 2025  
**Total Records:** 3,773 rows  
**Total Columns:** 8  
**Primary Use:** Broader economic reports beyond employment (housing, GDP, inflation, manufacturing, etc.)

### Column Structure

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| Date | object | Release date and time of economic data (format: YYYY-MM-DD HH:MM:SS) |
| Event | object | Name of the economic indicator/event |
| Survey | object | Consensus forecast/survey expectation |
| Actual | object | Actual reported value |
| Prior | object | Previous period's value |
| Revised | object | Revised value from previous period (if applicable) |
| Relevance | float64 | Bloomberg relevance score (0-100) indicating market impact |
| Ticker | object | Bloomberg ticker for the economic indicator |

### Sample Data Snapshot

```
Date                  Event                          Survey    Actual    Prior     Revised   Relevance  Ticker
2010-10-04 10:00:04   Pending Home Sales MoM        0.025     0.043     0.052     0.054     77.8523    USPHTMOM Index
2010-10-04 10:00:05   Pending Home Sales NSA YoY    --        -0.184    -0.201    -0.207    38.9262    USPHTYOY Index
2010-10-06 07:00:00   MBA Mortgage Applications     --        -0.002    -0.008    --        90.6040    MBAVCHNG Index
```

### Coverage Statistics
- **Unique Events:** 26 different economic indicators
- **Update Frequency:** Daily to quarterly depending on indicator
- **Key Categories Include:**
  - **Housing:** Pending Home Sales, New Home Sales, Existing Home Sales, Housing Starts, Building Permits, NAHB Index
  - **Manufacturing:** ISM Manufacturing, Durable Goods Orders, Industrial Production
  - **GDP & Growth:** GDP releases (Advance, Second, Final), Personal Income/Spending
  - **Inflation:** CPI, PPI, PCE Price Index
  - **Consumer:** Consumer Confidence, Retail Sales, Personal Consumption
  - **Business:** Business Inventories, Wholesale Sales
  - **Financial:** MBA Mortgage Applications, Credit conditions

---

## Data Quality Notes

### Missing Data
- Survey values may be missing (shown as "--") for some indicators
- Revised values are only present when historical revisions occur
- All relevance scores are populated (no missing values)

### Data Types
- Numeric fields (Survey, Actual, Prior, Revised) are stored as strings to preserve original formatting and handle "--" for missing data
- Convert to float/numeric types during analysis as needed
- Relevance scores are already numeric (float64)

### Date Handling
- Date fields include time components (HH:MM:SS)
- Times reflect official release times (typically 8:30 AM or 10:00 AM ET for major indicators)
- Last update timestamp is included in the final row of each dataset

### Bloomberg Ticker Format
- Format: `[MNEMONIC] Index` or `[MNEMONIC] Curncy`
- Used for direct Bloomberg Terminal lookups
- Consistent across all datasets

---

## Usage in Scenario Engine

These datasets work together to provide comprehensive economic context for interest rate scenario generation:

1. **RatesRegimes.parquet** provides the core yield curve and rates data
2. **econ_jobs.parquet** provides labor market context for regime changes
3. **econ_reports.parquet** provides broader economic indicators for scenario conditioning

All three datasets are merged in `data/processed/processed_data.parquet` for unified analysis.

---

**Last Updated:** October 25, 2025

