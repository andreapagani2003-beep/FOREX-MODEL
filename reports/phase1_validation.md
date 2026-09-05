# Phase 1 validation

Result: **FAIL**
errors: 1, warnings: 0

## Stats

| key | value |
|---|---|
| rows | 4164 |
| first_date | 2010-01-04 |
| last_date | 2026-09-03 |
| weekdays_in_range | 4349 |
| weekdays_missing | 185 |
| coverage_pct | 95.75 |
| largest_gap_weekdays | 2 |
| fx_lag_days_max | 0.0 |
| fx_lag_days_mean | 0.0 |
| us_lag_days_max | 0.0 |
| us_lag_days_mean | 0.0 |
| jgb_lag_days_max | 7.0 |
| jgb_lag_days_mean | 0.141 |
| timing_corr_same_day | -0.043 |
| timing_corr_spread_lag1 | 0.42 |
| timing_corr_spread_lead1 | 0.003 |
| usdjpy_min | 75.74 |
| us2y_min | 0.09 |
| us10y_min | 0.52 |
| jgb2y_min | -0.372 |
| jgb10y_min | -0.297 |
| spread2y_min | 0.025 |
| spread10y_min | 0.501 |
| usdjpy_max | 163.864 |
| us2y_max | 5.19 |
| us10y_max | 4.98 |
| jgb2y_max | 1.854 |
| jgb10y_max | 3.006 |
| spread2y_max | 5.131 |
| spread10y_max | 4.135 |

## errors

- `timing`: spot/spread daily-change correlation is not highest on the same day: same -0.04, spread lag1 +0.42, lead1 +0.00 -> a source is date-shifted
