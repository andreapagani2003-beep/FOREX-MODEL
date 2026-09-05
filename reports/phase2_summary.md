# Phase 2 — Statistical confirmation

Data: `daily.parquet`, 2010-01-04 → 2026-09-03, 4164 rows, convention `ny_close`. Significance 0.05. Half-life band 5–50 bars.

## Verdict

**Acceptance NOT MET.** Criterion (handoff §5): at least one specification cointegrated at 5% on post-2016 AND half-life in band AND no unexplained break in the last 24 months.

No primary/secondary recommendation: no specification passes on post_2016.

## Comparison across specifications

EG p = Engle-Granger p-value (USD/JPY on spread); Johansen trace r=0 vs 95% critical value; beta in USD/JPY per 1pp; half-life in trading days from the OU fit of the static-OLS residual with bootstrap 95% CI; breaks = sequential sup-F at 5% (sieve-bootstrap p-values).

| spec | sample | nobs | usdjpy_order | spread_order | eg_p_y_on_x | eg_p_x_on_y | johansen_trace_r0 | johansen_crit95 | johansen_rank | beta_static | beta_se | r2 | beta_kalman_last | beta_roll500_range | half_life | hl_ci_low | hl_ci_high | hl_kalman | breaks_found | breaks_unexplained_recent | anchors_pass | cointegrated_5pct | hl_in_band | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10y_nominal | full | 4164 | I(1) | I(1) | 0.968 | 0.635 | 4.9 | 15.5 | 0 | 18.96 | 0.35 | 0.409 | 19.7 | -34.0..24.4 | 359 | 116 | 643 | 91 | 0 | 0 | yes | no | no | no |
| 10y_nominal | post_2016 | 2664 | I(1) | I(1) | 0.994 | 0.940 | 2.8 | 15.5 | 0 | 13.91 | 0.38 | 0.336 | 15.5 | -6.6..16.3 | 648 | 90 | inf | 149 | 0 | 0 | yes | no | no | no |
| 10y_nominal | post_2022 | 1167 | I(1) | ambiguous | 0.361 | 0.764 | 12.0 | 15.5 | 0 | 0.79 | 0.52 | 0.002 | 9.0 | -6.6..16.1 | 122 | 51 | 219 | 48 | 2 | 1 | yes | no | no | no |
| 2y_nominal | full | 4164 | I(1) | I(1) | 0.955 | 0.874 | 1.6 | 15.5 | 0 | 12.71 | 0.15 | 0.636 | 16.5 | -24.4..87.1 | 438 | 116 | 819 | 96 | 0 | 0 | yes | no | no | no |
| 2y_nominal | post_2016 | 2664 | I(1) | I(1) | 0.986 | 0.905 | 2.4 | 15.5 | 0 | 10.32 | 0.19 | 0.534 | 12.1 | -5.5..9.2 | 455 | 82 | inf | 156 | 0 | 0 | yes | no | no | no |
| 2y_nominal | post_2022 | 1167 | I(1) | ambiguous | 0.516 | 0.650 | 18.0 | 15.5 | 1 | 1.59 | 0.34 | 0.018 | 8.7 | -5.5..7.9 | 135 | 50 | 257 | 62 | 1 | 0 | yes | yes | no | no |
| 10y_real | full | 4164 | I(1) | I(1) | 0.985 | 0.811 | 7.0 | 15.5 | 0 | 18.00 | 0.39 | 0.335 | 10.5 | -8.9..36.4 | 619 | 139 | inf | 157 | 0 | 0 | no | no | no | no |
| 10y_real | post_2016 | 2664 | I(1) | I(1) | 0.992 | 0.911 | 2.9 | 15.5 | 0 | 9.82 | 0.50 | 0.127 | 9.8 | -6.9..33.5 | 2101 | 109 | inf | 192 | 0 | 0 | yes | no | no | no |
| 10y_real | post_2022 | 1167 | I(1) | ambiguous | 0.494 | 0.849 | 11.4 | 15.5 | 0 | 2.24 | 0.49 | 0.018 | 5.4 | -6.9..17.3 | 132 | 50 | 249 | 83 | 1 | 0 | yes | no | no | no |

## Rolling cointegration (500-day Engle-Granger, full sample)

- `10y_nominal`: 6.7% of 733 windows reject at 5%. Median p by window-end year: 2011: 0.62, 2012: 0.20, 2013: 0.75, 2014: 0.59, 2015: 0.37, 2016: 0.62, 2017: 0.71, 2018: 0.28, 2019: 0.19, 2020: 0.01, 2021: 0.66, 2022: 0.74, 2023: 0.38, 2024: 0.87, 2025: 0.40, 2026: 0.44
- `2y_nominal`: 9.3% of 733 windows reject at 5%. Median p by window-end year: 2011: 0.13, 2012: 0.08, 2013: 0.98, 2014: 0.52, 2015: 0.14, 2016: 0.77, 2017: 0.66, 2018: 0.22, 2019: 0.19, 2020: 0.01, 2021: 0.71, 2022: 0.53, 2023: 0.65, 2024: 0.90, 2025: 0.38, 2026: 0.55
- `10y_real`: 10.6% of 733 windows reject at 5%. Median p by window-end year: 2011: 0.14, 2012: 0.23, 2013: 0.97, 2014: 0.59, 2015: 0.92, 2016: 0.83, 2017: 0.57, 2018: 0.23, 2019: 0.18, 2020: 0.01, 2021: 0.21, 2022: 0.51, 2023: 0.42, 2024: 0.81, 2025: 0.39, 2026: 0.42

## Structural breaks

- `10y_nominal` / post_2022: break 2024-01-17 (sup-F 3192, p=0.002), beta 15.0 → -4.0; nearest known: 2024-03-19 BoJ NIRP exit (62d) → UNEXPLAINED
- `10y_nominal` / post_2022: break 2025-03-19 (sup-F 562, p=0.018), beta 9.4 → -13.2; nearest known: 2024-07-31 BoJ hike / carry unwind (231d) → UNEXPLAINED, recent
- `2y_nominal` / post_2022: break 2023-08-28 (sup-F 2260, p=0.032), beta 7.0 → -2.0; nearest known: 2023-07-28 BoJ YCC change (31d) → explained
- `10y_real` / post_2022: break 2023-08-15 (sup-F 2923, p=0.012), beta 11.1 → -4.5; nearest known: 2023-07-28 BoJ YCC change (18d) → explained

Chow tests at the known policy dates (`10y_nominal`, full sample). Classical p assumes iid errors and is meaningless here (the residual is near-integrated); the sieve-bootstrap p is the one to read.

| date | event | F | p classical | p bootstrap | beta before → after | resid var ratio |
|---|---|---|---|---|---|---|
| 2022-09-22 | MoF intervention | 2703 | 0.0e+00 | 0.086 | 10.0 → -6.0 | 0.32 |
| 2022-12-20 | BoJ YCC tweak | 2525 | 0.0e+00 | 0.090 | 11.6 → -6.0 | 0.32 |
| 2023-07-28 | BoJ YCC change | 2535 | 0.0e+00 | 0.064 | 13.5 → -4.4 | 0.14 |
| 2023-10-31 | BoJ YCC change | 2416 | 0.0e+00 | 0.060 | 14.5 → -4.2 | 0.15 |
| 2024-03-19 | BoJ NIRP exit | 2217 | 0.0e+00 | 0.056 | 15.8 → -3.8 | 0.15 |
| 2024-04-29 | MoF intervention | 2156 | 0.0e+00 | 0.058 | 16.2 → -4.3 | 0.15 |
| 2024-07-31 | BoJ hike / carry unwind | 1905 | 0.0e+00 | 0.058 | 17.2 → -6.7 | 0.11 |

## Sanity anchors (handoff §5.7)

Sign of the static-OLS residual in the anchor windows (full sample):

- 2023-01-01..2023-01-31: expected negative, observed negative (mean residual -2.3, z-score mean -0.51) → pass
- 2024-07-01..2024-07-15: expected positive, observed positive (mean residual +24.8, z-score mean +1.61) → pass

## Figures

- `10y_nominal`: `reports/figures/phase2_10y_nominal.png`
- `2y_nominal`: `reports/figures/phase2_2y_nominal.png`
- `10y_real`: `reports/figures/phase2_10y_real.png`

## Notes on method

- Stationarity: ADF (H0 unit root) and KPSS (H0 stationary) on levels and first differences.
- Cointegration: Engle-Granger in both orderings (MacKinnon p-values); Johansen trace and max-eigenvalue with det_order 0, 1 lag difference.
- Beta: static OLS; rolling OLS 250/500/750; Kalman filter with random-walk (alpha, beta), delta 1e-05, observation variance from the first 250 obs.
- OU: AR(1) on the residual, theta = -ln(1 + slope), half-life ln2/theta; 95% CI from a 1000-draw residual bootstrap.
- Breaks: Chow at known dates with classical and sieve-bootstrap p; unknown breaks by sequential sup-F (trim 0.15, up to 3, 500 sieve-bootstrap draws). A found break within 45 days of a known policy date counts as explained.
- The `10y_real` spec uses US TIPS real yield minus *nominal* JGB yield: there is no JGB real-yield series in the pipeline, so it is a partial real spread.
