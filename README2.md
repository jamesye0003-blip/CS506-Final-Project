# Wildfire & Smoke Risk Prediction (Weather → Wildfire & AQI)

## 1) Project Description
We will build a short-term (6–72 hour) prediction pipeline for (a) **new wildfire ignition probability** and (b) **downwind smoke/AQI risk** at county/grid level. The system fuses near-real-time active fire detections (VIIRS/MODIS), weather forecasts (wind, humidity, temperature), weekly drought conditions, and historical air-quality observations. Data are ingested via open/public APIs and GIS files, then aligned on a common time–space grid for modeling and visualization. Primary users include city emergency ops, transit operators, and the general public.

## 2) Clear Goals
- **G1. Wildfire ignition risk (classification):** Predict whether at least one new active fire will be detected in a grid cell (e.g., 5–10 km) in the **next 24 hours** given recent weather and drought context.  
- **G2. Smoke/AQI short-term risk (regression):** Predict **PM2.5 (or AQI)** at monitor/county level for **+6h, +12h, +24h** horizons using upwind fire detections and forecast winds.  
- **G3. Alerting (thresholding):** Produce exceedance probabilities (e.g., **P(AQI ≥ USG)**) to drive simple “Heads-up” notifications.

**Success metrics:** AUC-PR for G1; RMSE / MAE and **pinball loss at τ=0.5/0.95** for G2; well-calibrated reliability curves for G3.

## 3) Data to Collect & How We’ll Collect It
We prioritize official/open feeds with stable docs:

### 3.1 Active Fires (near real-time + archive)
- **NASA/FIRMS** VIIRS & MODIS active fire detections (CSV/GeoJSON/HDF) with per-point detection time, confidence, FRP, and sensor.  
**Collection:** Pull rolling 24–72h windows (poll every 15–30 min) and archive daily snapshots.

### 3.2 Weather (forecast & observations)
- **NWS api.weather.gov**: gridpoint forecasts (wind speed/dir, temperature, relative humidity), watches/warnings, and hourly time series in JSON.  
**Collection:** For each grid cell or county centroid, query gridpoint endpoints for the next 72h.

### 3.3 Air Quality (targets & context)
- **AirNow API**: near-real-time AQI and hourly PM2.5/ozone for thousands of monitors (free key). Useful for real-time targets and evaluation.  
- **EPA AQS API**: validated historical PM2.5 concentrations at monitor level. Useful for backfill and robust training.  
**Collection:** Nightly backfills from AQS for history; hourly pulls from AirNow for “nowcast” targets.

### 3.4 Drought & fuel dryness (weekly)
- **U.S. Drought Monitor (USDM)** shapefiles and statistics (D0–D4) for each Thursday. Used as a slowly varying dryness/fuel proxy and prior for ignition risk.  
**Collection:** Weekly download of national shapefiles, then spatially join to our grid.

### 3.5 Optional enrichments (phase 2)
- Terrain & land cover (NLCD), population for exposure weighting, and satellite smoke plumes if we add dispersion calibration (not required for MVP).

**Ingestion mechanics:** All sources are polled via scheduled jobs (e.g., every 15–30 min for FIRMS/AirNow, every 1–3 h for NWS, weekly for USDM). Each pull writes raw JSON/CSV/GeoTIFF to object storage; a dbt/SQL layer materializes **tidy tables** keyed by `{grid_id, valid_time}` or `{monitor_id, valid_time}`.

## 4) Data Model & Feature Engineering Plan
(We’ll be explicit on inputs/targets; modeling choices may iterate.)

### 4.1 Spatial–temporal grid
- Define a conterminous-US grid (e.g., 0.1° ≈ 10 km) with H3/quadkey equivalents for fast joins.  
- **Targets:**  
  - **G1:** `ignite_24h` ∈ {0,1} per grid cell (any new FIRMS detection inside cell in next 24h).  
  - **G2:** `PM25_{+6,+12,+24}` at monitor level (and county averages).  
- **Key joins:** nearest-neighbor or polygon overlays from USDM to grid; upwind fire features using great-circle bearings & wind vectors.

### 4.2 Core features
- **Weather forecast features:** u/v wind components, wind speed & direction, RH, T, precip rate; last-k-hour trends.  
- **Fire proximity features:** counts/FRP sum of FIRMS detections **upwind** within 25/50/100 km and within the last 6/12/24h; distance to nearest fire; persistence (fires seen ≥2 consecutive overpasses).  
- **Dryness/fuel proxy:** USDM category (D0–D4) one-hot + rolling weeks in drought.  
- **Temporal calendar:** hour of day, day of week, week of year, holidays.

### 4.3 Modeling approaches
We will start simple and escalate as needed:
- **G1 (Ignition probability):**  
  - Baseline: logistic regression with L2.  
  - Primary: **XGBoost/LightGBM** with class-imbalance handling (scale_pos_weight or focal loss).  
  - Stretch: **spatiotemporal GNN** (message passing on grid adjacency) to capture neighborhood contagion.  
- **G2 (AQI/PM2.5):**  
  - Baseline: seasonal naive + persistence (y_t+h ≈ y_t); ARIMAX with weather.  
  - Primary: **Gradient boosted quantile regression** (τ=0.5, 0.95) with engineered upwind-fire × wind-vector features.  
  - Stretch: **Seq2Seq/TFT** (Temporal Fusion Transformer) with multi-horizon outputs.

### 4.4 Label construction & leakage control
- For **G1**, only fires first observed **after** the feature time are labeled positive; exclude cells with ongoing active fires at T=0 to focus on *new* ignitions.  
- For **G2**, when predicting +6h/+12h/+24h we freeze features to those available at T=0 (no peeking at later FIRMS or updated forecasts).

## 5) Visualization Plan
- **Risk maps:** Interactive web map with two layers:  
  - **Ignition probability** (quantiles) by grid cell;  
  - **AQI predictions** with monitor dots, confidence bands on hover.  
- **Event timelines:** For selected cities/regions, show predicted AQI vs. observed **and** upwind fire counts × wind direction glyphs.  
- **Reliability plots:** Calibration curves for `P(AQI≥threshold)` and for ignition risk.  
- **Case notebooks:** “Story” pages for notable events (e.g., plume intrusions), overlaying FIRMS detections & NWS winds.

## 6) Test Plan
- **Temporal splits (primary):** Rolling-origin evaluation by month/week: train on months 1…k, validate on month k+1. This mirrors real deployment.  
- **Spatial splits (robustness):** **Leave-region-out** (e.g., train on West+South, test on Northeast) to assess geographic generalization.  
- **Event-level splits (smoke):** **Leave-plume-event-out** (detect multi-day plume periods; hold out entire events).  
- **Metrics:**  
  - **G1:** AUC-PR, recall@fixed-precision (e.g., P=0.5), and **Brier score** with isotonic calibration.  
  - **G2:** MAE/RMSE; **pinball loss at τ=0.5/0.95**; coverage of 50%/90% prediction intervals.

## 7) System Architecture (high level)
1. **Ingest:** Scheduled jobs pull FIRMS, NWS, AirNow, AQS, and USDM (with keys where required).  
2. **Staging:** Store raw JSON/CSV; log provenance (source, query params, timestamp).  
3. **Transform:** Spatial joins to grid/monitors; feature generation (upwind convolution over fire points using NWS winds).  
4. **Train/Score:** Offline training with time-aware CV; batch prediction for horizons h∈{6,12,24,72}.  
5. **Serve & Viz:** Publish parquet/tiles to a lightweight dashboard (e.g., kepler.gl/Deck.gl or Mapbox) with downloadable CSV.

## 8) Data Management & Ethics
- Use only publicly available, non-PII data. Respect API terms and rate limits (AirNow/AQS require keys; NWS is public and cache-friendly).  
- Provide clear disclaimers: predictions are advisory and not a substitute for official alerts.  
- Open our **feature/label schema** so others can replicate.

## 9) Minimal Viable Product (2–3 weeks)
- **Scope:** Western US grid; horizons +6h/+24h; metrics dashboard.  
- **Data:** FIRMS (VIIRS S-NPP/NOAA-20), NWS gridpoint forecasts, AirNow hourly PM2.5, USDM weekly.  
- **Models:** G1 logistic + LGBM; G2 quantile LGBM.  
- **Outputs:** Risk map, monitor-level forecast vs. actual, reliability curves.

## 10) Stretch Goals
- Add **dispersion surrogates** (e.g., learned “upwind convolution” kernels per region).  
- Explore **graph neural nets** over grid adjacency for ignition risk.  
- Calibrate against additional archives (EPA modeled downscaler, etc.).

---

### Appendix: Draft Table Schemas (abridged)

**`fires_raw`**  
`{fire_id, time_utc, lat, lon, sensor, frp, confidence, source}`

**`weather_grid`**  
`{grid_id, valid_time, u10, v10, wspd, wdir, rh, t2m, precip_rate, source}`

**`aq_obs`**  
`{monitor_id, lat, lon, time_utc, parameter, value, aqi, source}`

**`usdm_weekly`**  
`{geom, week_date, d_class, dsci, source}`

**`features_grid_t`** (for G1)  
`{grid_id, valid_time, upwind_fire_count_{6,12,24}, upwind_frp_sum_{6,12,24}, dist_to_nearest_fire_km, u10, v10, rh, t2m, precip_rate, usdm_d0…d4, doy, dow, hod}`

**`labels_grid_t+24`**  
`{grid_id, target_ignite_24h}`

**`features_monitor_t`** (for G2)  
`{monitor_id, valid_time, upwind_fire_count_km_bands, wind_convolution_indices, recent_aq_{1,3,6}h, weather terms, calendar dummies}`

**`labels_monitor_t+h`**  
`{monitor_id, horizon_h, pm25, aqi}`

> At this stage we’ve been very explicit about the objectives and data collection plan (with sources and polling strategy). Modeling and visualization details are intentionally flexible to allow iteration after a first round of EDA and backtesting.
