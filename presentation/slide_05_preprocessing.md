# Slide 05 — Data Preprocessing

---

## Target Label Computation

- **MD proxy:** Mean of all non-sentinel TD values (TD < 90 dB) per visit
- **VFI proxy:** `clip(100 + 2 × MD_proxy, 0, 100)` — approximates the clinical VFI
- **Progression rate:** OLS linear regression of MD (or VFI) values over time-from-baseline (years)

---

## VF Input Normalization

- TD values clipped to **[−35, 0] dB**
  - Untested locations (TD ≥ 90 dB) set to 0 before clipping
- Linearly scaled to **[0, 1]**: `TD_norm = (TD_clipped + 35) / 35`

---

## Clinical Features (5 total)

| Feature | Normalization |
|---|---|
| Age at baseline | (age − 40) / 50 |
| Sex | 0 = female, 1 = male |
| Visit count | (n_visits − 3) / 17 |
| Follow-up duration | follow_up_years / 20 |
| Baseline MD proxy | (MD_baseline + 35) / 35 |
