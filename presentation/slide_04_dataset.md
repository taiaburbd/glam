# Slide 04 — Dataset

---

## UWHVF Dataset

- **Source:** University of Washington Humphrey Visual Field dataset (v1.0)
- **Size:** 28,943 HFA 24-2 VF tests from 3,871 patients
- **Each record contains:** 54-element Total Deviation (TD) array (dB), raw sensitivity, patient age, sex, visit index
- **No structural imaging** (no OCT, no fundus photos)

---

## Quality Filtering

- Retain patient-eyes with: **≥ 3 VF tests** AND **≥ 1 year follow-up**
- Outlier removal: MD rates outside [−10, +5] dB/year removed (24 of 4,300; 0.6%)
- **Final cohort: 4,276 patient-eyes**

---

## Dataset Characteristics (Post-Filtering)


| Metric                           | Value                      |
| -------------------------------- | -------------------------- |
| Patient-eyes                     | 4,276                      |
| Visits per eye (mean ± SD)       | 5.2 ± 2.8                  |
| Median follow-up                 | 4.3 years (range 1.0–20.8) |
| Mean MD rate                     | −0.126 ± 0.956 dB/year     |
| Fast progressors (MD < −1 dB/yr) | 394 (9.2%)                 |


---

## Train / Validation / Test Split

- Stratified by **MD-rate quartile** to balance fast/slow progressors
- **Train:** 2,992 | **Val:** 640 | **Test:** 644

