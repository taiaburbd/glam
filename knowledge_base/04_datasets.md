# Public Datasets for Glaucoma Progression Research

## Primary Datasets (Best Fit for This Project)

### 1. GRAPE — Multi-modal Glaucoma Progression Dataset
**Best fit for this project**

| Attribute | Value |
|-----------|-------|
| Size | 1,115 follow-up records, 263 eyes |
| Patients | Glaucoma or glaucoma suspect |
| Modalities | VF, fundus photos, OCT, IOP, clinical data |
| Labels | **VF progression annotations** (ground truth!) |
| Access | Open access (Nature Scientific Data, 2023) |
| Citation | https://doi.org/10.1038/s41597-023-02424-4 |
| Download | https://github.com/li-research/GRAPE |

**Notes for our project**:
- This is the closest match to our target task
- Check if myopia subset is labeled
- Mean follow-up: ~4.2 years

### 2. UWHVF — University of Washington Humphrey Visual Field
| Attribute | Value |
|-----------|-------|
| Size | 28,943 visual fields from 3,871 patients |
| Modalities | VF only (HFA 24-2 with reliability indices) |
| Labels | Raw sensitivity values + MD + VFI per test |
| Access | Open source (GitHub: uw-biomedical-ml/uwhvf) |
| Strength | Large, longitudinal, reliable indices available |

**Notes**:
- VF-only (no images) — use for temporal model pretraining
- Great for learning VF progression patterns before adding structural data

### 3. Harvard Glaucoma Detection and Progression (HGDP)
| Attribute | Value |
|-----------|-------|
| Size | Multi-modal, multitask |
| Modalities | OCT, fundus, VF, clinical |
| Labels | Detection AND progression |
| Access | ICCV 2023 paper release |
| Citation | Luo et al., ICCV 2023 |

---

## Secondary Datasets

### 4. Rotterdam Eye Study (Request Required)
- Prospective cohort with >10 years follow-up
- Strong myopia phenotyping (axial length measured)
- Request via official data sharing agreement

### 5. DIGS/ADAGES (UCSD)
- Diagnostic Innovations in Glaucoma Study
- Structural + functional longitudinal data
- Available for research collaborations (contact UCSD)
- Has myopia characterization

### 6. NTGS (Normal Tension Glaucoma Study)
- Relevant because NTG is common in myopes
- Low IOP despite progressive glaucoma

---

## Useful but Not Primary

| Dataset | Size | Modality | Issue for Us |
|---------|------|----------|-------------|
| REFUGE | 1,200 fundus | Fundus only | No VF, no progression |
| ORIGA | 650 fundus | Fundus only | No VF, no progression |
| DRISHTI-GS | 101 fundus | Fundus only | Too small |
| RIGA | 750 fundus | Fundus only | No progression |
| PAPILA | 488 fundus+VF | Fundus + VF | Single timepoint |

---

## Myopia-Specific Resources

### High Myopia Registries
- **HELM Study** (Hong Kong): High myopia with OCT longitudinal
- **COMET** (US): Myopia progression in children (not adults)
- **Taiwan NTU Dataset**: Includes myopia stratification

### Axial Length Databases
- Most ophthalmology departments collect this routinely
- IOLMaster data in electronic health records
- Partner with local ophthalmology department for access

---

## Data Acquisition Strategy

### Step 1: Start with Public Data
```bash
# Download GRAPE dataset
# (check the GitHub repo for current download instructions)
# https://github.com/li-research/GRAPE

# Download UWHVF
git clone https://github.com/uw-biomedical-ml/uwhvf.git
```

### Step 2: Data Harmonization
```python
# Different datasets use different VF formats
# GRAPE: .csv with point-wise sensitivities
# UWHVF: JSON format with metadata
# Always convert to unified Polars DataFrame

import polars as pl

def load_grape(path: str) -> pl.DataFrame:
    return pl.read_csv(path).with_columns(
        pl.col("date").str.to_date(),
        pl.col("eye").cast(pl.Categorical),
    )
```

### Step 3: Identify Myopic Patients
```python
# Filter for high myopia criteria
myopic_patients = patients.filter(
    (pl.col("spherical_equivalent") <= -6.0) |
    (pl.col("axial_length") >= 26.0)
)
```

### Step 4: Minimum Data Requirements
- At minimum: **3 reliable VF tests** over ≥2 years
- Ideal: **5+ VF tests** over ≥3 years
- With baseline fundus photo or OCT scan
- IOP measurements at each visit

---

## Synthetic Data Augmentation

When real myopic glaucoma data is scarce:

### 1. Mixup for VF Series
```python
def vf_mixup(vf1, vf2, label1, label2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    vf_mix = lam * vf1 + (1-lam) * vf2
    label_mix = lam * label1 + (1-lam) * label2
    return vf_mix, label_mix
```

### 2. Temporal Jitter
```python
def temporal_jitter(vf_series, times, sigma_days=30):
    """Add noise to visit timing to simulate irregular follow-up."""
    jitter = np.random.normal(0, sigma_days/365.25, len(times))
    times_jittered = times + jitter
    return vf_series, np.sort(times_jittered)
```

### 3. VF Simulator (Advanced)
- Reference: VirtualEye simulator (Henson et al.)
- Can generate realistic VF series with known progression rates
- Useful for curriculum learning (start with simulated, fine-tune on real)

---

## Ethical and Privacy Considerations

- **De-identification**: Remove patient name, DOB, medical record numbers
- **Minimum necessary data**: Don't collect unnecessary fields
- **IRB approval**: Required for clinical data — plan ahead
- **Data use agreements**: GRAPE and UWHVF have specific licenses — read them
- **Age of consent**: Pediatric data requires additional protections

---

## References
- GRAPE dataset: Cho et al., Nature Scientific Data, 2023
- UWHVF: Habib et al., Nature Machine Intelligence, 2022
- Harvard HGDP: Luo et al., ICCV, 2023
- "Retinal Glaucoma Public Datasets: What We Have and What's Missing." PMC, 2023
