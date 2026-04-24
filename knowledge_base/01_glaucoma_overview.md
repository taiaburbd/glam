# Glaucoma Overview for Deep Learning Researchers

## What is Glaucoma?
Glaucoma is a group of progressive optic neuropathies characterized by retinal ganglion cell (RGC) death and visual field loss. It is the **leading cause of irreversible blindness** worldwide (~80M affected).

### Types Relevant to This Project
| Type | Key Feature | Relevance |
|------|-------------|-----------|
| Primary Open-Angle (POAG) | Most common; elevated IOP | Baseline comparison |
| Normal-Tension (NTG) | IOP within normal range | Common in myopes |
| **Myopic Glaucoma** | High myopia + glaucoma | **This project's target** |

---

## Myopic Glaucoma — The Challenge
High myopia (≥-6D spherical equivalent or axial length ≥26mm) fundamentally changes the optic nerve head and retina in ways that confound standard glaucoma diagnosis and monitoring.

### Why Myopia Complicates Glaucoma
1. **Structural changes**: Tilted discs, peripapillary atrophy (PPA), staphyloma deform the optic nerve head appearance
2. **RNFL thinning baseline**: Myopic eyes have thinner RNFL at baseline (mimicking glaucoma damage)
3. **IOP underestimation**: Thinner corneas (CCT) in myopes cause tonometry to underread IOP
4. **VF artifacts**: Myopic VF defects (scotomas) overlap with glaucomatous patterns
5. **Faster progression**: Myopic eyes tend to progress faster when glaucoma is present
6. **Poorer structural-functional correlation**: The standard structure-function relationship breaks down

### Clinical Consequence
- Standard glaucoma grading tools (Glaucoma Hemifield Test, pattern deviation) are unreliable in myopes
- Diagnosis is often delayed — makes early prediction of **progression rate** even more critical

---

## Key Clinical Measurements

### Intraocular Pressure (IOP)
- Normal: 10–21 mmHg; in myopes, true IOP may be higher than measured
- Target for treatment; reduction slows progression

### Optic Nerve Head (ONH) Parameters
- **Cup-Disc Ratio (CDR)**: Glaucoma → larger cup relative to disc
- **Rim Area/ISNT rule**: Inferior > Superior > Nasal > Temporal RNFL thickness (often violated in myopes)
- **Lamina Cribrosa**: Deeper insertion in myopes → more vulnerable

### RNFL (Retinal Nerve Fiber Layer)
- Measured by OCT (Optical Coherence Tomography)
- Key sectors: superior, inferior, temporal, nasal
- **Global average**: single summary statistic most used in studies

### Axial Length
- **Critical feature for this model** — correlates with myopia severity
- Typical: 22–25mm; High myopia: >26mm

---

## Why MD and VFI?

### Mean Deviation (MD)
- Global index of VF sensitivity vs age-matched normals
- Units: **dB** (decibels)
- Range: 0 dB (normal) to −30 dB (severe loss)
- **Progression rate**: MD change per year (dB/year)
- Clinically meaningful change: >−1 dB/year = fast progressor

### Visual Field Index (VFI)
- Percentage of normal VF function remaining
- Range: 100% (normal) to 0% (end-stage)
- Less affected by cataract than MD
- **Progression rate**: VFI change per year (%/year)
- Used for estimating time-to-blindness

### Why Predict Both?
- MD and VFI capture slightly different aspects of progression
- VFI is more robust to media opacities (cataract — common in elderly myopes)
- Clinical guidelines use both for treatment decisions
- Dual prediction improves model calibration

---

## Visual Field Testing

### Humphrey Field Analyzer (HFA) 24-2
- Standard for glaucoma monitoring
- Tests 54 points in central 24° visual field
- Each point measured in dB
- Strategy: SITA-Standard or SITA-Fast

### Reliability Indices
| Index | Acceptable | Problem |
|-------|-----------|---------|
| Fixation Loss | <20% | Patient not fixating |
| False Positive Rate | <15% | "Trigger-happy" patient |
| False Negative Rate | <33% | Inattentive patient |

**Filter unreliable tests before training** — they add noise to progression estimates.

---

## Key Formulas for This Project

### OLS Progression Rate (used in preprocessing)
```
rate = Σ[(t_i - t̄)(VF_i - VF̄)] / Σ[(t_i - t̄)²]
```
Where `t_i` = years from baseline, `VF_i` = MD or VFI at visit i

### Conversion: MD to VFI (approximate)
```
VFI ≈ 100 × 10^(MD/10)   [simplified; actual computation uses pointwise values]
```

---

## References
- Weinreb RN et al. "Primary Open-Angle Glaucoma." NEJM, 2014
- Ohno-Matsui K et al. "Myopic Glaucoma." Progress in Retinal and Eye Research, 2021
- Leung CKS. "Diagnosing Glaucoma in the Myopic Eye." Eye, 2019
- Garway-Heath DF. "Structure-Function Relationship in Glaucoma." Ophthalmology, 2002
