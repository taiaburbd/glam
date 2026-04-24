# Key Papers — Reading List for GLAM Project

## Essential Papers (Read First)

### 1. Systematic Review: DL for Glaucoma Progression (2024)
- **Title**: "Deep Learning in Glaucoma Detection and Progression Prediction: A Systematic Review"
- **Journal**: PMC (2024), PMCID: PMC11852503
- **Why read**: Comprehensive overview of 50+ papers; identifies best architectures
- **Key finding**: Multimodal DL detects progression 3.5 years earlier than clinicians
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11852503/

### 2. Multimodal Glaucoma Progression (2025)
- **Title**: "Multimodal Deep Learning for Glaucoma Progression Using RNFL, VF, and Clinical Data"
- **Journal**: medRxiv (2025)
- **Why read**: Most recent multimodal approach; directly relevant architecture
- **Key result**: 0.97 AUC on 10,864-patient cohort with ConvNeXt + BiLSTM fusion

### 3. Hybrid-VF-Net for VF Forecasting (2025)
- **Title**: "A Hybrid Deep Learning Approach for Visual Field Test Forecasting"
- **Journal**: Ophthalmology Science (2025)
- **URL**: https://www.ophthalmologyscience.org/article/S2666-9145(25)00101-0/fulltext
- **Why read**: Purpose-built for VF progression — conv+RNN+transformer hybrid
- **Implementation insight**: Depthwise transformers for spatial VF patterns

### 4. Equity-Enhanced Glaucoma Progression (2025)
- **Title**: "Equity-Enhanced Glaucoma Progression Prediction with Knowledge Distillation"
- **Journal**: npj Digital Medicine (Nature, 2025)
- **URL**: https://www.nature.com/articles/s41746-025-01884-9
- **Why read**: Fairness across demographics; FairDist model architecture
- **Key finding**: Knowledge distillation maintains AUC while improving equity

### 5. Harvard Glaucoma Dataset (ICCV 2023)
- **Title**: "Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset"
- **Authors**: Luo et al.
- **Conference**: ICCV 2023
- **Why read**: Introduces large multimodal dataset + baseline models
- **URL**: https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Harvard_Glaucoma_Detection_and_Progression_A_Multimodal_Multitask_Dataset_and_ICCV_2023_paper.pdf

---

## Architecture Papers

### 6. GRAPE Dataset Paper
- **Title**: "GRAPE: A Multi-modal Glaucoma Progression Dataset"
- **Journal**: Nature Scientific Data (2023)
- **URL**: https://www.nature.com/articles/s41597-023-02424-4
- **Why read**: Understand the primary dataset we'll use

### 7. UWHVF Dataset + Baseline Models
- **Title**: "A machine learning approach for predicting visual field progression"
- **Journal**: Nature Machine Intelligence (2022)
- **Why read**: Establishes baseline VF-only performance; describes dataset

### 8. Predicting VF Progression with Structural + Functional (2024)
- **Title**: "Prediction of VF Progression with Baseline and Longitudinal Structural Measurements"
- **Journal**: American Journal of Ophthalmology (2024)
- **URL**: https://www.ajo.com/article/S0002-9394(24)00056-4/fulltext
- **Key result**: Time-to-progression prediction improvement with OCT integration

### 9. Generative Guidance for Progression Prediction
- **Title**: "Predicting Glaucoma Progression Using Deep Learning Framework with Generative Guidance"
- **Journal**: Scientific Reports (Nature, 2023)
- **URL**: https://www.nature.com/articles/s41598-023-46253-2
- **Why read**: Novel generative augmentation for small glaucoma datasets

---

## Myopia-Specific Papers

### 10. Myopic Glaucoma Review
- **Title**: "Myopic Glaucoma: Prevalence, Diagnosis, and Clinical Management"
- **Journal**: Progress in Retinal and Eye Research (2021)
- **Authors**: Ohno-Matsui K et al.
- **Why read**: Clinical foundation for understanding our patient population

### 11. Diagnosing Glaucoma in Myopic Eyes
- **Title**: "Challenges of Diagnosing Glaucoma in Highly Myopic Eyes"
- **Journal**: Eye (2019)
- **Authors**: Leung CKS
- **Why read**: Why standard tools fail; what features are reliable in myopes

### 12. RNFL in Myopic Glaucoma
- **Title**: "OCT-Measured Retinal Nerve Fiber Layer Thickness in Myopic Glaucoma"
- **Why read**: Understand why structural features need different interpretation in myopes

---

## Technical / Methods Papers

### 13. ConvNeXt V2
- **Title**: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
- **Authors**: Woo et al., CVPR 2023
- **URL**: https://arxiv.org/abs/2301.00808
- **Why read**: The backbone architecture we use

### 14. Long Short-Term Memory (LSTM)
- **Title**: "Long Short-Term Memory" — Hochreiter & Schmidhuber, 1997
- **Why read**: Foundation; understand bidirectional extension

### 15. Attention Is All You Need
- **Title**: "Attention Is All You Need" — Vaswani et al., NeurIPS 2017
- **Why read**: Transformer architecture used in temporal_model.py

### 16. Huber Loss for Regression
- **Title**: "Robust Estimation of a Location Parameter" — Huber, 1964
- **Why read**: Why we use Huber (L1+L2 hybrid) for VF progression regression

---

## Evaluation & Clinical Benchmarking

### 17. Measuring Rates of VF Change
- **Title**: "Practical Recommendations for Measuring Rates of Visual Field Change"
- **Authors**: Chauhan BC et al.
- **Journal**: British Journal of Ophthalmology (2008)
- **Why read**: Gold standard method for computing MD rates (our training labels)

### 18. Reproducibility in Medical DL
- **Title**: "Reproducibility Checklist for Deep Learning in Medical Imaging"
- **Journal**: PMC (2024), PMCID: PMC11300409
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11300409/
- **Why read**: 26-item checklist to make research reproducible and publishable

---

## Recommended Reading Order

```
Week 1: Clinical Foundation
  [1] Systematic Review (DL for glaucoma)
  [10] Myopic Glaucoma Review
  [17] Measuring Rates of VF Change

Week 2: Datasets & Baselines
  [6] GRAPE Dataset
  [7] UWHVF Dataset
  [5] Harvard HGDP

Week 3: Architecture
  [2] Multimodal DL (most relevant)
  [3] Hybrid-VF-Net
  [8] Structural + Functional Prediction

Week 4: Advanced Topics
  [4] Equity (FairDist)
  [9] Generative Augmentation
  [13] ConvNeXt V2

Week 5: Reproducibility & Publishing
  [18] Reproducibility Checklist
  [11] Diagnosing Glaucoma in Myopes
```

---

## Tools for Literature Management
- **Zotero**: Free citation manager (recommended)
- **Semantic Scholar**: AI-powered paper discovery (search "glaucoma progression deep learning")
- **PubMed**: For clinical ophthalmology papers
- **arXiv**: For preprints on new DL methods
- **Papers with Code**: For benchmark comparisons

## Search Queries
```
PubMed: ("glaucoma"[MeSH] AND "visual field"[MeSH] AND "deep learning") AND "progression"
Semantic Scholar: "myopic glaucoma progression prediction deep learning"
arXiv: cat:cs.CV "glaucoma" "visual field" "progression"
```
