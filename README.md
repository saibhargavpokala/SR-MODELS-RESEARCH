# Face Super-Resolution Identity Preservation Audit Research Foused

**PhD Preparatory Research — University of Birmingham** --

> Does SR enhancement of CCTV-quality face images improve machine-based identification accuracy equitably across race and gender?

---

## Overview

This project audits whether face super-resolution (SR) models preserve identity equitably across demographic groups when enhancing low-resolution CCTV-quality face images. Using a stratified sample of **2,996 images** from the FairFace dataset across **7 racial groups × 2 genders**, we evaluate three SR models at three degradation levels and measure identity preservation via ArcFace cosine similarity.

### Key Finding

All three SR models **reduce** identity similarity on average (mean Δ = −0.0602), meaning they hallucinate facial features that ArcFace reads as belonging to a different person — despite improving perceptual quality (PSNR/SSIM).

---

## Research Questions:

| # | Question | Finding |
|---|---------|---------|
| RQ1 | Does pixel-level improvement (PSNR/SSIM) translate to identity preservation? | **No** — PSNR/SSIM improve, but identity match worsens |
| RQ2 | Do SR models improve machine-based identification accuracy? | **No** — All models worsen identity similarity (Wilcoxon p < 0.001 at 64px) |
| RQ3 | Are outcomes equitable across demographics? | **Mostly** — gaps < 0.05, but significant disparities at 64px |
| RQ4 | When does SR reduce identification accuracy? | **Always** — negative improvement at every degradation level |

---

## Pipeline Architecture

```
Original Image (448×448)
    │
    ▼
Degradation Engine
    ├── Resize → 64×64 / 32×32 / 16×16
    ├── JPEG compression (Q=20)
    ├── Gaussian noise (σ=5.0)
    └── Bicubic upscale → 448×448
    │
    ▼
SR Enhancement
    ├── GFPGAN v1.3      → 896×896
    ├── CodeFormer (w=0.9) → 896×896
    └── RealESRGAN x4     → 1792×1792
    │
    ▼
ArcFace Embeddings (buffalo_l)
    │
    ▼
Cosine Similarity Analysis
    ├── sim(original, degraded)
    ├── sim(original, enhanced)
    └── Δ = sim_enhanced - sim_degraded
```

---

## Results Summary

### Descriptive Statistics

| Model | Deg (px) | N | Mean Δ | Median Δ | PSNR | SSIM |
|-------|----------|---|--------|----------|------|------|
| CodeFormer | 64 | 2,067 | −0.1111 | −0.1158 | 23.03 | 0.637 |
| CodeFormer | 32 | 614 | −0.0469 | −0.0463 | 20.05 | 0.569 |
| GFPGAN | 64 | 2,176 | −0.0622 | −0.0681 | 23.15 | 0.657 |
| GFPGAN | 32 | 912 | −0.0102 | −0.0017 | 20.34 | 0.596 |
| RealESRGAN | 64 | 1,620 | −0.0543 | −0.0452 | 23.24 | 0.673 |
| RealESRGAN | 32 | 692 | −0.0013 | −0.0017 | 20.46 | 0.614 |

### Demographic Equity Gaps

| Model | Best Group | Worst Group | Gap | Assessment |
|-------|-----------|-------------|-----|------------|
| CodeFormer | Black | Indian | 0.021 | ✅ OK |
| GFPGAN | East Asian | Indian | 0.033 | ✅ OK |
| RealESRGAN | White | Black | 0.014 | ✅ OK |

### Kruskal-Wallis Tests (Racial Equity)

Statistically significant disparities (p < 0.05) found in:
- **GFPGAN at 64px** — H = 28.45, p = 0.000077, η² = 0.010
- **RealESRGAN at 64px** — H = 12.66, p = 0.049, η² = 0.004
- **RealESRGAN at 16px** — H = 10.13, p = 0.038 (small N = 22, interpret cautiously)

### Gender Analysis

Males consistently show larger identity degradation than females across all racial groups. Indian males experience the worst outcome (mean Δ = −0.0757).

---

## Technical Stack

| Component | Tool / Library |
|-----------|---------------|
| **Compute** | Google Colab — Tesla T4 GPU (15.6 GB VRAM) |
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch 2.10, CUDA 13.0 |
| **SR Models** | GFPGAN, CodeFormer (basicsr), RealESRGAN |
| **Face Recognition** | InsightFace (ArcFace buffalo_l, ONNX Runtime GPU) |
| **Image Processing** | OpenCV, Pillow, scikit-image |
| **Statistics** | SciPy, scikit-posthocs |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | FairFace Padding 1.25 (CC BY 4.0) |

---

## Repository Structure

```
.
├── MINI_RESEARCH.ipynb        # Full experiment notebook (Colab-ready)
├── FaceSR_Identity_Audit_Report.pdf  # Formal research report
├── README.md                  # This file
├── results/
│   ├── embeddings_master.csv  # All 8,180 paired observations
│   ├── statistics.json        # Pre-computed test results
│   ├── config.json            # Experiment configuration
│   └── figures/
│       ├── demographic_heatmap.png
│       ├── improvement_boxplot.png
│       ├── degradation_effect.png
│       ├── gender_analysis.png
│       ├── psnr_vs_identity.png
│       └── visual_comparison.png
└── data/
    ├── sampled/               # 2,996 stratified face images
    ├── degraded/              # Degraded versions (64/32/16 px)
    └── enhanced/              # SR-enhanced outputs
```

---

## Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/face-sr-identity-audit/blob/main/MINI_RESEARCH.ipynb)

### 2. Run Steps in Order

1. **Step 1** — Install dependencies (then restart runtime)
2. **Steps 2–8** — Environment setup, sampling, model loading
3. **Step 9** — Main execution loop (~5 hours on T4)
4. **Steps 10–14** — Statistical analysis and visualizations

### 3. Requirements

- Google Colab with GPU runtime (T4 or better)
- ~15 GB disk space for dataset + models
- Dependencies auto-installed in Step 1

---

## Experiment Configuration

```json
{
  "SAMPLE_SIZE": 3000,
  "SAMPLES_PER_CELL": 214,
  "RACES": ["White", "Black", "Indian", "East Asian", "Southeast Asian", "Middle Eastern", "Latino_Hispanic"],
  "GENDERS": ["Male", "Female"],
  "DEGRADATION_LEVELS": [64, 32, 16],
  "JPEG_QUALITY": 20,
  "NOISE_SIGMA": 5.0,
  "SR_MODELS": ["GFPGAN", "CodeFormer", "RealESRGAN"],
  "CODEFORMER_FIDELITY": 0.9,
  "REALESRGAN_SCALE": 4
}
```

---

## Limitations

- Cosine similarity is a computational proxy, not ground-truth identification
- Single face recognition model (ArcFace) which may carry its own biases
- Frontal faces only — real CCTV includes oblique angles and motion blur
- 214 per cell: indicative, not definitive statistical power
- Three SR models tested; newer diffusion-based approaches may differ

```

---

## License

- **Code**: MIT License
- **Dataset**: FairFace (CC BY 4.0) — [Karkkainen & Joo, 2021](https://github.com/joojs/fairface)
- **Report**: CC BY-NC 4.0

---

## Author

**Saibharghav Pokala**
MSc DataScience & AI, University of Liverpool


---

*Generated: 5 March 2026*
