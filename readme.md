# Polynomial Continuum Fitting (MIR) with Y-only Silicate + Ice Correction

This repository provides a reproducible pipeline to fit a polynomial MIR continuum, apply Y-only (New_Silicates_YL.txt) silicate corrections (olivine + pyroxene), and combine water-ice components (amorphous 15K, crystalline 160K). It produces a 4‑panel PDF summarizing each step.

> **What’s Y‑only?**  
> The pipeline uses only `New_Silicates_YL.txt` (columns: wavelength, τ_olivine, τ_pyroxene, τ_enstatite) and **does not** depend on the W dataset.

---

## Features
- **Continuum fit**: Ridge‑regularized polynomial over MIR windows with bootstrap ±1σ uncertainty.
- **Silicate correction (Y‑only)**: Piecewise scaling (<13.5 μm / ≥13.5 μm) for olivine + pyroxene; combined optical‑depth model applied to the target.
- **GCS 3 overlay**: Optional τ×0.68 reference plotted alongside the fitted continuum.
- **Water ice**: Amorphous 15K (local continuum subtraction) and crystalline 160K components combined on the τ spectrum.
- **Clean CLI**: Paths are provided as arguments; no OS‑specific hardcoding.

---

## Installation
```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Expected Data Formats
- **Target spectrum (`--spectrum`)**: 3 columns `wave  flux  flux_err`  
  Units typically: `wave[μm]`, `flux[mJy]`, `flux_err[mJy]`.
- **GCS 3 silicate (`--gcs3`)**: 2 columns `wave  tau`  
  The model uses **τ×0.68** when overlaying on the continuum.
- **Y‑only silicate (`--lab-y`)**: 4 columns (skip first header line):  
  `wave  tau_oliv  tau_pyrox  tau_enst`
- **Ice lab directory (`--ice-dir`)**: Text files with wavenumber(cm⁻¹) and absorbance.  
  Required files (first match is used):
  - `Pure_H2O_15K*`
  - `Pure_H2O_160K*`

> The script automatically converts wavenumber→wavelength and resamples to the target grid.

---

## Usage
```bash
python continuum_fit_Y_only.py \
  --target-name "Per-emb 25" \
  --spectrum data/Per-emb25_IRS_spec.txt \
  --gcs3 data/Silicate_GCS3_tau.txt \
  --lab-y data/New_Silicates_YL.txt \
  --ice-dir data/Ice_labdata \
  --out outputs/Paper_JKAS_Fig2_Per-emb_25_Cont_Fit_Process_Github.pdf
```

### Output
- **PDF**: 4 panels saved to `--out`  
  (a) Raw spectrum  
  (b) Initial continuum (deg=7) with ±1σ  
  (c) Refined continuum (deg=6) + GCS3 overlay + Y‑only silicate models  
  (d) Silicate‑corrected τ + H₂O(15K) + H₂O(160K) + Combined

---

## Key Settings
- **MIR windows**: `(5.2–5.5), (7.8–8.0), (18.0–19.5), (20.0–21.5), (22.0–23.5), (24.0–25.8), (26.1–30.0)`
- **Continuum degrees**: (b) `deg=7` (guess mode, scales=1.0), (c) `deg=6` with scale = `[1.0, 1.18, 1.62, 1.42, 1.28, 1.18, 1.15]`
- **Silicate (Y‑only) coefficients**: `coe_oliv=0.25`, `coe_pyrox=1.15`, tail factors `(≥13.5μm)` = `0.8, 0.7`
- **GCS3 overlay**: τ×`0.68`
- **Ice scaling**: `15K × 18`, `160K × 2`; 15K uses local 2nd‑order continuum subtraction

> Random seed is fixed to 1 for bootstrap reproducibility.

---

## Folder Layout (suggested)
```
.
├─ continuum_fit_Y_only.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ Per-emb25_IRS_spec.txt
│  ├─ Silicate_GCS3_tau.txt
│  ├─ New_Silicates_YL.txt
│  └─ Ice_labdata/
│     ├─ Pure_H2O_15K...
│     └─ Pure_H2O_160K...
└─ outputs/          # created by the script
```

---

## Citing / Attribution
- If you use this code, please cite this repository and the relevant laboratory datasets you include (e.g., Y‑only silicates and GCS 3 template sources).

---

## License
MIT — see `LICENSE` for details. Include copyright notice and the license
text in any source or binary redistribution. No warranty is provided.

