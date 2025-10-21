#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial Continuum Fitting + Silicate Subtraction Pipeline

This version uses New_Silicates_YL.txt (YL dataset: wave, tau_oliv, tau_pyrox, tau_enst) and aligns behaviors with the provided
reference script:

- Continuum: degree=6 polynomial (Ridge) with two passes of MIR window scaling
  (first pass: [1.0, 1.18, 1.62, 1.42, 1.28, 1.18, 1.15], second pass:
   [1.0, 1.07, 1.91, 1.5, 1.4, 1.22, 1.15]).
- GCS3 overlay in panel (b) and (c), with 0.68 τ scaling.
- Optool-modeled silicate (olivine + pyroxene) combination uses λ<13.5 and λ≥13.5 piecewise
  with tail multipliers fixed to 1.0 (matching the reference code's use of "*1.").
- Ice: Pure_H2O_15K scaled by 21 and baseline-corrected; Pure_H2O_160K scaled by 2
  and smoothed (box=10). Combined ice = 15K(corrected) + 160K(smoothed).
- Produces a 4-panel PDF.

Author: Jaeyeong Kim
License: MIT
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import warnings
import logging

import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# external: pip install spectres
from spectres import spectres

# ----------------------------- Config & Types -----------------------------

def configure_warnings(suppress: bool = False):
"""Optionally silence common library warnings for cleaner CLI output."""
if not suppress:
return
# Python warnings
warnings.filterwarnings("ignore")
for cat in (FutureWarning, UserWarning, RuntimeWarning):
warnings.filterwarnings("ignore", category=cat)
# Matplotlib chatter
logging.getLogger("matplotlib").setLevel(logging.ERROR)
# Numpy runtime warnings (invalid/overflow/divide)
np.seterr(all="ignore")

np.random.seed(1)  # reproducibility

MIR_WINDOWS = [
    (5.2, 5.5),
    (7.8, 8.0),
    (18.0, 19.5),
    (20.0, 21.5),
    (22.0, 23.5),
    (24.0, 25.8),
    (26.1, 30.0),
]

@dataclass
class Spectrum:
    wave: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

# ----------------------------- IO Utilities ------------------------------

def load_three_column_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 3-column ASCII (wave, flux, flux_err)."""
    arrs = np.loadtxt(str(path), unpack=True)
    if len(arrs) != 3:
        raise ValueError(f"Expected 3 columns in {path}, got {len(arrs)}")
    return arrs[0], arrs[1], arrs[2]

def load_two_column_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2-column ASCII (x, y)."""
    arrs = np.loadtxt(str(path), unpack=True)
    if len(arrs) != 2:
        raise ValueError(f"Expected 2 columns in {path}, got {len(arrs)}")
    return arrs[0], arrs[1]

# -------------------------- Math / Modeling Utils ------------------------

def pick_windows(wave: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                 windows: List[Tuple[float, float]],
                 scales: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate fitting points from windows with optional scaling per window."""
    masks = [(wave >= lo) & (wave <= hi) for (lo, hi) in windows]
    xp = np.concatenate([wave[m] for m in masks])
    yp = np.concatenate([flux[m] * scales[i] for i, m in enumerate(masks)])
    yp_err = np.concatenate([flux_err[m] for m in masks])
    return xp, yp, yp_err

def poly_continuum_with_bootstrap(
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    degree: int,
    guess_mode: bool,
    windows: List[Tuple[float, float]],
    guess_scale_factors: List[float] | None = None,
    n_bootstrap: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fit a polynomial continuum on selected windows using Ridge regression and
    estimate 1σ predictive uncertainty via bootstrap.
    """
    scales = [1.0] * len(windows) if guess_mode else (guess_scale_factors or [1.0] * len(windows))
    xp, yp, yp_err = pick_windows(wave, flux, flux_err, windows, scales)

    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    x = xp[:, None]
    y = yp
    model.fit(x, y)

    X_plot = wave[:, None]
    y_fit = model.predict(X_plot).ravel()

    boot_preds = np.empty((n_bootstrap, len(wave)), dtype=float)
    for i in range(n_bootstrap):
        noise = np.random.normal(loc=0.0, scale=yp_err, size=yp.shape)
        y_boot = y + noise
        m_boot = make_pipeline(PolynomialFeatures(degree), Ridge())
        m_boot.fit(x, y_boot)
        boot_preds[i] = m_boot.predict(X_plot).ravel()
    y_std = boot_preds.std(axis=0)

    return wave, y_fit, y_std, (xp, yp, yp_err)

def to_optical_depth(flux_over_cont: np.ndarray) -> np.ndarray:
    """τ = -ln(F/Fc)."""
    return -np.log(np.clip(flux_over_cont, 1e-12, np.inf))

# --------------------------- Domain-Specific Bits -------------------------

def gcs3_silicate_template(
    wave_gcs: np.ndarray, tau_gcs: np.ndarray, scale_tau: float, wave_target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate GCS3 slicate template to target wavelengths with given scaling."""
    tau_scaled = tau_gcs * scale_tau
    f_tau = interp1d(np.sort(wave_gcs), tau_scaled[np.argsort(wave_gcs)], kind="linear", bounds_error=False, fill_value="extrapolate")
    tau_target = f_tau(wave_target)
    flux_factor = np.exp(-tau_target)
    return tau_target, flux_factor

# --------------------------- Y-only Silicates -----------------------------

def compose_lab_silicates_Y(
    wave_y: np.ndarray, tau_oliv_y: np.ndarray, tau_pyrox_y: np.ndarray,
    coe_oliv: float = 0.25, coe_pyrox: float = 1.15,
    coe_oliv_tail: float = 1.0, coe_pyrox_tail: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build olivine/pyroxene slicate absorption profiles using the YL dataset.

    Piecewise scaling:
      - Base scaling over all wavelengths: tau *= coe_oliv / coe_pyrox
      - Additional tail scaling for λ ≥ 13.5 µm: multiply by coe_*_tail (here fixed to 1.0).
    """
    s = np.argsort(wave_y)
    wy = wave_y[s]
    oliv = tau_oliv_y[s] * coe_oliv
    pyrox = tau_pyrox_y[s] * coe_pyrox

    # tail region scaling (here kept at 1.0 to match reference script)
    tail_mask = wy >= 13.5
    oliv_tail = oliv.copy(); oliv_tail[tail_mask] *= coe_oliv_tail
    pyrox_tail = pyrox.copy(); pyrox_tail[tail_mask] *= coe_pyrox_tail

    return wy, oliv_tail, pyrox_tail

def resample_tau_to_target(wave_lab: np.ndarray, tau_lab: np.ndarray, wave_target: np.ndarray) -> np.ndarray:
    f_tau = interp1d(wave_lab, tau_lab, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f_tau(wave_target)

# ------------------------------- Plotting --------------------------------

def setup_mpl():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "lines.linewidth": 3,
    })
    plt.rc("axes", titlesize=20, labelsize=26)
    plt.rc("xtick", labelsize=24)
    plt.rc("ytick", labelsize=24)

# ------------------------------- Pipeline --------------------------------

def run_pipeline(
    target_name: str,
    spec_path: Path,
    gcs3_path: Path,
    lab_y_path: Path,
    ice_dir: Path,
    out_pdf: Path,
):
    setup_mpl()

    # 1) Load target spectrum
    wave, flux, flux_err = load_three_column_txt(spec_path)
    spec = Spectrum(wave=wave, flux=flux, flux_err=flux_err)

    # Figure & Panel (a)
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 20), sharex=False)
    ax = axs[0]
    ax.plot(spec.wave, spec.flux, "k-", lw=1, label=f"{target_name} IRS Spectrum")
    ax.set_xlim(5.0, 30.0)
    ax.set_ylim(spec.flux.min(), spec.flux.max())
    ax.text(28.0, 10.0, "(a)", fontsize=26)
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelbottom=False)
    ax.minorticks_on()
    ax.legend(loc="upper left", fontsize=16)

    # Panel (b): initial continuum (guess mode Y) + first refined + GCS3 overlay
    x_all0, y_fit0g, y_std0g, (xp0, yp0, yp0e) = poly_continuum_with_bootstrap(
        wave=spec.wave, flux=spec.flux, flux_err=spec.flux_err,
        degree=6, guess_mode=True, windows=MIR_WINDOWS, n_bootstrap=100
    )

    # first refined scale (matches provided script)
    scale_refine_a = [1.0, 1.18, 1.62, 1.42, 1.28, 1.18, 1.15]
    x_all_a, y_fit_a, y_std_a, _ = poly_continuum_with_bootstrap(
        wave=spec.wave, flux=spec.flux, flux_err=spec.flux_err,
        degree=6, guess_mode=False, windows=MIR_WINDOWS, guess_scale_factors=scale_refine_a, n_bootstrap=100
    )

    ax = axs[1]
    ax.plot(spec.wave, spec.flux, "k-", lw=1, label=f"{target_name} IRS Spectrum")
    ax.set_xlim(5.0, 30.0)
    ax.set_ylim(spec.flux.min(), spec.flux.max())
    ax.errorbar(xp0, yp0, yp0e, fmt="ro", ecolor="r", elinewidth=0.5, markersize=2.5, label="Fitting points")
    ax.plot(x_all_a, y_fit_a, "g--", label="Polynomial deg. 6 (refine A)")
    ax.fill_between(x_all_a, y_fit_a - y_std_a, y_fit_a + y_std_a, alpha=0.2, label="Uncertainty")

    # GCS3 overlay (panel b)
    m_5_30_a = (x_all_a > 5.0) & (x_all_a < 30.0)
    x_train_a = x_all_a[m_5_30_a]
    y_train_a = y_fit_a[m_5_30_a]
    wave_gcs, tau_gcs = load_two_column_txt(gcs3_path)
    tau_gcs_a, flux_gcs_a = gcs3_silicate_template(wave_gcs, tau_gcs, scale_tau=0.68, wave_target=x_train_a)
    ax.plot(x_train_a, y_train_a * flux_gcs_a, ":", color="brown", lw=2, label="GCS 3 Spectrum")
    ax.text(28.0, 10.0, "(b)", fontsize=26)
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelbottom=False)
    ax.minorticks_on()
    ax.legend(loc="upper left", fontsize=16)

    # Panel (c): second refined scale + Optool-modeled silicates
    scale_refine_b = [1.0, 1.07, 1.91, 1.5, 1.4, 1.22, 1.15]
    x_all_b, y_fit_b, y_std_b, _ = poly_continuum_with_bootstrap(
        wave=spec.wave, flux=spec.flux, flux_err=spec.flux_err,
        degree=6, guess_mode=False, windows=MIR_WINDOWS, guess_scale_factors=scale_refine_b, n_bootstrap=100
    )
    m_5_30_b = (x_all_b > 5.0) & (x_all_b < 30.0)
    x_train_b = x_all_b[m_5_30_b]
    y_train_b = y_fit_b[m_5_30_b]

    ax = axs[2]
    ax.plot(spec.wave, spec.flux, "k-", lw=1, label=f"{target_name} IRS Spectrum")
    ax.set_xlim(5.0, 30.0)
    ax.set_ylim(spec.flux.min(), spec.flux.max())
    ax.plot(x_all_b, y_fit_b, "g--", label="Polynomial deg. 6 (refine B)")
    ax.fill_between(x_all_b, y_fit_b - y_std_b, y_fit_b + y_std_b, alpha=0.2, label="Uncertainty")

    # GCS3 overlay (panel c)
    tau_gcs_b, flux_gcs_b = gcs3_silicate_template(wave_gcs, tau_gcs, scale_tau=0.68, wave_target=x_train_b)
    ax.plot(x_train_b, y_train_b * flux_gcs_b, ":", color="brown", lw=2, label="Scaled GCS 3 Spectrum")

    # Y-only silicates
    wY, tau_oliv_Y, tau_pyrox_Y, tau_enst_Y = np.loadtxt(str(lab_y_path), unpack=True, skiprows=1)
    wave_lab, tau_oliv_corr, tau_pyrox_corr = compose_lab_silicates_Y(
        wY, tau_oliv_Y, tau_pyrox_Y,
        coe_oliv=0.25, coe_pyrox=1.15, coe_oliv_tail=1.0, coe_pyrox_tail=1.0
    )

    # Interpolate τ and convert to multiplicative factors
    tau_oliv_t = resample_tau_to_target(wave_lab, tau_oliv_corr, x_train_b)
    tau_pyrox_t = resample_tau_to_target(wave_lab, tau_pyrox_corr, x_train_b)
    tau_comb_t = tau_oliv_t + tau_pyrox_t

    f_oliv = np.exp(-tau_oliv_t)
    f_pyrox = np.exp(-tau_pyrox_t)
    f_comb = np.exp(-tau_comb_t)

    ax.plot(x_train_b, y_train_b * f_oliv, "-", color="violet", lw=2, label="Lab. Silicate Olivine")
    ax.plot(x_train_b, y_train_b * f_pyrox, "-", color="magenta", lw=2, label="Lab. Silicate Pyroxene")
    ax.plot(x_train_b, y_train_b * f_comb, "-", color="red", lw=2, label="Combined Lab. Silicate")
    ax.text(28.0, 10.0, "(c)", fontsize=26)
    ax.legend(loc="upper left", fontsize=14, ncol=2)
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    # Panel (d): τ spectrum (silicate-corrected) + water ice components
    ax = axs[3]

    # optical depth of target after continuum (use refine B)
    sel = (spec.wave > 5.0) & (spec.wave < 30.0)
    flux_c = spec.flux[sel] / y_train_b
    tau = to_optical_depth(flux_c)

    # subtract combined silicate τ
    tau_sil_corr = tau - tau_comb_t

    # 5.1 Load ice lab data from directory
    ice_files = sorted([p for p in ice_dir.iterdir() if p.is_file()], key=lambda p: len(p.name))

    def load_ice_tau(path_txt: Path) -> Tuple[np.ndarray, np.ndarray]:
        # file columns: wavenumber(cm^-1), absorbance
        freq, absorb = load_two_column_txt(path_txt)
        wave = 10000.0 / freq  # um
        tau_lab = -np.log(np.exp(-absorb))  # = absorb
        s = np.argsort(wave)
        return np.sort(wave), tau_lab[s]

    # Pure H2O 15K (amorphous) — scale 21 and smooth(box=10), then baseline correction
    pure15 = [p for p in ice_files if p.name.startswith("Pure_H2O_15K")]
    if not pure15:
        raise FileNotFoundError("No Pure_H2O_15K* file found in ice_dir")
    w15, t15 = load_ice_tau(pure15[0])
    m_band = (w15 >= x_train_b.min() - 0.025) & (w15 <= x_train_b.max() + 0.025)
    H2O15_on_target = spectres(x_train_b, w15[m_band], t15[m_band])
    H2O15_on_target = np.nan_to_num(H2O15_on_target * 21.0)

    # baseline correction (2nd order) on pure H2O ice spectrum at 15K
    def baseline_poly(w: np.ndarray, t: np.ndarray, deg: int = 2) -> np.ndarray:
        ind1 = (w >= 5.0) & (w <= 5.2)
        ind2 = (w >= 8.5) & (w <= 9.0)
        ind3 = (w >= 19.0) & (w <= 25.0)
        xp = np.concatenate([w[ind1], w[ind2], w[ind3]])
        yp = np.concatenate([t[ind1], t[ind2], t[ind3]])
        model = make_pipeline(PolynomialFeatures(deg), Ridge())
        model.fit(xp[:, None], yp)
        trend = model.predict(w[:, None]).ravel()
        return t - trend

    H2O15_corr = baseline_poly(x_train_b, H2O15_on_target, deg=2)

    # Crystalline H2O 160K — scale 2 and smooth(box=10)
    crystalline160 = [p for p in ice_files if p.name.startswith("Pure_H2O_160K")]
    if not crystalline160:
        raise FileNotFoundError("No Pure_H2O_160K* file found in ice_dir")
    w160, t160 = load_ice_tau(crystalline160[0])
    m_band2 = (w160 >= x_train_b.min() - 0.025) & (w160 <= x_train_b.max() + 0.025)
    H2O160_on_target = spectres(x_train_b, w160[m_band2], t160[m_band2])
    H2O160_on_target = np.nan_to_num(H2O160_on_target * 2.0)

    # Smooth both ice components with box=10
    def smooth_box10(y: np.ndarray) -> np.ndarray:
        box = np.ones(10) / 10.0
        return np.convolve(y, box, mode="same")

    H2O15_sm = smooth_box10(H2O15_corr)
    H2O160_sm = smooth_box10(H2O160_on_target)

    # Plot τ_sil_corr and ice components
    ax.plot(x_train_b, tau_sil_corr, "k-", lw=1, label=f"{target_name} Silicate-corrected spectrum")
    ax.plot(x_train_b, np.zeros_like(x_train_b), "k--", lw=1.0)
    ax.plot(x_train_b, H2O15_sm, color="blue", label="Pure H2O (15K) - continuum subtracted")
    ax.plot(x_train_b, H2O160_sm, color="green", label="Crystalline H2O (160K)")
    ax.plot(x_train_b, H2O15_sm + H2O160_sm, color="red", lw=2, label="Combined water ice component")

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"Optical depth")
    ax.set_xlim(5.0, 22.0)
    ax.set_ylim((H2O15_sm + H2O160_sm).max(), -0.2)
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.text(20.6, 0.8, "(d)", fontsize=26)
    ax.legend(loc="lower right", fontsize=16)

    fig.tight_layout(h_pad=0.1)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_pdf), format="pdf", bbox_inches="tight")
    print(f"[OK] Saved: {out_pdf}")

# --------------------------------- CLI -----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polynomial continuum fitting (MIR) with silicate (YL dataset) & ice components (updated).")
    p.add_argument("--target-name", default="Per-emb 25", help="Target display name")
    p.add_argument("--spectrum", required=True, type=Path, help="Path to 3-col spectrum txt (wave, flux, flux_err)")
    p.add_argument("--gcs3", required=True, type=Path, help="Path to GCS3 silicate tau txt (wave, tau)")
    p.add_argument("--lab-y", required=True, type=Path, help="Path to YL lab silicate file (4 cols, skiprows=1)")
    p.add_argument("--ice-dir", required=True, type=Path, help="Directory containing ice lab data txt files")
    p.add_argument("--out", required=True, type=Path, help="Output PDF path")
    p.add_argument("--suppress-warnings", action="store_true", help="Silence most library warnings and numpy runtime warnings")
    return p.parse_args()


def main():
    args = parse_args()
    configure_warnings(args.suppress_warnings)
    run_pipeline(
        target_name=args.target_name,
        spec_path=args.spectrum,
        gcs3_path=args.gcs3,
        lab_y_path=args.lab_y,
        ice_dir=args.ice_dir,
        out_pdf=args.out,
    )


if __name__ == "__main__":
    main()
