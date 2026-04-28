"""
═══════════════════════════════════════════════════════════════════════
MicroScope QC v3.0 — Microscopy Image Quality Analyzer (Cellpose AI)
═══════════════════════════════════════════════════════════════════════

Same audit-grade quality framework as v2, with one key upgrade:
    Cell detection now uses Cellpose (deep learning) instead of
    Hough Circle Transform. Significantly more accurate on real
    blood smears, histology sections, and other microscopy.

References:
    - Stringer et al. 2021, Nature Methods — "Cellpose: a generalist
      algorithm for cellular segmentation"
    - Pertuz et al. 2013, "Analysis of focus measure operators"
    - Immerkær 1996, "Fast noise variance estimation"
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ─── Cellpose model loading (one-time at module import) ──────────
# This MUST happen at module load, not per-request, otherwise every
# analysis would take 10+ seconds just to load the model weights.
_CELLPOSE_MODEL = None
_CELLPOSE_AVAILABLE = False

def _load_cellpose():
    global _CELLPOSE_MODEL, _CELLPOSE_AVAILABLE
    if _CELLPOSE_MODEL is not None:
        return
    try:
        from cellpose import models
        print("[MicroScope QC] Loading Cellpose cyto3 model...", flush=True)
        _CELLPOSE_MODEL = models.CellposeModel(gpu=False, pretrained_model='cyto3')
        _CELLPOSE_AVAILABLE = True
        print("[MicroScope QC] Cellpose ready.", flush=True)
    except Exception as e:
        print(f"[MicroScope QC] Cellpose unavailable, falling back to Hough: {e}", flush=True)
        _CELLPOSE_AVAILABLE = False

# Try loading at import time — failures are non-fatal (Hough fallback)
_load_cellpose()


VERSION = "3.0.0-cellpose"

# ═══════════ THRESHOLDS ═══════════
BLUR_LV_FAIL  =  50.0
BLUR_LV_WARN  = 150.0
BLUR_LV_PASS  = 400.0
TEN_FAIL  =   500.0
TEN_WARN  =  3000.0
TEN_PASS  = 12000.0
EDGE_DENSITY_FAIL  = 0.015
EDGE_DENSITY_WARN  = 0.040

EXPOSURE_DARK_FAIL    =  35
EXPOSURE_DARK_WARN    =  70
EXPOSURE_BRIGHT_WARN  = 195
EXPOSURE_BRIGHT_FAIL  = 225
SATURATED_HIGH_FAIL = 0.10
SATURATED_HIGH_WARN = 0.03
SATURATED_LOW_FAIL  = 0.10
SATURATED_LOW_WARN  = 0.03
DYNAMIC_RANGE_FAIL =  60
DYNAMIC_RANGE_WARN = 100
TILE_CV_FAIL = 0.25
TILE_CV_WARN = 0.12

NOISE_SIGMA_FAIL = 14.0
NOISE_SIGMA_WARN =  7.0
SNR_FAIL =  8.0
SNR_WARN = 18.0
SP_FAIL = 0.020
SP_WARN = 0.005

COVERAGE_SPARSE_FAIL =   2.0
COVERAGE_SPARSE_WARN =   8.0
COVERAGE_DENSE_WARN  =  50.0
COVERAGE_DENSE_FAIL  =  68.0
DENSITY_CV_FAIL = 1.20
DENSITY_CV_WARN = 0.70
TOUCHING_FAIL = 0.65
TOUCHING_WARN = 0.35

WEIGHT_BLUR     = 0.35
WEIGHT_EXPOSURE = 0.20
WEIGHT_NOISE    = 0.20
WEIGHT_DENSITY  = 0.25

SCORE_PASS     = 75.0
SCORE_REVIEW   = 55.0


# ═══════════ DATA CLASSES ═══════════
@dataclass
class Finding:
    rule_id: str
    severity: str
    metric: str
    measured: float
    threshold: float
    operator: str
    message: str
    impact: str

@dataclass
class MetricResult:
    name: str
    score: float
    severity: str
    measurements: dict
    findings: list

@dataclass
class Verdict:
    decision: str
    confidence: float
    reasoning: list
    blockers: list

@dataclass
class QualityReport:
    version: str
    timestamp: str
    image_info: dict
    metrics: dict
    overall_score: float
    verdict: Verdict
    annotated_image: Optional[np.ndarray] = None
    heatmap_image:   Optional[np.ndarray] = None
    histogram_data:  Optional[dict] = None


# ═══════════ HELPERS ═══════════
def _band_score(value, fail, warn, pass_, higher_is_better=True):
    if higher_is_better:
        if value <= fail:  return max(0.0, (value / fail) * 30)
        if value <= warn:  return 30 + (value - fail) / (warn - fail) * 30
        if value <= pass_: return 60 + (value - warn) / (pass_ - warn) * 30
        return min(100.0, 90 + (value - pass_) / pass_ * 10)
    else:
        if value >= fail:  return max(0.0, 30 - (value - fail) / fail * 30)
        if value >= warn:  return 30 + (fail - value) / (fail - warn) * 30
        if value >= pass_: return 60 + (warn - value) / (warn - pass_) * 30
        return min(100.0, 90 + (pass_ - value) / pass_ * 10)

def _worst(severities):
    if "fail" in severities: return "fail"
    if "warn" in severities: return "warn"
    return "pass"


# ═══════════ 1. BLUR (UNCHANGED FROM v2) ═══════════
def analyze_blur(gray):
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx*sx + sy*sy)
    tenengrad = float(np.mean(grad_mag ** 2))
    grad_8u = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(grad_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    f = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - w//2)**2 + (Y - h//2)**2)
    cutoff = min(h, w) * 0.15
    hf_ratio = float(mag[r > cutoff].sum()) / (float(mag.sum()) + 1e-9)

    s_lv  = _band_score(lap_var,    BLUR_LV_FAIL, BLUR_LV_WARN, BLUR_LV_PASS, True)
    s_ten = _band_score(tenengrad,  TEN_FAIL,     TEN_WARN,     TEN_PASS,     True)
    s_ed  = _band_score(edge_density, EDGE_DENSITY_FAIL, EDGE_DENSITY_WARN, 0.10, True)
    score = round(0.55*s_lv + 0.25*s_ten + 0.20*s_ed, 1)

    findings = []
    if lap_var < BLUR_LV_FAIL:
        findings.append(Finding("BLUR.LV.FAIL","fail","laplacian_variance",
            round(lap_var,2),BLUR_LV_FAIL,"<",
            f"Severe defocus or motion blur (LV={lap_var:.1f})",
            "Cell boundaries cannot be reliably segmented."))
    elif lap_var < BLUR_LV_WARN:
        findings.append(Finding("BLUR.LV.WARN","warn","laplacian_variance",
            round(lap_var,2),BLUR_LV_WARN,"<",
            f"Mild blur detected (LV={lap_var:.1f})",
            "Fine subcellular structures may be lost."))
    else:
        findings.append(Finding("BLUR.LV.PASS","pass","laplacian_variance",
            round(lap_var,2),BLUR_LV_WARN,"≥",
            f"Image is in focus (LV={lap_var:.1f})",
            "Sufficient sharpness for analysis."))
    if tenengrad < TEN_FAIL and lap_var < BLUR_LV_WARN:
        findings.append(Finding("BLUR.TEN.CONFIRM","fail","tenengrad",
            round(tenengrad,1),TEN_FAIL,"<",
            "Tenengrad confirms low gradient energy",
            "Independent confirmation of blur."))
    if edge_density < EDGE_DENSITY_FAIL:
        findings.append(Finding("BLUR.EDGES.FAIL","fail","edge_density",
            round(edge_density,4),EDGE_DENSITY_FAIL,"<",
            f"Almost no detectable edges ({edge_density*100:.2f}%)",
            "Image lacks expected cell structure."))

    return MetricResult(
        name="Sharpness",
        score=max(0.0, min(100.0, score)),
        severity=_worst([f.severity for f in findings]),
        measurements={
            "laplacian_variance": round(lap_var,2),
            "tenengrad": round(tenengrad,1),
            "edge_density": round(edge_density,4),
            "hf_energy_ratio": round(hf_ratio,4),
        },
        findings=findings,
    )


# ═══════════ 2. EXPOSURE (UNCHANGED FROM v2) ═══════════
def analyze_exposure(gray):
    g = gray.astype(np.float32)
    findings = []
    mean_v = float(g.mean())
    p1, p99 = np.percentile(g, [1, 99])
    dyn_range = float(p99 - p1)
    sat_high = float(np.mean(gray >= 254))
    sat_low  = float(np.mean(gray <= 1))

    h, w = gray.shape
    tiles = 5
    th, tw = h // tiles, w // tiles
    tile_means = []
    for r in range(tiles):
        for c in range(tiles):
            tile = g[r*th:(r+1)*th, c*tw:(c+1)*tw]
            tile_means.append(tile.mean())
    tile_means = np.array(tile_means, dtype=np.float32)
    tile_cv = float(tile_means.std() / (tile_means.mean() + 1e-6))

    small = cv2.resize(g, (64,64), interpolation=cv2.INTER_AREA)
    Y, X = np.mgrid[0:64, 0:64].astype(np.float32)
    A = np.column_stack([X.ravel()**2, Y.ravel()**2, (X*Y).ravel(),
                         X.ravel(), Y.ravel(), np.ones(64*64)])
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    surface = (A @ coef).reshape(64,64)
    vignette_strength = float((surface.max() - surface.min()) / (mean_v + 1e-6))

    if mean_v < EXPOSURE_DARK_FAIL or mean_v > EXPOSURE_BRIGHT_FAIL:
        s_exp = 0
    elif mean_v < EXPOSURE_DARK_WARN:
        s_exp = 30 + (mean_v - EXPOSURE_DARK_FAIL) / (EXPOSURE_DARK_WARN - EXPOSURE_DARK_FAIL) * 30
    elif mean_v > EXPOSURE_BRIGHT_WARN:
        s_exp = 30 + (EXPOSURE_BRIGHT_FAIL - mean_v) / (EXPOSURE_BRIGHT_FAIL - EXPOSURE_BRIGHT_WARN) * 30
    else:
        s_exp = 100

    s_uni = _band_score(tile_cv, TILE_CV_FAIL, TILE_CV_WARN, 0.04, False)
    s_dr  = _band_score(dyn_range, DYNAMIC_RANGE_FAIL, DYNAMIC_RANGE_WARN, 200, True)

    sat_penalty = 0
    if sat_high > SATURATED_HIGH_FAIL: sat_penalty += 30
    elif sat_high > SATURATED_HIGH_WARN: sat_penalty += 12
    if sat_low > SATURATED_LOW_FAIL: sat_penalty += 30
    elif sat_low > SATURATED_LOW_WARN: sat_penalty += 12

    score = round(0.50*s_exp + 0.30*s_uni + 0.20*s_dr - sat_penalty, 1)
    score = max(0.0, min(100.0, score))

    if mean_v < EXPOSURE_DARK_FAIL:
        findings.append(Finding("EXPOSURE.UNDER.FAIL","fail","mean_intensity",
            round(mean_v,1),EXPOSURE_DARK_FAIL,"<",
            f"Severely underexposed (mean = {mean_v:.0f}/255)",
            "Cell features may be lost in noise floor."))
    elif mean_v < EXPOSURE_DARK_WARN:
        findings.append(Finding("EXPOSURE.UNDER.WARN","warn","mean_intensity",
            round(mean_v,1),EXPOSURE_DARK_WARN,"<",
            f"Image is dim (mean = {mean_v:.0f}/255)",
            "Reduced contrast for staining-based identification."))
    elif mean_v > EXPOSURE_BRIGHT_FAIL:
        findings.append(Finding("EXPOSURE.OVER.FAIL","fail","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_FAIL,">",
            f"Severely overexposed (mean = {mean_v:.0f}/255)",
            "Highlights clipped — cell detail unrecoverable."))
    elif mean_v > EXPOSURE_BRIGHT_WARN:
        findings.append(Finding("EXPOSURE.OVER.WARN","warn","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_WARN,">",
            f"Image is bright (mean = {mean_v:.0f}/255)",
            "Risk of clipping in light cell regions."))
    else:
        findings.append(Finding("EXPOSURE.OK","pass","mean_intensity",
            round(mean_v,1),EXPOSURE_BRIGHT_WARN,"in_range",
            f"Exposure within target range (mean = {mean_v:.0f})",
            "Good histogram placement."))

    if sat_high > SATURATED_HIGH_FAIL:
        findings.append(Finding("EXPOSURE.CLIP.HIGH.FAIL","fail","saturated_high_fraction",
            round(sat_high,4),SATURATED_HIGH_FAIL,">",
            f"{sat_high*100:.1f}% of pixels are blown out (=255)",
            "Loss of detail in bright regions."))
    elif sat_high > SATURATED_HIGH_WARN:
        findings.append(Finding("EXPOSURE.CLIP.HIGH.WARN","warn","saturated_high_fraction",
            round(sat_high,4),SATURATED_HIGH_WARN,">",
            f"{sat_high*100:.1f}% pixels at maximum value",
            "Some highlight clipping."))
    if sat_low > SATURATED_LOW_FAIL:
        findings.append(Finding("EXPOSURE.CLIP.LOW.FAIL","fail","saturated_low_fraction",
            round(sat_low,4),SATURATED_LOW_FAIL,">",
            f"{sat_low*100:.1f}% of pixels are crushed to black",
            "Loss of detail in shadow regions."))

    if tile_cv > TILE_CV_FAIL:
        findings.append(Finding("EXPOSURE.UNIFORM.FAIL","fail","tile_cv",
            round(tile_cv,3),TILE_CV_FAIL,">",
            f"Strong illumination gradient (CV={tile_cv:.2f})",
            "Vignette or uneven lighting; consider flat-field correction."))
    elif tile_cv > TILE_CV_WARN:
        findings.append(Finding("EXPOSURE.UNIFORM.WARN","warn","tile_cv",
            round(tile_cv,3),TILE_CV_WARN,">",
            f"Mild illumination unevenness (CV={tile_cv:.2f})",
            "Edge regions may have biased measurements."))

    if dyn_range < DYNAMIC_RANGE_FAIL:
        findings.append(Finding("EXPOSURE.DR.FAIL","fail","dynamic_range",
            round(dyn_range,1),DYNAMIC_RANGE_FAIL,"<",
            f"Compressed dynamic range ({dyn_range:.0f}/255)",
            "Low contrast — staining differentiation impaired."))
    elif dyn_range < DYNAMIC_RANGE_WARN:
        findings.append(Finding("EXPOSURE.DR.WARN","warn","dynamic_range",
            round(dyn_range,1),DYNAMIC_RANGE_WARN,"<",
            f"Limited dynamic range ({dyn_range:.0f}/255)",
            "Lower contrast than ideal."))

    return MetricResult(
        name="Lighting",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "mean_intensity": round(mean_v,1),
            "p1": round(float(p1),1),
            "p99": round(float(p99),1),
            "dynamic_range": round(dyn_range,1),
            "saturated_high_pct": round(sat_high*100,3),
            "saturated_low_pct": round(sat_low*100,3),
            "tile_cv": round(tile_cv,3),
            "vignette_strength": round(vignette_strength,3),
        },
        findings=findings,
    )


# ═══════════ 3. NOISE (UNCHANGED FROM v2) ═══════════
def _immerkaer_sigma(gray):
    H = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, H)
    return float(np.sqrt(np.pi/2) * np.mean(np.abs(conv)) / 6.0)

def _salt_pepper_fraction(gray):
    g = gray.astype(np.int16)
    extreme = (gray <= 1) | (gray >= 254)
    if not extreme.any(): return 0.0
    kernel = np.ones((3,3), dtype=np.float32) / 8.0
    kernel[1,1] = 0
    neighbour_mean = cv2.filter2D(g.astype(np.float32), -1, kernel)
    isolation = np.abs(g - neighbour_mean) > 80
    sp = extreme & isolation
    return float(np.count_nonzero(sp)) / gray.size

def analyze_noise(gray):
    sigma = _immerkaer_sigma(gray)
    mean_v = float(np.mean(gray))
    snr = mean_v / (sigma + 1e-6)
    sp_frac = _salt_pepper_fraction(gray)
    h, w = gray.shape
    background_std = float(gray[0:h//5, 0:w//5].astype(np.float32).std())

    s_sigma = _band_score(sigma, NOISE_SIGMA_FAIL, NOISE_SIGMA_WARN, 2.0, False)
    s_snr   = _band_score(snr, SNR_FAIL, SNR_WARN, 50, True)
    s_sp    = 100 - min(100, sp_frac * 5000)
    score = round(0.45*s_sigma + 0.35*s_snr + 0.20*s_sp, 1)
    score = max(0.0, min(100.0, score))

    findings = []
    if sigma >= NOISE_SIGMA_FAIL:
        findings.append(Finding("NOISE.SIGMA.FAIL","fail","noise_sigma",
            round(sigma,2),NOISE_SIGMA_FAIL,"≥",
            f"High noise level (σ = {sigma:.1f})",
            "Cell boundaries obscured by sensor noise."))
    elif sigma >= NOISE_SIGMA_WARN:
        findings.append(Finding("NOISE.SIGMA.WARN","warn","noise_sigma",
            round(sigma,2),NOISE_SIGMA_WARN,"≥",
            f"Moderate noise (σ = {sigma:.1f})",
            "Some texture features may be obscured."))
    else:
        findings.append(Finding("NOISE.SIGMA.PASS","pass","noise_sigma",
            round(sigma,2),NOISE_SIGMA_WARN,"<",
            f"Low noise (σ = {sigma:.1f})",
            "Clean signal."))

    if snr < SNR_FAIL:
        findings.append(Finding("NOISE.SNR.FAIL","fail","snr",
            round(snr,1),SNR_FAIL,"<",
            f"Poor SNR ({snr:.1f})",
            "Signal barely above noise floor."))
    elif snr < SNR_WARN:
        findings.append(Finding("NOISE.SNR.WARN","warn","snr",
            round(snr,1),SNR_WARN,"<",
            f"Marginal SNR ({snr:.1f})",
            "Reduced confidence in pixel-level features."))

    if sp_frac >= SP_FAIL:
        findings.append(Finding("NOISE.SALTPEPPER.FAIL","fail","sp_fraction",
            round(sp_frac,4),SP_FAIL,"≥",
            f"Salt-and-pepper artifacts: {sp_frac*100:.2f}% of pixels",
            "Suggests sensor defects, transmission errors, or compression."))
    elif sp_frac >= SP_WARN:
        findings.append(Finding("NOISE.SALTPEPPER.WARN","warn","sp_fraction",
            round(sp_frac,4),SP_WARN,"≥",
            f"Some salt-and-pepper noise ({sp_frac*100:.2f}%)",
            "Median filter recommended."))

    return MetricResult(
        name="Noise",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "sigma": round(sigma,2),
            "snr": round(snr,2),
            "salt_pepper_pct": round(sp_frac*100,3),
            "background_std": round(background_std,2),
        },
        findings=findings,
    )


# ═══════════ 4. DENSITY (CELLPOSE-POWERED) ═══════════
def _detect_cells_cellpose(image_rgb, h, w):
    """
    Run Cellpose on the image to get cell masks.
    Returns:
      - centers: (N, 2) array of (x, y) centroids
      - radii:   (N,) array of equivalent radii
      - areas:   (N,) array of pixel areas
      - mask:    (H, W) integer mask (0 = background, 1..N = cell IDs)
    """
    # Cellpose expects RGB; we estimate diameter automatically with diameter=None
    # (it has a built-in size estimator), but for blood smears we can hint
    # at typical RBC diameter (~6-8 microns; in pixels at 40x ~ 25-35px)
    masks, flows, styles = _CELLPOSE_MODEL.eval(
        image_rgb,
        diameter=None,        # let Cellpose auto-estimate
        channels=[0, 0],      # grayscale mode (interprets RGB as single channel internally)
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )

    # masks: 2D array, 0 = background, 1..N = cell IDs
    n_cells = int(masks.max())
    if n_cells == 0:
        return np.zeros((0, 2), int), np.zeros((0,), float), np.zeros((0,), int), masks

    centers = []
    radii = []
    areas = []
    for cell_id in range(1, n_cells + 1):
        cell_mask = (masks == cell_id)
        area = int(cell_mask.sum())
        if area < 20:  # skip tiny noise
            continue
        ys, xs = np.where(cell_mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        # Equivalent radius assuming circular: A = πr² → r = √(A/π)
        r = float(np.sqrt(area / np.pi))
        centers.append([cx, cy])
        radii.append(r)
        areas.append(area)

    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)
    areas = np.array(areas, dtype=int)
    return centers, radii, areas, masks


def _detect_cells_hough_fallback(gray, h, w):
    """Fallback if Cellpose isn't available — basic Hough detection."""
    blurred = cv2.medianBlur(gray, 5)
    min_dim = min(h, w)
    min_r = max(6, int(min_dim * 0.013))
    max_r = max(20, int(min_dim * 0.050))
    min_dist = max(int(min_r * 1.8), 12)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_dist,
        param1=90, param2=28,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return np.zeros((0, 2), float), np.zeros((0,), float), np.zeros((0,), int), None
    circles = np.around(circles[0])
    centers = circles[:, :2].astype(float)
    radii = circles[:, 2].astype(float)
    areas = (np.pi * radii ** 2).astype(int)
    return centers, radii, areas, None


def analyze_density(gray, original_bgr):
    """
    Cellpose-powered cell density analysis with Hough fallback.
    """
    h, w = gray.shape
    img_area = h * w

    # Convert BGR → RGB for Cellpose
    if _CELLPOSE_AVAILABLE:
        rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        try:
            centers, radii, areas, cell_mask = _detect_cells_cellpose(rgb, h, w)
            method_used = "cellpose_cyto3"
        except Exception as e:
            print(f"[Density] Cellpose failed mid-analysis: {e}; using Hough.", flush=True)
            centers, radii, areas, cell_mask = _detect_cells_hough_fallback(gray, h, w)
            method_used = "hough_fallback"
    else:
        centers, radii, areas, cell_mask = _detect_cells_hough_fallback(gray, h, w)
        method_used = "hough_fallback"

    cell_count = len(centers)

    # --- Coverage from actual cell pixels (preferred) or circles ----
    if cell_mask is not None:
        coverage = float(np.mean(cell_mask > 0)) * 100
    elif cell_count > 0:
        total_area = float(areas.sum())
        coverage = min(95.0, (total_area / img_area) * 100)
    else:
        coverage = 0.0

    # --- Touching fraction: centre-distance based -------------------
    # Two cells are "touching" if their centre distance < 1.5 × median radius
    if cell_count > 1 and len(radii) > 0:
        median_r = float(np.median(radii))
        touch_dist = median_r * 1.5
        touching_count = 0
        n = min(cell_count, 400)
        sample = centers[:n]
        for i in range(n):
            closest = 9999.0
            for j in range(n):
                if i == j: continue
                d = float(np.linalg.norm(sample[i] - sample[j]))
                if d < closest:
                    closest = d
            if closest < touch_dist:
                touching_count += 1
        touching_frac = min(1.0, touching_count / n)
    else:
        touching_frac = 0.0

    # --- Spatial CV across 6×6 grid ---------------------------------
    grid_r, grid_c = 6, 6
    th, tw = h // grid_r, w // grid_c
    density_grid = np.zeros((grid_r, grid_c), dtype=np.float32)
    for cx, cy in centers:
        density_grid[
            min(int(cy) // th, grid_r - 1),
            min(int(cx) // tw, grid_c - 1)
        ] += 1
    g_mean = float(density_grid.mean())
    density_cv = float(density_grid.std() / (g_mean + 1e-6)) if g_mean > 0 else 0.0

    # --- Heatmap visualization --------------------------------------
    if density_grid.max() > 0:
        norm = density_grid / density_grid.max() * 255
    else:
        norm = density_grid
    norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(norm_resized, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)

    # --- Annotated image --------------------------------------------
    annotated = original_bgr.copy()
    if cell_mask is not None:
        # Cellpose: draw actual mask outlines (more informative than circles)
        contours_overlay = annotated.copy()
        n_cells = int(cell_mask.max())
        for cell_id in range(1, n_cells + 1):
            single = (cell_mask == cell_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_overlay, contours, -1, (0, 230, 180), 1, cv2.LINE_AA)
        annotated = contours_overlay
    else:
        # Hough fallback: draw circles
        for (cx, cy), r in zip(centers, radii):
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 230, 180), 1, lineType=cv2.LINE_AA)

    # --- Coverage scoring -------------------------------------------
    if coverage < COVERAGE_SPARSE_FAIL:
        s_cov = 5
    elif coverage < COVERAGE_SPARSE_WARN:
        s_cov = 30 + (coverage - COVERAGE_SPARSE_FAIL) / (COVERAGE_SPARSE_WARN - COVERAGE_SPARSE_FAIL) * 30
    elif coverage <= COVERAGE_DENSE_WARN:
        s_cov = 100
    elif coverage <= COVERAGE_DENSE_FAIL:
        s_cov = 30 + (COVERAGE_DENSE_FAIL - coverage) / (COVERAGE_DENSE_FAIL - COVERAGE_DENSE_WARN) * 30
    else:
        s_cov = 5

    s_cv = _band_score(density_cv, DENSITY_CV_FAIL, DENSITY_CV_WARN, 0.25, False)
    s_touch = _band_score(touching_frac, TOUCHING_FAIL, TOUCHING_WARN, 0.10, False)

    score = round(0.55*s_cov + 0.25*s_cv + 0.20*s_touch, 1)
    score = max(0.0, min(100.0, score))

    findings = []
    if coverage < COVERAGE_SPARSE_FAIL:
        findings.append(Finding("DENSITY.SPARSE.FAIL","fail","coverage_pct",
            round(coverage,2),COVERAGE_SPARSE_FAIL,"<",
            f"Field is essentially empty ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Insufficient sample for analysis."))
    elif coverage < COVERAGE_SPARSE_WARN:
        findings.append(Finding("DENSITY.SPARSE.WARN","warn","coverage_pct",
            round(coverage,2),COVERAGE_SPARSE_WARN,"<",
            f"Sparse field ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Larger field of view recommended."))
    elif coverage > COVERAGE_DENSE_FAIL:
        findings.append(Finding("DENSITY.DENSE.FAIL","fail","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_FAIL,">",
            f"Severely overcrowded ({coverage:.1f}% coverage)",
            "Heavy overlap prevents reliable single-cell segmentation."))
    elif coverage > COVERAGE_DENSE_WARN:
        findings.append(Finding("DENSITY.DENSE.WARN","warn","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_WARN,">",
            f"High cell density ({coverage:.1f}% coverage)",
            "Some cell overlap; counting may be less accurate."))
    else:
        findings.append(Finding("DENSITY.OK","pass","coverage_pct",
            round(coverage,2),COVERAGE_DENSE_WARN,"in_range",
            f"Optimal cell density ({coverage:.1f}% coverage, ~{cell_count} cells)",
            "Ideal monolayer for analysis."))

    if density_cv > DENSITY_CV_FAIL:
        findings.append(Finding("DENSITY.UNIFORM.FAIL","fail","spatial_cv",
            round(density_cv,3),DENSITY_CV_FAIL,">",
            f"Cells highly clustered (spatial CV = {density_cv:.2f})",
            "Some grid regions empty, others packed; non-representative."))
    elif density_cv > DENSITY_CV_WARN:
        findings.append(Finding("DENSITY.UNIFORM.WARN","warn","spatial_cv",
            round(density_cv,3),DENSITY_CV_WARN,">",
            f"Uneven cell distribution (CV = {density_cv:.2f})",
            "Avoid sampling local clusters."))

    if touching_frac > TOUCHING_FAIL:
        findings.append(Finding("DENSITY.TOUCHING.FAIL","fail","touching_fraction",
            round(touching_frac,3),TOUCHING_FAIL,">",
            f"{touching_frac*100:.0f}% of cells appear touching/overlapping",
            "Heavy rouleaux or overlap; counts may be unreliable."))
    elif touching_frac > TOUCHING_WARN:
        findings.append(Finding("DENSITY.TOUCHING.WARN","warn","touching_fraction",
            round(touching_frac,3),TOUCHING_WARN,">",
            f"{touching_frac*100:.0f}% of cells appear touching",
            "Moderate clumping; review recommended."))

    metric = MetricResult(
        name="Density",
        score=score,
        severity=_worst([f.severity for f in findings]),
        measurements={
            "coverage_pct": round(coverage, 2),
            "cell_count_estimate": int(cell_count),
            "spatial_cv": round(density_cv, 3),
            "touching_fraction": round(touching_frac, 3),
            "detection_method": method_used,
        },
        findings=findings,
    )
    return metric, annotated, heatmap_overlay


# ═══════════ DECISION ENGINE (UNCHANGED FROM v2) ═══════════
def make_verdict(metrics, overall_score):
    reasoning = []
    blockers = []
    critical_findings = []
    warn_findings = []
    for key, m in metrics.items():
        for f in m.findings:
            if f.severity == "fail": critical_findings.append((key, f))
            elif f.severity == "warn": warn_findings.append((key, f))

    reasoning.append({
        "step": "1. Aggregate score",
        "detail": f"Weighted overall score = {overall_score:.1f}/100",
        "outcome": ("strong" if overall_score >= SCORE_PASS
                    else "marginal" if overall_score >= SCORE_REVIEW else "weak"),
    })
    reasoning.append({
        "step": "2. Findings audit",
        "detail": f"{len(critical_findings)} critical, {len(warn_findings)} warning",
        "outcome": ("blocking" if critical_findings else "clear"),
    })

    blur_critical = any(f.severity=="fail" for f in metrics["blur"].findings)
    noise_critical = any(f.severity=="fail" for f in metrics["noise"].findings)

    if blur_critical:
        for f in metrics["blur"].findings:
            if f.severity=="fail": blockers.append(f.rule_id)
        reasoning.append({"step":"3. Blur veto",
            "detail":"Critical blur — focus is not recoverable","outcome":"REJECT"})
        return Verdict("REJECT", 0.95, reasoning, blockers)

    if noise_critical and metrics["noise"].score < 30:
        for f in metrics["noise"].findings:
            if f.severity=="fail": blockers.append(f.rule_id)
        reasoning.append({"step":"3. Noise veto",
            "detail":"Critical noise — signal not recoverable","outcome":"REJECT"})
        return Verdict("REJECT", 0.92, reasoning, blockers)

    if len(critical_findings) >= 2:
        for cat, f in critical_findings: blockers.append(f.rule_id)
        reasoning.append({"step":"3. Multiple criticals",
            "detail":f"{len(critical_findings)} critical issues; not safely correctable","outcome":"REJECT"})
        return Verdict("REJECT", 0.90, reasoning, blockers)

    if overall_score < SCORE_REVIEW:
        reasoning.append({"step":"3. Score threshold",
            "detail":f"Score {overall_score:.1f} below reject threshold ({SCORE_REVIEW})","outcome":"REJECT"})
        return Verdict("REJECT", 0.85, reasoning,
            [f.rule_id for _,f in critical_findings])

    if critical_findings or overall_score < SCORE_PASS:
        for cat, f in critical_findings: blockers.append(f.rule_id)
        reasoning.append({"step":"3. Borderline",
            "detail":("One critical issue present" if critical_findings
                      else f"Score in review band ({SCORE_REVIEW} ≤ {overall_score:.1f} < {SCORE_PASS})"),
            "outcome":"REVIEW"})
        confidence = 0.6 + (overall_score - SCORE_REVIEW) / (SCORE_PASS - SCORE_REVIEW) * 0.2
        return Verdict("REVIEW", round(confidence,2), reasoning, blockers)

    reasoning.append({"step":"3. All checks passed",
        "detail":f"No critical issues; score {overall_score:.1f} ≥ {SCORE_PASS}","outcome":"PASS"})
    return Verdict("PASS", round(0.85 + min(0.15, (overall_score-SCORE_PASS)/100), 2), reasoning, [])


# ═══════════ MASTER ENTRY POINT ═══════════
def analyze_image(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")
    if len(image_bgr.shape) == 2:
        gray = image_bgr
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_m = analyze_blur(gray)
    light_m = analyze_exposure(gray)
    noise_m = analyze_noise(gray)
    dens_m, annotated, heatmap = analyze_density(gray, image_bgr)

    overall = round(
        WEIGHT_BLUR*blur_m.score + WEIGHT_EXPOSURE*light_m.score +
        WEIGHT_NOISE*noise_m.score + WEIGHT_DENSITY*dens_m.score, 1
    )
    metrics = {"blur":blur_m, "lighting":light_m, "noise":noise_m, "density":dens_m}
    verdict = make_verdict(metrics, overall)

    hist = {}
    for ch_idx, ch_name in enumerate(["Blue","Green","Red"]):
        hist[ch_name] = cv2.calcHist([image_bgr],[ch_idx],None,[256],[0,256]).flatten().tolist()

    return QualityReport(
        version=VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_info={"width":int(image_bgr.shape[1]),"height":int(image_bgr.shape[0]),"channels":int(image_bgr.shape[2])},
        metrics=metrics,
        overall_score=overall,
        verdict=verdict,
        annotated_image=annotated,
        heatmap_image=heatmap,
        histogram_data=hist,
    )
