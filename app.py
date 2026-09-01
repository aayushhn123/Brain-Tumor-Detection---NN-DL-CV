# NeuroScan — Brain MRI Anomaly Detection
# UI v6.0 — Minimal · Apple-inspired · Native Streamlit components

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label as sp_label,
    uniform_filter,
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import time

st.set_page_config(
    page_title="NeuroScan · AI MRI Analysis",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  MINIMAL DESIGN SYSTEM
#  Philosophy: near-monochrome, one accent, generous whitespace, native
#  Streamlit widgets everywhere. CSS only reshapes spacing/typography/borders
#  — never invents new custom components.
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg:        #000000;
    --surface:   #161618;
    --surface-2: #1c1c1f;
    --surface-hover: #232326;
    --line:      rgba(255,255,255,0.14);
    --line-strong: rgba(255,255,255,0.22);
    --line-soft: rgba(255,255,255,0.07);
    --text:      #f5f5f7;
    --text-dim:  #a1a1a6;
    --text-faint:#6e6e73;
    --accent:    #2dd4bf;
    --accent-dim: rgba(45,212,191,0.10);
    --accent-line: rgba(45,212,191,0.35);
    --red:       #ff6b6b;
    --red-dim:   rgba(255,107,107,0.10);
    --amber:     #f5a623;
    --amber-dim: rgba(245,166,35,0.10);
    --radius:    12px;
    --radius-lg: 16px;
    --shadow-card: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.35);
    --shadow-card-hover: 0 2px 4px rgba(0,0,0,0.45), 0 14px 36px rgba(0,0,0,0.5);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
* { letter-spacing: -0.011em; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 980px !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* ── Fade-in for main flow ── */
.main .block-container { animation: fadeIn 0.5s ease both; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

/* ══════════════ HERO ══════════════ */
.hero { padding: 4.5rem 0 2.5rem; animation: fadeUp 0.6s ease both; }
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 1.1rem;
}
.hero-title {
    font-size: clamp(2.6rem, 5.5vw, 4rem); font-weight: 700; line-height: 1.04;
    letter-spacing: -0.035em; color: var(--text); margin: 0 0 1.1rem;
}
.hero-title .hi { color: var(--accent); }
.hero-desc {
    font-size: 1.2rem; font-weight: 400; color: var(--text-dim);
    line-height: 1.55; max-width: 620px; margin-bottom: 1.75rem;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.85rem; border-radius: 100px; font-size: 0.8rem;
    font-weight: 500; border: 1px solid var(--line-strong); color: var(--text-dim);
    background: var(--surface);
    transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
}
.pill:hover { border-color: var(--accent-line); background: var(--surface-hover); transform: translateY(-1px); }
.pill.warn { border-color: rgba(245,166,35,0.4); color: var(--amber); background: var(--amber-dim); }
.pill.warn:hover { border-color: var(--amber); }

/* ══════════════ SECTION LABELS ══════════════ */
.eyebrow-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-faint);
    margin: 3rem 0 0.6rem;
}
.section-title {
    font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em;
    color: var(--text); margin: 0 0 0.35rem;
}
.section-sub {
    font-size: 1rem; color: var(--text-dim); line-height: 1.55;
    max-width: 640px; margin-bottom: 1.5rem;
}
hr.divider { border: none; border-top: 1px solid var(--line-soft); margin: 3rem 0 0; }

/* ══════════════ CONTAINERS AS CARDS ══════════════ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--line-strong) !important;
    box-shadow: var(--shadow-card-hover) !important;
}

/* ══════════════ METRICS ══════════════ */
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 1.3rem 1.4rem;
    box-shadow: var(--shadow-card);
    animation: fadeUp 0.45s ease both;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--accent-line);
    box-shadow: var(--shadow-card-hover);
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-faint) !important;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 700 !important; color: var(--text) !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ══════════════ TABS ══════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.75rem !important; background: transparent !important;
    border-bottom: 1px solid var(--line) !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.95rem !important; font-weight: 500 !important;
    color: var(--text-faint) !important; padding: 0.7rem 0.2rem !important;
    background: transparent !important; border-bottom: 2px solid transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-dim) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important; border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"]:hover { color: var(--text) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.75rem !important; }

/* ══════════════ EXPANDER ══════════════ */
div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"]:hover { border-color: var(--line-strong) !important; }
div[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    transition: background 0.15s ease !important;
}
div[data-testid="stExpander"] summary:hover { background: var(--surface-hover) !important; }
div[data-testid="stExpander"] > div:nth-child(2) {
    border-top: 1px solid var(--line-soft) !important;
    background: var(--surface-2) !important;
}

/* ══════════════ SIDEBAR ══════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg) !important; border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem !important; }
.sb-logo {
    font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text);
    margin-bottom: 0.15rem;
}
.sb-logo span { color: var(--accent); }
.sb-tag { font-size: 0.78rem; color: var(--text-faint); margin-bottom: 1.75rem; }
.sb-heading {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-faint); margin: 1.75rem 0 0.75rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--line-soft);
}
section[data-testid="stSidebar"] label { font-size: 0.88rem !important; color: var(--text-dim) !important; }
section[data-testid="stSidebar"] hr { border-color: var(--line) !important; }

/* Sidebar checkboxes as distinct rows */
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 0.65rem 0.9rem; margin-bottom: 0.55rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"]:hover {
    border-color: var(--line-strong); background: var(--surface-hover);
}

/* ══════════════ BUTTONS ══════════════ */
.stButton > button {
    background: var(--text) !important; color: #000 !important; border: 1px solid var(--text) !important;
    border-radius: 100px !important; font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 0.65rem 1.9rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
    transition: opacity 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; opacity: 0.7 !important; }
.stButton > button[kind="secondary"] {
    background: transparent !important; color: var(--text) !important;
    border: 1px solid var(--line-strong) !important; box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--surface-hover) !important; border-color: var(--accent) !important;
}
div[data-testid="stCheckbox"] label:hover span:first-child { border-color: var(--accent) !important; }

/* ══════════════ FILE UPLOADER ══════════════ */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important; border: 1.5px dashed var(--line-strong) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--surface-hover) !important;
    box-shadow: var(--shadow-card-hover) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid var(--line-strong) !important; border-radius: 100px !important;
    background: var(--surface-2) !important; color: var(--text) !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important; background: var(--surface-hover) !important;
}
/* uploaded file chip */
div[data-testid="stFileUploaderFile"] {
    background: var(--surface-2) !important; border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important; padding: 0.3rem 0.6rem !important;
}

/* ══════════════ SLIDER ══════════════ */
div[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-dim) !important;
    transition: box-shadow 0.15s ease !important;
}
div[data-baseweb="slider"] div[role="slider"]:hover {
    box-shadow: 0 0 0 6px var(--accent-dim) !important;
}
.stSlider [data-testid="stTickBar"] { display: none !important; }

/* ══════════════ ALERTS ══════════════ */
div[data-testid="stAlertContentInfo"],
div[data-testid="stAlertContentWarning"],
div[data-testid="stAlertContentError"],
div[data-testid="stAlertContentSuccess"] {
    font-size: 0.92rem !important; line-height: 1.6 !important;
}
div[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--line) !important;
    box-shadow: var(--shadow-card) !important;
}

/* ══════════════ STATUS / SPINNER ══════════════ */
div[data-testid="stStatusWidget"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-card) !important;
}

/* ══════════════ SELECT SLIDER (sensitivity) ══════════════ */
div[data-baseweb="slider"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important; padding: 1.1rem 1rem 0.6rem !important;
}

/* ══════════════ IMAGES ══════════════ */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--line) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
}
[data-testid="stImage"] img:hover {
    border-color: var(--line-strong) !important;
    box-shadow: var(--shadow-card-hover) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stImageCaption"] {
    font-size: 0.8rem !important; color: var(--text-faint) !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
    font-weight: 600 !important; text-align: center !important;
    margin-top: 0.6rem !important;
}

/* ══════════════ RESULT BADGE (small, restrained) ══════════════ */
.result-badge {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.55rem 1.1rem; border-radius: 100px; font-size: 0.92rem;
    font-weight: 600; animation: fadeUp 0.4s ease both;
    box-shadow: var(--shadow-card);
}
.result-badge.found { background: var(--red-dim); color: var(--red); border: 1px solid rgba(255,107,107,0.35); }
.result-badge.clear { background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(45,212,191,0.35); }
.result-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.risk-pill {
    display: inline-flex; align-items: center; padding: 0.5rem 1rem; border-radius: 100px;
    font-size: 0.85rem; font-weight: 700; letter-spacing: 0.03em;
    box-shadow: var(--shadow-card);
}
.risk-pill.high   { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,107,107,0.35); }
.risk-pill.medium { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(245,166,35,0.35); }
.risk-pill.low    { background: var(--accent-dim);color: var(--accent);border: 1px solid rgba(45,212,191,0.35); }

/* ══════════════ DATAFRAME-LIKE ROWS FOR REPORT ══════════════ */
.report-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 0.95rem 1.25rem; border-bottom: 1px solid var(--line-soft); gap: 2rem;
    margin: 0 -1.25rem; transition: background 0.15s ease;
}
.report-row:hover { background: var(--surface-hover); }
.report-row:last-child { border-bottom: none; }
.report-key { font-size: 0.92rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
.report-explain { font-size: 0.82rem; color: var(--text-faint); line-height: 1.5; max-width: 480px; }
.report-val {
    font-size: 0.95rem; font-weight: 600; color: var(--text); text-align: right;
    flex-shrink: 0; min-width: 110px; font-variant-numeric: tabular-nums;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE — UNCHANGED — Ensemble multi-signal approach
# ═════════════════════════════════════════════════════════════════════════

def preprocess_mri(img: Image.Image) -> np.ndarray:
    """Normalize + denoise. Returns float32 in [0,1]."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    lo, hi = gray.min(), gray.max()
    norm = (gray - lo) / (hi - lo + 1e-8)
    return gaussian_filter(norm, sigma=0.8)


def extract_brain_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    rough = gray > 0.05
    rough = binary_fill_holes(rough)
    labeled, n = sp_label(rough)
    if n == 0:
        return rough
    sizes = ndimage.sum(rough, labeled, range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = (labeled == head_label)
    erode_px = max(5, int(min(h, w) * 0.035))
    brain = binary_erosion(head, iterations=erode_px)
    brain = binary_fill_holes(brain)
    brain = binary_dilation(brain, iterations=2)
    brain = brain & head
    if brain.sum() < (h * w * 0.03):
        brain = head
    return brain


def compute_local_contrast(gray: np.ndarray, window: int = 15) -> np.ndarray:
    local_mean = uniform_filter(gray, size=window)
    local_sq   = uniform_filter(gray ** 2, size=window)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    contrast = (gray - local_mean) / (local_std + 0.02)
    return contrast


def otsu_threshold_1d(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8
    best_thresh = bin_centers[len(bin_centers) // 2]
    best_var    = 0.0
    w0 = 0.0; mu0 = 0.0
    for i, (p, c) in enumerate(zip(hist, bin_centers)):
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            w0 += p; mu0 = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)
            continue
        mu1_num = np.dot(hist[i:], bin_centers[i:])
        mu1     = mu1_num / (w1 + 1e-8)
        var     = w0 * w1 * (mu0 / (w0 + 1e-8) - mu1) ** 2
        if var > best_var:
            best_var   = var
            best_thresh = c
        w0  += p
        mu0  = (mu0 * (w0 - p) + p * c) / (w0 + 1e-8)
    return best_thresh


def detect_tumor_region(
    gray: np.ndarray,
    brain_mask: np.ndarray,
    sensitivity: str = "balanced",
) -> tuple:
    h, w = gray.shape
    edge_margin = max(3, int(min(h, w) * 0.04))
    strict_mask = binary_erosion(brain_mask, iterations=edge_margin)
    brain_px = gray[strict_mask]
    if brain_px.size < 200:
        return None, None, {}
    mu = brain_px.mean()
    sig = brain_px.std()
    presets = {
        "low":      {"z_a": 3.2, "lc_z": 2.8, "min_area_frac": 0.003},
        "balanced": {"z_a": 2.5, "lc_z": 2.2, "min_area_frac": 0.002},
        "high":     {"z_a": 1.8, "lc_z": 1.6, "min_area_frac": 0.001},
    }
    p = presets.get(sensitivity, presets["balanced"])
    thr_a = mu + p["z_a"] * sig
    sig_a = (gray >= thr_a) & strict_mask
    lc = compute_local_contrast(gray, window=21)
    lc_brain = lc[strict_mask]
    lc_thr = lc_brain.mean() + p["lc_z"] * lc_brain.std()
    sig_b = (lc >= lc_thr) & strict_mask
    otsu_t = otsu_threshold_1d(brain_px)
    sig_c = (gray >= otsu_t) & strict_mask
    votes = sig_a.astype(np.uint8) + sig_b.astype(np.uint8) + sig_c.astype(np.uint8)
    anomaly = votes >= 2
    anomaly = binary_erosion(anomaly, iterations=2)
    anomaly = binary_dilation(anomaly, iterations=6)
    anomaly = binary_fill_holes(anomaly)
    anomaly = anomaly & strict_mask
    brain_area = strict_mask.sum()
    min_px = max(30, int(brain_area * p["min_area_frac"]))
    labeled, n = sp_label(anomaly)
    cleaned = np.zeros_like(anomaly)
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        if comp.sum() >= min_px:
            cleaned |= comp
    labeled2, n2 = sp_label(cleaned)
    if n2 == 0:
        return None, None, {}
    best_score = -1
    best_lbl = -1
    for lbl in range(1, n2 + 1):
        comp = (labeled2 == lbl)
        area = comp.sum()
        rows = np.where(np.any(comp, axis=1))[0]
        cols = np.where(np.any(comp, axis=0))[0]
        if rows.size == 0 or cols.size == 0: continue
        h_c = rows[-1] - rows[0] + 1
        w_c = cols[-1] - cols[0] + 1
        bbox_area = h_c * w_c
        solidity = area / (bbox_area + 1e-8)
        aspect_ratio = min(h_c, w_c) / (max(h_c, w_c) + 1e-8)
        score = area * (solidity ** 2) * aspect_ratio
        if score > best_score:
            best_score = score
            best_lbl = lbl
    tumor = (labeled2 == best_lbl)
    rows = np.where(np.any(tumor, axis=1))[0]
    cols = np.where(np.any(tumor, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, None, {}
    H, W = gray.shape
    pad = max(8, int(min(H, W) * 0.015))
    bbox = (
        max(0, cols[0] - pad),
        max(0, rows[0] - pad),
        min(W-1, cols[-1] + pad),
        min(H-1, rows[-1] + pad),
    )
    tumor_px = gray[tumor]
    tissue_px = gray[strict_mask & ~tumor]
    contrast = float((tumor_px.mean() - tissue_px.mean()) / (tissue_px.std() + 1e-8))
    area_frac = float(tumor.sum() / (brain_area + 1e-8))
    from skimage.measure import perimeter as sk_perimeter
    try:
        perim = float(sk_perimeter(tumor))
        circ = float(4 * np.pi * tumor.sum() / (perim ** 2 + 1e-8))
    except Exception:
        circ = 0.5
    circ = min(1.0, max(0.0, circ))
    diag = {
        "mu": float(mu), "sig": float(sig),
        "thr_a": float(thr_a), "otsu_t": float(otsu_t),
        "contrast": contrast, "area_frac": area_frac,
        "circularity": circ,
        "n_components_before_filter": n,
        "tumor_mean": float(tumor_px.mean()),
        "tissue_mean": float(tissue_px.mean()),
        "tissue_std": float(tissue_px.std()),
        "signal_votes_mean": float(votes[strict_mask].mean()),
        "lc": lc,
    }
    return bbox, tumor, diag


def estimate_confidence(diag: dict) -> float:
    contrast  = diag.get("contrast", 0)
    circ      = diag.get("circularity", 0)
    area      = diag.get("area_frac", 0)
    c_score   = min(1.0, contrast / 5.0)
    r_score   = 1.0 - abs(circ - 0.55) / 0.55
    r_score   = max(0, min(1, r_score))
    if 0.003 <= area <= 0.15:
        a_score = 1.0
    elif area < 0.003:
        a_score = area / 0.003
    else:
        a_score = max(0, 1 - (area - 0.15) / 0.15)
    raw = 0.55 * c_score + 0.25 * r_score + 0.20 * a_score
    return float(min(0.98, max(0.52, 0.50 + raw * 0.48)))


# ═════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — UNCHANGED
# ═════════════════════════════════════════════════════════════════════════

def draw_highlight(original: Image.Image, bbox: tuple, tumor_mask: np.ndarray) -> Image.Image:
    rgb   = np.array(original.convert("RGB"), dtype=np.float32)
    ovl   = rgb.copy()
    halo  = binary_dilation(tumor_mask, iterations=6) & ~tumor_mask
    ovl[halo]       = [255, 100,  40]
    ovl[tumor_mask] = [255,  40,  60]
    blended = (0.45 * ovl + 0.55 * rgb).clip(0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)
    draw    = ImageDraw.Draw(result)
    draw.rectangle(bbox, outline=(180, 30, 50), width=4)
    inner = (bbox[0]+3, bbox[1]+3, bbox[2]-3, bbox[3]-3)
    draw.rectangle(inner, outline=(255, 80, 80), width=1)
    tick = 14
    x0, y0, x1, y1 = bbox
    for (cx, cy, dx, dy) in [
        (x0, y0,  1,  1), (x1, y0, -1,  1),
        (x0, y1,  1, -1), (x1, y1, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255,220,0), width=3)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255,220,0), width=3)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    arm = max(12, int(min(original.size) * 0.028))
    draw.line([(cx-arm, cy), (cx+arm, cy)], fill=(255,230,0), width=2)
    draw.line([(cx, cy-arm), (cx, cy+arm)], fill=(255,230,0), width=2)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255,230,0))
    bh = 20
    bx, by = bbox[0], max(0, bbox[1] - bh - 3)
    draw.rectangle([bx, by, bx+148, by+bh], fill=(200, 25, 45))
    draw.text((bx+6, by+3), "ANOMALY DETECTED", fill=(255, 255, 255))
    return result


def make_heatmap(gray: np.ndarray, brain_mask: np.ndarray,
                 tumor_mask: np.ndarray | None) -> np.ndarray:
    disp = gray.copy()
    if brain_mask.any():
        lo, hi = gray[brain_mask].min(), gray[brain_mask].max()
        disp = (gray - lo) / (hi - lo + 1e-8)
    disp = np.clip(disp, 0, 1)
    disp[~brain_mask] = 0
    cmap = plt.get_cmap("inferno")
    rgba = (cmap(disp) * 255).astype(np.uint8)
    rgb  = rgba[:, :, :3]
    if tumor_mask is not None:
        rgb[tumor_mask] = [0, 220, 255]
        halo = binary_dilation(tumor_mask, iterations=3) & ~tumor_mask
        rgb[halo] = [0, 160, 200]
    return rgb


def make_brain_mask_visual(brain_mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    gray_u8 = (np.clip(gray, 0, 1) * 200).astype(np.uint8)
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    rgb[brain_mask, 0] = np.clip(rgb[brain_mask, 0].astype(int) * 0.3 + 0, 0, 255).astype(np.uint8)
    rgb[brain_mask, 1] = np.clip(rgb[brain_mask, 1].astype(int) * 0.6 + 30, 0, 255).astype(np.uint8)
    rgb[brain_mask, 2] = np.clip(rgb[brain_mask, 2].astype(int) * 0.5 + 160, 0, 255).astype(np.uint8)
    rgb[~brain_mask] = (rgb[~brain_mask].astype(float) * 0.15).astype(np.uint8)
    return rgb


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none", dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_histogram(gray: np.ndarray, brain_mask: np.ndarray,
                   tumor_mask: np.ndarray | None, diag: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    brain_vals = gray[brain_mask & (tumor_mask == False if tumor_mask is not None else brain_mask)]
    ax.hist(brain_vals, bins=60, color="#2dd4bf", alpha=0.5, label="Normal brain tissue",
            density=True, histtype="stepfilled")
    if tumor_mask is not None and tumor_mask.any():
        tumor_vals = gray[tumor_mask]
        ax.hist(tumor_vals, bins=30, color="#ff6b6b", alpha=0.8, label="Suspected tumor region",
                density=True, histtype="stepfilled")
    if "thr_a" in diag:
        ax.axvline(diag["thr_a"], color="#f5a623", linewidth=1.5,
                   linestyle="--", label="Brightness cutoff")
    if "otsu_t" in diag:
        ax.axvline(diag["otsu_t"], color="#a1a1a6", linewidth=1.5,
                   linestyle=":", label="Otsu split")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#6e6e73", labelsize=7)
    ax.set_xlabel("Pixel Brightness (0 = dark, 1 = bright)", color="#a1a1a6", fontsize=8)
    ax.set_ylabel("How common", color="#a1a1a6", fontsize=8)
    ax.set_title("Brightness Distribution — Brain vs Tumor", color="#f5f5f7", fontsize=9, pad=8)
    leg = ax.legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color("#a1a1a6")
    plt.tight_layout(pad=0.5)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_intensity_profile(gray: np.ndarray, tumor_mask: np.ndarray) -> Image.Image:
    rows = np.where(np.any(tumor_mask, axis=1))[0]
    cols = np.where(np.any(tumor_mask, axis=0))[0]
    cy   = int(rows.mean()) if rows.size else gray.shape[0] // 2
    cx   = int(cols.mean()) if cols.size else gray.shape[1] // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.6))
    fig.patch.set_facecolor("#0a0a0a")
    for ax, profile, label in [
        (ax1, gray[cy, :],  "Horizontal scan"),
        (ax2, gray[:, cx],  "Vertical scan"),
    ]:
        ax.set_facecolor("#0a0a0a")
        ax.plot(profile, color="#2dd4bf", linewidth=1.2)
        ax.fill_between(range(len(profile)), profile, alpha=0.15, color="#2dd4bf")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#6e6e73", labelsize=6)
        ax.set_title(f"{label} through center of region", color="#a1a1a6", fontsize=8)
        if label == "Horizontal scan":
            if cols.size:
                ax.axvspan(cols[0], cols[-1], alpha=0.2, color="#ff6b6b", label="Tumor span")
        else:
            if rows.size:
                ax.axvspan(rows[0], rows[-1], alpha=0.2, color="#ff6b6b", label="Tumor span")
    plt.tight_layout(pad=0.4)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_signal_votes(votes_map: np.ndarray, brain_mask: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    disp = votes_map.astype(float)
    disp[~brain_mask] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["#0a0a0a", "#2dd4bf", "#f5a623", "#ff6b6b"], N=4
    )
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.axis("off")
    ax.set_title("Agreement Map — Detectors 0–3", color="#a1a1a6", fontsize=8, pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["None", "1 of 3", "2 of 3", "All 3"])
    cbar.ax.tick_params(colors="#6e6e73", labelsize=7)
    cbar.outline.set_visible(False)
    plt.tight_layout(pad=0.3)
    img = fig_to_pil(fig)
    plt.close(fig)
    return img


# ═════════════════════════════════════════════════════════════════════════
#  RISK CLASSIFICATION — UNCHANGED
# ═════════════════════════════════════════════════════════════════════════

def classify_risk(confidence: float, area_frac: float, contrast: float) -> tuple:
    if confidence > 0.82 and area_frac > 0.005:
        return "HIGH", "high", "This scan shows strong signs of an abnormal region. We recommend consulting a neurologist or radiologist as soon as possible for a professional review."
    elif confidence > 0.68 or area_frac > 0.003:
        return "MODERATE", "medium", "There are moderate indicators of an unusual region. Additional imaging (like a contrast MRI or PET scan) would help clarify the finding."
    else:
        return "LOW", "low", "The signs are mild. No immediate action may be needed, but it's worth doing a follow-up scan in 3–6 months to monitor any changes."


PIPELINE_STEPS = [
    {"title": "Loading & preparing the image",
     "detail": "Converting to grayscale, normalizing brightness, and reducing noise."},
    {"title": "Skull stripping",
     "detail": "Isolating brain tissue from skull, scalp, and background."},
    {"title": "Computing a brightness map",
     "detail": "Comparing each pixel to its immediate neighbours."},
    {"title": "Running 3 independent detectors",
     "detail": "Z-score, local contrast, and Otsu auto-thresholding."},
    {"title": "Voting & noise cleanup",
     "detail": "Keeping only regions at least 2 of 3 detectors agree on."},
    {"title": "Selecting the most likely region",
     "detail": "Scoring remaining regions by size, compactness, and roundness."},
    {"title": "Measuring & scoring",
     "detail": "Calculating contrast, area, and a final confidence score."},
    {"title": "Generating report & visuals",
     "detail": "Assembling the overlay, heatmap, charts, and full report."},
]


# ═════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Brain MRI Analysis · v2.0 Ensemble Pipeline</div>
  <h1 class="hero-title">Neuro<span class="hi">Scan</span></h1>
  <p class="hero-desc">
    Upload a brain MRI slice. A three-detector ensemble pipeline finds unusual regions,
    measures their size and contrast, and returns a plain-English report.
  </p>
  <div class="pill-row">
    <span class="pill warn">⚠ Research use only — not a medical device</span>
    <span class="pill">Always consult a radiologist</span>
    <span class="pill">T1 · T1-CE · T2 · FLAIR</span>
    <span class="pill">3-detector ensemble</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  SIDEBAR — native widgets, minimal chrome
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-logo">Neuro<span>Scan</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Research prototype · v2.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">Detection Sensitivity</div>', unsafe_allow_html=True)
    sensitivity = st.select_slider(
        "How aggressively anomalies are flagged",
        options=["low", "balanced", "high"],
        value="balanced",
        label_visibility="collapsed",
    )

    _modes = {
        "low":      ("Conservative", "Flags only obvious, high-contrast anomalies. Fewer false alarms, may miss subtle lesions.", "info"),
        "balanced": ("Balanced — recommended", "The default. Works well for most standard brain MRI scans.", "success"),
        "high":     ("High sensitivity", "Catches faint or small anomalies. May produce more false positives.", "warning"),
    }
    _mt, _mb, _kind = _modes[sensitivity]
    getattr(st, _kind)(f"**{_mt}**  \n{_mb}")

    st.markdown('<div class="sb-heading">Display Options</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("Show pipeline intermediate images", value=False)
    show_votes = st.checkbox("Show detector agreement map", value=False)

    st.markdown('<div class="sb-heading">How It Works</div>', unsafe_allow_html=True)
    with st.expander("10-step algorithm breakdown"):
        _algo = [
            ("Preprocess",         "Grayscale, normalize, denoise"),
            ("Skull strip",        "Keep only brain tissue"),
            ("Local contrast map", "Compare pixels to neighbours"),
            ("Detector A · Z-score", "Flag pixels far above average"),
            ("Detector B · Local", "Flag pixels unusual vs neighbourhood"),
            ("Detector C · Otsu",  "Auto-threshold normal vs abnormal"),
            ("2-of-3 voting",      "Keep only pixels ≥2 detectors flag"),
            ("Cleanup",            "Remove noise, fill holes"),
            ("Blob scoring",       "Score by size, compactness, roundness"),
            ("Report",             "Confidence, overlays, full report"),
        ]
        for n, (t, d) in enumerate(_algo, 1):
            st.markdown(f"**{n:02d} · {t}**  \n<span style='color:var(--text-faint);font-size:0.85rem'>{d}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("NeuroScan v2.0 · Research prototype  \nNot validated for clinical use")


# ═════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ═════════════════════════════════════════════════════════════════════════

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow-label">Get started</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your MRI scan</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">A single 2D axial slice, PNG or JPEG — T1, T1-CE, T2, or FLAIR. '
    'Not a scout localizer or sagittal/coronal view.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drag and drop your MRI image here, or click to browse",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")

    prev_col, btn_col = st.columns([1, 2], vertical_alignment="center")
    with prev_col:
        st.image(raw_img, width=140)
    with btn_col:
        st.write("")
        run = st.button("Run Analysis", type="primary", use_container_width=False)

    if run:
        # ── Animated pipeline using native st.status ────────────────────
        with st.status("Running analysis pipeline…", expanded=True) as status:
            for step in PIPELINE_STEPS:
                st.write(f"**{step['title']}** — {step['detail']}")
                time.sleep(0.35)
            status.update(label="Analysis complete", state="complete", expanded=False)

        # ── Run detection (unchanged engine) ─────────────────────────────
        gray_norm  = preprocess_mri(raw_img)
        brain_mask = extract_brain_mask(gray_norm)
        bbox, tumor_mask, diag = detect_tumor_region(gray_norm, brain_mask, sensitivity=sensitivity)

        # ═════════════════════════════════════════════════════════════
        #  RESULTS — ANOMALY FOUND
        # ═════════════════════════════════════════════════════════════
        if bbox is not None and tumor_mask is not None:
            confidence = estimate_confidence(diag)
            area_pct   = diag["area_frac"] * 100
            risk, risk_cls, recommendation = classify_risk(confidence, diag["area_frac"], diag["contrast"])
            result_img  = draw_highlight(raw_img, bbox, tumor_mask)
            heatmap_arr = make_heatmap(gray_norm, brain_mask, tumor_mask)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow-label">Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)

            b1, b2 = st.columns([1, 1], vertical_alignment="center")
            with b1:
                st.markdown(
                    '<span class="result-badge found"><span class="dot"></span>Anomaly detected</span>',
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(f'<span class="risk-pill {risk_cls}">{risk} RISK</span>', unsafe_allow_html=True)

            st.error(
                "**What this means:** The algorithm found a region that looks statistically unusual "
                "compared to surrounding brain tissue — confirmed by at least 2 of 3 independent detectors. "
                "This *could* indicate a tumor, cyst, or other abnormality. "
                "**Only a qualified radiologist or neurologist can confirm this finding.**",
                icon="⚠️",
            )

            # ── Key measurements — native st.metric ──────────────────────
            st.markdown('<div class="eyebrow-label">Measurements</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Key measurements</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detection Confidence", f"{confidence*100:.1f}%")
            m2.metric("Brain Coverage", f"{area_pct:.2f}%")
            m3.metric("Brightness Contrast", f"{diag['contrast']:.2f}σ")
            m4.metric("Region Roundness", f"{diag['circularity']:.2f}")
            st.caption(
                "Confidence combines brightness contrast (55%), shape regularity (25%), and size (20%). "
                "Not a medical probability score."
            )

            # ── Visual analysis — native tabs ─────────────────────────────
            st.markdown('<div class="eyebrow-label">Imaging</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Visual analysis</div>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Detection Overlay", "Heatmap View", "Analysis Charts"])

            with tab1:
                st.caption(
                    "Right image: detected region in red with an orange halo. "
                    "Yellow crosshair marks the center; corner brackets show the bounding box."
                )
                c1, c2 = st.columns(2)
                c1.image(raw_img, caption="Original MRI Scan", use_container_width=True)
                c2.image(result_img, caption="Anomaly Highlighted", use_container_width=True)

            with tab2:
                st.caption(
                    "Left: brightness heatmap (dark → bright = low → high intensity), cyan = detected region. "
                    "Right: exact tissue analysed — everything else was excluded."
                )
                c1, c2 = st.columns(2)
                c1.image(heatmap_arr, caption="Intensity Heatmap + Overlay", use_container_width=True)
                c2.image(make_brain_mask_visual(brain_mask, gray_norm), caption="Brain Mask", use_container_width=True)

            with tab3:
                st.caption(
                    "Left: brightness distribution — teal is normal tissue, red is the anomaly region. "
                    "Right: brightness sliced through the region center, red zone marks the anomaly span."
                )
                c1, c2 = st.columns(2)
                c1.image(plot_histogram(gray_norm, brain_mask, tumor_mask, diag), caption="Pixel Brightness Distribution", use_container_width=True)
                c2.image(plot_intensity_profile(gray_norm, tumor_mask), caption="Brightness Profile Through Region", use_container_width=True)

                if show_votes and "lc" in diag:
                    brain_px  = gray_norm[brain_mask]
                    mu, sig_v = brain_px.mean(), brain_px.std()
                    _pre = {"low":(3.2,2.8),"balanced":(2.5,2.2),"high":(1.8,1.6)}
                    z_a, lc_z = _pre.get(sensitivity, (2.5,2.2))
                    lc_map    = diag["lc"]
                    lc_thr    = lc_map[brain_mask].mean() + lc_z * lc_map[brain_mask].std()
                    votes_disp = (
                        (gray_norm >= mu + z_a*sig_v).astype(np.uint8) +
                        (lc_map    >= lc_thr).astype(np.uint8) +
                        (gray_norm >= diag["otsu_t"]).astype(np.uint8)
                    )
                    st.image(plot_signal_votes(votes_disp, brain_mask), caption="Detector Agreement Map", use_container_width=True)

            # ── Pipeline debug images ────────────────────────────────────
            if show_debug:
                st.markdown('<div class="eyebrow-label">Debug</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Pipeline intermediate images</div>', unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                d1.image((gray_norm * 255).astype(np.uint8), caption="① Normalized Grayscale", use_container_width=True)
                d2.image(make_brain_mask_visual(brain_mask, gray_norm), caption="② Brain Mask", use_container_width=True)
                vis = np.zeros((*tumor_mask.shape, 3), dtype=np.uint8)
                vis[tumor_mask] = [255, 60, 60]
                d3.image(vis, caption="③ Raw Tumor Mask", use_container_width=True)

            # ── Full clinical report ─────────────────────────────────────
            st.markdown('<div class="eyebrow-label">Report</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Full clinical analysis report</div>', unsafe_allow_html=True)
            st.caption(f"NeuroScan v2.0 · Ensemble detection · Sensitivity: {sensitivity.upper()}")

            report_rows = [
                ("Detection Status", "Did the algorithm find a statistically unusual region?", "ANOMALY DETECTED"),
                ("Risk Classification", "Severity estimate from confidence, size, and contrast. A heuristic — not a diagnosis.", risk),
                ("Detection Confidence", "Algorithmic certainty a real anomaly exists. Not a medical probability.", f"{confidence*100:.1f}%"),
                ("Anomaly Size — % of Brain", "Fraction of total brain area covered by the region.", f"{area_pct:.2f}%"),
                ("Pixel Count — Anomaly Region", "Total pixels inside the detected region.", f"{int(tumor_mask.sum()):,} px"),
                ("Location — Bounding Box", "Pixel coordinates of the box around the anomaly.", f"x: {bbox[0]}–{bbox[2]}, y: {bbox[1]}–{bbox[3]}"),
                ("Brightness Contrast (σ)", "Standard deviations above normal brain brightness.", f"{diag['contrast']:.3f} σ"),
                ("Anomaly Mean Brightness", "Average pixel brightness inside the region (0=black, 1=white).", f"{diag['tumor_mean']:.4f}"),
                ("Surrounding Tissue Brightness", "Average brightness of normal tissue around the anomaly.", f"{diag['tissue_mean']:.4f}"),
                ("Tissue Variability (Std Dev)", "How much brightness varies across normal tissue.", f"{diag['tissue_std']:.4f}"),
                ("Z-Score Brightness Cutoff", "Brightness cutoff used by Detector A.", f"{diag['thr_a']:.4f}"),
                ("Otsu Auto-Split Threshold", "Threshold auto-chosen by Detector C.", f"{diag['otsu_t']:.4f}"),
                ("Region Roundness (Circularity)", "1.0 = perfect circle. Tumors typically score 0.3–0.8.", f"{diag['circularity']:.3f}"),
                ("Brain Pixels Analysed", "Total pixels identified as brain tissue.", f"{int(brain_mask.sum()):,} px"),
                ("Sensitivity Mode Used", "Detection sensitivity active for this scan.", sensitivity.upper()),
            ]
            with st.container(border=True):
                rows_html = "".join(
                    f'<div class="report-row"><div><div class="report-key">{k}</div>'
                    f'<div class="report-explain">{e}</div></div>'
                    f'<div class="report-val">{v}</div></div>'
                    for k, e, v in report_rows
                )
                st.markdown(rows_html, unsafe_allow_html=True)

            # ── Recommendation + disclaimer ──────────────────────────────
            _kind_map = {"HIGH": "error", "MODERATE": "warning", "LOW": "info"}
            getattr(st, _kind_map.get(risk, "info"))(f"**Recommendation — {risk} risk**  \n{recommendation}")
            st.warning(
                "**Medical disclaimer:** This analysis uses classical image processing — it is not a trained "
                "medical AI and has not been validated for clinical use. All findings must be reviewed by a "
                "qualified radiologist or neurologist before any medical decisions are made. This tool must "
                "never replace professional medical evaluation.",
                icon="⚠️",
            )

        # ═════════════════════════════════════════════════════════════
        #  RESULTS — NO DETECTION
        # ═════════════════════════════════════════════════════════════
        else:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow-label">Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)

            st.markdown(
                '<span class="result-badge clear"><span class="dot"></span>No anomaly detected</span>',
                unsafe_allow_html=True,
            )

            st.info(
                "**What this means:** At the current sensitivity, no region was flagged as statistically "
                "unusual — the 3 detectors did not reach 2-of-3 agreement on any significant region.  \n\n"
                "**This does not mean the scan is definitively clear.** Subtle, very small, or diffuse "
                "lesions may not be detectable. If you expect a lesion is present, try *High Sensitivity* "
                "in the sidebar and re-run. Also confirm the image is an axial brain MRI slice."
            )

            c1, c2 = st.columns(2)
            c1.image(raw_img, caption="Uploaded MRI Scan", use_container_width=True)
            c2.image(make_heatmap(gray_norm, brain_mask, None), caption="Brain Region Identified", use_container_width=True)
