"""Directional resolution (MonoDir) — placeholder core.

The real algorithm is xmipp's ``xmipp_resolution_directional`` (MonoDir),
C++ source in I2PC/xmipp:
  src/xmipp/libraries/reconstruction/resolution_directional.{h,cpp}

Workflow (from scipion's protocol_resolution_directional.py):
  1. Run MonoRes on the map + mask (step 0.25, maxRes 18) -> monoresResolutionMap
  2. Run MonoDir on the map + mask, taking the MonoRes map as an extra input
  3. Per-voxel outputs (each a volume the same shape as the map):
       doa        - Degree of Anisotropy, 0..1          (doaMetric.vol)
       mean_res   - mean directional resolution, A      (meanResdoa.vol)
       radial     - radial resolution, A                (radial_resolution.vol)
       azimuthal  - azimuthal/tangential resolution, A  (azimuthal_resolution.vol)
       highest    - best resolution over directions, A  (highestResolution.vol)
       lowest     - worst resolution over directions, A (lowestResolution.vol)
  4. Histograms of doa (range 0-1) and mean/radial/azimuthal (range 0-30),
     30 bins each, over the mask.

This module currently returns FAKE data with realistic ranges so the GUI tab
(inputs, threaded compute, metric selector, histogram, map coloring) can be
built and reviewed before the real algorithm is ported.
"""

import math
import time
import numpy as np

# Metric key -> (display label, units label for axes, default display range)
METRICS = {
    "doa":       ("Degree of anisotropy (DoA)", "DoA",            (0.0, 1.0)),
    "mean_res":  ("Mean directional resolution", "Resolution (Å)", None),
    "radial":    ("Radial resolution",           "Resolution (Å)", None),
    "azimuthal": ("Azimuthal resolution",        "Resolution (Å)", None),
    "highest":   ("Highest resolution",          "Resolution (Å)", None),
    "lowest":    ("Lowest resolution",           "Resolution (Å)", None),
}


def fso_crossings(res_axis, fso, thresholds=(0.9, 0.5, 0.1)):
    """Resolution (Å) where the FSO curve crosses each threshold.

    `res_axis` is resolution in Å (descending as frequency rises) and `fso`
    the occupancy 1..0. Linear interpolation between bracketing samples, like
    xmipp's interpolRes. Returns {threshold: resolution or None}.
    """
    out = {}
    r = np.asarray(res_axis, dtype=float)
    y = np.asarray(fso, dtype=float)
    for thr in thresholds:
        hit = None
        for i in range(len(y) - 1):
            y1, y2 = y[i], y[i + 1]
            if (y1 >= thr >= y2) or (y1 <= thr <= y2):
                if y2 == y1:
                    hit = float(r[i])
                else:
                    t = (thr - y1) / (y2 - y1)
                    hit = float(r[i] + t * (r[i + 1] - r[i]))
                break
        out[thr] = hit
    return out


def direction_resolution(dr_grid, direction):
    """Look up directional resolution (Å) for a 3D direction vector.

    `dr_grid` is the (zenith 0..90, azimuth 0..360) array from the DR polar
    plot. Directions are antipodally symmetric (a Fourier shell direction and
    its opposite carry the same information), so |z| folds onto one hemisphere.
    """
    x, y, z = [float(v) for v in direction[:3]]
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return None
    x, y, z = x / n, y / n, z / n
    zenith = math.degrees(math.acos(min(1.0, abs(z))))
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    nz, na = dr_grid.shape
    iz = min(nz - 1, max(0, int(round(zenith * (nz - 1) / 90.0))))
    ia = int(round(azimuth * na / 360.0)) % na
    val = float(dr_grid[iz, ia])
    return val if np.isfinite(val) else None


def fake_fso(sampling_rate=1.0, n=120, drop_from=5.7, drop_to=4.0):
    """Placeholder FSO curve + Bingham p-value.

    Returns dict with:
      freq      spatial frequency (1/Å), ascending
      res       resolution (Å) = 1/freq
      fso       Fourier Shell Occupancy, ~1 at low freq falling to ~0
      bingham   Bingham test p-value curve (plotted red dashed)
      crossings {0.9: Å, 0.5: Å, 0.1: Å}
    The width of the fall-off (drop_from -> drop_to) is the anisotropy band.
    """
    nyquist = 1.0 / (2.0 * sampling_rate)
    freq = np.linspace(1.0 / 40.0, nyquist, n)
    res = 1.0 / freq
    mid = 0.5 * (drop_from + drop_to)
    k = max(1e-6, (drop_from - drop_to) / 5.0)
    fso = 1.0 / (1.0 + np.exp(-(res - mid) / k))        # ~1 at coarse res
    fso = np.clip(fso, 0.0, 1.0)
    bingham = np.clip(1.0 - 0.85 * (1.0 - fso), 0.0, 1.0)
    return {"freq": freq, "res": res, "fso": fso, "bingham": bingham,
            "crossings": fso_crossings(res, fso)}


def fake_dr(best=4.0, worst=6.7, n_zenith=91, n_azimuth=360):
    """Placeholder directional resolution grid for the polar plot.

    Shape (zenith 0..90, azimuth 0..360), values in Å. Smoothly anisotropic:
    best along the pole, worst near the equator, with a mild azimuthal ripple.
    """
    zen = np.radians(np.linspace(0, 90, n_zenith))[:, None]
    azi = np.radians(np.linspace(0, 360, n_azimuth, endpoint=False))[None, :]
    t = np.sin(zen) ** 2 * (0.85 + 0.15 * np.cos(2 * azi))
    return (best + (worst - best) * t).astype(np.float32)


def fake_monodir(map1, mask, sampling_rate=1.0, significance=0.95,
                 res_step=0.5, fast=False, premasked=False, volume_radius=-1.0,
                 monores_map=None, progress_callback=None):
    """Placeholder MonoDir. Returns a dict of per-voxel metric arrays (float32),
    NaN outside the mask. Mirrors the real algorithm's direction sweep so the
    progress bar wiring is identical once the real port drops in.
    """
    shape = map1.shape
    inside = mask > 0

    # The real MonoDir sweeps a set of directions on the projection sphere;
    # loop over a plausible count so progress reports per-direction.
    n_directions = 5 if fast else 20
    for i in range(n_directions):
        time.sleep(0.02)   # simulate per-direction work — REMOVE with real algo
        if progress_callback is not None:
            progress_callback((i + 1) / n_directions)

    rng = np.random.default_rng()
    base = 2.0 + 10.0 * rng.random(shape)            # ~2..12 A
    spread = 0.5 + 2.0 * rng.random(shape)           # per-voxel anisotropy
    highest = base
    lowest = base + spread
    out = {
        "highest":   highest,
        "lowest":    lowest,
        "mean_res":  0.5 * (highest + lowest),
        "radial":    base + 0.5 * spread * rng.random(shape),
        "azimuthal": base + 0.5 * spread * rng.random(shape),
        "doa":       np.clip(spread / (lowest + 1e-9), 0.0, 1.0),
    }
    for k, v in out.items():
        v = v.astype(np.float32)
        v[~inside] = np.nan
        out[k] = v
    return out
