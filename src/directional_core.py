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
