"""Directional resolution (FSO) — placeholder core.

The real algorithm is xmipp's ``xmipp_resolution_fso`` (Fourier Shell
Occupancy), C++ source in I2PC/xmipp:
  src/xmipp/libraries/reconstruction/resolution_fso.{h,cpp}

Inputs (from scipion's protocol_resolution_fso.py): two half maps, a mask,
a cone angle (default 17 deg) and an FSC threshold (default 0.143).

Outputs used here:
  * FSO curve   - fraction of directions in each resolution shell whose FSC is
                  still above the threshold. Falls from 1 to 0; the width of
                  that fall is the anisotropy. Plotted with the Bingham test
                  p-value (red dashed).
  * DR grid     - resolution as a function of direction (zenith x azimuth),
                  shown as the polar plot and used to colour the map by the
                  current viewing direction.

This module currently returns FAKE data with realistic ranges so the GUI can be
built and reviewed before the real algorithm is ported.
"""

import math
import numpy as np

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


def direction_angles(direction):
    """(azimuth, zenith) in degrees for a 3D direction vector.

    Same convention as the DR polar plot: azimuth is the angle around the
    circle, zenith the angle from the pole used as the radius. Directions are
    antipodally symmetric, so |z| folds onto one hemisphere.
    Returns None for a zero vector.
    """
    x, y, z = [float(v) for v in direction[:3]]
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return None
    x, y, z = x / n, y / n, z / n
    zenith = math.degrees(math.acos(min(1.0, abs(z))))
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    return azimuth, zenith


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


def compute_directional(half1, half2, mask, sampling_rate=1.0, cone_angle=17.0,
                        threshold=0.143, fast=False, progress_callback=None):
    """Run the real FSO on two half maps.

    Returns the dict the GUI consumes: freq, res, fso, bingham, crossings,
    global_fsc, plus dr_grid (zenith x azimuth resolutions in A).
    """
    from .fso_core import compute_fso
    return compute_fso(half1, half2, mask,
                       sampling_rate=sampling_rate, cone_angle=cone_angle,
                       threshold=threshold, fast=fast,
                       progress_callback=progress_callback)


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
