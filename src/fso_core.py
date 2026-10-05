"""Pure-numpy port of Xmipp's FSO (Fourier Shell Occupancy).

Reference: ``ProgFSO`` in I2PC/xmipp,
``src/xmipp/libraries/reconstruction/resolution_fso.{h,cpp}``.

Pipeline (mirrors the C++):

  prepareData              mask both halves, fftshift
  FourierTransform         one rFFT per half map (the only transforms)
  defineFrequencies        per-axis digital frequencies, |u| map, shell index
  arrangeFSC_and_fscGlobal flatten the sub-Nyquist voxels into 1-D arrays and
                           compute the global FSC.  Done ONCE
  generateDirections       321 (or 81) directions on the sphere
  fscDir_fast  x N         per direction: cone weighting + per-shell
                           accumulation -> directional FSC -> resolution
  anistropyParameter       FSO = fraction of directions per shell above the
                           FSC threshold
  Bingham test             per-shell scatter matrices -> p-value curve
  resolutionDistribution   cone-weighted average of the directional
                           resolutions onto a 360 x 91 grid

Not ported (not needed for the plugin's three views): the 3DFSC volume and the
directional filter.
"""

import math
import numpy as np

from .fso_directions import DIRECTIONS_321, DIRECTIONS_81

# Multi-core FFT when scipy is present, same wrapper as monores_core.
try:
    from scipy import fft as _sfft

    def _rfftn(x):
        return _sfft.rfftn(x, workers=-1)
except ImportError:
    def _rfftn(x):
        return np.fft.rfftn(x)


# Lower incomplete gamma, gammainc(x, 5), tabulated exactly as the C++ does in
# ProgFSO::incompleteGammaFunction (index = round(2x), clamped to 0..40).
_INCOMPLETE_GAMMA = np.array([
    0.0, 0.00017212, 0.0036598, 0.018576, 0.052653, 0.10882, 0.18474, 0.27456,
    0.37116, 0.4679, 0.55951, 0.64248, 0.71494, 0.77633, 0.82701, 0.86794,
    0.90037, 0.92564, 0.94504, 0.95974, 0.97075, 0.97891, 0.9849, 0.98925,
    0.9924, 0.99465, 0.99626, 0.9974, 0.99819, 0.99875, 0.99914, 0.99941,
    0.9996, 0.99973, 0.99982, 0.99988, 0.99992, 0.99994, 0.99996, 0.99997,
    0.99998], dtype=np.float64)


def _incomplete_gamma(x):
    idx = np.rint(2.0 * np.asarray(x, dtype=np.float64)).astype(np.int64)
    np.clip(idx, 0, 40, out=idx)
    return _INCOMPLETE_GAMMA[idx]


def _fft_idx2digfreq(n_idx, n_real):
    """FFT_IDX2DIGFREQ over an index vector (same convention as monores_core)."""
    idx = np.arange(n_idx)
    half = int(n_real) >> 1
    num = np.where(idx <= half, idx, idx - int(n_real))
    return num.astype(np.float64) / float(n_real)


def _precompute(half1, half2, mask, sampling_rate):
    """prepareData + defineFrequencies + arrangeFSC_and_fscGlobal.

    Returns a dict of flat per-voxel arrays covering only the voxels with
    |u| <= 0.5, plus the shell axis and the global FSC.
    """
    h1 = np.asarray(half1, dtype=np.float64)
    h2 = np.asarray(half2, dtype=np.float64)
    if h1.shape != h2.shape:
        raise ValueError(f"Half maps differ in shape: {h1.shape} vs {h2.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != h1.shape:
            raise ValueError(f"Mask shape {m.shape} does not match maps {h1.shape}")
        h1 = h1 * m
        h2 = h2 * m

    # CenterFFT(..., true) before the transform. For FSC this only adds a linear
    # phase that cancels in z1*conj(z2), but we keep it to match the C++.
    h1 = np.fft.fftshift(h1)
    h2 = np.fft.fftshift(h2)

    nz, ny, nx = h1.shape
    ft1 = _rfftn(h1)
    ft2 = _rfftn(h2)
    del h1, h2

    uz = _fft_idx2digfreq(ft1.shape[0], nz).reshape(-1, 1, 1)
    uy = _fft_idx2digfreq(ft1.shape[1], ny).reshape(1, -1, 1)
    ux = _fft_idx2digfreq(ft1.shape[2], nx).reshape(1, 1, -1)

    u = np.sqrt(uz * uz + uy * uy + ux * ux)          # |u| on the rFFT grid
    keep = u <= 0.5                                   # reachable frequencies

    # The rFFT keeps both halves of the j == 0 plane, so those coefficients
    # appear twice as conjugate pairs. The C++ drops the duplicates; without
    # this the j == 0 plane would be double counted.
    jz = np.zeros_like(keep)
    jz[:, :, 0] = True
    uy_b = np.broadcast_to(uy, u.shape)
    uz_b = np.broadcast_to(uz, u.shape)
    dup = (jz & (uy_b < 0)) | (jz & (uy_b == 0) & (uz_b < 0))
    keep &= ~dup

    uu = u[keep]
    # |u| == 0 only at DC; its direction is undefined, so leave it as a zero
    # vector rather than dividing by zero. Shell 0 is dropped from the output.
    inv = np.where(uu > 0, 1.0 / np.where(uu > 0, uu, 1.0), 0.0)
    shell_idx = np.rint(uu * nx).astype(np.int64)
    n_shells = nx // 2 + 1
    np.clip(shell_idx, 0, n_shells - 1, out=shell_idx)

    z1 = ft1[keep]
    z2 = ft2[keep]
    del ft1, ft2

    data = {
        "shell_idx": shell_idx,
        "real_z1z2": np.real(z1 * np.conj(z2)).astype(np.float32),
        "absz1_sq": (np.abs(z1) ** 2).astype(np.float32),
        "absz2_sq": (np.abs(z2) ** 2).astype(np.float32),
        "fx": (np.broadcast_to(ux, u.shape)[keep] * inv).astype(np.float32),
        "fy": (np.broadcast_to(uy, u.shape)[keep] * inv).astype(np.float32),
        "fz": (np.broadcast_to(uz, u.shape)[keep] * inv).astype(np.float32),
        "n_shells": n_shells,
        "nx": nx,
    }
    del z1, z2

    # Global FSC, same per-shell accumulation as fsc_core.compute_fsc
    num = np.bincount(shell_idx, weights=data["real_z1z2"].astype(np.float64),
                      minlength=n_shells)
    d1 = np.bincount(shell_idx, weights=data["absz1_sq"].astype(np.float64),
                     minlength=n_shells)
    d2 = np.bincount(shell_idx, weights=data["absz2_sq"].astype(np.float64),
                     minlength=n_shells)
    with np.errstate(invalid="ignore", divide="ignore"):
        data["global_fsc"] = np.nan_to_num(num / np.sqrt(d1 * d2))
    # freq[i] = i / (xvoldim * sampling), as in arrangeFSC_and_fscGlobal
    data["freq"] = np.arange(n_shells) / (nx * sampling_rate)
    return data


def _direction_fsc(pre, dx, dy, dz, cos_angle, aux, threshold, sampling_rate):
    """fscDir_fast: directional FSC for one cone, and its resolution."""
    n_shells = pre["n_shells"]
    cosine = np.abs(pre["fx"] * dx + pre["fy"] * dy + pre["fz"] * dz)
    inside = cosine >= cos_angle
    # Gaussian taper on the cone; zero outside so one bincount covers everything
    w = np.where(inside, np.exp(-((cosine - 1.0) ** 2) * aux), 0.0).astype(np.float64)

    si = pre["shell_idx"]
    num = np.bincount(si, weights=w * pre["real_z1z2"], minlength=n_shells)
    d1 = np.bincount(si, weights=w * pre["absz1_sq"], minlength=n_shells)
    d2 = np.bincount(si, weights=w * pre["absz2_sq"], minlength=n_shells)

    fsc = np.maximum(0.0, num / (np.sqrt(d1 * d2) + 1e-38))

    # First shell past index 2 where the FSC drops to the threshold
    below = np.nonzero((np.arange(n_shells) > 2) & (fsc <= threshold))[0]
    if below.size:
        i = int(below[0])
        ff = i / (pre["nx"] * sampling_rate)
        resol = 1.0 / ff if ff > 0 else float("nan")
    else:
        resol = float("nan")
    return fsc, resol


def _resolution_distribution(res_dir, dirs_rad, cos_angle, aux,
                             n_rot=360, n_tilt=91):
    """resolutionDistribution: cone-weighted average of the per-direction
    resolutions onto a dense (tilt, rot) grid. Returned as (zenith, azimuth)
    to match the polar plot and the viewing-direction lookup."""
    rot = np.radians(np.arange(n_rot)).reshape(-1, 1)      # (n_rot, 1)
    tilt = np.radians(np.arange(n_tilt)).reshape(1, -1)    # (1, n_tilt)
    st = np.sin(tilt)
    gx = st * np.cos(rot)
    gy = st * np.sin(rot)
    gz = np.broadcast_to(np.cos(tilt), gx.shape)

    w_sum = np.zeros(gx.shape, dtype=np.float64)
    wt_sum = np.zeros(gx.shape, dtype=np.float64)
    for (r, t), res in zip(dirs_rad, res_dir):
        if not np.isfinite(res):
            continue
        dx = math.sin(t) * math.cos(r)
        dy = math.sin(t) * math.sin(r)
        dz = math.cos(t)
        cosine = np.abs(gx * dx + gy * dy + gz * dz)
        inside = cosine >= cos_angle
        if not inside.any():
            continue
        cw = np.where(inside, np.exp(-((cosine - 1.0) ** 2) * aux), 0.0)
        w_sum += cw * res
        wt_sum += cw

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(wt_sum > 0, w_sum / wt_sum, np.nan)
    return grid.T.astype(np.float32)        # -> (tilt/zenith, rot/azimuth)


def compute_fso(half1, half2, mask=None, sampling_rate=1.0,
                cone_angle=17.0, threshold=0.143, fast=False,
                n_threads=None, progress_callback=None):
    """Run FSO on two half maps.

    Returns a dict with:
      freq        spatial frequency per shell (1/A)
      res         resolution per shell (A)
      fso         Fourier Shell Occupancy, 1 -> 0
      bingham     Bingham test p-value curve
      crossings   {0.9: A, 0.5: A, 0.1: A}
      dr_grid     (zenith 0..90, azimuth 0..359) resolution in A
      global_fsc  global FSC per shell (cross-check against fsc_core)
      directions  the (rot, tilt) table used, in degrees
    """
    from concurrent.futures import ThreadPoolExecutor
    from .directional_core import fso_crossings

    pre = _precompute(half1, half2, mask, sampling_rate)
    n_shells = pre["n_shells"]

    dirs_deg = DIRECTIONS_81 if fast else DIRECTIONS_321
    dirs_rad = [(math.radians(r), math.radians(t)) for r, t in dirs_deg]
    n_dirs = len(dirs_rad)

    cos_angle = math.cos(math.radians(cone_angle))
    # aux = 4 / (cos(ang)-1)^2, the taper width used by the C++
    aux = 4.0 / ((cos_angle - 1.0) ** 2) if cos_angle != 1.0 else 0.0

    res_dir = np.full(n_dirs, np.nan)
    occupancy = np.zeros(n_shells, dtype=np.float64)   # aniParam
    scatter = np.zeros((n_shells, 3, 3), dtype=np.float64)

    if n_threads is None:
        import os
        n_threads = max(1, min(4, (os.cpu_count() or 2)))

    done = [0]

    def one(k):
        r, t = dirs_rad[k]
        dx = math.sin(t) * math.cos(r)
        dy = math.sin(t) * math.sin(r)
        dz = math.cos(t)
        fsc, resol = _direction_fsc(pre, dx, dy, dz, cos_angle, aux,
                                    threshold, sampling_rate)
        d = np.array([dx, dy, dz])
        return k, fsc, resol, np.outer(d, d)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for k, fsc, resol, T in ex.map(one, range(n_dirs)):
            res_dir[k] = resol
            hit = fsc >= threshold
            occupancy += hit                      # anistropyParameter
            scatter[hit] += T                     # Bingham accumulation
            done[0] += 1
            if progress_callback is not None:
                progress_callback(done[0] / n_dirs)

    # Bingham test: mean scatter matrix per shell, then
    # S = 0.5 p (p+2) (2n) (trace(T^2) - 1/p), p = 3
    counts = occupancy.copy()
    bingham_stat = np.zeros(n_shells, dtype=np.float64)
    nz = counts > 0
    if nz.any():
        T = scatter[nz] / counts[nz][:, None, None]
        trT2 = np.einsum("nij,nji->n", T, T)
        p = 3.0
        bingham_stat[nz] = 0.5 * p * (p + 2) * (2 * counts[nz]) * (trT2 - 1.0 / p)
    bingham = _incomplete_gamma(bingham_stat)

    # FSO curve: fraction of directions above the threshold, first 5 shells
    # forced to 1 exactly as the C++ does
    fso = occupancy / float(n_dirs)
    fso[:5] = 1.0

    dr_grid = _resolution_distribution(res_dir, dirs_rad, cos_angle, aux)

    freq = pre["freq"]
    with np.errstate(divide="ignore"):
        res = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-38), np.inf)

    # Drop the DC shell: it carries no resolution and would break the plots
    return {
        "freq": freq[1:],
        "res": res[1:],
        "fso": fso[1:],
        "bingham": bingham[1:],
        "crossings": fso_crossings(res[1:], fso[1:]),
        "dr_grid": dr_grid,
        "global_fsc": pre["global_fsc"][1:],
        "directions": dirs_deg,
    }
