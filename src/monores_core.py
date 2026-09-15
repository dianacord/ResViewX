#!/usr/bin/env python3
"""Pure-numpy port of MonoRes (Vilax/monores torch implementation).

Faithful translation: torch is used there only as an array/FFT library, so
every op maps 1:1 to numpy/scipy. float64 throughout to match the C++/torch.
"""
from __future__ import annotations
import math
import numpy as np

PI = math.pi
DBL_MIN = 2.2250738585072014e-308
MAX_RESOLUTION_CANDIDATES = 100000

# Use scipy's FFT with multi-core support when available (workers=-1 spreads
# each transform across all CPU cores). Falls back to numpy if scipy is absent.
try:
    from scipy import fft as _sfft

    def rfftn(x, axes=None):
        return _sfft.rfftn(x, axes=axes, workers=-1)

    def irfftn(x, s=None, axes=None):
        return _sfft.irfftn(x, s=s, axes=axes, workers=-1)
except ImportError:
    def rfftn(x, axes=None):
        return np.fft.rfftn(x, axes=axes)

    def irfftn(x, s=None, axes=None):
        return np.fft.irfftn(x, s=s, axes=axes)


# ---------------------------------------------------------------------------
# scalar helpers (c_functions.py)
# ---------------------------------------------------------------------------

def icdf_gauss(p: float) -> float:
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        z = t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
        return -z
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)


def _c_round(x: float) -> int:
    if math.isnan(x) or math.isinf(x):
        raise ValueError("cannot round %r to an FFT index" % x)
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))


def _c_div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0:
            return float("nan")
        return math.copysign(float("inf"), a) * math.copysign(1.0, b)
    return a / b


def _c_sqrt(x: float) -> float:
    if x < 0.0:
        return float("nan")
    return math.sqrt(x)


def digfreq2fft_idx(freq: float, size: int) -> int:
    idx = _c_round(size * freq)
    if idx < 0:
        idx += int(size)
    return idx


def fft_idx2digfreq(idx: int, size: int) -> float:
    if size <= 1:
        return 0.0
    idx = int(idx); size = int(size)
    num = idx if idx <= (size >> 1) else (-size + idx)
    return num / float(size)


def fft_idx2digfreq_t(idx: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(idx, dtype=np.float64)
    half = int(size) >> 1
    num = np.where(idx <= half, idx, idx - int(size))
    return num.astype(np.float64) / float(size)


# ---------------------------------------------------------------------------
# geometry / frequency grids (utils.py)
# ---------------------------------------------------------------------------

def xmipp_centered_radius2(shape) -> np.ndarray:
    nz, ny, nx = shape
    k = np.arange(nz, dtype=np.float64) - (nz // 2)
    i = np.arange(ny, dtype=np.float64) - (ny // 2)
    j = np.arange(nx, dtype=np.float64) - (nx // 2)
    return (k * k).reshape(-1, 1, 1) + (i * i).reshape(1, -1, 1) + (j * j).reshape(1, 1, -1)


def fourier_freqs_3d(real_shape):
    nz, ny, nx = real_shape
    fz_n, fy_n, fx_n = nz, ny, nx // 2 + 1

    def _freq_vector(n_fourier, n_real):
        idx = np.arange(n_fourier)
        vec = fft_idx2digfreq_t(idx, n_real)
        vec[0] = 1e-38
        return vec

    freq_fourier_z = _freq_vector(fz_n, nz)
    freq_fourier_y = _freq_vector(fy_n, ny)
    freq_fourier_x = _freq_vector(fx_n, nx)

    uz = fft_idx2digfreq_t(np.arange(fz_n), nz).reshape(-1, 1, 1)
    uy = fft_idx2digfreq_t(np.arange(fy_n), ny).reshape(1, -1, 1)
    ux = fft_idx2digfreq_t(np.arange(fx_n), nx).reshape(1, 1, -1)
    u2 = np.broadcast_to(uz * uz + uy * uy + ux * ux, (fz_n, fy_n, fx_n)).copy()
    iu = 1.0 / np.sqrt(np.where(u2 > 0, u2, np.ones_like(u2)))
    iu[0, 0, 0] = 1e38

    return iu, freq_fourier_x, freq_fourier_y, freq_fourier_z


def real_gaussian_filter(volume: np.ndarray, sigma: float) -> np.ndarray:
    nz, ny, nx = volume.shape
    spec = rfftn(volume, axes=(-3, -2, -1))
    uz = fft_idx2digfreq_t(np.arange(nz), nz).reshape(-1, 1, 1)
    uy = fft_idx2digfreq_t(np.arange(ny), ny).reshape(1, -1, 1)
    ux = fft_idx2digfreq_t(np.arange(nx // 2 + 1), nx).reshape(1, 1, -1)
    w2 = uz * uz + uy * uy + ux * ux
    spec = spec * np.exp(-(PI * PI) * w2 * (sigma * sigma))
    return irfftn(spec, s=(nz, ny, nx), axes=(-3, -2, -1))


# ---------------------------------------------------------------------------
# mask preparation
# ---------------------------------------------------------------------------

def protein_radius_and_volume(mask: np.ndarray):
    r2 = xmipp_centered_radius2(mask.shape)
    sel = mask == 1
    n_voxels = int(sel.sum())
    max_r2 = float(r2[sel].max()) if n_voxels > 0 else 0.0
    radius = int(math.sqrt(max_r2))
    return radius, n_voxels


def find_cliff_value(input_vol, radius, mask, rsmooth=0.0):
    nx = input_vol.shape[2]
    radiuslimit = nx // 2
    r2 = xmipp_centered_radius2(input_vol.shape)
    last_std2 = 1e-38
    for rad in range(radius, radiuslimit):
        inf = float(rad * rad); sup = float((rad + 1) * (rad + 1))
        shell = (r2 < sup) & (r2 >= inf)
        n = float(shell.sum())
        if n == 0.0:
            std2 = float("nan")
        else:
            values = input_vol[shell]
            s = float(values.sum()); s2 = float((values * values).sum())
            mean = s / n
            std2 = s2 / n - mean * mean
        if _c_div(std2, last_std2) < 0.01:
            radiuslimit = rad - 1
            break
        last_std2 = std2
    raux = float(radiuslimit) - float(rsmooth)
    raux *= raux
    mask[r2 >= raux] = -1
    return radiuslimit


def exclude_area(mask, mask_excl, half_maps_given, noise_only_in_halves):
    if half_maps_given:
        if noise_only_in_halves:
            inside = mask == 1
            mask[inside & (mask_excl == 1)] = -1
            mask[~inside] = -1
    else:
        mask[(mask == 1) & (mask_excl == 1)] = -1


# ---------------------------------------------------------------------------
# monogenic amplitude
# ---------------------------------------------------------------------------

def monogenic_amplitude_3d_fourier(myfftV, iu, fx, fy, fz, real_shape):
    nz, ny, nx = real_shape
    F = myfftV
    F_aux = (-1j) * F * iu
    amplitude = irfftn(F, s=(nz, ny, nx), axes=(-3, -2, -1))
    amplitude = amplitude * amplitude
    for u in (fx.reshape(1, 1, -1), fy.reshape(1, -1, 1), fz.reshape(-1, 1, 1)):
        comp = irfftn(u * F_aux, s=(nz, ny, nx), axes=(-3, -2, -1))
        amplitude = amplitude + comp * comp
    return np.sqrt(amplitude)


def amplitude_mono_sig_3d_lpf(myfftV, freq, freqH, freqL, iu, fx, fy, fz, real_shape):
    nz, ny, nx = real_shape
    un = 1.0 / iu

    ideltal = _c_div(PI, freq - freqH)
    ramp = 0.5 * (1.0 + np.cos((un - freq) * ideltal))
    in_ramp = (un >= freqH) & (un <= freq)
    above = un > freq
    gain = np.where(in_ramp, ramp, np.zeros_like(ramp))
    gain = np.where(above, np.ones_like(gain), gain)
    F = myfftV * gain
    F_aux = (-1j) * F * iu

    amplitude = irfftn(F, s=(nz, ny, nx), axes=(-3, -2, -1))
    amplitude = amplitude * amplitude

    for u in (fx.reshape(1, 1, -1), fz.reshape(-1, 1, 1), fy.reshape(1, -1, 1)):
        comp = irfftn(u * F_aux, s=(nz, ny, nx), axes=(-3, -2, -1))
        amplitude = amplitude + comp * comp

    amplitude = np.sqrt(amplitude)

    spec = rfftn(amplitude, axes=(-3, -2, -1))
    raised_w = _c_div(PI, freqL - freq)
    lp_ramp = 0.5 * (1.0 + np.cos(raised_w * (un - freq)))
    in_lp = (un <= freqL) & (un >= freq)
    over = un > freqL
    lp_gain = np.where(in_lp, lp_ramp, np.ones_like(lp_ramp))
    lp_gain = np.where(over, np.zeros_like(lp_gain), lp_gain)
    spec = spec * lp_gain
    return irfftn(spec, s=(nz, ny, nx), axes=(-3, -2, -1))


def refine_mask(myfftV, iu, fx, fy, fz, mask, real_shape):
    amplitude = monogenic_amplitude_3d_fourier(myfftV, iu, fx, fy, fz, real_shape)
    amplitude = real_gaussian_filter(amplitude, 4.0)
    signal_sel = mask >= 1
    noise_sel = mask == 0
    n_s = int(signal_sel.sum()); n_n = int(noise_sel.sum())
    if n_n == 0:
        raise RuntimeError("refiningMask: noise region empty")
    if n_s == 0:
        raise RuntimeError("refiningMask: signal region empty")
    noise_values = amplitude[noise_sel]
    noise_sorted = np.sort(noise_values)
    thr_idx = int(n_n * 0.95)
    thr_idx = min(thr_idx, n_n - 1)
    threshold_first_estimation = float(noise_sorted[thr_idx])
    drop = signal_sel & (amplitude < threshold_first_estimation)
    mask[drop] = 0
    return int((mask >= 1).sum())


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _mean_var(values):
    n = float(values.size)
    s = float(values.sum()); s2 = float((values * values).sum())
    mean = s / n
    return mean, s2 / n - mean * mean, n


def _nearest_rank(values, significance):
    n = values.size
    srt = np.sort(values)
    idx = int(n * significance)
    if idx >= n:
        idx = n - 1
    return float(srt[idx])


def statistics_in_binary_mask(volS, volN, mask, significance):
    sig = volS[mask > 0]
    noi = volN[mask >= 0]
    if sig.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if noi.size == 0:
        raise RuntimeError("no noise voxels (mask >= 0)")
    meanS, sdS2, NS = _mean_var(sig)
    meanN, sdN2, NN = _mean_var(noi)
    thr95 = _nearest_rank(noi, significance)
    return meanS, sdS2, meanN, sdN2, thr95, NS, NN


def statistics_in_out_binary_mask(volS, mask, significance):
    sig = volS[mask >= 1]
    noi = volS[mask == 0]
    if sig.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if noi.size == 0:
        raise RuntimeError("no noise voxels (mask == 0)")
    meanS, sdS2, NS = _mean_var(sig)
    meanN, sdN2, NN = _mean_var(noi)
    thr95 = _nearest_rank(noi, significance)
    return meanS, sdS2, meanN, sdN2, thr95, NS, NN


# ---------------------------------------------------------------------------
# local-resolution assignment
# ---------------------------------------------------------------------------

def set_local_resolution(amplitudeMS, mask, resolution_map, threshold_noise,
                         resolution, resolution_2, half_maps_given):
    sel = mask >= 1
    passed = sel & (amplitudeMS > threshold_noise)
    failed = sel & ~(amplitudeMS > threshold_noise)
    mask[passed] = 1
    resolution_map[passed] = resolution
    mask[failed] += 1
    give_up = failed & (mask > 2)
    mask[give_up] = 0 if half_maps_given else -1
    resolution_map[give_up] = resolution_2


# ---------------------------------------------------------------------------
# frequency sweep
# ---------------------------------------------------------------------------

def resolution2eval(freq_step, sampling, max_res, volsize):
    count_res = 0
    last_fourier_idx = -1
    nyquist = 2.0 * sampling
    guard = 0
    while True:
        guard += 1
        if guard > MAX_RESOLUTION_CANDIDATES:
            return
        resolution = max_res - count_res * freq_step
        freq = _c_div(sampling, resolution)
        count_res += 1
        try:
            fourier_idx = digfreq2fft_idx(freq, volsize)
        except ValueError:
            return
        aux_frequency = fft_idx2digfreq(fourier_idx, volsize)
        freq = aux_frequency
        if fourier_idx == last_fourier_idx:
            continue
        last_fourier_idx = fourier_idx
        resolution = _c_div(sampling, aux_frequency)
        if resolution < nyquist:
            return
        freqL = _c_div(sampling, resolution + freq_step)
        try:
            fourier_idx_2 = digfreq2fft_idx(freqL, volsize)
        except ValueError:
            fourier_idx_2 = None
        if fourier_idx_2 == fourier_idx:
            if fourier_idx > 0:
                freqL = fft_idx2digfreq(fourier_idx - 1, volsize)
            else:
                freqL = _c_div(sampling, resolution + freq_step)
        yield resolution, freq, freqL


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------

def post_process_local_resolutions(resolution_vol, list_res, mask, sampling):
    nyquist = 2.0 * sampling
    filtered_map = resolution_vol.copy()

    last_res = list_res[-1] - 0.001
    n_count = int((filtered_map >= last_res).sum())
    values = filtered_map[filtered_map > last_res]
    if n_count == 0 or values.size == 0:
        raise RuntimeError("no voxel reached the last analysed resolution")
    if values.size < n_count:
        pad = np.zeros(n_count - values.size, dtype=values.dtype)
        values = np.concatenate([values, pad])
    values = np.sort(values)
    filling_value = float(values[int(0.5 * n_count)])

    last_res = list_res[-1]
    bad = filtered_map < last_res
    filtered_map[bad] = filling_value
    mask[bad] = 0
    mask[~bad] = 1

    filtered_map = real_gaussian_filter(filtered_map, 3.0)

    inside = mask > 0
    revert = inside & ((filtered_map > resolution_vol) | (filtered_map < nyquist))
    filtered_map = np.where(revert, resolution_vol, filtered_map)

    chimera_map = filtered_map.copy()
    resolution_map_out = np.where(mask == 0, np.zeros_like(filtered_map), filtered_map)
    return resolution_map_out, chimera_map, mask


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------

def run_monores(v1, v2, mask_np, sampling_rate, max_res, step=0.25,
                significance=0.95, noiseonlyinhalves=False, gaussian=False,
                progress_callback=None):
    dt = np.float64
    v1 = np.asarray(v1, dtype=dt)
    if v2 is not None:
        v2 = np.asarray(v2, dtype=dt)
        input_vol = 0.5 * (v1 + v2)
        noise_vol = (v1 - v2) / 2.0
        fftN = rfftn(noise_vol, axes=(-3, -2, -1))
        half_maps_given = True
    else:
        input_vol = v1
        fftN = None
        half_maps_given = False

    real_shape = tuple(input_vol.shape)
    nz, ny, nx = real_shape
    mask = np.trunc(mask_np).astype(np.int64)

    radius, n_voxels_original_mask = protein_radius_and_volume(mask)
    find_cliff_value(input_vol, radius, mask, 0.0)

    if half_maps_given and noiseonlyinhalves:
        mask[mask < 1] = -1

    fftV = rfftn(input_vol, axes=(-3, -2, -1))
    iu, fx, fy, fz = fourier_freqs_3d(real_shape)

    freq_step = step if step >= 0.25 else 0.25

    critical_z = icdf_gauss(significance)
    max_meanS = -DBL_MIN
    cut_value = 0.025
    sampling = sampling_rate
    min_res = 2.0 * sampling
    volsize = nx

    if not noiseonlyinhalves:
        n_voxels_original_mask = refine_mask(fftV, iu, fx, fy, fz, mask, real_shape)
    if n_voxels_original_mask == 0:
        raise RuntimeError("mask contains no protein voxels after refinement")

    resolution_map = np.zeros(real_shape, dtype=dt)
    list_res = []
    iteration = 0
    do_next_iteration = True
    lefttrimming = False

    # count candidates for progress
    candidates = list(resolution2eval(freq_step, sampling, max_res, volsize))
    n_cand = max(len(candidates), 1)

    for ci, (resolution, freq, _fl) in enumerate(candidates):
        list_res.append(resolution)
        resolution_2 = list_res[0] if iteration < 2 else list_res[iteration - 2]
        freqL = freq + 0.02
        freqH = freq - 0.02
        if freqL >= 0.5:
            freqL = 0.5
        if freqH <= 0.0:
            freqH = 0.0

        amplitudeMS = amplitude_mono_sig_3d_lpf(fftV, freq, freqH, freqL,
                                                iu, fx, fy, fz, real_shape)
        if half_maps_given:
            amplitudeMN = amplitude_mono_sig_3d_lpf(fftN, freq, freqH, freqL,
                                                    iu, fx, fy, fz, real_shape)
            meanS, sdS2, meanN, sdN2, thr95, NS, NN = statistics_in_binary_mask(
                amplitudeMS, amplitudeMN, mask, significance)
        else:
            meanS, sdS2, meanN, sdN2, thr95, NS, NN = statistics_in_out_binary_mask(
                amplitudeMS, mask, significance)

        if (NS / n_voxels_original_mask) < cut_value:
            do_next_iteration = False
            resolved = resolution_map != 0
            mask[~resolved] = 0
            mask[resolved] = 1
            lefttrimming = True
        else:
            if NS == 0:
                break
            if gaussian:
                threshold_noise = meanN + critical_z * _c_sqrt(sdN2)
            else:
                threshold_noise = thr95
            if meanS > max_meanS:
                max_meanS = meanS
            if meanS < 0.001 * max_meanS:
                break
            set_local_resolution(amplitudeMS, mask, resolution_map,
                                 threshold_noise, resolution, resolution_2,
                                 half_maps_given)
            z = _c_div(meanS - meanN, _c_sqrt(sdS2 / NS + sdN2 / NN))
            if z < critical_z:
                do_next_iteration = False
            if do_next_iteration and resolution < min_res:
                do_next_iteration = False

        iteration += 1
        if progress_callback is not None:
            progress_callback((ci + 1) / n_cand)
        if not do_next_iteration:
            break

    if not list_res:
        raise RuntimeError("No frequency could be analysed")

    if not lefttrimming:
        resolved = resolution_map != 0
        mask[~resolved] = 0
        mask[resolved] = 1

    res_map, chimera_map, refined_mask = post_process_local_resolutions(
        resolution_map, list_res, mask, sampling)
    return res_map, chimera_map, refined_mask, list_res
