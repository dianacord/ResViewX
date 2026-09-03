"""Pure numpy FSC computation — no torch dependency."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def compute_fsc(vol1, vol2, mask=None, log_fn=None, progress_callback=None):
    def report(frac):
        if progress_callback is not None:
            progress_callback(frac)

    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes must match: {vol1.shape} vs {vol2.shape}")

    vol1 = vol1.astype(np.float32, copy=False)
    vol2 = vol2.astype(np.float32, copy=False)

    if mask is not None:
        vol1 = vol1 * mask
        vol2 = vol2 * mask
    report(0.15)

    # Run both FFTs in parallel — numpy FFT releases the GIL so threads help
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(np.fft.rfftn, vol1)
        f2 = ex.submit(np.fft.rfftn, vol2)
        ft1 = f1.result()
        ft2 = f2.result()
    report(0.55)

    freq_grid = _radial_freq_grid(vol1.shape)

    n_shells = min(vol1.shape) // 2
    bins = np.linspace(0.0, 0.5, n_shells + 1)
    shell_idx = np.digitize(freq_grid.ravel(), bins) - 1

    ft1_flat = ft1.ravel()
    ft2_flat = ft2.ravel()

    cross  = np.real(ft1_flat * np.conj(ft2_flat))
    power1 = np.abs(ft1_flat) ** 2
    power2 = np.abs(ft2_flat) ** 2
    report(0.8)

    valid = (shell_idx >= 0) & (shell_idx < n_shells)
    C_s = np.bincount(shell_idx[valid], weights=cross[valid],  minlength=n_shells)
    A_s = np.bincount(shell_idx[valid], weights=power1[valid], minlength=n_shells)
    B_s = np.bincount(shell_idx[valid], weights=power2[valid], minlength=n_shells)

    denom = np.sqrt(A_s * B_s)
    fsc_values = np.where(denom > 0, C_s / denom, 0.0)
    fsc_values[0] = 1.0  # DC component
    report(1.0)

    if log_fn is not None:
        centers = (bins[:-1] + bins[1:]) / 2.0
        lines = [f"FSC values (volume shape: {vol1.shape}, n_shells: {n_shells})",
                 f"  {'Shell':>5}  {'Freq':>8}  {'FSC':>8}"]
        for i in range(1, n_shells):
            lines.append(f"  {i:>5}  {centers[i]:>8.4f}  {fsc_values[i]:>8.4f}")
        log_fn("\n".join(lines))

    return fsc_values


def _radial_freq_grid(vol_shape):
    d, h, w = vol_shape
    fd = np.fft.fftfreq(d)
    fh = np.fft.fftfreq(h)
    fw = np.fft.rfftfreq(w)
    FD, FH, FW = np.meshgrid(fd, fh, fw, indexing='ij')
    return np.sqrt(FD ** 2 + FH ** 2 + FW ** 2)


def find_resolution(fsc_values, pixel_size, threshold=0.143):
    n = len(fsc_values)
    bins = np.linspace(0.0, 0.5, n + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    for i in range(1, n - 1):
        if fsc_values[i] >= threshold >= fsc_values[i + 1]:
            t = (threshold - fsc_values[i]) / (fsc_values[i + 1] - fsc_values[i])
            freq = centers[i] + t * (centers[i + 1] - centers[i])
            freq_per_angstrom = freq / pixel_size
            return 1.0 / freq_per_angstrom if freq_per_angstrom > 0 else None

    return None
