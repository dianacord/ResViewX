"""Local resolution — real MonoRes (pure numpy, no torch).

Thin wrapper over monores_core.run_monores that returns a single per-voxel
local-resolution array with NaN outside the resolved region, which is what the
GUI histogram and coloring expect.
"""

import numpy as np


def monores(map1, map2=None, mask=None,
            sampling_rate=1.0, max_res=15.0, step=0.5, significance=0.95,
            progress_callback=None):
    """Compute a local-resolution map with MonoRes.

    Parameters
    ----------
    map1 : ndarray              first half map (or the full map in single-map mode)
    map2 : ndarray or None      second half map (optional)
    mask : ndarray              binary mask (REQUIRED by MonoRes)
    sampling_rate : float       pixel/voxel size in A/px (from the map header)
    max_res : float             low-resolution start of the sweep (A)
    step : float                resolution step (A), clamped to >= 0.25 internally
    significance : float        confidence for the signal/noise test

    Returns
    -------
    ndarray  local resolution per voxel in A; NaN outside the resolved region.
    """
    from .monores_core import run_monores

    if mask is None:
        raise ValueError("MonoRes requires a mask.")

    res_map, _chimera, _refined_mask, _list_res = run_monores(
        map1, map2, mask,
        sampling_rate=sampling_rate,
        max_res=max_res,
        step=step,
        significance=significance,
        progress_callback=progress_callback,
    )

    # run_monores returns 0 outside the resolved region; convert to NaN so the
    # histogram / percentile logic in the GUI ignores those voxels.
    out = np.asarray(res_map, dtype=np.float32)
    out[out <= 0] = np.nan
    return out
