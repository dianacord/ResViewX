#!/usr/bin/env python3
"""Run the pure-numpy MonoRes port and write MRC outputs (for comparison/viewing).

Usage:
    python run_numpy_monores.py half1.map half2.map mask.map sampling maxRes outdir

Writes monoresResolutionMap.mrc and monoresResolutionChimera.mrc to outdir.
Run inside an env with numpy + mrcfile (e.g. the `monores` conda env).
"""
import os
import sys
import time
import numpy as np
import mrcfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monores_numpy import run_monores


def load(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float64)


def write(path, arr, sampling):
    data = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = float(sampling)


def main():
    if len(sys.argv) != 7:
        print("usage: python run_numpy_monores.py half1 half2 mask sampling maxRes outdir")
        sys.exit(1)
    half1, half2, mask, sampling, max_res, outdir = sys.argv[1:]
    sampling = float(sampling); max_res = float(max_res)
    os.makedirs(outdir, exist_ok=True)

    v1 = load(half1); v2 = load(half2); m = load(mask)
    t0 = time.time()
    res_map, chimera, refined, list_res = run_monores(
        v1, v2, m, sampling_rate=sampling, max_res=max_res, step=0.25,
        significance=0.95,
        progress_callback=lambda f: print("  progress %.0f%%" % (f * 100), flush=True))
    print("numpy MonoRes took %.1f s" % (time.time() - t0))

    write(os.path.join(outdir, "monoresResolutionMap.mrc"), res_map, sampling)
    write(os.path.join(outdir, "monoresResolutionChimera.mrc"), chimera, sampling)
    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()
