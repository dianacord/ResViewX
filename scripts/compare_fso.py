#!/usr/bin/env python3
"""Compare our FSO output against an Xmipp reference.

Xmipp writes two metadata files we care about:
  fso.xmd                       the FSO curve (MDL_RESOLUTION_FREQ, _FSO, _ANISOTROPY)
  Resolution_Distribution.xmd   the DR grid (MDL_ANGLE_ROT, _ANGLE_TILT, _RESOLUTION_FRC)

Both are plain-text Xmipp metadata, so we parse them without needing xmipp installed.

Usage:
    python compare_fso.py half1.mrc half2.mrc mask.mrc sampling \\
           ref_fso.xmd ref_Resolution_Distribution.xmd [--cone 17] [--thr 0.143]

Run in an environment with numpy + mrcfile (e.g. the `monores` conda env).
"""
import os
import sys
import argparse
import numpy as np
import mrcfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _load_src():
    """Import the plugin's fso_core without needing ChimeraX."""
    import types
    import importlib.util
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    pkg = types.ModuleType("fsopkg")
    pkg.__path__ = [src]
    sys.modules["fsopkg"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"fsopkg.{name}", os.path.join(src, f"{name}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"fsopkg.{name}"] = m
        spec.loader.exec_module(m)
        return m

    load("fso_directions")
    load("directional_core")
    return load("fso_core")


def read_xmd(path):
    """Minimal Xmipp metadata reader: returns {column_name: [values]}."""
    cols, rows = [], []
    with open(path) as fh:
        in_loop = False
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("data_"):
                continue
            if s.startswith("loop_"):
                in_loop = True
                cols = []
                continue
            if s.startswith("_"):
                if in_loop:
                    cols.append(s.split()[0].lstrip("_"))
                continue
            if in_loop and cols:
                parts = s.split()
                if len(parts) == len(cols):
                    rows.append([float(p) for p in parts])
    if not cols or not rows:
        raise RuntimeError(f"No table found in {path}")
    arr = np.array(rows, dtype=float)
    return {c: arr[:, i] for i, c in enumerate(cols)}


def report(name, ours, ref):
    ours = np.asarray(ours, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = min(ours.size, ref.size)
    ours, ref = ours[:n], ref[:n]
    ok = np.isfinite(ours) & np.isfinite(ref)
    d = ours[ok] - ref[ok]
    print(f"\n=== {name} ({ok.sum()} comparable points) ===")
    if d.size == 0:
        print("  nothing comparable")
        return
    print("  mean abs diff : %.6f" % np.mean(np.abs(d)))
    print("  max  abs diff : %.6f" % np.max(np.abs(d)))
    print("  RMS  diff     : %.6f" % np.sqrt(np.mean(d ** 2)))
    if d.size > 1 and np.std(ours[ok]) > 0 and np.std(ref[ok]) > 0:
        print("  pearson r     : %.6f" % np.corrcoef(ours[ok], ref[ok])[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("half1"); p.add_argument("half2"); p.add_argument("mask")
    p.add_argument("sampling", type=float)
    p.add_argument("ref_fso")
    p.add_argument("ref_dist")
    p.add_argument("--cone", type=float, default=17.0)
    p.add_argument("--thr", type=float, default=0.143)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()

    fso_core = _load_src()

    def rd(path):
        with mrcfile.open(path, permissive=True) as m:
            return np.asarray(m.data, dtype=np.float64)

    out = fso_core.compute_fso(rd(a.half1), rd(a.half2), rd(a.mask),
                               sampling_rate=a.sampling, cone_angle=a.cone,
                               threshold=a.thr, fast=a.fast)

    ref = read_xmd(a.ref_fso)
    fso_key = next((k for k in ref if "FSO" in k.upper()), None)
    ani_key = next((k for k in ref if "ANISOTROPY" in k.upper()), None)
    if fso_key:
        report("FSO curve", out["fso"], ref[fso_key])
    if ani_key:
        report("Bingham p-value", out["bingham"], ref[ani_key])

    rd_md = read_xmd(a.ref_dist)
    rot_k = next((k for k in rd_md if "ROT" in k.upper()), None)
    tilt_k = next((k for k in rd_md if "TILT" in k.upper()), None)
    res_k = next((k for k in rd_md if "FRC" in k.upper() or "RESOLUTION" in k.upper()), None)
    if rot_k and tilt_k and res_k:
        grid = out["dr_grid"]          # (zenith, azimuth)
        rows_rot = rd_md[rot_k].astype(int)
        rows_tilt = rd_md[tilt_k].astype(int)
        ok = (rows_tilt < grid.shape[0]) & (rows_rot < grid.shape[1])
        mine = grid[rows_tilt[ok], rows_rot[ok]]
        report("DR grid", mine, rd_md[res_k][ok])


if __name__ == "__main__":
    main()
