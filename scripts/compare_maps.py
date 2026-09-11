#!/usr/bin/env python3
"""Compare two MonoRes local-resolution maps voxel-by-voxel.

Usage:
    python compare_maps.py mapA.mrc mapB.mrc

Reports resolved-voxel counts, mean/max/RMS absolute difference (A),
Pearson correlation, and the fraction of voxels agreeing within tolerance.
Run inside an environment that has numpy + mrcfile (e.g. the `monores` conda env).
"""
import sys
import numpy as np
import mrcfile


def load(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float64)


def main():
    if len(sys.argv) != 3:
        print("usage: python compare_maps.py mapA.mrc mapB.mrc")
        sys.exit(1)
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    if a.shape != b.shape:
        print("Shape mismatch: %s vs %s" % (a.shape, b.shape))
        sys.exit(1)

    both = (a > 0) & (b > 0)
    print("=== resolved voxels ===")
    print("map A:", int((a > 0).sum()))
    print("map B:", int((b > 0).sum()))
    print("both :", int(both.sum()))

    da = a[both]
    db = b[both]
    diff = da - db
    print("\n=== agreement (voxels resolved in both) ===")
    print("mean abs diff (A): %.6f" % np.mean(np.abs(diff)))
    print("max  abs diff (A): %.6f" % np.max(np.abs(diff)))
    print("RMS  diff     (A): %.6f" % np.sqrt(np.mean(diff ** 2)))
    if da.size > 1:
        print("pearson r        : %.6f" % np.corrcoef(da, db)[0, 1])
    for tol in (0.001, 0.01, 0.1, 0.25):
        print("  within %.3f A : %.3f%%" % (tol, np.mean(np.abs(diff) <= tol) * 100))

    print("\nbitwise identical:", np.array_equal(a, b))
    print("allclose(1e-6)   :", np.allclose(a, b, atol=1e-6))


if __name__ == "__main__":
    main()
