# ChimeraX-FSC

A ChimeraX plugin for cryo-EM map validation: **global Fourier Shell Correlation (FSC)** and **local resolution** analysis, with an embedded plotting panel.

## Features

### Global FSC tab
- Compute FSC between two half-maps (**Map vs Map**) or between a map and a fitted PDB model (**Map vs PDB**, simulated via `molmap`).
- Reports resolution at an adjustable threshold (gold-standard 0.143 for half-maps, 0.5 for map-to-model).
- Spatial-frequency (1/Å) x-axis; parallelized FFTs; per-shell values logged.
- Auto pixel size from map metadata; save the plot as an image or the data as CSV.

### Local Resolution tab
- Per-voxel local resolution (currently a fake prototype function — real algorithm to be dropped in).
- Interactive histogram with a matching color scale.
- Colors a copy of the map surface by local resolution using a matplotlib `rainbow` palette (so the map and histogram colors match).

## Requirements
- ChimeraX 1.11
- No PyTorch dependency — FSC math is pure NumPy.

## Install (developer mode)
```bash
cd my_bundle
ChimeraX --nogui --cmd "devel install . exit true"
```

Then open **Tools → Volume Data → FSC Plot**, or run `fsc #1 #2` from the command line.

## Project layout
```
src/
├── __init__.py       # BundleAPI entry point
├── cmd.py            # `fsc` command (headless)
├── fsc_core.py       # Pure-numpy FSC math
├── localres_core.py  # Local resolution (prototype)
└── tool.py           # Qt GUI panel (both tabs)
```
