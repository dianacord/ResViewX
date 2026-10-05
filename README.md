# ChimeraX-FSC

A ChimeraX plugin for cryo-EM map validation: **global Fourier Shell Correlation (FSC)**, **local resolution**, and **directional resolution** analysis, with an embedded plotting panel.

## Features

### Global FSC tab
- Compute FSC between two half-maps (**Map vs Map**) or between a map and a fitted atomic model (**Map vs Atomic model**, simulated via `molmap`).
- Reports resolution at an adjustable threshold (gold-standard 0.143 for half-maps, 0.5 for map-to-model).
- Spatial-frequency (1/Å) x-axis; parallelized FFTs; per-shell values logged.
- Auto pixel size from map metadata; save the plot as an image or the data as CSV.

### Local Resolution tab
- Per-voxel local resolution with MonoRes, ported to pure NumPy and validated voxel-by-voxel against the original implementation.
- Multi-core FFTs (`scipy.fft`) and a background thread keep it fast and the UI responsive.
- Interactive histogram with a matching color scale.
- Colors a copy of the map surface by local resolution using a matplotlib `rainbow` palette, with a matching color key in the 3D view.

### Directional Resolution tab (in progress)
Based on Xmipp's **FSO** (Fourier Shell Occupancy, `xmipp_resolution_fso`). Takes two half maps + a mask, a cone angle and an FSC threshold.
- **FSO curve**: fraction of directions per resolution shell still above the FSC threshold (green), with the Bingham test p-value (red dashed) and the anisotropy band shaded between the FSO = 0.9 and 0.1 crossings. The wider the band, the more anisotropic the map.
- **DR polar plot**: resolution as a function of direction — azimuth around the circle, zenith angle as the radius.
- **Color by viewing direction**: paints the map a single colour, the directional resolution along the current viewing direction, updating as the map is rotated.
- The computation itself is a placeholder until `xmipp_resolution_fso` is ported.

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
├── fsc_core.py          # Pure-numpy FSC math
├── localres_core.py     # Local resolution (MonoRes wrapper)
├── monores_core.py      # MonoRes, pure NumPy
├── directional_core.py  # Directional resolution / FSO (placeholder)
└── tool.py           # Qt GUI panel (both tabs)
```
