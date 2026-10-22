# ChimeraX-FSC

A [UCSF ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) plugin that brings cryo-EM map
validation into the viewer where maps are actually inspected: **global Fourier Shell
Correlation**, **local resolution**, and **directional resolution**, each with an
interactive plot and map colouring.

The algorithms are reimplemented in pure NumPy/SciPy, so nothing beyond ChimeraX is
required at runtime

---

## Requirements

| | |
|---|---|
| ChimeraX | 1.11 (tested on 1.11.1, macOS) |
| Python | the one bundled with ChimeraX (3.11) |
| Packages | `numpy`, `scipy`, `matplotlib`, Qt — all ship with ChimeraX |

Nothing extra to install for normal use. The optional validation scripts in `scripts/`
run outside ChimeraX and additionally need `mrcfile` (see
[Reproducing the validation](#reproducing-the-validation)).

---

## Installation

Clone the repository and install it as a ChimeraX bundle:

```bash
git clone https://github.com/dianacord/ChimeraX-FSC.git
cd ChimeraX-FSC
ChimeraX --nogui --cmd "devel install . exit true"
```

If `ChimeraX` is not on your `PATH`, use the full path to the executable, for example on
macOS:

```bash
/Applications/ChimeraX-1.11.1.app/Contents/MacOS/ChimeraX --nogui --cmd "devel install . exit true"
```

> **Note.** If ChimeraX is already running, or the bundle is already installed, you will
> see `Deferring bundle installation … ChimeraX must be restarted`. That is expected:
> **restart ChimeraX once** and the install completes.

**Verify the install** — start ChimeraX and run:

```
usage fsc
```

It should print the command's arguments. The panel is at
**Tools → Volume Data → FSC Plot**.

To uninstall: `toolshed uninstall ChimeraX-FSC`.

---

## Quick start

Open two half maps and a mask, then:

```
open emd_70903_half_map_1.map
open emd_70903_half_map_2.map
open emd_70903_msk_1.map
fsc #1 #2 mask #3
```

which prints, to both the Log and the terminal:

```
FSC: emd_70903_half_map_1.map vs emd_70903_half_map_2.map (mask emd_70903_msk_1.map)
  Box            : 400 x 400 x 400
  Pixel size     : 0.8240 Å/vox  (from emd_70903_half_map_1.map header)
  Threshold      : 0.143
  Resolution     : 1.93 Å
```

Test data: [EMD-70903](https://www.ebi.ac.uk/emdb/EMD-70903) half maps and mask, from EMDB.

---

## The `fsc` command

```
fsc #1 #2                     two half maps; pixel size from the map header
fsc #1,2                      same thing
fsc #1 #2 mask #3             masked (recommended; solvent noise worsens the result)
fsc #1 #2 pixelSize 0.824     override the pixel size
fsc #1 #2 threshold 0.5       map-to-model threshold
fsc #1 #2 plot false          number only, no panel
fsc #1 #2 table true          also log the per-shell table
```

It returns the resolution, so it can be scripted:

```python
from chimerax.core.commands import run
resolution = run(session, "fsc #1 #2 mask #3")
```

---

## Features

### Global FSC
- **Map vs Map** (two half maps) or **Map vs Atomic model** (a map simulated from the
  model with `molmap`, resampled onto the same grid).
- Resolution at an adjustable threshold: 0.143 for half maps, 0.5 for map-to-model.
- X axis is spatial frequency, labelled in Å. Hover or click the curve to read off
  resolution, frequency and FSC at that point.
- Pixel size auto-filled from the map header; save the plot as an image or the data as CSV.

### Local Resolution
- Per-voxel resolution via **MonoRes**, so you can see which parts of the map are well
  resolved. The colour varies across the structure and does not change as you rotate.
- Requires a map and a mask; a second half map is optional and improves the noise estimate.
- Histogram with a matching colour scale (viridis or blue-white-red), export as image or CSV.
- Colours a **copy** of the map, leaving your original untouched.

### Directional Resolution
- **FSO** (Fourier Shell Occupancy) with the **Bingham test**, measuring how resolution
  varies with direction — which reveals anisotropy from preferred particle orientation.
- **FSO curve**: fraction of directions per shell still above the FSC threshold, with the
  Bingham p-value and the anisotropy band shaded between the 0.9 and 0.1 crossings. The
  wider the band, the more anisotropic the map.
- **DR polar plot**: resolution per direction, rot around the circle and tilt as the radius.
- **Colour by viewing direction**: paints the map one colour, the resolution along the
  direction you are looking from, updating as you rotate.
- **Colour by directional reliability**: the same measurement inverted, on an arbitrary
  0–1 scale. The inversion is deliberate — the resolution limiting what you see lies
  perpendicular to the direction you look along, so the direction with the worst measured
  resolution is the view you can trust most.
- 321 directions by default, 81 in fast mode.

Every tab has **Help** and **Citations** buttons, and a **?** beside each input.

---

## Performance

Measured on EMD-70903 (400³ box, 0.824 Å/vox), Apple Silicon, CPU only:

| | Time | Peak RAM |
|---|---|---|
| Global FSC | < 1 s | small |
| Local resolution (MonoRes) | ~80 s | moderate |
| Directional (FSO, 321 directions) | ~19 s | ~4.8 GB |
| Directional (FSO, 81 directions) | ~9 s | ~3.1 GB |

FFTs are spread across CPU cores with `scipy.fft(workers=-1)`, and the long computations
run on a background thread so the interface stays responsive. Rotating the map in
directional mode is a table lookup, so it stays smooth regardless of map size.

---

## Reproducing the validation

The ports were checked against their reference implementations. To repeat this you need
an environment outside ChimeraX with `numpy`, `scipy` and `mrcfile`:

```bash
conda create -n monores python=3.10 numpy scipy mrcfile -c conda-forge
conda activate monores
```

**Local resolution (MonoRes).** Run the original PyTorch implementation
([Vilax/monores](https://github.com/Vilax/monores)) and compare:

```bash
python monores.py --vol half1.map --vol2 half2.map --mask mask.map \
       --sampling_rate 0.824 --maxRes 15 -o reference/
python scripts/run_numpy_monores.py half1.map half2.map mask.map 0.824 15 ours/
python scripts/compare_maps.py reference/monoresResolutionMap.mrc ours/monoresResolutionMap.mrc
```

Result on EMD-70903: identical resolved-voxel set, **0 Å** mean and maximum difference.

**Directional resolution (FSO).** There is no public Python reference — the original is
compiled C++ in [I2PC/xmipp](https://github.com/I2PC/xmipp)
(`src/xmipp/libraries/reconstruction/resolution_fso.cpp`). Validation is therefore:

- the global FSC computed inside the FSO pipeline gives **1.93 Å** on EMD-70903, matching
  the resolution EMDB reports;
- synthetic controls behave correctly — an isotropic map gives a pole/equator ratio of
  1.00, and an imposed missing wedge gives 5.93, with the FSO band widening from 0.66 to
  1.95 Å.

`scripts/compare_fso.py` compares our FSO curve, Bingham curve and DR grid against Xmipp
`.xmd` output, for when a reference becomes available.

---

## Project layout

```
src/
├── __init__.py           # BundleAPI entry point
├── cmd.py                # the `fsc` command
├── tool.py               # Qt panel: all three tabs
├── fsc_core.py           # FSC, pure NumPy
├── localres_core.py      # local resolution wrapper
├── monores_core.py       # MonoRes, pure NumPy
├── directional_core.py   # directional helpers + FSO entry point
├── fso_core.py           # FSO, pure NumPy
└── fso_directions.py     # direction tables transcribed from Xmipp
scripts/                  # validation, run outside ChimeraX
├── compare_maps.py       # voxel-by-voxel comparison of two maps
├── compare_fso.py        # compare FSO output against Xmipp .xmd
├── run_numpy_monores.py  # run the NumPy MonoRes and write MRCs
└── monores_numpy.py      # standalone copy of the MonoRes port
```

---

## Citing

If you use this plugin, please cite the method you used as well as the plugin. The
**Citations** button in each tab lists them and copies them to the clipboard.

- **This plugin** — Cordovez, D., Vilas, J.L. & Sánchez-García, R. (in preparation).
  *Interactive platform for structural resolution analysis in cryogenic electron microscopy.*
- **FSC 0.143 criterion** — Rosenthal, P.B. & Henderson, R. (2003). *J Mol Biol* 333(4),
  721–745. doi:10.1016/j.jmb.2003.07.013
- **MonoRes** — Vilas, J.L. et al. (2018). *Structure* 26(2), 337–344.e4.
  doi:10.1016/j.str.2017.12.018
- **FSO and the Bingham test** — Vilas, J.L., Tagare, H.D., Vargas, J., Carazo, J.M. &
  Sorzano, C.O.S. (2023). *Nat Methods* 20, 1021–1024. doi:10.1038/s41592-023-01874-3
- **ChimeraX** — Meng, E.C. et al. (2023). *Protein Sci* 32(11), e4792.
  doi:10.1002/pro.4792

---

## Use of AI assistance

This software was developed with the assistance of a large language model
(Claude, Anthropic), used for code implementation, porting the reference algorithms to
NumPy, and drafting documentation. All scientific decisions, validation and
interpretation are the authors'.

The AI is **not an author**: it cannot take responsibility for the accuracy or integrity
of the work. Commits made with AI assistance carry an
`Assisted-by:` trailer, following the convention adopted by the Linux kernel, Fedora and
LLVM, which deliberately frames the model as a tool rather than a co-author. Earlier
commits in this repository use a `Co-Authored-By:` trailer, which was the tool's default;
the history has been left intact rather than rewritten, and the meaning is the same —
assistance, not authorship.

All AI-assisted output was reviewed by the authors, and the ported algorithms were
validated against their reference implementations as described in
[Reproducing the validation](#reproducing-the-validation).

---

## License

See the reference implementations for the licensing of the algorithms ported here
([Vilax/monores](https://github.com/Vilax/monores),
[I2PC/xmipp](https://github.com/I2PC/xmipp), GPL).
