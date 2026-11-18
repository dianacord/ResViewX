"""Command registration for the ResViewX plugin.

Usage from the ChimeraX command line:

    fsc #1 #2                                   two half maps, pixel size from header
    fsc #1,2                                    same thing
    fsc #1 #2 pixelSize 0.824                   override the pixel size
    fsc #1 #2 mask #3 threshold 0.5             masked, map-to-model threshold
    fsc #1 #2 plot false                        number only, no panel
    fsc #1 #2 table true                        also log the per-shell table

The resolution is printed to the console and written to the Log, and returned
so it can be used from Python:

    from chimerax.core.commands import run
    resolution = run(session, "fsc #1 #2")
"""

from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg
from chimerax.core.errors import UserError
from chimerax.map import MapsArg


def register_command(session):
    # One positional specifier holding both half maps. ChimeraX's spec parsing
    # is greedy, so two separate positional MapsArg arguments cannot work:
    # "#1 #2" is swallowed whole by the first one.
    desc = CmdDesc(
        required=[("maps", MapsArg)],
        keyword=[
            ("pixel_size", FloatArg),
            ("mask", MapsArg),
            ("threshold", FloatArg),
            ("plot", BoolArg),
            ("table", BoolArg),
        ],
        synopsis="Compute FSC between two half-maps and report the resolution",
    )
    register("fsc", desc, fsc_command, logger=session.logger)


def fsc_command(session, maps, pixel_size=None, mask=None,
                threshold=0.143, plot=None, table=False):
    """Compute the FSC between two half maps and report the resolution."""
    import numpy as np
    from .fsc_core import compute_fsc, find_resolution

    if maps is None or len(maps) != 2:
        n = 0 if maps is None else len(maps)
        raise UserError(
            f"fsc needs exactly two maps, got {n}. "
            "Usage: fsc #1 #2  (or fsc #1,2)")
    map1, map2 = maps[0], maps[1]

    if mask is not None and len(mask) != 1:
        raise UserError("Specify exactly one volume for mask")
    mask_vol = mask[0] if mask else None

    # Pixel size from the map header unless the user overrides it. Getting this
    # wrong rescales every resolution, so say which one was used.
    if pixel_size is None:
        try:
            pixel_size = float(map1.data.step[0])
        except Exception:
            pixel_size = 1.0
        pixel_source = f"{map1.name} header"
    else:
        pixel_source = "command line"
    if pixel_size <= 0:
        raise UserError("pixelSize must be positive")

    vol1 = map1.data.matrix().astype(np.float32, copy=True)
    vol2 = map2.data.matrix().astype(np.float32, copy=True)
    if vol1.shape != vol2.shape:
        raise UserError(
            f"Shape mismatch: {map1.name} {vol1.shape} vs {map2.name} {vol2.shape}")

    mask_arr = None
    if mask_vol is not None:
        mask_arr = mask_vol.data.matrix().astype(np.float32, copy=True)
        if mask_arr.shape != vol1.shape:
            raise UserError(
                f"Mask shape {mask_arr.shape} does not match the maps {vol1.shape}")

    session.logger.status("Computing FSC …")
    fsc_values = compute_fsc(vol1, vol2, mask_arr)
    resolution = find_resolution(fsc_values, pixel_size, threshold=threshold)

    # Shell axis, shared by the table and the summary
    n = len(fsc_values)
    bins = np.linspace(0.0, 0.5, n + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    lines = [
        f"FSC: {map1.name} vs {map2.name}"
        + (f" (mask {mask_vol.name})" if mask_vol is not None else ""),
        f"  Box            : {vol1.shape[0]} x {vol1.shape[1]} x {vol1.shape[2]}",
        f"  Pixel size     : {pixel_size:.4f} Å/vox  (from {pixel_source})",
        f"  Threshold      : {threshold:.3f}",
    ]
    if resolution is not None:
        lines.append(f"  Resolution     : {resolution:.2f} Å")
    else:
        lines.append(f"  Resolution     : not reached — FSC stays above "
                     f"{threshold:.3f} out to Nyquist "
                     f"({2.0 * pixel_size:.2f} Å)")

    if table:
        lines.append("")
        lines.append(f"  {'Shell':>5}  {'Freq':>8}  {'1/Å':>10}  "
                     f"{'Resolution (Å)':>15}  {'FSC':>7}")
        for i in range(1, n):
            spatial = centers[i] / pixel_size
            res_i = pixel_size / centers[i] if centers[i] > 0 else float("inf")
            res_s = f"{res_i:.3f}" if np.isfinite(res_i) else "inf"
            lines.append(f"  {i:>5}  {centers[i]:>8.4f}  {spatial:>10.4f}  "
                         f"{res_s:>15}  {fsc_values[i]:>7.4f}")

    report = "\n".join(lines)
    is_gui = bool(getattr(session, "ui", None) and session.ui.is_gui)
    session.logger.info(report)          # Log panel in the GUI; stdout otherwise
    if is_gui:
        print(report, flush=True)        # also the terminal, when there is one

    session.logger.status(
        "FSC done." if resolution is None else f"FSC resolution {resolution:.2f} Å")

    # Open the panel too when there is a GUI, unless told not to
    if plot is None:
        plot = is_gui
    if plot:
        try:
            from .tool import FSCTool
            # Reuse the panel if one is already open instead of stacking a new
            # one on every command.
            existing = session.tools.find_by_class(FSCTool)
            tool = existing[0] if existing else FSCTool(session, "ResViewX")
            # Show what was actually computed, not whatever the dropdowns held
            tool.set_fsc_inputs(map1, map2, mask_vol, pixel_size, threshold)
            tool.compute_and_show(map1, map2, pixel_size, mask_vol)
            try:
                tool.tool_window.shown = True
            except Exception:
                pass
        except Exception as e:
            session.logger.warning(f"Could not open the FSC panel: {e}")

    return resolution
