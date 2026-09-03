"""Command registration for the FSC plugin."""

from chimerax.core.commands import CmdDesc, register, FloatArg
from chimerax.map import MapsArg


def register_command(session):
    desc = CmdDesc(
        keyword=[
            ("half_map1", MapsArg),
            ("half_map2", MapsArg),
            ("pixel_size", FloatArg),
            ("mask", MapsArg),
        ],
        required_arguments=["half_map1", "half_map2"],
        synopsis="Compute FSC between two half-maps and show the plot",
    )
    register("fsc", desc, fsc_command, logger=session.logger)


def fsc_command(session, half_map1, half_map2, pixel_size=1.0, mask=None):
    if len(half_map1) != 1:
        from chimerax.core.errors import UserError
        raise UserError("Specify exactly one volume for halfMap1")
    if len(half_map2) != 1:
        from chimerax.core.errors import UserError
        raise UserError("Specify exactly one volume for halfMap2")

    from .tool import FSCTool
    tool = FSCTool(session, "FSC Plot")
    tool.compute_and_show(
        half_map1[0],
        half_map2[0],
        pixel_size,
        mask[0] if mask else None,
    )
