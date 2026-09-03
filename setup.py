from setuptools import setup

setup(
    name="ChimeraX-FSC",
    version="1.0",
    packages=["chimerax.fsc"],
    package_dir={"chimerax.fsc": "src"},
    package_data={"chimerax.fsc": ["bundle_info.xml"]},
)
