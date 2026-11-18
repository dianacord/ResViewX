from setuptools import setup

setup(
    name="ChimeraX-ResViewX",
    version="1.0",
    packages=["chimerax.resviewx"],
    package_dir={"chimerax.resviewx": "src"},
    package_data={"chimerax.resviewx": ["bundle_info.xml"]},
)
