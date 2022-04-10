from setuptools import find_packages, setup

setup(
    name="AmpOpt",
    version="0.0.1",
    packages=["ampopt", "ampopt_cli"],
    package_dir={"": "src"},
    entry_points={
        "console_scripts": ["ampopt=ampopt_cli.cli:app", "hpopt=ampopt_cli.cli:app"],
    },
)
