import setuptools

setuptools.setup(
    name="AmpOpt",
    version="dev",
    packages=["ampopt"],
    entry_points={
        "console_scripts": ["ampopt=ampopt.cli:app", "hpopt=ampopt.cli:app"],
    },
)
