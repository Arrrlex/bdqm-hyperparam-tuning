import setuptools

setuptools.setup(
    name="bdqm-hyperparam-tuning",
    version="dev",
    packages=["hpopt"],
    entry_points={
        "console_scripts": ["hpopt=hpopt.cli:app"],
    },
)
