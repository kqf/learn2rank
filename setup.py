from setuptools import setup, find_packages

setup(
    name="ranking",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pointwise=model.main:pointwise",
            "pairwise=model.main:pairwise",
        ],
    },
)
