""" paths.py
Exposes common paths useful for manipulating datasets and generating figures.
"""
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the project folder
src = root

# Absolute path to the `data` folder (contains datasets)
data = src / "data"

# Absolute path to the `scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"

# Absolute path to the `figures` folder (contains generated figures)
figures = src / "figures"

# Absolute path to the  `results` folder
results = src / "results"