from pathlib import Path
from typing import Dict, Sequence, List

import numpy as np
from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData
from tqdm.contrib import tenumerate
import re
import json

rng = np.random.default_rng()

# Path to root of bdqm-hyperparam-tuning repo
bdqm_hpopt_path = Path(__file__).resolve().parents[1]

def get_electron_densities() -> Dict[str, Path]:
    gaussians_path = bdqm_hpopt_path / "data/GMP/valence_gaussians"
    regex = r"(.+?)_"
    return {re.match(regex, p.name).group(1): p for p in gaussians_path.iterdir()}


def get_sigmas() -> Dict[int, List[int]]:
    with open(bdqm_hpopt_path / "data/GMP/sigmas.json") as f:
        d = json.load(f)
    return {int(k): v for k,v in d.items()}

ELECTRON_DENSITIES = get_electron_densities()
SIGMAS = get_sigmas()

class GMPTransformer:
    """Scikit-learn compatible wrapper for GMP descriptor."""
    def __init__(self, n_gaussians, n_mcsh, cutoff, **a2d_kwargs):
        sigmas=SIGMAS[n_gaussians]

        def mcsh_groups(i): return [1] if i == 0 else list(range(1,i+1))

        MCSHs = {
            "MCSHs": {
                str(i): {
                    "groups": mcsh_groups(i),
                    "sigmas": sigmas,
                } for i in range(n_mcsh)
            },
            "atom_gaussians": ELECTRON_DENSITIES,
            "cutoff": cutoff,
        }

        self.elements = list(ELECTRON_DENSITIES.keys())
        self.a2d = AtomsToData(
            descriptor=GMP(MCSHs=MCSHs, elements=self.elements),
            **a2d_kwargs
        )
        self.setup = ("gmp", MCSHs, {"cutoff": cutoff}, self.elements)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = len(X)
        return [
            self.a2d.convert(img, idx=idx)
            for idx, img in tenumerate(
                X, desc="Calculating descriptors", total=n, unit=" images"
            )
        ]


class ScalerTransformer:
    """Scikit-learn compatible wrapper for FeatureScaler and TargetScaler."""

    def __init__(self, cls, *args, **kwargs):
        self._cls = cls
        self._cls_args = args
        self._cls_kwargs = kwargs

    def fit(self, X, y=None):
        self.scaler = self._cls(X, *self._cls_args, **self._cls_kwargs)
        return self

    def transform(self, X):
        return self.scaler.norm(X)


def split_data(data: Sequence, valid_pct: float):
    """
    Split data into train & validation sets.

    `valid_pct` sets how big the validation set is; setting `valid_pct=0.1` will
    mean that 10% of the data goes into the valid dataset.

    Args:
        data: the data to split
        valid_pct: a float between 0 and 1, size of the validation split
    Returns:
        train_data: the training split
        valid_data: the validation split
    """
    n = len(data)
    indices = np.arange(n)
    n_valid = int(round(valid_pct * n))
    valid_indices = rng.choice(indices, size=n_valid, replace=False)
    train_indices = np.setdiff1d(indices, valid_indices)

    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    return train_data, valid_data
