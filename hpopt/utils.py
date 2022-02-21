from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Sequence

from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData
from ase import Atom
from tqdm.contrib import tenumerate
import numpy as np

# Path to data directory (../data)
bdqm_hpopt_path = Path(__file__).resolve().parents[1]
# Path to amptorch git repo (assumed to be ../../amptorch)
amptorch_path = Path(__file__).resolve().parents[2] / "amptorch"


def get_path_to_gaussian(element: str) -> Path:
    """Get path to gaussian file given element name."""
    gaussians_path = amptorch_path / "examples/GMP/valence_gaussians"
    return next(gaussians_path.glob(f"{element}_*"))


def get_all_elements(traj: Iterable[Iterable[Atom]]) -> List[str]:
    """Get list of elements given iterable of images."""
    return list({atom.symbol for image in traj for atom in image})


MCSH = Dict[str, Dict[str, Iterable[float]]]


def gen_mcshs(sigmas: Iterable[float], n: int) -> MCSH:
    def mcsh(i):
        groups = [1] if i == 0 else list(np.arange(i) + 1)
        return {"groups": groups, "sigmas": sigmas}
    return {str(i): mcsh(i) for i in range(n)}


class GMPTransformer:
    """Scikit-learn compatible wrapper around GMP featurizing code."""

    def __init__(
        self,
        sigmas: Sequence[float],
        atom_gaussians: Dict[str, Path],
        cutoff: float,
        **a2d_kwargs,
    ):
        MCSHs = {
            "MCSHs": gen_mcshs(sigmas, 3),
            "atom_gaussians": atom_gaussians,
            "cutoff": cutoff,
        }
        self.descriptor = GMP(MCSHs=MCSHs, elements=list(atom_gaussians.keys()))
        self.a2d = AtomsToData(descriptor=self.descriptor, **a2d_kwargs)

    @property
    def setup(self) -> Tuple[str, MCSH, Any, Sequence[str]]:
        return ("gmp", self.descriptor.MCSHs, None, self.descriptor.elements)

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
