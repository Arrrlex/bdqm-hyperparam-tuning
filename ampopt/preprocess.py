"""
Functions and classes for preprocessing data.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from functools import lru_cache

import lmdb
import torch
from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from ase import Atoms
from ase.io import Trajectory
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm.contrib import tenumerate

from ampopt.utils import ampopt_path

def compute_gmp(
    train: str,
    *others: str
) -> None:
    """Compute GMP features and save to lmdb."""
    fnames = [train] + list(others)
    print(f"Creating LMDBs from files {', '.join(fnames)}")

    data_dir = ampopt_path / "data"
    lmdb_paths = [data_dir / Path(fname).with_suffix(".lmdb") for fname in fnames]

    for path in lmdb_paths:
        if path.exists():
            print(f"{path} already exists, aborting")
            return

    torch.set_default_tensor_type(torch.DoubleTensor)

    print(f"Fitting to {train}...")
    featurizer = mk_feature_pipeline(Trajectory(data_dir / train))

    for fname, lmdb_fname in zip(fnames, lmdb_paths):
        print(f"\nLooking at {fname}:")
        imgs = Trajectory(data_dir / fname)
        feats = featurizer.transform(imgs)
        save_to_lmdb(feats, featurizer, lmdb_fname)


def mk_feature_pipeline(train_imgs: Sequence) -> Pipeline:
    """
    Compute fitted featurizer given train data.

    Args:
        train_imgs (Sequence): the training data
    Returns:
        preprocess_pipeline (Pipeline): the sklearn pipeline object
    """
    featurizer_pipeline = Pipeline(
        steps=[
            (
                "GMP",
                GMPTransformer(
                    n_gaussians=8,
                    n_mcsh=3,
                    cutoff=5,
                    r_energy=True,
                    r_forces=True,
                    save_fps=False,
                    fprimes=False,
                ),
            ),
            (
                "FeatureScaler",
                ScalerTransformer(
                    FeatureScaler,
                    forcetraining=False,
                    scaling={"type": "normalize", "range": (0, 1)},
                ),
            ),
            (
                "TargetScaler",
                ScalerTransformer(
                    TargetScaler,
                    forcetraining=False,
                ),
            ),
        ]
    )

    return featurizer_pipeline.fit(train_imgs)

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


class GMPTransformer:
    """Scikit-learn compatible wrapper for GMP descriptor."""

    def __init__(self, n_gaussians, n_mcsh, cutoff, **a2d_kwargs):
        sigmas = sigmas_dict()[n_gaussians]

        def mcsh_groups(i):
            return [1] if i == 0 else list(range(1, i + 1))

        MCSHs = {
            "MCSHs": {
                str(i): {
                    "groups": mcsh_groups(i),
                    "sigmas": sigmas,
                }
                for i in range(n_mcsh)
            },
            "atom_gaussians": electron_densities(),
            "cutoff": cutoff,
        }

        self.elements = list(electron_densities().keys())
        self.a2d = AtomsToData(
            descriptor=GMP(MCSHs=MCSHs, elements=self.elements), **a2d_kwargs
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


@lru_cache
def electron_densities() -> Dict[str, Path]:
    gaussians_path = ampopt_path / "data/GMP/valence_gaussians"
    regex = r"(.+?)_"
    return {re.match(regex, p.name).group(1): p for p in gaussians_path.iterdir()}

@lru_cache
def sigmas_dict() -> Dict[int, List[int]]:
    with open(ampopt_path / "data/GMP/sigmas.json") as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


def save_to_lmdb(feats: Sequence, pipeline: Pipeline, lmdb_path: Path) -> None:
    """
    Save the features and pipeline information to the lmdb file.

    Args:
        feats: the features to save
        params: the parameters of the preprocess pipeline
        pipeline: the preprocess pipeline
    """

    feature_scaler = pipeline.named_steps["FeatureScaler"]
    target_scaler = pipeline.named_steps["TargetScaler"]
    gmp = pipeline.named_steps["GMP"]

    to_save = {
        **{str(i): f for i, f in enumerate(feats)},
        **{
            "length": len(feats),
            "feature_scaler": feature_scaler.scaler,
            "target_scaler": target_scaler.scaler,
            "descriptor_setup": gmp.setup,
            "elements": gmp.elements,
        },
    }

    db = lmdb.open(
        str(lmdb_path),
        map_size=64_393_216 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for key, val in tqdm(to_save.items(), desc="Writing data to LMDB"):
        txn = db.begin(write=True)
        txn.put(key.encode("ascii"), pickle.dumps(val, protocol=-1))
        txn.commit()

    db.sync()
    db.close()


def save_to_traj(imgs: Iterable[Atoms], path: Path):
    """Save `imgs`."""
    with Trajectory(path, "w") as t:
        for img in tqdm(imgs, desc="Writing data to .traj"):
            t.write(img)
