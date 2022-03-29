"""
Functions and classes for preprocessing data.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import lmdb
import torch
from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from ase import Atoms
from ase.io import Trajectory
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm.contrib import tenumerate

from hpopt.utils import bdqm_hpopt_path, split_data


def get_electron_densities() -> Dict[str, Path]:
    gaussians_path = bdqm_hpopt_path / "data/GMP/valence_gaussians"
    regex = r"(.+?)_"
    return {re.match(regex, p.name).group(1): p for p in gaussians_path.iterdir()}


def get_sigmas() -> Dict[int, List[int]]:
    with open(bdqm_hpopt_path / "data/GMP/sigmas.json") as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


ELECTRON_DENSITIES = get_electron_densities()
SIGMAS = get_sigmas()


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
        sigmas = SIGMAS[n_gaussians]

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
            "atom_gaussians": ELECTRON_DENSITIES,
            "cutoff": cutoff,
        }

        self.elements = list(ELECTRON_DENSITIES.keys())
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


def mk_feature_pipeline(train_imgs: Sequence) -> Tuple[Sequence, Pipeline]:
    """
    Compute scaled feature values for train data.

    Args:
        train_imgs (Sequence): the training data
    Returns:
        train_feats (list): the features for the train data
        preprocess_pipeline (Pipeline): the actual pipeline object
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

    train_feats = featurizer_pipeline.fit_transform(train_imgs)
    return train_feats, featurizer_pipeline


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


def create_lmdbs(train_fname: str, valid_fname: str, test_fname: str) -> None:
    """Calculate features and save to lmdb."""
    data_dir = bdqm_hpopt_path / "data"
    train_lmdb_path = data_dir / "train.lmdb"
    test_lmdb_path = data_dir / "test.lmdb"
    valid_lmdb_path = data_dir / "valid.lmdb"

    for path in [train_lmdb_path, test_lmdb_path, valid_lmdb_path]:
        if path.exists():
            print(f"{path} already exists, aborting")
            return

    torch.set_default_tensor_type(torch.DoubleTensor)
    print("Loading data...")
    train_imgs = Trajectory(data_dir / train_fname)
    valid_imgs = Trajectory(data_dir / valid_fname)
    test_imgs = Trajectory(data_dir / test_fname)

    print("\nFitting pipeline & computing train features...")
    train_feats, featurizer = mk_feature_pipeline(train_imgs)
    print("Computing valid features...")
    valid_feats = featurizer.transform(valid_imgs)
    print("Computing test features...")
    test_feats = featurizer.transform(test_imgs)

    print("\nSaving train data...")
    save_to_lmdb(train_feats, featurizer, train_lmdb_path)
    print("\nSaving valid data...")
    save_to_lmdb(test_feats, featurizer, test_lmdb_path)
    print("\nSaving valid data...")
    save_to_lmdb(valid_feats, featurizer, valid_lmdb_path)


def save_to_traj(imgs: Iterable[Atoms], path: Path):
    """Save `imgs`."""
    with Trajectory(path, "w") as t:
        for img in tqdm(imgs, desc="Writing data to .traj"):
            t.write(img)


def create_validation_split(
    train_fname: str, valid_split: int, train_out_fname: str, valid_out_fname: str
) -> None:
    """Split train into train & valid, save to .traj files."""
    data_dir = bdqm_hpopt_path / "data"

    valid_traj_path = data_dir / valid_out_fname
    train_traj_path = data_dir / train_out_fname

    for path in [train_traj_path, valid_traj_path]:
        if path.exists():
            print(f"{path} already exists, aborting")
            return

    print("Loading and splitting data...")
    train_imgs = Trajectory(data_dir / train_fname)
    train_imgs, valid_imgs = split_data(train_imgs, valid_pct=valid_split)

    print("\nSaving data...")
    save_to_traj(train_imgs, train_traj_path)
    save_to_traj(valid_imgs, valid_traj_path)
