"""
Script to pre-generate features and save in lmdb format.

This lets us save time during hyperparameter optimization.
"""

import pickle
from pathlib import Path
from typing import Sequence, Tuple

import lmdb
import numpy as np
import torch
from amptorch.preprocessing import FeatureScaler, TargetScaler
from ase.io import Trajectory
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from utils import (GMPTransformer, ScalerTransformer, bdqm_hpopt_path,
                   get_all_elements, get_path_to_gaussian)


def mk_feature_pipeline(train_imgs: Sequence) -> Tuple[Sequence, Pipeline]:
    """
    Compute scaled feature values for train data

    Args:
        train_imgs (Sequence): the training data
    Returns:
        train_feats (list): the features for the train data
        preprocess_pipeline (Pipeline): the actual pipeline object
    """

    elements = get_all_elements(train_imgs)
    atom_gaussians = {el: get_path_to_gaussian(el) for el in elements}

    featurizer_pipeline = Pipeline(
        steps=[
            (
                "GMP",
                GMPTransformer(
                    atom_gaussians=atom_gaussians,
                    # sigmas=[0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4],
                    sigmas=np.exp(np.linspace(-2, 1.5, 8)),
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


def main(data_dir: Path, train_fname: str, valid_fname: str, test_fname: str) -> None:
    """Calculate features and save to lmdb."""
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

    print("\nFitting pipeline & computing features...")
    train_feats, featurizer = mk_feature_pipeline(train_imgs)
    valid_feats = featurizer.transform(valid_imgs)
    test_feats = featurizer.transform(test_imgs)

    print("\nSaving data...")
    save_to_lmdb(train_feats, featurizer, train_lmdb_path)
    save_to_lmdb(test_feats, featurizer, test_lmdb_path)
    save_to_lmdb(valid_feats, featurizer, valid_lmdb_path)


if __name__ == "__main__":
    main(
        data_dir=bdqm_hpopt_path / "data",
        train_fname="train.traj",
        valid_fname="valid.traj",
        test_fname="oc20_300_test.traj",
    )
