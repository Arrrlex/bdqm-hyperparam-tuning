"""
Script to pre-generate features and save in lmdb format.

This lets us save time during hyperparameter optimization.
"""

import pickle
from pathlib import Path
from typing import Tuple, Sequence

import lmdb
import numpy as np
from amptorch.preprocessing import FeatureScaler, TargetScaler
from ase.io import Trajectory
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import bdqm_hpopt_path, get_path_to_gaussian, get_all_elements, GMPTransformer, ScalerTransformer


def featurize(
    data_dir: Path, train_fname: str, test_fname: str
) -> Tuple[Sequence, Sequence, Pipeline]:
    """
    Compute scaled feature values for train and test data.

    The train and test live in the `data_dir` directory.

    Returns:
        train_feats (list): the features for the train data
        test_feats (list): the features for the test data
        params (dict): the parameters for the preprocess pipeline
        preprocess_pipeline (Pipeline): the actual pipeline object
    """
    print(f"Loading data")
    train_images = Trajectory(data_dir / train_fname)
    test_images = Trajectory(data_dir / test_fname)

    elements = get_all_elements(train_images)
    atom_gaussians = {el: get_path_to_gaussian(el) for el in elements}

    preprocess_pipeline = Pipeline(
        steps=[
            (
                "GMP",
                GMPTransformer(
                    atom_gaussians=atom_gaussians,
                    sigmas=[0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4],
                    # sigmas=np.exp(np.linspace(-2, 1.5, 8)),
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

    print("\nFitting and transforming train data")
    train_feats = preprocess_pipeline.fit_transform(train_images)
    print("\nTransforming test data")
    test_feats = preprocess_pipeline.transform(test_images)

    return train_feats, test_feats, preprocess_pipeline


def save_to_lmdb(feats: Sequence, pipeline: Pipeline, lmdb_path: Path) -> None:
    """
    Save the features and pipeline information to the lmdb file.

    Args:
        feats: the features to save
        params: the parameters of the preprocess pipeline
        pipeline: the preprocess pipeline
    """

    to_save = {
        **{str(i): f for i, f in enumerate(feats)},
        **{
            "length": len(feats),
            "feature_scaler": pipeline.named_steps["FeatureScaler"].scaler,
            "target_scaler": pipeline.named_steps["TargetScaler"].scaler,
            "descriptor_setup": pipeline.named_steps["GMP"].setup,
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


def main(data_dir: Path, train_fname: str, test_fname: str) -> None:
    """Calculate features and save to lmdb."""
    train_lmdb_path = (data_dir / train_fname).with_suffix(".lmdb")
    test_lmdb_path = (data_dir / test_fname).with_suffix(".lmdb")

    for path in [train_lmdb_path, test_lmdb_path]:
        if path.exists():
            print(f"{path} already exists, aborting")
            return

    train_feats, test_feats, pipeline = featurize(data_dir, train_fname, test_fname)

    save_to_lmdb(train_feats, pipeline, train_lmdb_path)
    save_to_lmdb(test_feats, pipeline, test_lmdb_path)


if __name__ == "__main__":
    main(
        data_dir=bdqm_hpopt_path / "data",
        train_fname="oc20_3k_train.traj",
        test_fname="oc20_300_test.traj",
    )
