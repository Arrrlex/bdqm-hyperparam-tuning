"""
Script to pre-generate features and save in lmdb format.

This lets us save time during hyperparameter optimization.
"""

import pickle
from pathlib import Path
from typing import Iterable, List

import lmdb
from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from ase import Atom
from ase.io import Trajectory
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm.contrib import tenumerate

# Path to data directory (../data)
bdqm_hpopt_path = Path(__file__).resolve().parents[1]
# Path to amptorch git repo (assumed to be ../../amptorch)
amptorch_path = Path(__file__).resolve().parents[2] / "amptorch"


# Utils

def get_path_to_gaussian(element: str) -> Path:
    """Get path to gaussian file given element name."""
    gaussians_path = amptorch_path / "examples/GMP/valence_gaussians"
    return next(gaussians_path.glob(f"{element}_*"))


def get_all_elements(traj: Iterable[Iterable[Atom]]) -> List[str]:
    """Get list of elements given iterable of images."""
    return list({atom.symbol for image in traj for atom in image})


class GMPTransformer:
    """Scikit-learn compatible wrapper around GMP featurizing code."""

    def __init__(self, sigmas, elements, atom_gaussians, cutoff, **a2d_kwargs):
        MCSHs = {
            "MCSHs": {
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
                "2": {"groups": [1, 2], "sigmas": sigmas},
                "3": {"groups": [1, 2, 3], "sigmas": sigmas},
            },
            "atom_gaussians": atom_gaussians,
            "cutoff": cutoff,
        }

        descriptor = GMP(MCSHs=MCSHs, elements=elements)

        self.a2d = AtomsToData(descriptor=descriptor, **a2d_kwargs)

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


def featurize(data_dir: Path, train_fname: str, test_fname: str):
    """
    Compute scaled feature values for train and test data.

    The train and test live in the `data_dir` directory.

    Returns:
        train_feats (list): the features for the train data
        test_feats (list): the features for the test data
        params (dict): the parameters for the preprocess pipeline
        preprocess_pipeline (Pipeline): the actual pipeline object
    """
    print(f"Loading {train_fname}")
    train_images = Trajectory(data_dir / train_fname)
    print(f"Loading {test_fname}")
    test_images = Trajectory(data_dir / test_fname)

    elements = get_all_elements(train_images)
    atom_gaussians = {el: get_path_to_gaussian(el) for el in elements}

    params = {
        "GMP": {
            "elements": elements,
            "atom_gaussians": atom_gaussians,
            "sigmas": [0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4],
            "cutoff": 5,
            "r_energy": True,
            "r_forces": True,
            "save_fps": False,
            "fprimes": False,
        },
        "FeatureScaler": {
            "forcetraining": False,
            "scaling": {"type": "normalize", "range": (0, 1)},
        },
        "TargetScaler": {
            "forcetraining": False,
        },
    }

    preprocess_pipeline = Pipeline(
        steps=[
            ("GMP", GMPTransformer(**params["GMP"])),
            (
                "FeatureScaler",
                ScalerTransformer(FeatureScaler, **params["FeatureScaler"]),
            ),
            ("TargetScaler", ScalerTransformer(TargetScaler, **params["TargetScaler"])),
        ]
    )

    print("\nFitting and transforming train data")
    train_feats = preprocess_pipeline.fit_transform(train_images)
    print("\nTransforming test data")
    test_feats = preprocess_pipeline.transform(test_images)

    return train_feats, test_feats, params, preprocess_pipeline


def save_to_lmdb(feats, params, pipeline, lmdb_path):
    """
    Save the features and pipeline information to the lmdb file.

    The following key-values are saved:
     - "row_{i}", the i-th row of features
     - "params", the params dict of the preprocess pipeline
     - "feature_scaler", the FeatureScaler object
     - "target_scaler", the TargetScaler object

    Args:
        feats: the features to save
        params: the parameters of the preprocess pipeline
        pipeline: the preprocess pipeline
    """
    to_save = {
        **{f"row_{i}": f for i, f in enumerate(feats)},
        **{
            "params": params,
            "feature_scaler": pipeline.named_steps["FeatureScaler"].scaler,
            "target_scaler": pipeline.named_steps["TargetScaler"].scaler,
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
        key = str(key).encode("ascii")
        val = pickle.dumps(val, protocol=-1)
        txn = db.begin(write=True)
        txn.put(key, val)
        txn.commit()

    db.sync()
    db.close()


def main(data_dir, train_fname, test_fname):
    """Calculate features and save to lmdb."""
    train_lmdb_path = (data_dir / train_fname).with_suffix(".lmdb")
    test_lmdb_path = (data_dir / test_fname).with_suffix(".lmdb")

    if train_lmdb_path.exists():
        print(f"{train_lmdb_path} already exists, aborting")
        return
    if test_lmdb_path.exists():
        print(f"{test_lmdb_path} already exists, aborting")
        return

    train_feats, test_feats, params, pipeline = featurize(
        data_dir, train_fname, test_fname
    )

    save_to_lmdb(train_feats, params, pipeline, train_lmdb_path)
    save_to_lmdb(test_feats, params, pipeline, test_lmdb_path)


if __name__ == "__main__":
    main(
        data_dir=bdqm_hpopt_path / "data",
        train_fname="oc20_3k_train.traj",
        test_fname="oc20_300_test.traj",
    )
