import pickle
import lmdb
from ase.io import Trajectory
from tqdm import tqdm
from tqdm.contrib import tenumerate
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.GMP import GMP

from pathlib import Path

bdqm_hpopt_path = Path(__file__).resolve().parents[1]
amptorch_path = Path(__file__).resolve().parents[2] / 'amptorch'


def get_path_to_gaussian(element):
    gaussians_path = amptorch_path / "examples/GMP/valence_gaussians"
    return next(gaussians_path.glob(f"{element}_*"))

def get_all_elements(traj):
    return list({atom.symbol for image in traj for atom in image})


def construct_lmdb(data_dir, train_fname, test_fname):
    print(f"Loading {train_fname}")
    train_images = Trajectory(data_dir / train_fname)
    print(f"Loading {test_fname}")
    test_images = Trajectory(data_dir / test_fname)


    sigmas = [0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4]

    elements = get_all_elements(train_images)
    gaussians = {el: get_path_to_gaussian(el) for el in elements}

    MCSHs = {
        'MCSHs': {
            '0': {'groups': [1], 'sigmas': sigmas},
            '1': {'groups': [1], 'sigmas': sigmas},
            '2': {'groups': [1, 2], 'sigmas': sigmas},
            '3': {'groups': [1, 2, 3], 'sigmas': sigmas},
        },
        'atom_gaussians': gaussians,
        'cutoff': 5,
    }

    descriptor_setup = ('GMP', MCSHs, None, elements)
    descriptor = GMP(MCSHs=MCSHs, elements=elements)

    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=True,
        save_fps=False,
        fprimes=False)

    n_train = len(train_images)
    n_test = len(test_images)

    train_feats = [
        a2d.convert(img, idx=idx)
        for idx, img in tenumerate(train_images, desc='Calculating train descriptors', total=n_train, unit=' images')
    ]

    test_feats = [
        a2d.convert(img, idx=idx)
        for idx, img in tenumerate(test_images, desc='Calculating test descriptors', total=n_test, unit=' images')
    ]

    scaling = {"type": "normalize", "range": (0, 1)}
    feature_scaler = FeatureScaler(train_feats, False, scaling)
    target_scaler = TargetScaler(train_feats, False)

    feature_scaler.norm(train_feats)
    target_scaler.norm(train_feats)

    feature_scaler.norm(test_feats)
    target_scaler.norm(test_feats)

    train_data = {
        **dict(enumerate(train_feats)),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "length": n_train,
        "elements": elements,
        "descriptor_setup": descriptor_setup,
    }
    save_to_lmdb(train_data, lmdb_path=(data_dir / train_fname).with_suffix(".lmdb"))

    test_data = {
        **dict(enumerate(test_feats)),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "length": n_test,
        "elements": elements,
        "descriptor_setup": descriptor_setup,
    }
    save_to_lmdb(test_data, lmdb_path=(data_dir / test_fname).with_suffix(".lmdb"))

def save_to_lmdb(to_save, lmdb_path):
    db = lmdb.open(
        str(lmdb_path),
        map_size=64393216 * 2,
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

if __name__ == '__main__':
    construct_lmdb(
        data_dir=bdqm_hpopt_path / "data",
        train_fname="oc20_3k_train.traj",
        test_fname="oc20_300_test.traj",
    )
