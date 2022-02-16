import os
import pickle
import lmdb
import numpy as np
import ase.io
from ase.io import Trajectory
import torch
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.GMP import GMP

from pathlib import Path

bdqm_hpopt_path = Path(__file__).resolve().parents[1]
amptorch_path = bdqm_hpopt_path.parent / 'amptorch'


def get_path_to_gaussian(element):
    gaussians_path = amptorch_path / "examples/GMP/valence_gaussians"
    return next(p for p in gaussians_path.iterdir() if p.name.startswith(element + "_"))

def get_all_elements(*trajs):
    return list({atom.symbol for traj in trajs for image in traj for atom in image})

def construct_lmdb(train_images, test_images, lmdb_path, normalizers_path):
    db = lmdb.open(
        lmdb_path,
        map_size=64393216 * 2,
        subdir=False,
        meminit=False,
        map_async=True)

    sigmas = [0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4]# [0.2, 1.13, 2.07, 3.0]

    elements = get_all_elements(train_images, test_images)

    MCSHs = {
        'MCSHs': {
            '0': {'groups': [1], 'sigmas': sigmas},
            '1': {'groups': [1], 'sigmas': sigmas},
            '2': {'groups': [1, 2], 'sigmas': sigmas},
            '3': {'groups': [1, 2, 3], 'sigmas': sigmas},
        },
        'atom_gaussians': {element: get_path_to_gaussian(element) for element in elements},
        'cutoff': 5,
    }


    descriptor_setup = ('gmp', MCSHs, None, elements)
    descriptor = GMP(MCSHs=MCSHs, elements=elements)

    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=True,
        save_fps=False,
        fprimes=False)

    idx = 0
    data_list = []
    for image in tqdm(train_images, desc='calculating fps', total=len(train_images), unit=' images'):
        data = a2d.convert(image, idx=idx)
        data_list.append(data)
        idx += 1

    if os.path.isfile(normalizers_path):
        normalizers = torch.load(normalizers_path)
        feature_scaler = normalizers["feature"]
        target_scaler = normalizers["target"]
    else:
        scaling = {"type": "normalize", "range": (0, 1)}
        feature_scaler = FeatureScaler(data_list, False, scaling)
        target_scaler = TargetScaler(data_list, False)
        normalizers = {
            "target": target_scaler,
            "feature": feature_scaler,
        }
        torch.save(normalizers, normalizers_path)

    feature_scaler.norm(data_list)
    target_scaler.norm(data_list)

    idx = 0
    for do in tqdm(data_list, desc="Writing images to LMDB"):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
        txn.commit()
        idx += 1

    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("target_scaler".encode("ascii"), pickle.dumps(target_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("elements".encode("ascii"), pickle.dumps(elements, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put(
        "descriptor_setup".encode("ascii"), pickle.dumps(descriptor_setup, protocol=-1)
    )
    txn.commit()

    db.sync()
    db.close()

if __name__ == '__main__':
    train_images = Trajectory(bdqm_hpopt_path / 'data' / 'oc20_3k_train.traj')
    test_images = Trajectory(bdqm_hpopt_path / 'data' / 'oc20_300_test.traj')
    construct_lmdb(train_images, test_images, 
            lmdb_path=str(bdqm_hpopt_path / 'data' / 'train.lmdb'), 
            normalizers_path=str(bdqm_hpopt_path / 'data' / 'normalizers.pt'),
            )
