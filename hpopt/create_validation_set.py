"""
Script to split train data into train and validate.

This lets us reserve the test data to do a true test of our final hyperparameters.
"""

from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import Trajectory
from tqdm import tqdm
from utils import bdqm_hpopt_path, split_data


def save_to_traj(imgs: Iterable[Atoms], path: Path):
    """Save `imgs`"""
    with Trajectory(path, "w") as t:
        for img in tqdm(imgs, desc="Writing data to .traj"):
            t.write(img)


def main(data_dir: Path, train_fname: str) -> None:
    """Split train into train & valid, save to .traj files."""
    valid_traj_path = data_dir / "valid.traj"
    train_traj_path = data_dir / "train.traj"

    for path in [train_traj_path, valid_traj_path]:
        if path.exists():
            print(f"{path} already exists, aborting")
            return

    print("Loading and splitting data...")
    train_imgs = Trajectory(data_dir / train_fname)
    train_imgs, valid_imgs = split_data(train_imgs, valid_pct=0.1)

    print("\nSaving data...")
    save_to_traj(train_imgs, train_traj_path)
    save_to_traj(valid_imgs, valid_traj_path)


if __name__ == "__main__":
    main(
        data_dir=bdqm_hpopt_path / "data",
        train_fname="oc20_3k_train.traj",
    )
