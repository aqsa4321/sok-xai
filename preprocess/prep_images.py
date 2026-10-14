#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image preprocessing helpers.

Currently handles: splitting the flat cell_cancer dataset into
train/{class}/ and test/{class}/ subfolders, matching the layout
of skin_cancer. Run once; results are reproducible with a fixed seed.
"""

import os
import shutil
import random
from pathlib import Path


def split_flat_dataset(src_dir, dst_dir, test_ratio=0.2, seed=42, move=False):
    """
    Turn a flat class-per-folder dataset into a Keras-style train/test split.

    src_dir layout (input):
        src_dir/
            class_a/
                img1.jpg
                img2.jpg
                ...
            class_b/
                ...

    dst_dir layout (output):
        dst_dir/
            train/class_a/...
            train/class_b/...
            test/class_a/...
            test/class_b/...

    Parameters
    ----------
    src_dir : str
        Path to the flat dataset (one subfolder per class).
    dst_dir : str
        Where to write the split. Can be the same as src_dir; the
        train/ and test/ subfolders will be created alongside the
        existing class folders.
    test_ratio : float
        Fraction of files per class to route to test/.
    seed : int
        RNG seed for the shuffle. Fixed for reproducibility.
    move : bool
        If True, move files (destructive). If False (default), copy.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    rng = random.Random(seed)

    # Find class subfolders (skip anything that isn't a directory,
    # and skip 'train'/'test' if they already exist)
    class_dirs = [p for p in src.iterdir()
                  if p.is_dir() and p.name not in {"train", "test"}]

    if not class_dirs:
        raise RuntimeError(f"No class subfolders found in {src}")

    print(f"Found {len(class_dirs)} classes in {src}:")
    for c in class_dirs:
        n = len(list(c.glob("*")))
        print(f"  {c.name}: {n} files")

    # Create train/ and test/ target folders
    for split in ("train", "test"):
        for c in class_dirs:
            (dst / split / c.name).mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    op_name = "move" if move else "copy"

    total_train, total_test = 0, 0
    for c in class_dirs:
        files = sorted([f for f in c.iterdir() if f.is_file()])
        rng.shuffle(files)
        n_test = int(round(len(files) * test_ratio))
        test_files = files[:n_test]
        train_files = files[n_test:]

        for f in train_files:
            op(str(f), str(dst / "train" / c.name / f.name))
        for f in test_files:
            op(str(f), str(dst / "test" / c.name / f.name))

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"  {c.name}: {len(train_files)} -> train, {len(test_files)} -> test")

    print(f"\nDone ({op_name}). Total: {total_train} train, {total_test} test.")


if __name__ == "__main__":
    # Split cell_cancer (flat structure) into train/test.
    # Non-destructive: files are copied, originals stay in place.
    # Re-running is safe -- shutil.copy2 will overwrite.
    CELL_DIR = "xai-cdss/data/cell_cancer"
    split_flat_dataset(CELL_DIR, CELL_DIR, test_ratio=0.2, seed=42, move=False)