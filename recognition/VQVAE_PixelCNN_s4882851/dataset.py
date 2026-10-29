"""Data loading and preprocessing for the HipMRI Study 2D prostate MRI slices.

This module provides:
  * file discovery and patient-level split verification (leakage check),
  * per-slice loading with normalisation and the one known shape fix,
  * a PyTorch Dataset for training/validation/test slices.

Dataset facts (verified in the project data audit):
  * 12,660 slices total: 11,460 train (31 patients), 660 validation (4 patients),
    540 test (3 patients). Splits are provided as separate directories and contain
    zero patient overlap (re-checked at runtime by verify_patient_splits).
  * Images are 256x128 float32 Nifti volumes with one extra trailing dimension.
  * One scan (patient 019, week 1) is 256x144 and is centre-cropped to 256x128.
  * Raw intensities span 0-1212 (median slice max 397), so each slice is scaled
    by its own maximum to [0, 1] (per-slice normalisation).

Reference: the Nifti loading pattern follows the course-provided Appendix B
example (nibabel get_fdata with caching='unchanged', drop trailing dim).
"""
import glob
import os
import re

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# Rangpur path to the 2D slice data (override with HIPMRI_ROOT env var if needed).
DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

# Filenames look like .../keras_slices_train/case_00000_week_1_slice_000.nii.gz
_SLICE_RE = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")

# All slices are resized (cropped) to this fixed shape.
ROWS, COLS = 256, 128

SPLITS = ("train", "validate", "test")


def _slice_key(path):
    """Sort key so slices order deterministically by (case, week, slice)."""
    m = _SLICE_RE.search(os.path.basename(path))
    return tuple(int(g) for g in m.groups())


def patient_id(path):
    """Extract the patient number from a slice path (for leakage checks)."""
    return int(_SLICE_RE.search(os.path.basename(path)).group(1))


def list_split(root, split):
    """Return the sorted list of slice file paths for one split directory."""
    files = sorted(glob.glob(os.path.join(root, f"keras_slices_{split}", "*.nii.gz")),
                   key=_slice_key)
    if not files:
        raise FileNotFoundError(f"no slices found for split '{split}' under {root}")
    return files


def verify_patient_splits(root):
    """Assert the provided splits are patient-disjoint (no data leakage).

    Every slice of a patient must live in exactly one split. Raises AssertionError
    otherwise. Returns a small summary dict for logging.
    """
    patients = {s: {patient_id(p) for p in list_split(root, s)} for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = patients[a] & patients[b]
                assert not overlap, f"patient leakage between {a} and {b}: {overlap}"
    return {s: len(p) for s, p in patients.items()}


def load_slice(path):
    """Load one Nifti slice as a float32 array in [0, 1], shape (256, 128).

    * drops the trailing dimension some HipMRI volumes carry,
    * centre-crops the width to 128 for the single 256x144 scan,
    * scales by the slice's own maximum (per-slice normalisation).
    """
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape == (ROWS, 144):  # the audited width anomaly only
        arr = arr[:, 8:136]
    if arr.shape != (ROWS, COLS):
        raise ValueError(f"unexpected slice shape {arr.shape}: {path}")
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"non-finite or negative MRI intensity: {path}")
    return arr / max(float(arr.max()), 1e-8)


class HipMRISlices(Dataset):
    """PyTorch dataset over one split of the HipMRI 2D slices.

    Args:
        root: dataset root (defaults to the Rangpur course path).
        split: one of 'train', 'validate', 'test'.
        preload: if True, load every slice into a single tensor up front
            (~1.7 GB RAM for the full train split, ~230 slices/s) for fast
            training; if False, read each slice from disk on access.
        flip_p: probability of a random horizontal flip (augmentation).
            Defaults to 0.0 so evaluation is deterministic; enable for training
            only if desired (left-right flipping is anatomically plausible for
            axial pelvic MRI but is kept off by default for reproducibility).
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", preload=True, flip_p=0.0,
                 limit=0):
        """Build the dataset for one split.

        Args:
            root: dataset root directory (Rangpur course path by default).
            split: 'train', 'validate' or 'test'.
            preload: load all slices into one tensor up front (fast training,
                ~1.7 GB RAM) instead of reading from disk per item.
            flip_p: probability of a random horizontal flip; 0.0 keeps
                evaluation deterministic.
            limit: debug-only cap on the number of slices (0 = all).
        """
        assert split in SPLITS, f"unknown split '{split}'"
        self.paths = list_split(root, split)
        if limit:
            self.paths = self.paths[:limit]
        self.flip_p = float(flip_p)
        self._cache = None
        if preload:
            # Preallocate to avoid a second full-dataset copy at np.stack.
            batch = np.empty((len(self.paths), ROWS, COLS), dtype=np.float32)
            for i, path in enumerate(tqdm(self.paths, desc=f"loading {split}",
                                          unit="slice", mininterval=10)):
                batch[i] = load_slice(path)
            self._cache = torch.from_numpy(batch).unsqueeze(1)  # [N,1,256,128]

    def __len__(self):
        """Number of slices in this split."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Return one slice as a float tensor of shape [1, 256, 128].

        Served from the preloaded cache when available, otherwise read from
        disk. Applies the optional random horizontal flip for augmentation.
        """
        if self._cache is not None:
            img = self._cache[idx]
        else:
            img = torch.from_numpy(load_slice(self.paths[idx])).unsqueeze(0)
        if self.flip_p > 0 and torch.rand(1).item() < self.flip_p:
            img = torch.flip(img, dims=[-1])
        return img

# --------------------------------------------------------------------------- #
# Segmentation masks (exploratory: used to condition the prior on anatomy)
# --------------------------------------------------------------------------- #
MASK_CLASSES = 6  # label values 0..5 in keras_slices_seg_*

# Masks mirror the images but use a seg_ prefix:
#   images  case_004_week_0_slice_0.nii.gz
#   masks    seg_004_week_0_slice_0.nii.gz
# so they need their own key; _slice_key would not match and would raise.
_MASK_RE = re.compile(r"seg_(\d+)_week_(\d+)_slice_(\d+)")


def _mask_key(path):
    """Sort key for mask files, by (case, week, slice), matching _slice_key."""
    m = _MASK_RE.search(os.path.basename(path))
    return tuple(int(g) for g in m.groups())


def list_mask_split(root, split):
    """Sorted mask paths for one split (keras_slices_seg_<split>)."""
    files = sorted(glob.glob(os.path.join(root, f"keras_slices_seg_{split}", "*.nii.gz")),
                   key=_mask_key)

    if not files:
        raise FileNotFoundError(f"no masks found for split '{split}' under {root}")
    return files


def load_mask(path):
    """Load one label slice as int64 (256, 128), same crop rule as load_slice."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape == (ROWS, 144):
        arr = arr[:, 8:136]
    if arr.shape != (ROWS, COLS):
        raise ValueError(f"unexpected mask shape {arr.shape}: {path}")
    return np.rint(arr).astype(np.int64)


class HipMRIMasks(Dataset):
    """Label masks for one split, aligned with HipMRISlices by sorted filename.

    Returns one-hot masks [MASK_CLASSES, 256, 128] as float32 so they can be fed
    straight into the conditioning path of CondPixelCNNPrior.
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", preload=True, limit=0,
                 one_hot=True):
        """Build the mask dataset for one split.

        Args:
            root: dataset root directory.
            split: 'train', 'validate' or 'test'.
            preload: cache all label slices in memory as int64.
            limit: debug-only cap on the number of masks (0 = all).
            one_hot: return one-hot [MASK_CLASSES, 256, 128] masks when True,
                otherwise the raw integer label grid.
        """
        assert split in SPLITS, f"unknown split '{split}'"
        self.paths = list_mask_split(root, split)
        if limit:
            self.paths = self.paths[:limit]
        self.one_hot = one_hot
        self._cache = None
        if preload:
            batch = np.empty((len(self.paths), ROWS, COLS), dtype=np.int64)
            for i, path in enumerate(tqdm(self.paths, desc=f"loading {split} masks",
                                          unit="slice", mininterval=10)):
                batch[i] = load_mask(path)
            self._cache = torch.from_numpy(batch)  # [N,256,128] int64

    def __len__(self):
        """Number of masks in this split."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Return one mask, one-hot [MASK_CLASSES, 256, 128] or raw int grid."""
        m = (self._cache[idx] if self._cache is not None
             else torch.from_numpy(load_mask(self.paths[idx])))
        if not self.one_hot:
            return m
        return torch.nn.functional.one_hot(m.clamp(0, MASK_CLASSES - 1),
                                           MASK_CLASSES).permute(2, 0, 1).float()


def verify_image_mask_alignment(root, split):
    """Check image and mask filenames correspond one-to-one by (case, week, slice)."""
    imgs = [_slice_key(p) for p in list_split(root, split)]
    masks = [_mask_key(p) for p in list_mask_split(root, split)]

    assert imgs == masks, f"image/mask ordering mismatch in split '{split}'"
    return {"split": split, "n": len(imgs)}
