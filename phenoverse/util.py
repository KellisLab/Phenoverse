#!/usr/bin/python

#########################################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

#########################################################


import logging
import random
from collections import Counter

import numpy as np
import scanpy as sc
import torch

logger = logging.getLogger(__name__)


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_data(adata):
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    logger.info(f"Preprocessed data shape: {adata.shape}")
    return adata


def _to_dense(X):
    return X.toarray() if not isinstance(X, np.ndarray) else X


def _build_sample_bags(adata, sampleidcol, celltypelabel, ct_to_idx):
    X = _to_dense(adata.X)
    sample_bags = {}
    sample_ids = adata.obs[sampleidcol].unique().tolist()
    for d in sample_ids:
        mask = (adata.obs[sampleidcol] == d).values
        idx = np.where(mask)[0]
        if len(idx) == 0:
            sample_bags[d] = (np.zeros((0, adata.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            continue
        ct_names = adata.obs[celltypelabel].values[idx]
        ct_idx = np.array([ct_to_idx[ct] for ct in ct_names], dtype=np.int64)
        sample_bags[d] = (X[idx].astype(np.float32), ct_idx)
    return sample_bags


def stratified_subsample_bag(x_arr, ct_arr, target, min_per_type=5):
    n_cells = x_arr.shape[0]
    if n_cells <= target:
        return x_arr, ct_arr

    groups = {}
    for i, ct in enumerate(ct_arr):
        groups.setdefault(int(ct), []).append(i)

    allocated = {}
    for ct, inds in groups.items():
        desired = int(round(len(inds) / n_cells * target))
        allocated[ct] = min(len(inds), max(min_per_type, desired))

    current_sum = sum(allocated.values())
    ct_list = list(allocated.keys())
    while current_sum < target:
        for ct in ct_list:
            if allocated[ct] < len(groups[ct]):
                allocated[ct] += 1
                current_sum += 1
                if current_sum == target:
                    break
    while current_sum > target:
        for ct in ct_list:
            if allocated[ct] > min_per_type:
                allocated[ct] -= 1
                current_sum -= 1
                if current_sum == target:
                    break

    sampled = []
    for ct, inds in groups.items():
        chosen = np.random.choice(inds, allocated[ct], replace=False)
        sampled.extend(chosen.tolist())
    sampled = np.array(sampled)
    return x_arr[sampled], ct_arr[sampled]


def build_balanced_sample_list(train_samples, sample_to_label, num_classes, le=None, log=False):
    class_samples = {}
    for d in train_samples:
        c = sample_to_label[d]
        class_samples.setdefault(c, []).append(d)

    max_count = max(len(ds) for ds in class_samples.values())

    if log and le is not None:
        before_str = ", ".join(f"{le.classes_[c]}: {len(ds)}" for c, ds in sorted(class_samples.items()))
        logger.info(f'Sample counts before balancing: {{{before_str}}}')

    balanced = []
    for c in range(num_classes):
        ds = class_samples.get(c, [])
        if len(ds) == 0:
            continue
        if len(ds) >= max_count:
            balanced.extend(ds)
        else:
            balanced.extend(ds)
            extra = np.random.choice(ds, max_count - len(ds), replace=True).tolist()
            balanced.extend(extra)

    if log and le is not None:
        after_counts = Counter([sample_to_label[d] for d in balanced])
        after_str = ", ".join(f"{le.classes_[c]}: {v}" for c, v in sorted(after_counts.items()))
        logger.info(f'Sample counts after balancing:  {{{after_str}}}')
        logger.info(f'Total samples per epoch: {len(balanced)}')

    return balanced
