#!/usr/bin/python

#########################################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

#########################################################


import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .model import PhenoverseModel
from .util import (
    _build_sample_bags,
    build_balanced_sample_list,
    preprocess_data,
    stratified_subsample_bag,
)

logger = logging.getLogger(__name__)


def train_main(
    adata_path,
    label,
    sampleidcol,
    celltypelabel,
    embedding_dim=128,
    token_dim=128,
    num_prototypes=4,
    encoder_hidden_dim=256,
    encoder_blocks=2,
    n_heads=4,
    n_latents=8,
    dropout=0.3,
    batch_size=1,
    test_size=0.2,
    out_checkpoint='Phenoverse_best_model.pth',
    max_epochs=400,
    patience=10,
    accum_steps=8):

    logger.info('Starting to run...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adata = sc.read_h5ad(adata_path)
    logger.info(f'Raw data shape: {adata.shape}')

    for col in (label, sampleidcol, celltypelabel):
        if col not in adata.obs.columns:
            logger.error(f'{col} not found!!')
            sys.exit(1)

    le = LabelEncoder()
    le.fit(adata.obs[label].values)
    num_classes = len(le.classes_)

    celltypes = [ct for ct in adata.obs[celltypelabel].unique() if pd.notna(ct)]
    celltypes_sorted = sorted(celltypes)
    ct_to_idx = {ct: i for i, ct in enumerate(celltypes_sorted)}
    num_cell_types = len(celltypes_sorted)
    logger.info(f'Found {num_cell_types} cell types: {celltypes_sorted}')

    sample_ids_all = adata.obs[sampleidcol].unique()
    sample_labels_str = {
        d: adata.obs.loc[adata.obs[sampleidcol] == d, label].iloc[0] for d in sample_ids_all
    }
    sample_labels_enc = [le.transform([sample_labels_str[d]])[0] for d in sample_ids_all]

    train_samples, val_samples = train_test_split(
        list(sample_ids_all),
        test_size=test_size,
        stratify=sample_labels_enc,
        random_state=42,
    )
    logger.info(f'Total samples: {len(sample_ids_all)}')
    logger.info(f'Training samples: {len(train_samples)}')
    logger.info(f'Validation samples: {len(val_samples)}')

    train_mask = adata.obs[sampleidcol].isin(train_samples)
    adata_train = adata[train_mask].copy()
    adata_val = adata[~train_mask].copy()

    adata_train = preprocess_data(adata_train)
    adata_val = preprocess_data(adata_val)

    sample_bags_train = _build_sample_bags(adata_train, sampleidcol, celltypelabel, ct_to_idx)
    sample_bags_val = _build_sample_bags(adata_val, sampleidcol, celltypelabel, ct_to_idx)

    sample_to_label = {d: le.transform([sample_labels_str[d]])[0] for d in sample_ids_all}
    criterion = nn.CrossEntropyLoss()

    model = PhenoverseModel(
        input_dim=adata_train.shape[1],
        embedding_dim=embedding_dim,
        token_dim=token_dim,
        num_cell_types=num_cell_types,
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_blocks=encoder_blocks,
        n_heads=n_heads,
        n_latents=n_latents,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    SUBSAMPLE_RANGE = (500, 1000)

    build_balanced_sample_list(train_samples, sample_to_label, num_classes, le=le, log=True)

    for epoch in range(max_epochs):
        model.train()

        epoch_samples = build_balanced_sample_list(train_samples, sample_to_label, num_classes)
        random.shuffle(epoch_samples)

        train_loss_accum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)

        for i, d in enumerate(epoch_samples):
            x_arr, ct_arr = sample_bags_train[d]
            if x_arr.shape[0] == 0:
                continue

            target = np.random.randint(SUBSAMPLE_RANGE[0], SUBSAMPLE_RANGE[1] + 1)
            x_sub, ct_sub = stratified_subsample_bag(x_arr, ct_arr, target=target)

            x_t = torch.tensor(x_sub, dtype=torch.float32, device=device)
            ct_t = torch.tensor(ct_sub, dtype=torch.long, device=device)
            y_t = torch.tensor([sample_to_label[d]], dtype=torch.long, device=device)

            out = model(x_t, ct_t)
            loss = criterion(out['logits'].unsqueeze(0), y_t)
            (loss / accum_steps).backward()

            if ((i + 1) % accum_steps == 0) or (i == len(epoch_samples) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_accum += loss.item()
            train_count += 1

        train_avg_loss = train_loss_accum / max(train_count, 1)

        model.eval()
        val_loss_accum = 0.0
        val_count = 0
        with torch.no_grad():
            for d in val_samples:
                x_arr, ct_arr = sample_bags_val[d]
                if x_arr.shape[0] == 0:
                    continue
                x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
                ct_t = torch.tensor(ct_arr, dtype=torch.long, device=device)
                y_t = torch.tensor([sample_to_label[d]], dtype=torch.long, device=device)
                out = model(x_t, ct_t)
                loss = criterion(out['logits'].unsqueeze(0), y_t)
                val_loss_accum += loss.item()
                val_count += 1

        val_avg_loss = val_loss_accum / max(val_count, 1)
        logger.info(
            f'[Epoch {epoch+1}/{max_epochs}] TrainLoss={train_avg_loss:.4f}, ValLoss={val_avg_loss:.4f}'
        )

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f'[Model] Early stopping at epoch {epoch+1}')
                break

    if best_state is None:
        logger.error('No best model state captured during training.')
        sys.exit(1)

    checkpoint_dict = {
        'label_encoder': le,
        'ct_to_idx': ct_to_idx,
        'embedding_dim': embedding_dim,
        'token_dim': token_dim,
        'num_prototypes': num_prototypes,
        'num_cell_types': num_cell_types,
        'num_classes': num_classes,
        'encoder_hidden_dim': encoder_hidden_dim,
        'encoder_blocks': encoder_blocks,
        'n_heads': n_heads,
        'n_latents': n_latents,
        'dropout': dropout,
        'model_state_dict': best_state,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'input_dim': adata_train.shape[1],
    }
    checkpoint_dir = os.path.dirname(out_checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint_dict, out_checkpoint)
    logger.info(f'Saved checkpoint to {out_checkpoint}')
    logger.info('Completed successfully!!')
