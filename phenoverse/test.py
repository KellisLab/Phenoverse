#!/usr/bin/python

#########################################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

#########################################################


import logging
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from .model import PhenoverseModel
from .util import _to_dense, preprocess_data

logger = logging.getLogger(__name__)


def test_main(
    adata_path,
    checkpoint_path,
    sampleidcol,
    celltypelabel,
    output_dir='.'):

    logger.info('Starting to run...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adata = sc.read_h5ad(adata_path)

    for col in (sampleidcol, celltypelabel):
        if col not in adata.obs.columns:
            logger.error(f'{col} not found!!')
            sys.exit(1)

    adata_original = adata.copy()
    adata = preprocess_data(adata)
    data_X = _to_dense(adata.X)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    le = checkpoint['label_encoder']
    ct_to_idx = checkpoint['ct_to_idx']
    embedding_dim = checkpoint['embedding_dim']
    token_dim = checkpoint['token_dim']
    num_prototypes = checkpoint['num_prototypes']
    num_cell_types = checkpoint['num_cell_types']
    num_classes = checkpoint['num_classes']
    input_dim = checkpoint['input_dim']
    encoder_hidden_dim = checkpoint['encoder_hidden_dim']
    encoder_blocks = checkpoint['encoder_blocks']
    n_heads = checkpoint['n_heads']
    n_latents = checkpoint['n_latents']
    dropout = checkpoint['dropout']

    if data_X.shape[1] != input_dim:
        logger.error(f'Feature mismatch: checkpoint expects {input_dim} features, input data has {data_X.shape[1]}!!')
        sys.exit(1)

    test_cell_types = set(adata.obs[celltypelabel].dropna().unique())
    unseen_cell_types = test_cell_types - set(ct_to_idx.keys())
    if unseen_cell_types:
        logger.error(f'Unseen cell type(s) not present in the trained model: {sorted(unseen_cell_types)}!!')
        sys.exit(1)

    model = PhenoverseModel(
        input_dim=input_dim,
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sample_ids = adata.obs[sampleidcol].unique().tolist()

    sample_logits_dict = {}
    sample_rows = []

    n_cells = adata.shape[0]
    cell_embs_all = np.zeros((n_cells, embedding_dim), dtype=np.float32)
    cell_proto_all = np.zeros((n_cells, num_prototypes), dtype=np.float32)

    with torch.no_grad():
        for d in sample_ids:
            mask = (adata.obs[sampleidcol] == d).values
            idx = np.where(mask)[0]
            if len(idx) == 0:
                sample_logits_dict[d] = np.zeros((num_classes,), dtype=np.float32)
                continue

            x_arr = data_X[idx].astype(np.float32)
            ct_names = adata.obs[celltypelabel].values[idx]
            ct_arr = np.array([ct_to_idx[ct] for ct in ct_names], dtype=np.int64)

            x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
            ct_t = torch.tensor(ct_arr, dtype=torch.long, device=device)
            out = model(x_t, ct_t)

            logits = out['logits'].cpu().numpy()
            sample_logits_dict[d] = logits
            sample_emb = out['sample_embedding'].cpu().numpy()
            sample_rows.append([d] + sample_emb.tolist())

            cell_embs_all[idx] = out['cell_embeddings'].cpu().numpy()
            cell_proto_all[idx] = out['cell_prototype_assignments'].cpu().numpy()

    sample_ids_list = list(sample_logits_dict.keys())
    sample_probs = []
    for d in sample_ids_list:
        p = torch.softmax(torch.tensor(sample_logits_dict[d]), dim=0).numpy()
        sample_probs.append(p)
    sample_probs = np.array(sample_probs)
    preds = np.argmax(sample_probs, axis=1)

    os.makedirs(output_dir, exist_ok=True)

    predicted_labels = [le.classes_[p] for p in preds]
    predictions_df = pd.DataFrame({'Sample': sample_ids_list, 'Predicted': predicted_labels})
    for c in range(num_classes):
        predictions_df[f'Prob_{le.classes_[c]}'] = sample_probs[:, c]

    sample_columns = ['Sample'] + [f'Phenoverse_latent_{i+1}' for i in range(token_dim)]
    sample_df = pd.DataFrame(sample_rows, columns=sample_columns)

    adata_original.obsm['Phenoverse_cell_embeddings'] = cell_embs_all
    adata_original.obsm['Phenoverse_prototype_assignments'] = cell_proto_all
    adata_original.uns['predicted_phenotype'] = predictions_df
    adata_original.uns['sample_representations'] = sample_df

    annotated_path = os.path.join(output_dir, 'Phenoverse_annotated.h5ad')
    ad.settings.allow_write_nullable_strings = True
    adata_original.write(annotated_path)
    logger.info(f'Wrote annotated AnnData to {annotated_path}')

    logger.info('Completed successfully!!')
