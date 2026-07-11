#!/usr/bin/python

#########################################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

#########################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(self.norm2(h))
        h = self.drop2(h)
        return x + h


class CellEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_cell_types, hidden_dim=512, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ct_embed = nn.Embedding(num_cell_types, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, ct_idx):
        h = self.input_proj(x)
        h = h + self.ct_embed(ct_idx)
        for block in self.blocks:
            h = block(h)
        return F.normalize(self.out(h), dim=-1)


class WithinCellTypePrototypePool(nn.Module):
    def __init__(self, embedding_dim, num_cell_types, num_prototypes=4, token_dim=128, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_cell_types = num_cell_types
        self.num_prototypes = num_prototypes
        self.token_dim = token_dim

        self.prototypes = nn.Parameter(torch.randn(num_cell_types, num_prototypes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.proto_temp = nn.Parameter(torch.tensor(10.0))

        self.cell_gate = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )

        token_input_dim = embedding_dim + embedding_dim + num_prototypes + 2
        self.token_mlp = nn.Sequential(
            nn.Linear(token_input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.ct_token_embed = nn.Embedding(num_cell_types, token_dim)

    def forward(self, cell_embs, cell_types):
        device = cell_embs.device
        n_cells = cell_embs.shape[0]
        tokens = torch.zeros(self.num_cell_types, self.token_dim, device=device)
        prototype_occupancies = torch.zeros(self.num_cell_types, self.num_prototypes, device=device)
        proportions = torch.zeros(self.num_cell_types, device=device)
        heterogeneity = torch.zeros(self.num_cell_types, device=device)
        cell_prototype_assignments = torch.zeros(n_cells, self.num_prototypes, device=device)

        unique_cts = torch.unique(cell_types)
        for ct in unique_cts:
            ct_int = int(ct.item())
            mask = (cell_types == ct)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            z = cell_embs[mask]
            n_ct = z.shape[0]
            if n_ct == 0:
                continue

            proportions[ct_int] = n_ct / max(float(n_cells), 1.0)
            gate_logits = self.cell_gate(z).squeeze(-1)
            attn = torch.softmax(gate_logits, dim=0)
            mean_state = torch.sum(attn.unsqueeze(-1) * z, dim=0)

            centered = z - mean_state.unsqueeze(0)
            std_state = torch.sqrt(torch.mean(centered ** 2, dim=0) + 1e-6)
            heterogeneity[ct_int] = std_state.mean()

            prot = F.normalize(self.prototypes[ct_int], dim=-1)
            logits_proto = torch.matmul(F.normalize(z, dim=-1), prot.t()) * self.proto_temp
            assign = torch.softmax(logits_proto, dim=-1)
            cell_prototype_assignments[idx] = assign
            proto_occ = assign.mean(dim=0)
            prototype_occupancies[ct_int] = proto_occ

            scalar_feats = torch.stack([proportions[ct_int], heterogeneity[ct_int]])
            token_input = torch.cat([mean_state, std_state, proto_occ, scalar_feats], dim=0)
            tokens[ct_int] = self.token_mlp(token_input) + self.ct_token_embed(ct)

        missing = (proportions == 0)
        if missing.any():
            missing_idx = torch.arange(self.num_cell_types, device=device)[missing]
            tokens[missing] = self.ct_token_embed(missing_idx)

        return tokens, prototype_occupancies, proportions, cell_prototype_assignments


class PerceiverCrossAttentionAggregator(nn.Module):
    def __init__(self, token_dim, num_cell_types, num_classes, n_latents=8, n_heads=4, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.n_latents = n_latents

        self.latents = nn.Parameter(torch.randn(n_latents, token_dim))
        nn.init.xavier_uniform_(self.latents)

        self.norm_q  = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )
        self.class_head = nn.Linear(token_dim, num_classes)

    def forward(self, tokens, missing_mask=None):
        q  = self.norm_q(self.latents).unsqueeze(0)
        kv = self.norm_kv(tokens).unsqueeze(0)
        kpm = missing_mask.unsqueeze(0) if missing_mask is not None else None

        latent_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kpm)

        latent_scores = self.pool_gate(latent_out).squeeze(-1)
        latent_attn   = torch.softmax(latent_scores, dim=-1)
        sample_emb = torch.sum(latent_attn.unsqueeze(-1) * latent_out, dim=1).squeeze(0)

        logits = self.class_head(sample_emb)
        return logits, sample_emb


class PhenoverseModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        token_dim,
        num_cell_types,
        num_classes,
        num_prototypes=4,
        encoder_hidden_dim=512,
        encoder_blocks=2,
        n_heads=4,
        n_latents=8,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_cell_types = num_cell_types
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.cell_encoder = CellEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_cell_types=num_cell_types,
            hidden_dim=encoder_hidden_dim,
            num_blocks=encoder_blocks,
            dropout=dropout,
        )
        self.prototype_pool = WithinCellTypePrototypePool(
            embedding_dim=embedding_dim,
            num_cell_types=num_cell_types,
            num_prototypes=num_prototypes,
            token_dim=token_dim,
            dropout=dropout,
        )
        self.aggregator = PerceiverCrossAttentionAggregator(
            token_dim=token_dim,
            num_cell_types=num_cell_types,
            num_classes=num_classes,
            n_latents=n_latents,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x, ct_idx):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if ct_idx.dim() == 0:
            ct_idx = ct_idx.unsqueeze(0)

        cell_embs = self.cell_encoder(x, ct_idx)
        tokens, proto_occ, proportions, cell_proto_assign = self.prototype_pool(cell_embs, ct_idx)
        missing_mask = (proportions == 0)
        logits, sample_emb = self.aggregator(tokens, missing_mask)

        return {
            'logits': logits,
            'sample_embedding': sample_emb,
            'cell_embeddings': cell_embs,
            'prototype_occupancies': proto_occ,
            'cell_prototype_assignments': cell_proto_assign,
        }

    @torch.no_grad()
    def get_sample_representation(self, x, ct_idx):
        self.eval()
        return self.forward(x, ct_idx)['sample_embedding']
