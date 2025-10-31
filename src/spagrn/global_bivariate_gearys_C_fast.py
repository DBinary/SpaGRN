#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GPU-accelerated co-expression utilities (bivariate Geary's C)
# Added by assistant to provide a GPU version of global_bivariate_gearys_C.
#
# Usage:
#   from corexp_fast import global_bivariate_gearys_C_gpu
#   results_df = global_bivariate_gearys_C_gpu(
#       adata, fw, tfs_in_data, select_genes, layer_key='raw_counts',
#       num_workers=6, torch_device='cuda:0', pair_batch_size=4096
#   )
#
# The function preserves the original inputs but performs the heavy inner loop
# (many gene-pair computations) on the GPU. The parameter torch_device controls
# which CUDA device to use (string or torch.device). If torch_device is None or
# CUDA is unavailable, the function will raise an error.
#
# Notes:
# - The implementation computes bivariate Geary's C exactly as in corexp.py:
#     numerator = sum_e w_e * (x_i - x_j) * (y_i - y_j)
#     denominator = sqrt(sum (x - mean_x)^2) * sqrt(sum (y - mean_y)^2)
#   where e indexes the provided flattened neighbor list (Cell_x, Cell_y, Weight).
# - Work is processed in batches of gene-pairs to limit GPU memory usage.
# - The output is a pandas.DataFrame with columns ['TF', 'target', 'importance'].
#
# Limitations:
# - This version densifies the requested gene columns for the batch to GPU memory;
#   the batch sizes should be tuned (pair_batch_size) based on available VRAM.
# - For extremely large numbers of TFs / targets you may still hit memory limits;
#   in that case reduce pair_batch_size.
#
# Author: assistant
# Date: 2025-10-31
"""
GPU-accelerated bivariate Geary's C computation
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from scipy.sparse import issparse

# optional torch import
try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def _ensure_torch_device(torch_device: Optional[str]):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for global_bivariate_gearys_C_gpu but it's not installed.")
    if torch_device is None:
        # default to 'cuda' device if available
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(torch_device, torch.device):
        dev = torch_device
    else:
        dev = torch.device(str(torch_device))
    if dev.type != "cuda":
        raise RuntimeError("global_bivariate_gearys_C_gpu requires a CUDA device (torch_device).")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine (torch.cuda.is_available() is False).")
    return dev


def _format_matrix_to_numpy(adata, layer_key: Optional[str]):
    """
    Return dense numpy matrix (n_cells x n_genes) for the requested layer.
    This matches the original behavior in corexp.py (toarray() when needed).
    """
    if layer_key:
        X = adata.layers[layer_key]
    else:
        X = adata.X
    if issparse(X):
        X_np = X.toarray()
    else:
        X_np = np.asarray(X)
    return X_np.astype(np.float64, copy=False)


def _chunked_indices(total: int, chunk_size: int):
    """Yield (start, end) pairs covering [0, total) in chunks of chunk_size."""
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end


def global_bivariate_gearys_C_fast(
    adata,
    weights: pd.DataFrame,
    tfs_in_data: List[str],
    select_genes: List[str],
    layer_key: Optional[str] = "raw_counts",
    num_workers: int = 4,              # kept for compatibility with original signature
    torch_device: Optional[str] = "cuda:0",
    pair_batch_size: int = 1024,      # number of gene-pairs to evaluate per GPU batch
) -> pd.DataFrame:
    """
    GPU-accelerated replacement for global_bivariate_gearys_C.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (n_obs x n_vars). Must contain requested genes.
    weights : pd.DataFrame
        Flattened neighbor DataFrame with columns ['Cell_x','Cell_y','Weight'].
    tfs_in_data : list[str]
        List of TF names (subset of adata.var_names) to use as gene_x.
    select_genes : list[str]
        List of target gene names (subset of adata.var_names) to use as gene_y.
    layer_key : str or None
        adata.layers key to use; if None, use adata.X.
    num_workers : int
        Present for API compatibility; GPU implementation ignores this for inner loop.
    torch_device : str or torch.device or None
        CUDA device specifier (e.g., 'cuda:0'). If None, defaults to 'cuda' if available.
    pair_batch_size : int
        Number of gene-pairs to evaluate per GPU batch. Tune based on GPU VRAM.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns ['TF', 'target', 'importance'] containing bivariate Geary's C values.
    """
    # Validate & resolve device
    dev = _ensure_torch_device(torch_device)

    # Prepare gene name arrays and index mapping (like original)
    gene_names = np.array(adata.var.index, dtype=object)
    tf_ids = np.asarray(adata.var.index.get_indexer(tfs_in_data), dtype=np.int64)
    target_ids = np.asarray(adata.var.index.get_indexer(select_genes), dtype=np.int64)

    # Prepare cell index mapping as in original: create tmp_obs with integer ids
    tmp_obs = adata.obs.copy()
    tmp_obs["__corexp_idx__"] = np.arange(len(adata.obs))
    # weights['Cell_x'] / 'Cell_y' are assumed to contain observation names present in adata.obs_names
    cell_x_names = weights["Cell_x"].to_list()
    cell_y_names = weights["Cell_y"].to_list()
    try:
        cell_x_id = tmp_obs.loc[cell_x_names]["__corexp_idx__"].to_numpy(dtype=np.int64)
        cell_y_id = tmp_obs.loc[cell_y_names]["__corexp_idx__"].to_numpy(dtype=np.int64)
    except Exception as e:
        raise KeyError("Cell names in weights do not match adata.obs_names.") from e

    # Edge weights (flattened)
    wij = weights["Weight"].to_numpy(dtype=np.float64)
    if wij.ndim != 1:
        wij = wij.ravel()

    # Load expression matrix to dense numpy (matches original code's .toarray() on layer)
    X_np = _format_matrix_to_numpy(adata, layer_key)  # shape (n_cells, n_genes)
    n_cells, n_genes = X_np.shape

    # Precompute per-gene denominators: mean and denom-sqrt factors used in denominator
    # denominator = sqrt(sum (x - mean_x)^2) * sqrt(sum (y - mean_y)^2)
    # We'll compute per-gene sqrt(sum((x - mean)^2)) once on CPU and move subset to GPU per batch.
    gene_means = X_np.mean(axis=0)                # (n_genes,)
    gene_centered_sq_sums = ((X_np - gene_means) ** 2).sum(axis=0)  # (n_genes,)
    # To avoid zero-division, clamp small denominators (match original behaviour implicitly)
    gene_denom_factor = np.sqrt(np.maximum(gene_centered_sq_sums, 1e-24))  # (n_genes,)

    # Prepare tensors for edge-level indexing on GPU (constant for whole run)
    # cell_x_id and cell_y_id are arrays of length E (#edges)
    cell_x_id_np = np.asarray(cell_x_id, dtype=np.int64)
    cell_y_id_np = np.asarray(cell_y_id, dtype=np.int64)
    wij_np = np.asarray(wij, dtype=np.float64)

    # Move edge index arrays and weights to GPU once
    cell_x_id_t = torch.from_numpy(cell_x_id_np).to(device=dev, dtype=torch.long)  # (E,)
    cell_y_id_t = torch.from_numpy(cell_y_id_np).to(device=dev, dtype=torch.long)  # (E,)
    wij_t = torch.from_numpy(wij_np).to(device=dev, dtype=torch.float64)           # (E,)

    # Prepare all gene-pair combinations (we will batch over these pairs)
    # We'll create arrays of gene_x indices and gene_y indices of length P = len(tf_ids) * len(target_ids)
    tfs_len = tf_ids.size
    targets_len = target_ids.size
    if tfs_len == 0 or targets_len == 0:
        # Nothing to compute
        return pd.DataFrame(columns=["TF", "target", "importance"])

    # We'll produce meshgrid in a memory-efficient manner by iterating tf outer loop
    # Build list of pair starts for batching: we will flatten pairs as (i_tf * targets_len + j_target)
    total_pairs = int(tfs_len) * int(targets_len)

    # Prepare outputs (pre-allocated)
    TF_out = []
    target_out = []
    importance_out = []

    start_time = time.time()

    # We'll iterate outer over TFs to reduce intermediate memory then batch their corresponding target pairs
    # For each TF (single gene_x index), we compute gene_x values at edge endpoints once and reuse for many gene_y.
    # This reduces repeated loads.
    for tf_idx_idx, gene_x_id in tqdm(enumerate(tf_ids)):
        # gene_x_id is an integer index into genes
        if gene_x_id < 0:
            # unmapped TF name -> skip
            continue

        # Prepare gene_x vector at the edge endpoints (GPU)
        # Extract column of X for gene_x (dense) and move the endpoints to GPU
        gene_x_col = X_np[:, gene_x_id].astype(np.float64, copy=False)  # (n_cells,)
        # gather x_i and x_j for all edges
        x_at_x_t = torch.from_numpy(gene_x_col).to(device=dev, dtype=torch.float64).index_select(0, cell_x_id_t)  # (E,)
        x_at_y_t = torch.from_numpy(gene_x_col).to(device=dev, dtype=torch.float64).index_select(0, cell_y_id_t)  # (E,)

        # Precompute (x_i - x_j) at edges for this gene_x (used in numerator)
        dx_edges_t = x_at_x_t - x_at_y_t  # (E,)

        # Also compute mean and denom factor for gene_x (CPU scalars)
        mean_x = float(gene_means[gene_x_id])
        denom_x = float(gene_denom_factor[gene_x_id])
        if denom_x == 0.0:
            denom_x = 1e-24  # guard

        # Now iterate targets in batches (to limit memory) and compute numerator and denominator
        # We'll build blocks of gene_y indices of size up to pair_batch_size
        # Each inner batch will compute (for b genes): numerator_b and denominator_b vectorized.
        # number of gene_y per inner batch:
        # To make pair_batch_size be total gene-pairs per batch, we set inner_batch_size accordingly:
        inner_batch_size = max(1, int(pair_batch_size // 1))  # we only vary gene_y per tf here

        # iterate over targets in chunks
        for t_start in range(0, targets_len, inner_batch_size):
            t_end = min(t_start + inner_batch_size, targets_len)
            batch_target_ids = target_ids[t_start:t_end]  # (b,)

            # Filter out any unmapped (negative) indices
            valid_mask = batch_target_ids >= 0
            if not valid_mask.any():
                # All unmapped in this chunk -> append placeholders and continue
                for idx_invalid in batch_target_ids:
                    TF_out.append(gene_names[gene_x_id])
                    target_out.append("<UNMAPPED>")
                    importance_out.append(np.nan)
                continue
            # Only process valid ones
            valid_targets = batch_target_ids[valid_mask]
            b = valid_targets.size

            # Collect gene_y columns for valid_targets, shape (n_cells, b)
            # Move selected columns to GPU
            Xb_cols = X_np[:, valid_targets]  # (n_cells, b)
            Xb_t = torch.from_numpy(Xb_cols.astype(np.float64, copy=False)).to(device=dev, dtype=torch.float64)  # (n_cells, b)

            # Gather values at edge endpoints: (E, b)
            Xb_at_x_t = Xb_t.index_select(0, cell_x_id_t)  # (E, b)
            Xb_at_y_t = Xb_t.index_select(0, cell_y_id_t)  # (E, b)

            # Compute dy_edges: (x_i - x_j) for gene_y
            dy_edges_t = Xb_at_x_t - Xb_at_y_t  # (E, b)

            # Numerator per gene_y in batch: sum_e wij * dx_edges_e * dy_edges_e
            # dx_edges_t is (E,), dy_edges_t is (E,b), wij_t is (E,)
            # compute (wij * dx) as (E,1) then multiply elementwise with dy (E,b) then sum dim=0 -> (b,)
            mul_factor = (wij_t * dx_edges_t).unsqueeze(1)  # (E,1)
            numer_b_t = (mul_factor * dy_edges_t).sum(dim=0)  # (b,)

            # Denominator per gene_y: denom_x * denom_y
            denom_y_arr = gene_denom_factor[valid_targets]  # (b,)
            # Guard zeros
            denom_y_arr = np.where(denom_y_arr == 0.0, 1e-24, denom_y_arr)
            denom_b = denom_x * denom_y_arr  # (b,)

            # Compute importance = numerator / denom
            numer_b_np = numer_b_t.detach().cpu().numpy().astype(np.float64)  # (b,)
            importance_b = numer_b_np / denom_b  # (b,)

            # Append results into outputs in the original order (including unmapped placeholders)
            # We must interleave valid and invalid entries according to valid_mask
            out_idx = 0
            for i_mask, orig_target in zip(valid_mask, batch_target_ids):
                TF_out.append(gene_names[gene_x_id])
                if not i_mask:
                    # unmapped
                    target_out.append("<UNMAPPED>")
                    importance_out.append(np.nan)
                else:
                    # take next from importance_b
                    target_out.append(gene_names[valid_targets[out_idx]])
                    importance_out.append(float(importance_b[out_idx]))
                    out_idx += 1

            # explicit GPU memory cleanup for this inner block
            del Xb_t, Xb_at_x_t, Xb_at_y_t, dy_edges_t, mul_factor, numer_b_t
            torch.cuda.empty_cache()

        # cleanup per-TF temporary tensors
        del x_at_x_t, x_at_y_t, dx_edges_t
        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed = end_time - start_time

    # Build DataFrame preserving original column names
    results_df = pd.DataFrame({"TF": TF_out, "target": target_out, "importance": importance_out})
    # Optionally, you can filter out unmapped rows if desired:
    results_df = results_df[results_df["target"] != "<UNMAPPED>"].reset_index(drop=True)

    # Print timing similar to original
    print(f"[global_bivariate_gearys_C_gpu] Completed {len(results_df)} pairs in {elapsed:.4f} seconds (device={dev})")

    return results_df