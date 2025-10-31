"""
GPU-accelerated Geary batch wrapper that calls the GPU-faithful geary implementation.

This c_autocor_fast.py uses geary.geary_batch to compute per-gene Geary C,
z-scores and p-values on GPU in a batched manner while preserving esda.Geary numerics.

It expects the geary.py module (GPU implementation) to be available in import path.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import norm

try:
    import torch
except Exception:
    torch = None

# import the GPU-faithful implementation
from .geary import geary_batch, _ensure_csr_zero_diag  # _ensure_csr_zero_diag is used to sanitize Weights

def gearys_c_fast(
    adata,
    Weights,
    layer_key='raw_counts',
    mode='pvalue',           # 'pvalue' | 'zscore'
    batch_size=1024,         # per-batch genes to upload
    torch_device='cuda',     # 'cuda' or 'cuda:0'
    torch_dtype='float64',   # 'float64' recommended for fidelity
):
    """
    High-fidelity GPU batch Geary. Uses geary.geary_batch internally.

    Returns numpy array length n_genes of either p-values (mode='pvalue')
    or z-scores (mode='zscore'). p-values follow esda.Geary one-tailed convention.
    """
    assert mode in ['pvalue', 'zscore']

    # load expression
    if layer_key:
        X = adata.layers[layer_key]
    else:
        X = adata.X
    n, g = X.shape
    if n < 4:
        raise ValueError("Geary's C normal approximation requires n >= 4.")

    # prepare Weights CSR with zero diag
    W_csr = _ensure_csr_zero_diag(Weights)

    out = np.empty(g, dtype=np.float64)

    # process in batches; geary_batch expects dense numpy block (n, b)
    for start in range(0, g, batch_size):
        end = min(start + batch_size, g)
        if issparse(X):
            Xb = X[:, start:end].toarray()
        else:
            Xb = np.asarray(X[:, start:end])
        # call geary_batch which computes C, Z, p_norm with esda-consistent one-tailed p
        C_arr, Z_arr, p_norm = geary_batch(Xb, W_csr, device=torch_device, torch_dtype=torch_dtype)
        if mode == 'zscore':
            out[start:end] = Z_arr
        else:
            # return p_norm (one-tailed as esda.Geary.p_norm)
            out[start:end] = p_norm

    # cleanup
    if mode == 'pvalue':
        out[~np.isfinite(out)] = 1.0
    else:
        out[~np.isfinite(out)] = 0.0
    return out