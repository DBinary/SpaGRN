#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU-accelerated Getis-Ord General G for AnnData (high-fidelity to esda.getisord.G)

This module implements getis_g_fast(...) which mirrors the computation flow of
esda.getisord.G but performs heavy per-gene arithmetic on GPU using PyTorch.
It preserves the original function signature expected by callers:

    getis_g_fast(adata, Weights, n_processes=n_processes, layer_key=layer_key, mode='pvalue')

Key points:
- Structural constants S0, S1, S2 are computed on CPU via scipy.sparse to keep
  exactness with esda formulas.
- Weight matrix W is converted to torch.sparse_csr_tensor and used for fast
  GPU sparse-dense operations (W @ x and related).
- For each gene (batched), we compute:
    - G_obs = (x^T W x) / ((sum x)^2 - sum x^2)
    - higher order column sums ∑x^2, ∑x^3, ∑x^4 used in EG2 numerator
    - EG2 and VG per gene using the same algebra as esda.getisord.G.__moments
    - z = (G_obs - EG) / sqrt(VG) and p_norm = 1 - Phi(|z|) (as in esda)
- The function signature is compatible (n_processes parameter is accepted for compatibility but not used).
- Uses float64 by default for numerical fidelity; accepts torch_device / torch_dtype but these are optional
  and default chosen to mimic CPU/esda behavior.

Usage:
    pvals = getis_g_fast(adata, W, n_processes=None, layer_key='raw_counts', mode='pvalue')

Author: assistant
Date: 2025-10-30
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from scipy.sparse import csr_matrix, issparse
from scipy.stats import norm
import scipy

# optional GPU backend
try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def _ensure_csr_zero_diag(W) -> csr_matrix:
    """
    Accept libpysal.weights.W (has .sparse) or a scipy.sparse matrix.
    Return CSR with diagonal zeroed, zeros eliminated, indices sorted.
    """
    if hasattr(W, "sparse"):  # libpysal.W
        Wc = W.sparse
        if not isinstance(Wc, csr_matrix):
            Wc = Wc.tocsr()
    elif issparse(W):
        Wc = W.tocsr()
    else:
        raise TypeError("Weights must be libpysal.weights.W or scipy.sparse matrix.")
    Wc.setdiag(0.0)
    Wc.eliminate_zeros()
    Wc.sort_indices()
    return Wc


def _struct_constants_S0_S1_S2(W_csr: csr_matrix) -> Tuple[float, float, float]:
    """
    Compute S0, S1, S2 consistent with esda.getisord.G.__moments.
    """
    S0 = float(W_csr.sum())
    A = W_csr + W_csr.T
    S1 = 0.5 * float((A.multiply(A)).sum())
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    S2 = float(np.square(row_sum).sum())
    return S0, S1, S2


def _getis_b_coeffs(n: int, S0: float, S1: float, S2: float) -> Tuple[float, float, float, float, float]:
    """
    Coefficients b0..b4 used in EG2 calculation (same algebra as esda).
    """
    n = float(n)
    n2 = n * n
    S0sq = S0 * S0
    b0 = (n2 - 3.0 * n + 3.0) * S1 - n * S2 + 3.0 * S0sq
    b1 = -((n2 - n) * S1 - 2.0 * n * S2 + 6.0 * S0sq)
    b2 = -(2.0 * n * S1 - (n + 3.0) * S2 + 6.0 * S0sq)
    b3 = 4.0 * (n - 1.0) * S1 - 2.0 * (n + 1.0) * S2 + 8.0 * S0sq
    b4 = S1 - S2 + S0sq
    return b0, b1, b2, b3, b4


def _torch_device_and_dtype(torch_device="cuda", torch_dtype="float64"):
    """
    Resolve a torch.device and torch.dtype for GPU operations. Requires CUDA.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for getis_g_fast GPU path.")
    # device
    if isinstance(torch_device, torch.device):
        dev = torch_device
    elif isinstance(torch_device, bool):
        dev = torch.device("cuda") if torch_device else torch.device("cpu")
    elif torch_device is None or (isinstance(torch_device, str) and torch_device.strip().lower() == "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(str(torch_device))
    if dev.type != "cuda":
        raise RuntimeError("getis_g_fast requires a CUDA device. Set torch_device='cuda' or 'cuda:0'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available (torch.cuda.is_available() is False).")
    # dtype
    if isinstance(torch_dtype, torch.dtype):
        tdtype = torch_dtype
    else:
        s = str(torch_dtype).lower()
        tdtype = torch.float64 if ("64" in s or "double" in s or "fp64" in s) else torch.float32
    return dev, tdtype


def _torch_sparse_csr_from_scipy(W_csr: csr_matrix, device, dtype):
    """
    Convert scipy CSR -> torch.sparse_csr_tensor on device/dtype.
    """
    Wc = W_csr.tocsr(copy=False)
    Wc.sort_indices()
    indptr = torch.from_numpy(Wc.indptr.astype(np.int64)).to(device=device)
    indices = torch.from_numpy(Wc.indices.astype(np.int64)).to(device=device)
    values_np = Wc.data.astype(np.float64 if dtype == torch.float64 else np.float32, copy=False)
    values = torch.from_numpy(values_np).to(device=device)
    return torch.sparse_csr_tensor(
        crow_indices=indptr,
        col_indices=indices,
        values=values,
        size=Wc.shape,
        dtype=dtype,
        device=device,
    )


def getis_g_fast(
    adata,
    Weights,
    n_processes=None,      # kept for compatibility; not used in GPU path
    layer_key=None,
    mode="pvalue",         # 'pvalue' or 'zscore'
    batch_size: int = 1024,
    torch_device: str = "cuda",
    torch_dtype: str = "float64",
):
    """
    GPU-accelerated Getis-Ord General G, compatible with g_autocor.getis_g signature.

    Parameters kept identical to original wrapper (n_processes accepted for API compatibility).
    Additional optional args (batch_size/torch_device/torch_dtype) have defaults so callers do not need to pass them.

    Returns:
        numpy array of length n_genes with either p-values (mode='pvalue') or z-scores (mode='zscore').
    """
    assert mode in ["pvalue", "zscore"]

    # Resolve device/dtype
    dev, tdtype = _torch_device_and_dtype(torch_device, torch_dtype)

    # Expression matrix (n_cells x n_genes)
    X = adata.layers[layer_key] if layer_key else adata.X
    n, g = X.shape
    if n < 4:
        raise ValueError("General G normal approximation requires n >= 4.")

    # Prepare weights CSR and structural constants on CPU (preserve esda math)
    W_csr = _ensure_csr_zero_diag(Weights)
    S0, S1, S2 = _struct_constants_S0_S1_S2(W_csr)
    b0, b1, b2, b3, b4 = _getis_b_coeffs(n, S0, S1, S2)

    # Tensors/constants on GPU
    n_t = torch.tensor(float(n), device=dev, dtype=tdtype)
    S0_t = torch.tensor(float(S0), device=dev, dtype=tdtype)
    EG_t = S0_t / (n_t * (n_t - 1.0))  # expected G (scalar)

    # Move sparse W to GPU
    W_t = _torch_sparse_csr_from_scipy(W_csr, device=dev, dtype=tdtype)

    # Prepare output
    out = np.empty(g, dtype=np.float64)

    # Process in batches to limit memory
    for start in range(0, g, batch_size):
        end = min(start + batch_size, g)
        bcount = end - start

        # Extract batch (n, bcount) and send to GPU (dense block)
        if issparse(X):
            Xb = X[:, start:end].toarray()
        else:
            Xb = np.asarray(X[:, start:end])
        Xb_np = Xb.astype(np.float64 if tdtype == torch.float64 else np.float32, copy=False)
        Xb_t = torch.from_numpy(Xb_np).to(device=dev, dtype=tdtype)  # (n, bcount)

        # Numerator: x^T W x  -> (bcount,)
        WX_t = torch.sparse.mm(W_t, Xb_t)             # (n, bcount)
        numer_t = (Xb_t * WX_t).sum(dim=0)            # (bcount,)

        # Denominator basic terms
        sum1_t = Xb_t.sum(dim=0)                      # ∑x
        sum2_t = (Xb_t * Xb_t).sum(dim=0)             # ∑x^2
        denom_pairs_t = (sum1_t * sum1_t - sum2_t).clamp_min(1e-24)  # (bcount,)
        G_obs_t = numer_t / denom_pairs_t

        # Higher order sums for EG2 numerator
        X2_t = Xb_t * Xb_t
        X3_t = X2_t * Xb_t
        X4_t = X2_t * X2_t
        sum3_t = X3_t.sum(dim=0)
        sum4_t = X4_t.sum(dim=0)

        # Coeffs as tensors
        b0_t = torch.tensor(float(b0), device=dev, dtype=tdtype)
        b1_t = torch.tensor(float(b1), device=dev, dtype=tdtype)
        b2_t = torch.tensor(float(b2), device=dev, dtype=tdtype)
        b3_t = torch.tensor(float(b3), device=dev, dtype=tdtype)
        b4_t = torch.tensor(float(b4), device=dev, dtype=tdtype)

        # EG2 numerator/denominator (vectorized)
        num_var_t = (
            b0_t * (sum2_t * sum2_t)
            + b1_t * sum4_t
            + b2_t * (sum1_t * sum1_t) * sum2_t
            + b3_t * sum1_t * sum3_t
            + b4_t * (sum1_t ** 4)
        )

        denom_var_t = (
            (denom_pairs_t * denom_pairs_t)
            * n_t
            * (n_t - 1.0)
            * (n_t - 2.0)
            * (n_t - 3.0)
        ).clamp_min(1e-24)

        EG2_t = num_var_t / denom_var_t

        VG_t = EG2_t - (EG_t * EG_t)
        VG_t = VG_t.clamp_min(1e-24)

        # Z and p (esda uses one-sided p_norm = 1 - Phi(|z|))
        Z_t = (G_obs_t - EG_t) / torch.sqrt(VG_t)

        if mode == "zscore":
            out[start:end] = Z_t.detach().cpu().numpy()
        else:
            Z_np = Z_t.detach().cpu().numpy()
            # esda.G.p_norm uses p = 1 - Phi(|z|)
            p_np = 1.0 - norm.cdf(np.abs(Z_np))
            p_np[~np.isfinite(p_np)] = 1.0
            out[start:end] = p_np

        # Cleanup and free GPU mem
        del Xb_t, WX_t, numer_t, sum1_t, sum2_t, denom_pairs_t, G_obs_t
        del X2_t, X3_t, X4_t, sum3_t, sum4_t, num_var_t, denom_var_t, EG2_t, VG_t, Z_t
        torch.cuda.empty_cache()

    out = np.asarray(out, dtype=np.float64)
    if mode == "pvalue":
        out[~np.isfinite(out)] = 1.0
    else:
        out[~np.isfinite(out)] = 0.0
    return out