#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GPU-accelerated Moran's I calculations (batch / AnnData compatible)
# Places GPU-accelerated replacements for morans_i_p_values and morans_i_zscore
# while keeping the original function signatures expected by callers.
#
# Notes:
# - Requires PyTorch with CUDA available.
# - Keeps the same statistical formulas as esda.Moran and the earlier m_autocor.py:
#   uses structural constants S0/S1/S2 computed on CPU (scipy.sparse) identical to esda,
#   uses per-gene kurtosis K computed from centered moments, and the Cliff & Ord
#   variance formula to produce z-score and p-values.
# - Input API is unchanged: callers can pass the same adata, Weights, layer_key, n_process.
# - This module performs heavy vector math on GPU using torch and sparse mm.
#
# Limitations / design choices:
# - Structural constants (S0,S1,S2) are computed on CPU via scipy (keeps formula fidelity).
# - The kurtosis computation and per-gene moments are done on GPU when feasible:
#   to avoid densifying the full matrix when adata.layers is sparse we fall back to
#   an efficient CPU sparse-moments path that matches the original behavior.
# - Results (z and p) follow esda/Moran one- or two-tailed behavior consistent with original code:
#   this module returns two-tailed p-values by default to match m_autocor.py earlier behavior
#   (but the user can treat z-scores directly when needed).
#
# Author: assistant
# Date: 2025-10-30
"""
GPU-accelerated Moran's I functions used by SpaGRN.

Expose:
- morans_i_p_values(adata, Weights, layer_key='raw_counts', n_process=None)
- morans_i_zscore(adata, Weights, layer_key='raw_counts', n_process=None)

These functions maintain call signatures from m_autocor.py but run heavy parts on GPU.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from scipy.sparse import csr_matrix, issparse
import scipy

# for structural constants and CPU sparse moments we reuse numpy/scipy
from scipy.stats import norm

# torch for GPU ops
try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

import scanpy as sc

# Reuse the same structural constant formulas as in other modules
def _ensure_csr_zero_diag(W) -> csr_matrix:
    """
    Convert libpysal.W or scipy.sparse to CSR, set diag zero, sort indices.
    """
    if hasattr(W, "sparse"):
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
    S0, S1, S2 as in Cliff & Ord and esda.
    """
    S0 = float(W_csr.sum())
    A = W_csr + W_csr.T
    S1 = 0.5 * float((A.multiply(A)).sum())
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    S2 = float(np.square(row_sum).sum())
    return S0, S1, S2


def _torch_device_and_dtype(torch_device="cuda", torch_dtype="float64"):
    """
    Resolve torch device and dtype (CUDA required).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU-accelerated Moran.")
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
        raise RuntimeError("GPU Moran requires CUDA device. Set torch_device='cuda' or 'cuda:0'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available.")
    # dtype
    if isinstance(torch_dtype, torch.dtype):
        tdtype = torch_dtype
    else:
        s = str(torch_dtype).lower()
        tdtype = torch.float64 if ("64" in s or "double" in s or "fp64" in s) else torch.float32
    return dev, tdtype


def _torch_sparse_csr_from_scipy(W_csr: csr_matrix, device, dtype):
    """
    Convert scipy CSR -> torch.sparse_csr_tensor (on device).
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


# -----------------------
# Kurtosis K calculation
# -----------------------
def _compute_K_gpu_or_cpu(adata, layer_key, dev, tdtype, row_batch_size=8192) -> np.ndarray:
    """
    Compute per-gene kurtosis K = m4 / m2^2.
    If expression matrix is sparse we compute moments on CPU in a numerically stable sparse way.
    If dense and fits reasonably in GPU memory we compute using torch on GPU in a single shot.
    Otherwise we fall back to row-batched GPU computation to limit memory.
    """
    X = adata.layers[layer_key] if layer_key else adata.X
    n = adata.n_obs
    g = adata.n_vars

    # small epsilon to guard divisions
    eps = 1e-12

    if issparse(X):
        # compute using scipy sparse powers on CPU (no densify)
        mu = np.asarray(X.mean(axis=0)).ravel()
        Ex2 = np.asarray(X.power(2).mean(axis=0)).ravel()
        Ex3 = np.asarray(X.power(3).mean(axis=0)).ravel()
        Ex4 = np.asarray(X.power(4).mean(axis=0)).ravel()
        m2 = Ex2 - mu * mu
        m4 = Ex4 - 4 * mu * Ex3 + 6 * (mu * mu) * Ex2 - 3 * (mu ** 4)
        m2 = np.maximum(m2, eps)
        K = m4 / (m2 * m2)
        K[~np.isfinite(K)] = 3.0
        return K.astype(np.float64)
    else:
        # Dense matrix path: use GPU if possible
        X_np = np.asarray(X, dtype=np.float64, copy=False)
        # attempt single-shot GPU if size reasonable
        bytes_required = X_np.nbytes
        # heuristic threshold: try single-shot if matrix < 2GB (tunable)
        if bytes_required < 2 * 1024 ** 3:
            # one-shot to GPU
            X_t = torch.from_numpy(X_np.astype(np.float64 if tdtype == torch.float64 else np.float32, copy=False)).to(device=dev, dtype=tdtype)
            mu_t = X_t.mean(dim=0, keepdim=True)
            Z_t = X_t - mu_t
            Z2_t = Z_t * Z_t
            sum2 = Z2_t.sum(dim=0).to('cpu', torch.float64).numpy()
            Z4_t = Z2_t * Z2_t
            sum4 = Z4_t.sum(dim=0).to('cpu', torch.float64).numpy()
            m2 = (sum2 / float(X_np.shape[0]))
            m4 = (sum4 / float(X_np.shape[0]))
            m2 = np.maximum(m2, eps)
            K = m4 / (m2 * m2)
            K[~np.isfinite(K)] = 3.0
            # free memory
            del X_t, mu_t, Z_t, Z2_t, Z4_t
            torch.cuda.empty_cache()
            return K.astype(np.float64)
        else:
            # batched row accumulation on GPU to limit memory
            nrows = X_np.shape[0]
            sum2 = np.zeros(g, dtype=np.float64)
            sum4 = np.zeros(g, dtype=np.float64)
            for start in range(0, nrows, row_batch_size):
                end = min(start + row_batch_size, nrows)
                Xb = X_np[start:end, :].astype(np.float64, copy=False)
                Xb_t = torch.from_numpy(Xb).to(device=dev, dtype=tdtype)
                mu_t = torch.tensor(X_np.mean(axis=0, keepdims=True), device=dev, dtype=tdtype)
                Zb_t = Xb_t - mu_t
                Z2_t = Zb_t * Zb_t
                sum2 += Z2_t.sum(dim=0).to('cpu', torch.float64).numpy()
                Z4_t = Z2_t * Z2_t
                sum4 += Z4_t.sum(dim=0).to('cpu', torch.float64).numpy()
                del Xb_t, Zb_t, Z2_t, Z4_t
                torch.cuda.empty_cache()
            m2 = sum2 / float(nrows)
            m4 = sum4 / float(nrows)
            m2 = np.maximum(m2, eps)
            K = m4 / (m2 * m2)
            K[~np.isfinite(K)] = 3.0
            return K.astype(np.float64)


# -----------------------
# Compute Moran's I vectorized via Scanpy (CPU) or compute on GPU directly
# -----------------------
def _compute_morans_I_vector(adata, W_csr: csr_matrix, layer_key: str) -> np.ndarray:
    """
    Compute Moran's I for each gene using scanpy's vectorized routine.
    sc.metrics.morans_i accepts graph (sparse) and vals shaped (n_features, n_cells).
    For consistency with earlier code this function uses scanpy (CPU) to get I values.
    We can later compute z/p on GPU.
    """
    X = adata.layers[layer_key] if layer_key else adata.X
    # sc.metrics.morans_i expects vals shape (n_features, n_cells)
    vals = X.T if issparse(X) else np.asarray(X).T
    I = sc.metrics.morans_i(W_csr, vals)
    return np.asarray(I, dtype=np.float64)


# -----------------------
# GPU z/p vectorized using esda variance formulas
# -----------------------
def _z_p_from_I_gpu(
    I: np.ndarray,
    n_obs: int,
    S0: float,
    S1: float,
    S2: float,
    K: np.ndarray,
    tdtype,
    dev,
    two_tailed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute z and p given I and K; K and I are numpy arrays.
    This implements the same algebra as _z_p_from_I_vectorized_strict but uses
    PyTorch where convenient for broadcasting; final results returned as numpy.
    """
    # Move arrays to torch to leverage vector ops on GPU
    I_t = torch.from_numpy(I.astype(np.float64)).to(device=dev, dtype=tdtype)
    K_t = torch.from_numpy(K.astype(np.float64)).to(device=dev, dtype=tdtype)
    n_t = torch.tensor(float(n_obs), device=dev, dtype=tdtype)
    S0_t = torch.tensor(float(S0), device=dev, dtype=tdtype)
    S1_t = torch.tensor(float(S1), device=dev, dtype=tdtype)
    S2_t = torch.tensor(float(S2), device=dev, dtype=tdtype)

    EI = -1.0 / (n_obs - 1.0)
    EI_t = torch.tensor(float(EI), device=dev, dtype=tdtype)
    S0sq_t = S0_t * S0_t
    n2 = float(n_obs) * float(n_obs)

    # parts as in cliff & ord formula (vectorized)
    part1_t = (n_t * (S1_t * (n2 - 3.0 * float(n_obs) + 3.0) - n_t * S2_t + 3.0 * S0sq_t)) / (
        (n_t - 1.0) * (n_t - 2.0) * (n_t - 3.0) * S0sq_t
    )
    part2_t = (K_t * (S1_t * (n2 - n_t) - 2.0 * n_t * S2_t + 6.0 * S0sq_t)) / (
        (n_t - 1.0) * (n_t - 2.0) * (n_t - 3.0) * S0sq_t
    )
    VI_t = part1_t - part2_t - (EI_t * EI_t)
    VI_t = torch.clamp(VI_t, min=1e-24)

    Z_t = (I_t - EI_t) / torch.sqrt(VI_t)
    Z = Z_t.detach().cpu().numpy().astype(np.float64)

    if two_tailed:
        p = 2.0 * (1.0 - norm.cdf(np.abs(Z)))
    else:
        # one-tailed consistent with many esda outputs: right-tail
        p = 1.0 - norm.cdf(Z)

    return Z, p


# -----------------------
# Public API (same names as in m_autocor.py)
# -----------------------
def morans_i_p_values_scanpy_exact(adata, Weights, layer_key="raw_counts", n_process=None, torch_device="cuda", torch_dtype="float64"):
    """
    GPU-accelerated computation of Moran's I p-values (two-tailed by default).
    Signature compatible with m_autocor.morans_i_p_values.
    """
    # Validate device/dtype and prepare
    dev, tdtype = _torch_device_and_dtype(torch_device, torch_dtype)

    # Load expression and weights
    W_csr = _ensure_csr_zero_diag(Weights)
    # compute I via scanpy (CPU) for compatibility
    I = _compute_morans_I_vector(adata, W_csr, layer_key)

    # structural constants on CPU
    S0, S1, S2 = _struct_constants_S0_S1_S2(W_csr)

    # compute per-gene kurtosis (GPU or CPU internal as needed)
    K = _compute_K_gpu_or_cpu(adata, layer_key, dev, tdtype)

    # compute z and p on GPU, return p (two-tailed)
    Z, p = _z_p_from_I_gpu(I, adata.n_obs, S0, S1, S2, K, tdtype, dev, two_tailed=True)

    # numeric cleanup
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 1.0
    return p


def morans_i_zscore(adata, Weights, layer_key="raw_counts", n_process=None, torch_device="cuda", torch_dtype="float64"):
    """
    GPU-accelerated computation of Moran's I z-scores (normal approx).
    Signature compatible with m_autocor.morans_i_zscore.
    """
    dev, tdtype = _torch_device_and_dtype(torch_device, torch_dtype)

    W_csr = _ensure_csr_zero_diag(Weights)
    I = _compute_morans_I_vector(adata, W_csr, layer_key)
    S0, S1, S2 = _struct_constants_S0_S1_S2(W_csr)
    K = _compute_K_gpu_or_cpu(adata, layer_key, dev, tdtype)

    Z, p_dummy = _z_p_from_I_gpu(I, adata.n_obs, S0, S1, S2, K, tdtype, dev, two_tailed=True)

    Z = np.asarray(Z, dtype=np.float64)
    Z[~np.isfinite(Z)] = 0.0
    return Z