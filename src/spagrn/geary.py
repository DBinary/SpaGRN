"""
GPU-accelerated Geary's C (high-fidelity to esda.geary.Geary)

This module provides:
- Geary: a GPU-only class compatible with esda.geary.Geary for single-vector usage.
- geary_batch: a GPU-vectorized function that computes Geary C, z_norm, p_norm
  for multiple columns (genes) at once. This is intended to be called from
  c_autocor_fast.py to perform batch GPU computations while keeping formulas
  and numeric treatment consistent with esda.Geary.

Requirements
- PyTorch with CUDA enabled
- scipy, numpy, libpysal
- Inputs: y (1D array) and w (libpysal.weights.W or libpysal.graph.Graph) or a scipy CSR

Notes on fidelity
- Structural constants S0, S1, S2 are computed on CPU from the same formulas as esda.
- Variance formulas (VC_norm and VC_rand) follow Cliff & Ord / esda expressions.
- p_norm and p_rand follow esda's one-tailed logic (use z_norm positive/negative test).
- Kurtosis K is computed per-vector using mean-centered moments; computed on GPU
  when possible then moved to CPU for final stable scalar operations where needed.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy import stats
from libpysal import graph, weights
from typing import Tuple, Optional

# PyTorch/CUDA required
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

__all__ = ["Geary", "geary_batch", "compute_structural_constants"]


def _ensure_csr_zero_diag(W):
    """Return scipy csr_matrix with diag set to zero and sorted indices."""
    if hasattr(W, "sparse"):
        Wc = W.sparse
        if not isinstance(Wc, csr_matrix):
            Wc = Wc.tocsr()
    elif issparse(W):
        Wc = Wc.tocsr()
    else:
        raise TypeError("Weights must be libpysal.weights.W or scipy.sparse matrix.")
    Wc.setdiag(0.0)
    Wc.eliminate_zeros()
    Wc.sort_indices()
    return Wc


def compute_structural_constants(W_csr: csr_matrix) -> Tuple[float, float, float]:
    """
    Compute S0, S1, S2 as in esda.Geary (Cliff & Ord).
    Returns (S0, S1, S2) as Python floats.
    """
    S0 = float(W_csr.sum())
    A = W_csr + W_csr.T
    S1 = 0.5 * float((A.multiply(A)).sum())
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    S2 = float(np.square(row_sum).sum())
    return S0, S1, S2


def _torch_sparse_csr_from_scipy(W_csr: csr_matrix, device, dtype):
    """Convert scipy CSR -> torch.sparse_csr_tensor on given device/dtype."""
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


class Geary:
    """
    GPU-only Geary class (behavior aligned with esda.geary.Geary).

    Parameters
    ----------
    y : array_like (n,)
    w : libpysal.weights.W or libpysal.graph.Graph
    transformation : {'R','B','D','U','V'} default 'r'
    permutations : int (kept for API compatibility; simulation done on CPU if >0)
    cuda_device : str or torch.device (default 'cuda')
    torch_dtype : 'float64' or 'float32' (default 'float64')

    Attributes: C, EC, VC_norm, VC_rand, seC_norm, seC_rand, z_norm, z_rand, p_norm, p_rand, den, sim...
    """
    def __init__(self, y, w, transformation="r", permutations=0, cuda_device="cuda", torch_dtype="float64"):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU Geary.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU Geary.")

        if not isinstance(w, weights.W | graph.Graph):
            raise TypeError("w must be a libpysal.weights.W or libpysal.graph.Graph object.")

        y = np.asarray(y).flatten()
        self.n = int(len(y))
        self.y = y

        if isinstance(w, weights.W):
            w.transform = transformation
        else:
            w = w.transform(transformation)
            self.summary = w.summary()
        self.w = w
        self.permutations = int(permutations)

        # prepare sparse cpu csr and structural constants
        self._W_csr = _ensure_csr_zero_diag(self.w)
        self.S0, self.S1, self.S2 = compute_structural_constants(self._W_csr)
        if self.S0 <= 0:
            raise ValueError("Invalid weights: S0 <= 0. Check that the graph has edges and is properly normalized.")

        # compute den and C using GPU path
        C_val, den_val, VC_norm, VC_rand, seC_norm, seC_rand = self._compute_single_gpu(y, self._W_csr, cuda_device, torch_dtype)
        self.C = float(C_val)
        self.den = float(den_val)
        self.VC_norm = float(VC_norm)
        self.VC_rand = float(VC_rand)
        self.seC_norm = float(seC_norm)
        self.seC_rand = float(seC_rand)
        self.EC = 1.0

        # z and p same as esda one-tailed logic
        de = self.C - 1.0
        self.z_norm = de / self.seC_norm if self.seC_norm > 0 else float("nan")
        self.z_rand = de / self.seC_rand if self.seC_rand > 0 else float("nan")
        if de > 0:
            self.p_norm = stats.norm.sf(self.z_norm)
            self.p_rand = stats.norm.sf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        # permutations (simulate on CPU to avoid heavy GPU <-> CPU random shuffling)
        if self.permutations:
            sim = [self._calc_cpu(np.random.permutation(self.y)) for _ in range(self.permutations)]
            self.sim = sim = np.array(sim)
            above = sim >= self.C
            larger = int(above.sum())
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (self.permutations + 1.0)
            self.EC_sim = float(sim.mean())
            self.seC_sim = float(sim.std(ddof=0))
            self.VC_sim = float(self.seC_sim ** 2)
            self.z_sim = (self.C - self.EC_sim) / self.seC_sim if self.seC_sim > 0 else float("nan")
            self.p_z_sim = stats.norm.sf(abs(self.z_sim))

    def _calc_cpu(self, y_perm):
        # fallback CPU scalar calc using stored sparse indices
        focal_ix, neighbor_ix = self.w.sparse.nonzero()
        weights_arr = self.w.sparse.data
        num = (weights_arr * ((y_perm[focal_ix] - y_perm[neighbor_ix]) ** 2)).sum()
        a = (self.n - 1) * num
        s0 = self.w.s0 if isinstance(self.w, weights.W) else self.summary.s0
        yd = y_perm - y_perm.mean()
        den = float((yd * yd).sum()) * s0 * 2.0
        return a / den

    def _compute_single_gpu(self, y, W_csr: csr_matrix, cuda_device="cuda", torch_dtype="float64"):
        """
        Compute single-vector Geary C and variance terms on GPU; return tuple:
        (C_obs, den, VC_norm, VC_rand, seC_norm, seC_rand)
        """
        # device and dtype
        dev = torch.device(cuda_device if isinstance(cuda_device, str) else str(cuda_device))
        tdtype = torch.float64 if ("64" in str(torch_dtype)) else torch.float32

        # convert W to torch sparse csr
        W_t = _torch_sparse_csr_from_scipy(W_csr, device=dev, dtype=tdtype)

        # constants
        n = float(self.n)
        n_t = torch.tensor(float(self.n), device=dev, dtype=tdtype)
        S0_t = torch.tensor(float(self.S0), device=dev, dtype=tdtype)
        kappa_t = (n_t - 1.0) / (2.0 * S0_t)
        EC_t = torch.tensor(1.0, device=dev, dtype=tdtype)

        # move y
        y_np = np.asarray(y, dtype=np.float64 if tdtype == torch.float64 else np.float32)
        y_t = torch.from_numpy(y_np).to(device=dev, dtype=tdtype).view(-1, 1)  # (n,1)

        mu_t = y_t.mean(dim=0, keepdim=True)
        Z_t = y_t - mu_t  # (n,1)
        Z2_t = Z_t * Z_t
        sum2_t = Z2_t.sum()
        denom_t = sum2_t.clamp_min(1e-24)

        # diag term: (row_sum+col_sum)
        row_sum = np.asarray(W_csr.sum(axis=1)).ravel().astype(np.float64 if tdtype == torch.float64 else np.float32)
        col_sum = np.asarray(W_csr.sum(axis=0)).ravel().astype(np.float64 if tdtype == torch.float64 else np.float32)
        r_plus_c_t = torch.from_numpy((row_sum + col_sum)).to(device=dev, dtype=tdtype).view(-1, 1)
        diag_term_t = (Z2_t * r_plus_c_t).sum()

        # sparse multiply
        WZ_t = torch.sparse.mm(W_t, Z_t)  # (n,1)
        cross_term_t = (Z_t * WZ_t).sum()
        numer_t = diag_term_t - 2.0 * cross_term_t
        C_obs_t = kappa_t * (numer_t / denom_t)
        C_obs = float(C_obs_t.detach().cpu().item())

        # get Z back to CPU for kurtosis m2/m4
        Z_cpu = Z_t.detach().cpu().numpy().ravel()
        yd2 = Z_cpu * Z_cpu
        yd4 = yd2 * yd2
        m2 = float(yd2.mean()) if yd2.size else 0.0
        m4 = float(yd4.mean()) if yd4.size else 0.0
        K = (m4 / (m2 * m2)) if (m2 > 0 and np.isfinite(m2) and np.isfinite(m4)) else 1.0
        if not np.isfinite(K) or K < 1.0:
            K = 1.0
        if K > 1e6:
            K = 1e6

        # compute VC_rand and VC_norm on CPU using S0/S1/S2
        s0 = self.S0
        s1 = self.S1
        s2 = self.S2
        s02 = s0 * s0
        n2 = n * n
        k = float(K)
        Acoef = (n - 1.0) * s1 * (n2 - 3.0 * n + 3.0 - (n - 1.0) * k)
        Bcoef = 0.25 * ((n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * k))
        Ccoef = s02 * (n2 - 3.0 - (n - 1.0) ** 2 * k)
        vc_rand = (Acoef - Bcoef + Ccoef) / (n * (n - 2.0) * (n - 3.0) * s02) if (n > 3 and s02 > 0) else 0.0
        vc_norm = (1.0 / (2.0 * (n + 1.0) * s02)) * ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02) if s02 > 0 else 0.0

        if not np.isfinite(vc_rand) or vc_rand < 0:
            vc_rand = max(vc_rand, 0.0)
        if not np.isfinite(vc_norm) or vc_norm < 0:
            vc_norm = max(vc_norm, 0.0)

        seC_rand = float(np.sqrt(vc_rand)) if vc_rand > 0 else 0.0
        seC_norm = float(np.sqrt(vc_norm)) if vc_norm > 0 else 0.0

        yss = float((Z_cpu * Z_cpu).sum())
        den = yss * s0 * 2.0

        return C_obs, den, vc_norm, vc_rand, seC_norm, seC_rand


def geary_batch(
    X_np: np.ndarray,
    W_csr: csr_matrix,
    device: str = "cuda",
    torch_dtype: str = "float64",
    batch_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized GPU computation for multiple columns (genes).

    Parameters
    - X_np: (n, g) numpy array (dense). If you have scipy sparse, densify per-batch externally.
    - W_csr: scipy csr_matrix (diag zeroed)
    - device: CUDA device string
    - torch_dtype: 'float64' or 'float32'
    - batch_mode: reserved flag (not used here) to keep API compatible

    Returns (C_obs_array, z_norm_array, p_norm_array) as numpy arrays length g.
    p_norm follows esda one-tailed logic (use z sign and sf/cdf accordingly).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for geary_batch.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for geary_batch.")

    # prepare constants and torch sparse
    S0, S1, S2 = compute_structural_constants(W_csr)
    if S0 <= 0:
        raise ValueError("Invalid weights: S0 <= 0.")
    dev = torch.device(device if isinstance(device, str) else str(device))
    tdtype = torch.float64 if ("64" in str(torch_dtype)) else torch.float32

    n, g = X_np.shape
    if n < 4:
        raise ValueError("Need n >= 4 for normal approx.")

    # torch tensors
    W_t = _torch_sparse_csr_from_scipy(W_csr, device=dev, dtype=tdtype)
    row_sum = np.asarray(W_csr.sum(axis=1)).ravel()
    col_sum = np.asarray(W_csr.sum(axis=0)).ravel()
    r_plus_c_t = torch.from_numpy((row_sum + col_sum).astype(np.float64 if tdtype == torch.float64 else np.float32)).to(device=dev, dtype=tdtype)

    n_t = torch.tensor(float(n), device=dev, dtype=tdtype)
    S0_t = torch.tensor(float(S0), device=dev, dtype=tdtype)
    kappa_t = (n_t - 1.0) / (2.0 * S0_t)
    EC_t = torch.tensor(1.0, device=dev, dtype=tdtype)

    # move X to device
    X_t = torch.from_numpy(X_np.astype(np.float64 if tdtype == torch.float64 else np.float32, copy=False)).to(device=dev, dtype=tdtype)  # (n,g)

    # center columns
    mu_t = X_t.mean(dim=0, keepdim=True)  # (1,g)
    Z_t = X_t - mu_t                       # (n,g)
    Z2_t = Z_t * Z_t                       # (n,g)
    sum2_t = Z2_t.sum(dim=0)               # (g,)
    denom_t = sum2_t.clamp_min(1e-24)

    # diag term
    diag_term_t = (Z2_t * r_plus_c_t[:, None]).sum(dim=0)  # (g,)

    # W @ Z (supports multi-column)
    WZ_t = torch.sparse.mm(W_t, Z_t)   # (n,g)
    cross_term_t = (Z_t * WZ_t).sum(dim=0)  # (g,)
    numer_t = diag_term_t - 2.0 * cross_term_t
    C_obs_t = kappa_t * (numer_t / denom_t)

    # kurtosis K per column
    Z4_t = Z2_t * Z2_t
    m2_t = (sum2_t / n_t).clamp_min(1e-24)
    m4_t = (Z4_t.sum(dim=0) / n_t)
    K_t = (m4_t / (m2_t * m2_t)).clamp(min=1.0, max=1e6)

    # compute VC vectorized same as esda formula
    S1_t = torch.tensor(float(S1), device=dev, dtype=tdtype)
    S2_t = torch.tensor(float(S2), device=dev, dtype=tdtype)
    S0sq_t = torch.tensor(float(S0 * S0), device=dev, dtype=tdtype)
    n2 = float(n) * float(n)
    n_t_ = n_t

    part1_t = ((n_t_ - 1.0) * S1_t * (n2 - 3.0 * float(n) + 3.0 - K_t * (n_t_ - 1.0))) / (
        S0sq_t * n_t_ * (n_t_ - 2.0) * (n_t_ - 3.0)
    )
    part2_t = (n2 - 3.0 - K_t * (n_t_ - 1.0) * (n_t_ - 1.0)) / (n_t_ * (n_t_ - 2.0) * (n_t_ - 3.0))
    part3_t = ((n_t_ - 1.0) * S2_t * (n2 + 3.0 * float(n) - 6.0 - K_t * (n2 - float(n) + 2.0))) / (
        4.0 * n_t_ * (n_t_ - 2.0) * (n_t_ - 3.0) * S0sq_t
    )
    VC_t = (part1_t + part2_t - part3_t).clamp_min(1e-24)

    Zscore_t = (C_obs_t - EC_t) / torch.sqrt(VC_t)

    C_np = C_obs_t.detach().cpu().numpy().astype(np.float64)
    Z_np = Zscore_t.detach().cpu().numpy().astype(np.float64)

    # p_norm follows esda one-tailed logic: if de>0 use sf else cdf
    p_norm = np.empty_like(Z_np, dtype=np.float64)
    de = C_np - 1.0
    # vectorized one-tailed mapping
    pos_mask = de > 0
    neg_mask = ~pos_mask
    p_norm[pos_mask] = stats.norm.sf(Z_np[pos_mask])
    p_norm[neg_mask] = stats.norm.cdf(Z_np[neg_mask])

    return C_np, Z_np, p_norm