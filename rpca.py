import numpy as np

def rpca(X, lambd=None, max_iter=100, tol=1e-7):
    """
    Perform Robust PCA on matrix X.
    Returns:
        L: low‑rank component
        S: sparse component
    """
    if lambd is None:
        lambd = 1.0 / np.sqrt(max(X.shape))
    # Scale initial Y by the maximum of the spectral and infinity norms
    norm2 = np.linalg.norm(X, 2)
    norm_inf = np.linalg.norm(X, np.inf)
    scaling = max(norm2, norm_inf)
    Y = X / scaling if scaling != 0 else X
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    mu = 1.25 / norm2 if norm2 != 0 else 1.0
    rho = 1.5
    for _ in range(max_iter):
        # Update L
        U, sigma, Vt = np.linalg.svd(X - S + Y/mu, full_matrices=False)
        svp = np.sum(sigma > 1.0/mu)
        if svp < 1:
            svp = 1
        L = U[:, :svp] @ np.diag(sigma[:svp] - 1.0/mu) @ Vt[:svp, :]
        # Update S
        S = np.maximum(X - L + Y/mu - lambd/mu, 0) + np.minimum(X - L + Y/mu + lambd/mu, 0)
        # Update Y
        Z = X - L - S
        Y = Y + mu * Z
        mu = mu * rho
        if np.linalg.norm(Z, 'fro') < tol:
            break
    return L, S

def decompose_correlation(corr, lambd=None, max_iter=100, tol=1e-7):
    """Apply RPCA to correlation matrix."""
    L, S = rpca(corr, lambd, max_iter, tol)
    return L, S

def compute_sparse_scores(S, etf_names):
    """Score per ETF = sum of absolute values of its row in S (idiosyncratic shock magnitude)."""
    scores = {etf_names[i]: np.sum(np.abs(S[i, :])) for i in range(len(etf_names))}
    return scores
