import numpy as np
from scipy import sparse

def calc_scatters(K):
    """Calculate scatter matrix"""
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)

    scatters = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            scatter = K1[j+1] - K1[i]
            if i > 0:
                scatter += K2[i][i] - K2[i][j+1] - K2[j+1][i] + K2[j+1][j+1]
            scatters[i][j] = scatter
    
    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True):
    """Change point detection with dynamic programming"""
    n = K.shape[0]
    J = calc_scatters(K)

    if lmin < 1:
        lmin = 1
    if lmax > n:
        lmax = n

    I = 1e101 * np.ones((n+1, ncp+1))
    I[lmin-1:lmax, 0] = 0
    I[0, 0] = 0

    if backtrack:
        P = np.zeros((n+1, ncp+1), dtype=int)
    
    for k in range(1, ncp+1):
        for i in range((k+1)*lmin-1, n+1):
            tmin = max(k*lmin-1, i-lmax)
            tmax = i-lmin+1
            for t in range(tmin, tmax):
                cost = J[t, i-1]
                if cost + I[t, k-1] < I[i, k]:
                    I[i, k] = cost + I[t, k-1]
                    if backtrack:
                        P[i, k] = t+1

    # Backtrack
    if backtrack:
        cp = np.zeros(ncp, dtype=int)
        cur = n
        for k in range(ncp, 0, -1):
            cp[k-1] = P[cur, k]
            cur = P[cur, k]
        return cp
    else:
        return I

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Automatic selection of number of change points"""
    n = K.shape[0]
    m = np.zeros(ncp)
    
    for k in range(ncp):
        cp = cpd_nonlin(K, k+1, **kwargs)
        start_idx = int(cp[max(0, k-1)])
        end_idx = int(cp[k])
        if start_idx >= end_idx or end_idx > n:
            continue
        m[k] = np.sum(np.diag(K)[start_idx:end_idx]) / (end_idx - start_idx)
    
    m = np.concatenate(([0], m, [vmax]))
    scores = np.zeros(ncp)
    
    for k in range(ncp):
        scores[k] = (m[k+1] - m[k]) / (m[k+2:].max() - m[k+1] + 1e-10) * desc_rate
    
    n_changes = scores.argmax() + 1
    cp = cpd_nonlin(K, n_changes, **kwargs)
    
    return cp.astype(int)