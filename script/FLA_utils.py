import numpy as np


def acf_1d(f, max_lag):
    """
    Unbiased normalized autocorrelation (mean-centered, variance normalized)
    Returns rho[0..max_lag], with rho[0]=1.
    input 
    - f: 1D array of values (e.g. fitness values along a trajectory)
    - max_lag: maximum lag to compute autocorrelation for (non-negative integer)
    output
    - rho: 1D array of autocorrelation values for lags 0..max
    """
    f = np.asarray(f, dtype=float)
    f = f - f.mean()
    denom = np.dot(f, f)
    if denom == 0 or np.isnan(denom):
        return np.full(max_lag + 1, np.nan)
    rho = np.empty(max_lag + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        rho[k] = np.dot(f[:-k], f[k:]) / denom
    return rho

def p1_rho1(f):
    """
    p(1) interpreted as lag-1 autocorrelation rho(1).
    ---
    input
    - f: 1D array of values (e.g. fitness values along a trajectory)
    ---
    output
    - float: the lag-1 autocorrelation value
    """
    
    rho = acf_1d(f, max_lag=1)
    return float(rho[1])

def autocorr_length_first_e_fold(rho):
    """
    Correlation length as first lag where rho(lag) <= 1/e.
    If it never drops below 1/e, returns max_lag.
    ---
    input
    - rho: 1D array of autocorrelation values for lags 0..max_lag, with rho[0]=1.
    ---
    output
    - float: the correlation length defined as the first lag where rho(lag) <= 1/e, or max_lag if it never drops below 1/e.
    """

    if rho is None or len(rho) < 2:
        return np.nan
    thresh = 1.0 / np.e
    for k in range(1, len(rho)):
        if np.isfinite(rho[k]) and rho[k] <= thresh:
            return float(k)
    return float(len(rho) - 1)

def neutrality_rate(f, eps):
    """
    Neutrality = fraction of steps with |Δf| <= eps.
    ---
    input
    - f: 1D array of values (e.g. fitness values along a trajectory)
    - eps: non-negative float, the threshold for considering a step as neutral
    ---
    output
    - float: the neutrality rate
    """
    f = np.asarray(f, dtype=float)
    df = np.diff(f)
    if len(df) == 0:
        return np.nan
    return float(np.mean(np.abs(df) <= eps))

def choose_eps(df_all, mode= "global_range_frac", frac= 1e-3):
    """
    Choose eps for neutrality rate based on the distribution of |Δf| values across all trajectories.
    ---
    input
    - df_all: list of 1D arrays of |Δf| values across all trajectories
    - mode: method for choosing eps, either "global_range_frac" or "global_std_frac"    
    - frac: non-negative float, the fraction to apply to the global range or std to get eps
    ---
    output
    - float: the chosen eps value
    """
    x = np.asarray(df_all, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if mode == "global_range_frac":
        return float(frac * (x.max() - x.min()))
    elif mode == "global_std_frac":
        return float(frac * x.std(ddof=0))
    else:
        raise ValueError(f"Unknown eps mode: {mode}")
