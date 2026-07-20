import numpy as np

def autocorr(x, lag=1):
    return np.corrcoef(x[:-lag], x[lag:])[0,1]

def mixing_curve(series, max_lag=30):
    return [autocorr(series, lag) for lag in range(1, max_lag)]