import numpy as np

def dv_rate(series):
    threshold = np.percentile(series, 95)
    tail_prob = np.mean(series > threshold)
    return -np.log(tail_prob + 1e-8)