import numpy as np
from scipy.stats import spearmanr, pearsonr

def ic(signal, returns):
    return pearsonr(signal, returns)[0]

def rank_ic(signal, returns):
    return spearmanr(signal, returns)[0]