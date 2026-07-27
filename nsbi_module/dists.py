
import numpy as np
from scipy.stats import gamma

def truncated_gamma(a, scale, low, upp, size):
    samples = []
    while len(samples) < size:
        sample = gamma.rvs(a, scale=scale, size=size - len(samples))
        truncated_sample = sample[(sample >= low) & (sample <= upp)]
        samples.extend(truncated_sample)
    return np.array(samples)


