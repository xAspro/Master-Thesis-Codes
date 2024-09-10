import numpy as np
# Checked once.. Seems like no file is importing this file
# Warning: This file is not run in the project

def bootstrap(data, nsamples, statistic, alpha):

    n = len(data)
    rindices = np.random.randint(n, size=(nsamples, n))
    samples = data[rindices]
    
    stats = statistic(samples, axis=1)

    a = np.percentile(stats, alpha)
    b = np.percentile(stats, 100.0-alpha)

    return np.abs(a-b)/2.0

