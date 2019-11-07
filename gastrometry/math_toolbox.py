import numpy as np
import copy
from astropy.stats import median_absolute_deviation as mad_astropy

def biweight_median(sample, CSTD=6.):
    """
    Median with outlier rejection using mad clipping.
    Using the biweight described in Beers 1990 (Used originaly
    for finding galaxy clusters redshfit).

    :param sample: 1d numpy array. The sample where you want
                   to compute the median with outlier rejection.
    :param CSTD:   float. Constant used in the algorithm of the
                   Beers 1990. [default: 6.]
    """
    M = np.median(sample)
    iterate = [copy.deepcopy(M)]
    mu = (sample-M) / (CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = (sample-M) * ((1.-mu**2)**2)
    down = (1.-mu**2)**2
    M += np.sum(up[Filtre])/np.sum(down[Filtre])

    iterate.append(copy.deepcopy(M))
    i = 1
    while abs((iterate[i-1]-iterate[i])/iterate[i])<0.001:
        mu = (sample-M) / (CSTD*mad_astropy(sample))
        Filtre = (abs(mu)<1)
        up = (sample-M) * ((1.-mu**2)**2)
        down = (1.-mu**2)**2
        M += np.sum(up[Filtre])/np.sum(down[Filtre])
        iterate.append(copy.deepcopy(M))
        i += 1
        if i == 100 :
            print('Fail to converge')
            break
    return M

def biweight_mad(sample, CSTD=9.):
    """
    Median absolute deviation with outlier rejection using mad clipping.
    Using the biweight described in Beers 1990 (Used originaly
    for finding galaxy clusters peculiar velocity dispersion).

    :param sample: 1d numpy array. The sample where you want
                   to compute the mad with outlier rejection.
    :param CSTD:   float. Constant used in the algorithm of the
                   Beers 1990. [default: 9.]
    """
    M = biweight_median(sample)
    mu = (sample-M) / (CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = ((sample-M)**2)*((1.-mu**2)**4)
    down = (1.-mu**2)*(1.-5.*mu**2)
    mad = np.sqrt(len(sample)) * (np.sqrt(np.sum(up[Filtre]))/abs(np.sum(down[Filtre])))
    return mad
