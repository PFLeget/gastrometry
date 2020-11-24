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

def median_check_finite(x):
    """
    Median using biweight_median, but remove the nan.

    :param sample: 1d numpy array. The sample where you want
                   to compute the median with outlier rejection.
    """
    median = np.zeros_like(x[0])
    for i in range(len(median)):
        Filtre = np.isfinite(x[:,i])
        if np.sum(Filtre) == 0:
            median[i] = np.nan
        else:
            median[i] = biweight_median(x[:,i][Filtre])
    return median

def return_var_map(weight, xi):
    N = int(np.sqrt(len(xi)))
    var = np.diag(np.linalg.inv(weight))
    VAR = np.zeros(N*N)
    I = 0
    for i in range(N*N):
        if xi[i] !=0:
            VAR[i] = var[I]
            I+=1
        if I == len(var):
            break
    VAR = VAR.reshape(N,N) + np.flipud(np.fliplr(VAR.reshape(N,N)))
    if N%2 == 1:
        VAR[int(N/2), int(N/2)] /= 2. 
    return VAR

def vcorr(x,y,dx,dy,
          rmin=5./3600., rmax=1.5, dlogr=0.05,
          maxpts = 30000):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error
    for the supplied sample of data, using brute-force pair counting.
    Output are the following functions:
    logr - mean log of radius in each bin
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    xi_- - <vr1 vr2 - vt1 vt2>
    xi_x - <vr1 vt2 + vt1 vr2>
    xi_z2 - <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2>
    """
    if len(x) > maxpts:
        # Subsample array to get desired number of points
        rate = float(maxpts) / len(x)
        print("Subsampling rate {:5.3f}%".format(rate*100.))
        use = np.random.random(len(x)) <= rate
        x = x[use]
        y = y[use]
        dx = dx[use]
        dy = dy[use]
    print("Length ",len(x))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))
    # Omit self-pairs
    use = i1!=i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make complex separation vector
    dr = 1j * (y[i2]-y[i1])
    dr += x[i2]-x[i1]

    # log radius vector used to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    bins = int(np.ceil(np.log(rmax/rmin)/dlogr))
    hrange = (logrmin, logrmin+bins*dlogr)
    counts = np.histogram(logdr, bins=bins, range=hrange)[0]
    logr = np.histogram(logdr, bins=bins, range=hrange, weights=logdr)[0] / counts

    # First accumulate un-rotated stats
    v =  dx + 1j*dy
    vv = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0]/counts
    vv = v[i1] * v[i2]
    xiz2 = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0]/counts

    # Now rotate into radial / perp components
    vv *= np.conj(dr)
    vv *= np.conj(dr)
    dr = dr.real*dr.real + dr.imag*dr.imag
    vv /= dr
    del dr
    ximinus = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0]/counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    return logr, xiplus, ximinus, xicross, xiz2

def xiB(logr, xiplus, ximinus):
    """
    Return estimate of pure B-mode correlation function
    """
    # Integral of d(log r) ximinus(r) from r to infty:
    dlogr = np.zeros_like(logr)
    dlogr[1:-1] = 0.5*(logr[2:] - logr[:-2])
    tmp = np.array(ximinus) * dlogr
    integral = np.cumsum(tmp[::-1])[::-1]
    return 0.5*(xiplus-ximinus) + integral

def vcorr2d(x,y,dx,dy,
            rmax=1., bins=513):
    """
    Produce 2d 2-point correlation function of total displacement power
    for the supplied sample of data, using brute-force pair counting.
    Output are 2d arrays giving the 2PCF and then the number of pairs that
    went into each bin.  The 2PCF calculated is
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    Note that each pair is counted only once.  So to count all pairs one can
    average xi_+ with itself reflected about the origin.
    """

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]

    print("Length ",len(x))

    ind = np.linspace(0,len(x)-1,len(x)).astype(int)
    i1, i2 = np.meshgrid(ind,ind)
    Filtre = (i1 != i2)
    i1 = i1.reshape(len(x)**2)
    i2 = i2.reshape(len(x)**2)
    Filtre = Filtre.reshape(len(x)**2)

    i1 = i1[Filtre]
    i2 = i2[Filtre]
    del Filtre
    print(np.shape(i1))

    # Make separation vectors and count pairs
    yshift = y[i2]-y[i1]
    xshift = x[i2]-x[i1]
    counts = np.histogram2d(xshift,yshift, bins=bins, range=hrange)[0]

    # Accumulate displacement sums
    v =  dx + 1j*dy
    #print 'xiplus' ##
    vv = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus, X, Y  = np.histogram2d(xshift,yshift, bins=bins, range=hrange, weights=vv)

    xiplus /= counts
    x = copy.deepcopy(X[:-1]) + (X[1] - X[0])/2.
    y = copy.deepcopy(Y[:-1]) + (Y[1] - Y[0])/2.
    x , y = np.meshgrid(x,y)

    return xiplus.T, x, y
