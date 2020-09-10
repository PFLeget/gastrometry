"""
.. module:: meanify
"""
import numpy as np
from scipy.stats import binned_statistic
import fitsio
import os


class meanify1D_wrms(object):
    """Take data, build a 1d average, and write output average.

    :param bin_spacing: Bin_size, resolution on the mean function. (default=0.3)
    """
    def __init__(self, bin_spacing=0.3):

        self.bin_spacing = bin_spacing

        self.coords = []
        self.params = []
        self.params_err = []

    def add_data(self, coord, param, params_err=None):
        """
        Add new data to compute the mean function. 

        :param coord: Array of coordinate of the parameter.
        :param param: Array of parameter.
        """
        self.coords.append(coord)
        self.params.append(param)
        if params_err is None:
            self.params_err = None
        else:
            self.params_err.append(params_err)
    
    def meanify(self, x_min=None, x_max=None):
        """
        Compute the mean function.
        """
        params = np.concatenate(self.params)
        coords = np.concatenate(self.coords)
        if self.params_err is not None:
            params_err = np.concatenate(self.params_err)
        else:
            params_err = np.ones_like(params)

        weights = 1./params_err**2

        if x_min is None:
            x_min = np.min(coords)
        if x_max is None:
            x_max = np.max(coords)

        nbin = int((x_max - x_min) / self.bin_spacing)

        binning = np.linspace(x_min, x_max, nbin)
        Filter = np.array([True]*nbin)


        sum_wpp, x0, bin_target = binned_statistic(coords, weights*params*params,
                                                   bins=binning, statistic='sum')

        sum_wp, x0, bin_target = binned_statistic(coords, weights*params,
                                                  bins=binning, statistic='sum')

        sum_w, x0, bin_target = binned_statistic(coords, weights,
                                                 bins=binning, statistic='sum')

        average = sum_wp / sum_w
        wvar = (1. / sum_w) * (sum_wpp - 2.*average*sum_wp + average*average*sum_w)
        wrms = np.sqrt(wvar)

        # get center of each bin 
        x0 = x0[:-1] + (x0[1] - x0[0])/2.

        # remove any entries with nan (counts == 0 and non finite value in
        # the 1D statistic computation) 
        self.x0 = x0
        self.average = average
        self.wrms= wrms

    def save_results(self, name_output='wrms_mag.fits'):
        """
        Write output mean function.
        
        :param name_output: Name of the output fits file. (default: 'mean_gp.fits')
        """
        dtypes = [('X0', self.x0.dtype, self.x0.shape),
                  ('AVERAGE', self.average.dtype, self.average.shape),
                  ('WRMS', self.wrms.dtype, self.wrms.shape),
                  ]
        data = np.empty(1, dtype=dtypes)
        
        data['X0'] = self.x0
        data['AVERAGE'] = self.average
        data['WRMS'] = self.wrms

        with fitsio.FITS(name_output,'rw',clobber=True) as f:
            f.write_table(data, extname='average_solution')


if __name__ == "__main__":

    import pylab as plt

    def polynomial(x):
        return 8e-2*x*x - 5e-1*x -1.
    
    m = meanify1D_wrms(bin_spacing=0.05)
    np.random.seed(42)
    for i in range(300):
        print(i+1)
        N = 10000
        x = np.random.uniform(-6,6, size=N)
        y = polynomial(x)
        y += np.random.normal(scale=0.5, size=N)
        x += np.random.normal(scale=0.5, size=N)
        
        plt.scatter(x, y, c='b', s=1, alpha=0.05)
        
        m.add_data(x, y, params_err=None)

    m.meanify(x_min=-4.9, x_max=4.9)
    X = np.linspace(-5,5, 100)
    plt.plot(X, polynomial(X), 'k--', lw=5)
    plt.plot(m.x0, m.average, 'r', lw=3)
    plt.plot(m.x0, m.average+3*m.wrms, 'r--', lw=3)
    plt.plot(m.x0, m.average-3*m.wrms, 'r--', lw=3)

    m.save_results()
