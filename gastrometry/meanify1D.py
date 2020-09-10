"""
.. module:: meanify
"""
import numpy as np
from scipy.stats import binned_statistic_2d
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
    
    def meanify(self, x_min=None, x_max=None)
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


        self._average = average
        self._wrms = wrms
        self._x0 = x0

        # get center of each bin 
        x0 = x0[:-1] + (x0[1] - x0[0])/2.

        Filter &= np.isfinite(average)
        Filter &= np.isfinite(wrms)

        # remove any entries with nan (counts == 0 and non finite value in
        # the 1D statistic computation) 
        self.x0 = x0[Filter]
        self.average = average[Filter]
        self.wrms= wrms[Filter]

    def save_results(self, name_output='wrms_mag.fits'):
        """
        Write output mean function.
        
        :param name_output: Name of the output fits file. (default: 'mean_gp.fits')
        """
        dtypes = [('X0', self.coords0.dtype, self.coords0.shape),
                  ('AVERAGE', self.params0.dtype, self.params0.shape),
                  ('WRMS', self.wrms0.dtype, self.wrms0.shape),
                  ('_AVERAGE', self._average.dtype, self._average.shape),
                  ('_WRMS', self._wrms.dtype, self._wrms.shape),
                  ('_X0', self._u0.dtype, self._u0.shape),
                  ]
        data = np.empty(1, dtype=dtypes)
        
        data['X0'] = self.x0
        data['AVERAGE'] = self.average
        data['WRMS'] = self.wrms
        data['_AVERAGE'] = self._average
        data['_WRMS'] = self._wrms
        data['_X0'] = self._x0

        with fitsio.FITS(name_output,'rw',clobber=True) as f:
            f.write_table(data, extname='average_solution')
