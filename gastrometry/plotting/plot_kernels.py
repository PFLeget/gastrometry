import pylab as plt
import treegp
import numpy as np
import warnings
from iminuit import Minuit

dist = np.linspace(0.0, 6, 100)
coord_corr = np.array([dist, np.zeros_like(dist)]).T

kernel_rbf = "1 * RBF(length_scale=1)"
interp_rbf = treegp.GPInterpolation(kernel=kernel_rbf,
                                    normalize=False,
                                    white_noise=0.)
ker_rbf = interp_rbf.kernel_template
corr_rbf = ker_rbf.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]


def vk(l):
    kernel_vk = "1 * VonKarman(length_scale=%f)"%(l)
    interp_vk = treegp.GPInterpolation(kernel=kernel_vk,
                                       normalize=False,
                                       white_noise=0.)
    ker_vk = interp_vk.kernel_template
    corr_vk = ker_vk.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]
    return corr_vk

def chi2_fct(param):
    residuals = corr_rbf - vk(param[0])
    return np.sum(residuals**2)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = Minuit.from_array_func(chi2_fct, [1.1], print_level=0)
    m.migrad()
    results = [m.values[key] for key in m.values.keys()]
    L = results[0]
    
plt.figure(figsize=(5,3))
plt.subplots_adjust(top=0.99, bottom=0.15, right=0.99)
plt.plot(dist, corr_rbf, 'b', lw=3, label='Gaussian')
plt.plot(dist, vk(L), 'r', lw=3, label='Von Karman')
plt.xlabel('$r$', fontsize=14)
plt.ylabel('$\\xi(r)$', fontsize=14)
plt.xlim(0, 6)
plt.ylim(0, 1.1)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.legend(loc=1)
plt.savefig('rbf_vs_vk_isotropic.pdf')
