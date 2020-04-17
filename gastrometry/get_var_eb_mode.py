import cPickle
import os
import glob
import numpy as np
import pylab as plt
from iminuit import Minuit
import treegp
import warnings

def vk(d, var=1, l=1, cst=0):
    coord = np.array([d, np.zeros_like(d)]).T
    kernel = treegp.VonKarman(length_scale=l)
    pcf = kernel.__call__(coord,Y=np.zeros_like(coord))[:,0]
    return (var * pcf) + cst

class get_var_ebmode:

    def __init__(self, pkl_file):

        dic = cPickle.load(open(pkl_file))

        self.exp_id = dic['exp_id']
        self.xie = dic['2pcf_stat']['xie']
        self.xib = dic['2pcf_stat']['xib']
        self.d = np.exp(dic['2pcf_stat']['logr'])

        self.Filtre = np.array([True]*len(self.xie))
        self.Filtre &= np.isfinite(self.xie)
        self.Filtre &= np.isfinite(self.xib)
        self.Filtre &= np.isfinite(self.d)
        self.Filtre &= (self.d>2e-2)

    def chi2(self, l, xi):
        model = vk(self.d[self.Filtre], var=1., l=l, cst=0.)
        F = np.array([model, np.ones_like(model)]).T
        FWF = np.dot(F.T, np.eye(len(model))).dot(F)
        Y = xi[self.Filtre].reshape((len(xi[self.Filtre]), 1))
        try:
            alpha = np.linalg.inv(FWF).dot(np.dot(F.T, np.eye(len(model))).dot(Y))
            alpha[0] = abs(alpha[0])
            residuals = xi[self.Filtre] - ((alpha[0] * model) + alpha[1])
            self.chi2_value = residuals.dot(np.eye(len(model))).dot(residuals.reshape((len(model), 1)))
            self.alpha = alpha
            if np.isfinite(self.alpha[0][0]) and np.isfinite(self.alpha[1][0]):
                return self.chi2_value
            else:
                return +999 * len(residuals)
        except:
            return +999 * np.sum(self.Filtre)

    def chi2e(self, params):
        chi2 = self.chi2(params[0], self.xie)
        return chi2

    def chi2b(self, params):
        chi2 = self.chi2(params[0], self.xib)
        return chi2

    def _minimize_minuit_e(self, p0 = [3000./3600.]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = Minuit.from_array_func(self.chi2e, p0, print_level=0)
            self.m.migrad()
        results = [self.m.values[key] for key in self.m.values.keys()]
        Filtre = np.isfinite(self.xie)
        self.result_e = [self.alpha[0][0], results[0], self.alpha[1][0]]

    def _minimize_minuit_b(self, p0 = [3000./3600.]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = Minuit.from_array_func(self.chi2b, p0, print_level=0)
            self.m.migrad()
        results = [self.m.values[key] for key in self.m.values.keys()]
        Filtre = np.isfinite(self.xib)
        VAR = np.std(self.xib[Filtre & ~self.Filtre])
        if self.alpha[0][0]> 5*VAR:
            self.alpha[0][0] = VAR
        self.result_b = [self.alpha[0][0], results[0], self.alpha[1][0]]

    def minimize_minuit(self, p0 = [1.5]):

        self._minimize_minuit_e(p0=p0)
        self._minimize_minuit_b(p0=p0)

    def plot_eb_results(self):

        plt.figure()
        plt.scatter(self.d, self.xie, c='b', alpha=0.5)
        plt.plot(self.d, vk(self.d, var=self.result_e[0],
                       l=self.result_e[1],
                       cst=self.result_e[2]), 'b', lw=3)
        plt.scatter(self.d, self.xib, c='r', alpha=0.5)
        plt.plot(self.d, vk(self.d, var=self.result_b[0],
                       l=self.result_b[1],
                       cst=self.result_b[2]), 'r', lw=3)
        plt.xscale('log')
        plt.ylim(np.min(self.xie[np.isfinite(self.xie)]), np.max(self.xie[np.isfinite(self.xie)]))
        plt.xlim(self.d[np.isfinite(self.d)][0], self.d[np.isfinite(self.d)][-1])

def comp_eb_mode_info(gp_outputs_rep="/sps/lsst/users/leget/HSC/v2/astro_VK/", rep_out=''):

    dirs = os.path.join(gp_outputs_rep, "*/gp_output_*.pkl")
    gp_outputs = glob.glob(dirs)
    i = 0
    nrep = len(gp_outputs)
    dic_out = {}
    for pkl_name in gp_outputs:
        print "%i / %i"%((i+1, nrep))
        gve = get_var_ebmode(pkl_name)
        gve.minimize_minuit()
        dic_out.update({gve.exp_id:{'e_stat':gve.result_e,
                                            'b_stat':gve.result_b}})
        i += 1

    pkl_name = os.path.join(rep_out, 'eb_stat.pkl')
    pkl_file = open(pkl_name, 'w')
    cPickle.dump(dic_out, pkl_file)
    pkl_file.close()
    

if __name__ == "__main__":

    #pkl_name = "/sps/lsst/users/leget/HSC/v2/astro_VK/195834_i2/gp_output_195834.pkl"
    #pkl_name = "/sps/lsst/users/leget/HSC/v2/astro_VK/143162_z/gp_output_143162.pkl"
    comp_eb_mode_info(gp_outputs_rep="/sps/lsst/users/leget/HSC/v2/astro_VK/", rep_out='')

