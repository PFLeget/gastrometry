import numpy as np
import pylab as plt
import pickle
import os
import glob
import warnings
from astropy.utils.console import ProgressBar

def get_varE(logr, xie):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask0 = ((np.exp(logr)>2e-2) & (np.exp(logr)<1e-1))
        mask0 &= np.isfinite(xie)
        return np.mean(xie[mask0])

def get_shot_noise_stat(rep_out = '/pbs/home/l/leget/sps_lsst/HSC/v3.3/astro_VK_with_mean/'):

    gp_outputs = glob.glob(os.path.join(rep_out, '*/gp_output*'))

    n_train = []
    n_valid = []

    varestart = []
    vareend = []
    delta_var_e_mode = []
    delta_var_e_mode_normed = []

    nvisits = len(gp_outputs)

    with ProgressBar(nvisits) as bar:
        for i in range(nvisits):
            dic = pickle.load(open(gp_outputs[i], 'rb'))
            n_train.append(len(dic['input_data']['indice_train']))
            n_valid.append(len(dic['input_data']['indice_test']))
            var_e_start = get_varE(dic['2pcf_stat']['logr_test'], dic['2pcf_stat']['xie_test'])
            var_e_end = get_varE(dic['2pcf_stat']['logr_residuals'], dic['2pcf_stat']['xie_residuals'])
            varestart.append(var_e_start)
            vareend.append(var_e_end)

            delta_var_e_mode.append(var_e_start - var_e_end)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                delta_var_e_mode_normed.append((var_e_start - var_e_end) / var_e_start)
            bar.update()

    pkl_output = open('shot_noise.pkl', 'wb')
    dic = {'n_train':np.array(n_train),
           'n_valid':np.array(n_valid),
           'delta_var_e_mode':np.array(delta_var_e_mode),
           'delta_var_e_mode_normed': np.array(delta_var_e_mode_normed),
           'var_e_start':np.array(varestart),
           'var_e_end':np.array(vareend),}
    pickle.dump(dic, pkl_output)
    pkl_output.close()

if __name__ == '__main__':

    get_shot_noise_stat(rep_out = '/pbs/home/l/leget/sps_lsst/HSC/v3.3/astro_VK_with_mean/')

