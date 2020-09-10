import numpy as np
import pylab as plt
import fitsio
import os
import pickle
import glob
from gastrometry import meanify1D_wrms

class comp_wrms_vs_mag(object):

    def __init__(self, rep_out, bin_spacing=0.05, gp_corrected=True):

        self.rep_out = rep_out
        self.bin_spacing = bin_spacing
        self.mean = {}
        self.gp_corrected = gp_corrected

        mean_du = meanify1D_wrms(bin_spacing=self.bin_spacing)
        mean_dv = meanify1D_wrms(bin_spacing=self.bin_spacing)

        self.mean.update({'du':mean_du,
                          'dv':mean_dv})

    def stack_visits(self):

        for f in self.rep_out:
            print(f)
            file_out = glob.glob(os.path.join(f, 'gp_output*'))
            try:
                dic_out = pickle.load(open(file_out[0], 'rb'))
                dic_in = pickle.load(open(os.path.join(f,'input.pkl'), 'rb'))
            except:
                print('for this rep there is nothing: ', file_out)
            
            for coord in ['u', 'v']:
                mag = dic_in['dic_all']['magic_mag']
                residuals = dic_out['gp_output']['gp%s.d%s_all'%((coord, coord))]
                if self.gp_corrected:
                    residuals -= dic_out['gp_output']['gp%s.d%s_all_predict'%((coord, coord))]
                residuals_err = dic_out['gp_output']['gp%s.d%s_err_all'%((coord, coord))]
                    
                self.mean['d%s'%(coord)].add_data(mag, residuals, params_err=residuals_err)

    def comp_mean(self):
            
        for coord in ['u', 'v']:
            if len(self.mean['d%s'%(coord)].params) !=0:
                self.mean['d%s'%(coord)].meanify(x_min=None, x_max=None)

    def save_results(self, rep_out):
        if self.gp_corrected:
            gp_corr = 'gp_corrected'
        else:
            gp_corr = 'no_gp_corrected'
            
        for coord in ['u', 'v']:
            file_name = 'wrms_vs_mag_d%s_%s.fits'%((coord, gp_corr))
            fits_file = os.path.join(rep_out, file_name)
            if len(self.mean['d%s'%(coord)].params)!=0:
                self.mean['d%s'%(coord)].save_results(name_output=fits_file)

def run_ma_poule_wrms_vs_mag(rep_out, bin_spacing=0.05,
                             gp_corrected=True):

    # across all filters
    reps_out = glob.glob(os.path.join(rep_out,'*'))

    path_wrms = os.path.join(rep_out, 'wrms_vs_mag')
    os.system('mkdir %s'%(path_wrms))

    cm = comp_wrms_vs_mag(reps_out, bin_spacing=bin_spacing,
                          gp_corrected=gp_corrected)
    cm.stack_visits()
    cm.comp_mean()
    cm.save_results(path_wrms)

if __name__ == "__main__":

    rep_out = '/pbs/home/l/leget/sps_lsst/HSC/v3.3/astro_VK'
    run_ma_poule_wrms_vs_mag(rep_out, bin_spacing=0.05,
                             gp_corrected=True)
