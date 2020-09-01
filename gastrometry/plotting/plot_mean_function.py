import numpy as np
import pylab as plt
import pyloka
import os
import fitsio
import pickle

def get_ccd_sting(ccd_id):
    if ccd_id / 10. < 1.:
        return '00' + str(ccd_id)
    if ccd_id / 10. >= 1. and ccd_id /10. < 10.:
        return '0' + str(ccd_id)
    if ccd_id /10. >= 10.:
        return str(ccd_id)

def get_tp_coord(xccd, yccd, ccd_id=0):
    rep_ccd = '/sps/lsst/HSC/prod.2020-03.calib/dbimage_UI5XG7I/data/7969_SSP_COSMOS_DEEP_NW_57451_0_i2_58258/'
    rep_ccd += '58258p' + get_ccd_sting(ccd_id)
    ccd_fits_name = os.path.join(rep_ccd, 'calibrated.fz')
    u, v = pyloka.pix2tp(xccd, yccd, ccd_fits_name)
    return u, v


class get_mean(object):

    def __init__(self, fits_file):

        self.mean = fitsio.read(fits_file)
        self.params0 = self.mean['PARAMS0'][0]
        self.coord0 = self.mean['COORDS0'][0]
        self.y0 = self.mean['_AVERAGE'][0]
        self.u0 = self.mean['_U0'][0]
        self.v0 = self.mean['_V0'][0]


def build_mean_in_tp(rep_mean='~/sps_lsst/HSC/v3.3/astro_VK/mean_function/all/'):

    #'mean_du_23_all.fits'
    dic_mean = {'du':{},
                'dv':{}}

    for comp in ['du', 'dv']:
        y0 = []
        u0 = []
        v0 = []

        for i in range(105):
            try:
            #if True:
                print(comp, i)
                File = os.path.join(rep_mean, 'mean_%s_%i_all.fits'%((comp, i)))
                gm = get_mean(File)
                y0.append(gm.params0)
                u, v = get_tp_coord(gm.coord0[:,0], gm.coord0[:,1], ccd_id=i)
                u0.append(u)
                v0.append(v)
            except:
                print('file does not exist')

        dic_mean[comp]['y0'] = np.concatenate(y0)
        dic_mean[comp]['u0'] = np.concatenate(u0)
        dic_mean[comp]['v0'] = np.concatenate(v0)

    File = open('mean_tp.pkl', 'wb')
    pickle.dump(dic_mean, File)
    File.close()

    return dic_mean

if __name__ == '__main__':

    dic = build_mean_in_tp(rep_mean='~/sps_lsst/HSC/v3.3/astro_VK/mean_function/all/')
