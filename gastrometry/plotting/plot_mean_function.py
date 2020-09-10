import numpy as np
import pylab as plt
try:
    import pyloka
except:
    print('poloka-core is not installed')
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

def plot_fov_mean(file_tp):

    dic = pickle.load(open(file_tp, 'rb'))

    for comp in ['du', 'dv']:
        plt.figure(figsize=(13, 9))
        plt.subplots_adjust(left=0.1, bottom=0.1, top=0.97, right=0.95)
        plt.scatter(dic[comp]['u0'], dic[comp]['v0'], c=dic[comp]['y0'],
                    vmin=-2, vmax=2, s=1, cmap=plt.cm.inferno)
        cb = plt.colorbar()
        cb.set_label(comp+' (mas)', fontsize=20)
        cb.ax.tick_params(labelsize=20)
        plt.xticks(fontsize=20)
        plt.xlabel('u (degree)', fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('v (degree)', fontsize=20)
        plt.axis('equal')
        plt.savefig(comp+'_mean.png')

def plot_mean_ccd(fits_file_du,
                  fits_file_dv, name= '',
                  cmap=None, MAX=2, name_fig=None):

    FONT = 10
    
    plt.figure(figsize=(6.5,5))
    plt.subplots_adjust(wspace=0.4, top=0.9)#, right=0.95, left=0.07)
    plt.subplot(1,2,1)
    mean = fitsio.read(fits_file_du)
    y0 = mean['PARAMS0'][0]
    coord0 = mean['COORDS0'][0]
    plt.scatter(coord0[:,0], coord0[:,1],
                c=y0, cmap = cmap, vmin=-MAX,vmax=MAX,
                lw=0, s=4)
    plt.xlabel('x (pixel)',fontsize=FONT)
    plt.ylabel('y (pixel)',fontsize=FONT)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    cb = plt.colorbar()
    cb.set_label('$du$ (mas)', fontsize=FONT)
    cb.ax.tick_params(labelsize=8)

    plt.subplot(1,2,2)
    mean = fitsio.read(fits_file_dv)
    y0 = mean['PARAMS0'][0]
    coord0 = mean['COORDS0'][0]
    plt.scatter(coord0[:,0], coord0[:,1],
                c=y0, cmap = cmap, vmin=-MAX,vmax=MAX,
                lw=0, s=6)
    plt.xlabel('x (pixel)',fontsize=FONT)
    plt.yticks([],[])
    plt.xticks(fontsize=8)
    cb = plt.colorbar()
    cb.set_label('$dv$ (mas)', fontsize=FONT)
    cb.ax.tick_params(labelsize=8)

    plt.suptitle(name, fontsize=FONT)
    if name_fig is not None:
        plt.savefig(name_fig)
        plt.close()


if __name__ == '__main__':

    #dic = build_mean_in_tp(rep_mean='~/sps_lsst/HSC/v3.3/astro_VK/mean_function/all/')
    plot_fov_mean('../../../hsc_outputs/v3.3/astro_VK/mean_tp.pkl')

    #for i in [7, 14, 42]:
    #    plot_mean_ccd('../../../hsc_outputs/v3.3/astro_VK/mean_function_20/all/mean_du_%i_all.fits'%(i),
    #                  '../../../hsc_outputs/v3.3/astro_VK/mean_function_20/all/mean_dv_%i_all.fits'%(i),
    #                  name= 'CCD %i'%(i), cmap=plt.cm.inferno, MAX=2, name_fig='ccd_%i_mean.png'%(i))

    #plt.show()

    #plot_mean_ccd('../../../hsc_outputs/v3.3/astro_VK/mean_function_20/all/mean_du_11_all.fits',
    #              '../../../hsc_outputs/v3.3/astro_VK/mean_function_20/all/mean_dv_11_all.fits',
    #              name= 'CCD 11', cmap=plt.cm.inferno, MAX=2, name_fig='ccd_11_mean.png')
    #plot_mean_ccd('../../../hsc_outputs/v3.3/astro_VK_with_mean/mean_function/all/mean_du_11_all.fits',
    #              '../../../hsc_outputs/v3.3/astro_VK_with_mean/mean_function/all/mean_dv_11_all.fits',
    #              name= 'CCD 11 (corrected)', cmap=plt.cm.inferno, MAX=2, name_fig='ccd_11_mean_corrected.png')