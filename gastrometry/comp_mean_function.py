import numpy as np
import treegp
import pylab as plt
import fitsio
import os
import pickle
import glob

class comp_mean(object):

    def __init__(self, files_out, bin_spacing=10.,
                 statistics='weighted', nccd=104,
                 gp_corrected=True):

        self.files_out = files_out
        self.bin_spacing = bin_spacing
        self.statistics = statistics
        self.nccd = nccd
        self.mean = {}
        self.gp_corrected = gp_corrected

        for i in range(nccd):
            mean_du = treegp.meanify(bin_spacing=self.bin_spacing,
                                     statistics=self.statistics)
            mean_dv = treegp.meanify(bin_spacing=self.bin_spacing,
                                     statistics=self.statistics)
            self.mean.update({i+1: {'du':mean_du,
                                    'dv':mean_dv,}})

    def stack_fields(self):

        for f in self.files_out:
            print(f)
            dic = pickle.load(open(os.path.join(f), 'rb'))
            coord_ccd = {}
            residuals = {}
            residuals_err = {}
            for coord in ['u', 'v']:
                coord_ccd[coord] = np.array([dic['gp_output']['gp%s.xccd'%(coord)],
                                             dic['gp_output']['gp%s.yccd'%(coord)]]).T
                residuals[coord] = dic['gp_output']['gp%s.d%s'%((coord, coord))]
                if self.gp_corrected:
                    residuals[coord]-= dic['gp_output']['gp%s.d%s_predict'%((coord, coord))]
                if self.statistics == 'weighted':
                    residuals_err[coord] = dic['input_data']['d%s_err'%(coord)]

            for chipnum in self.mean:
                for coord in ['u', 'v']:
                    filtre = (dic['gp_output']['gp%s.chipnum'%(coord)] == chipnum)
                    if np.sum(filtre) != 0:
                        if self.statistics == 'weighted':
                            error = residuals_err[coord][filtre]
                        else:
                            error = None
                        self.mean[chipnum]['d%s'%(coord)].add_field(coord_ccd[coord][filtre],
                                                                    residuals[coord][filtre],
                                                                    params_err=error)

    def comp_mean(self):
        for chipnum in self.mean:
            print(chipnum)
            for coord in ['u', 'v']:
                if len(self.mean[chipnum]['d%s'%(coord)].params) !=0:
                    self.mean[chipnum]['d%s'%(coord)].meanify()

    def save_results(self, rep_out):
        for chipnum in self.mean:
            print(chipnum)
            for coord in ['u', 'v']:
                file_name = 'mean_d%s_%i.fits'%((coord, chipnum))
                fits_file = os.path.join(rep_out, file_name)
                if len(self.mean[chipnum]['d%s'%(coord)].params) !=0:
                    self.mean[chipnum]['d%s'%(coord)].save_results(name_output=fits_file)

def plot_mean(fits_file_du,
              fits_file_dv, name= '',
              cmap=None, MAX=2, name_fig=None):


    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1,2,1)
    mean = fitsio.read(fits_file_du)
    y0 = mean['PARAMS0'][0]
    coord0 = mean['COORDS0'][0]
    plt.scatter(coord0[:,0], coord0[:,1],
                c=y0, cmap = cmap, vmin=-MAX,vmax=MAX,
                lw=0, s=8)
    plt.xlabel('x (pixel)',fontsize=14)
    plt.ylabel('y (pixel)',fontsize=14)
    cb = plt.colorbar()
    cb.set_label('$<du>$ (mas)', fontsize=14)

    plt.subplot(1,2,2)
    mean = fitsio.read(fits_file_dv)
    y0 = mean['PARAMS0'][0]
    coord0 = mean['COORDS0'][0]
    plt.scatter(coord0[:,0], coord0[:,1],
                c=y0, cmap = cmap, vmin=-MAX,vmax=MAX,
                lw=0, s=8)

    plt.xlabel('x (pixel)',fontsize=14)
    cb = plt.colorbar()
    cb.set_label('$<dv>$ (mas)', fontsize=14)


    plt.suptitle(name, fontsize=14)
    if name_fig is not None:
        plt.savefig(name_fig)
        plt.close()

if __name__ == "__main__":

    #files_out = glob.glob('../../../../sps_lsst/HSC/v3.2/astro_VK/*/gp_output*.pkl')

    #cm = comp_mean(files_out, bin_spacing=10., 
    #               statistics='weighted', nccd=104,
    #               gp_corrected=True)
    #cm.stack_fields()
    #cm.comp_mean()
    #cm.save_results('../../../../sps_lsst/HSC/v3.2/astro_VK/mean_function_weighted/')

    for i in range(104):
        print(i+1)
        try:
            plot_mean('../../../../sps_lsst/HSC/v3.2/astro_VK/mean_function_weighted/mean_du_%i.fits'%(i+1),
                      '../../../../sps_lsst/HSC/v3.2/astro_VK/mean_function_weighted/mean_dv_%i.fits'%(i+1),
                      name='CCD %i'%(i+1),
                      cmap=None,
                      name_fig='../../../../sps_lsst/HSC/v3.2/astro_VK/mean_function_weighted/plotting/CCD_%i_weighted.png'%(i+1))
        except:
            print('files %i does not exist'%(i+1))
    #plt.show()
