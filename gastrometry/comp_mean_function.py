import numpy as np
import treegp
import pylab as plt
import fitsio
import os
import pickle
import glob

class comp_mean(object):

    def __init__(self, files_out, bin_spacing=10.,
                 statistics='weighted', nccd=105,
                 gp_corrected=True, alias=None):

        self.files_out = files_out
        self.bin_spacing = bin_spacing
        self.statistics = statistics
        self.nccd = nccd
        self.mean = {}
        self.gp_corrected = gp_corrected
        self.alias = alias

        for i in range(nccd):
            mean_du = treegp.meanify(bin_spacing=self.bin_spacing,
                                     statistics=self.statistics)
            mean_dv = treegp.meanify(bin_spacing=self.bin_spacing,
                                     statistics=self.statistics)
            self.mean.update({i: {'du':mean_du,
                                  'dv':mean_dv,}})

    def stack_fields(self):

        for f in self.files_out:
            print(f)
            dic = pickle.load(open(os.path.join(f), 'rb'))
            coord_ccd = {}
            residuals = {}
            residuals_err = {}
            for coord in ['u', 'v']:
                coord_ccd[coord] = np.array([dic['gp_output']['gp%s.xccd_all'%(coord)],
                                             dic['gp_output']['gp%s.yccd_all'%(coord)]]).T
                residuals[coord] = dic['gp_output']['gp%s.d%s_all'%((coord, coord))]
                if self.gp_corrected:
                    residuals[coord]-= dic['gp_output']['gp%s.d%s_all_predict'%((coord, coord))]
                if self.statistics == 'weighted':
                    residuals_err[coord] = dic['gp_output']['gp%s.d%s_err_all'%((coord, coord))]

            for chipnum in self.mean:
                for coord in ['u', 'v']:
                    filtre = (dic['gp_output']['gp%s.chip_num_all'%(coord)] == chipnum)
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
                    self.mean[chipnum]['d%s'%(coord)].meanify(lu_min=26., lu_max=2021.,
                                                              lv_min=17., lv_max=4149.,)

    def save_results(self, rep_out):
        for chipnum in self.mean:
            print(chipnum)
            for coord in ['u', 'v']:
                if self.alias is None:
                    file_name = 'mean_d%s_%i.fits'%((coord, chipnum))
                else:
                    file_name = 'mean_d%s_%i_%s.fits'%((coord, chipnum, self.alias))
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

def run_ma_poule_mean(rep_out, bin_spacing=30.,
                      statistics='weighted', nccd=105,
                      gp_corrected=True, sub_rep='all'):

    if sub_rep == 'all':
        # across all filters
        files_out = glob.glob(os.path.join(rep_out, '*/gp_output*.pkl'))
        path_mean = os.path.join(rep_out, 'mean_function')
        os.system('mkdir %s'%(path_mean))
        path_mean = os.path.join(path_mean, 'all')
        os.system('mkdir %s'%(path_mean))
    else:
        if sub_rep not in ['i', 'r']:
            files_out = glob.glob(os.path.join(rep_out, '*%s/gp_output*.pkl'%(sub_rep)))
        else:
            files_out1 = glob.glob(os.path.join(rep_out, '*%s/gp_output*.pkl'%(sub_rep)))
            files_out2 = glob.glob(os.path.join(rep_out, '*%s2/gp_output*.pkl'%(sub_rep)))
            files_out = files_out1 + files_out2
        path_mean = os.path.join(rep_out, 'mean_function')
        path_mean = os.path.join(path_mean, sub_rep)
        os.system('mkdir %s'%(path_mean))

    cm = comp_mean(files_out, bin_spacing=bin_spacing,
                   statistics=statistics, nccd=nccd,
                   gp_corrected=gp_corrected, alias=sub_rep)
    cm.stack_fields()
    cm.comp_mean()
    cm.save_results(path_mean)

    os.system('mkdir %s'%(os.path.join(path_mean, 'plotting')))
    for i in range(nccd):
        print(i)
        try:
            plot_mean(os.path.join(path_mean, 'mean_du_%i_%s.fits'%((i, sub_rep))),
                      os.path.join(path_mean, 'mean_dv_%i_%s.fits'%((i, sub_rep))),
                      name='CCD %i'%(i),
                      cmap=None,
                      name_fig=os.path.join(path_mean,'plotting/CCD_%i_%s.png'%((i, sub_rep))))
        except:
            print('files %i does not exist'%(i))

    if sub_rep == 'all':
        for hsc_filter in ['g', 'r', 'i', 'z', 'y']:
            run_ma_poule_mean(rep_out, bin_spacing=bin_spacing,
                              statistics=statistics, nccd=nccd,
                              gp_corrected=gp_corrected,
                              sub_rep=hsc_filter)

if __name__ == "__main__":

    rep_out = '/pbs/home/l/leget/sps_lsst/HSC/v3.2/astro_VK'
    run_ma_poule_mean(rep_out, bin_spacing=15,
                      statistics='weighted', nccd=105,
                      gp_corrected=True)
