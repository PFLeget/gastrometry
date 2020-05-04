import numpy as np
import copy
import treecorr
import treegp
from gastrometry import vcorr, xiB, plotting
from sklearn.model_selection import train_test_split
import os
import pickle
import parser, optparse

def read_option():

    usage = "launch gp_job"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--rep","-r",dest="rep",help="dir input/output", default=100)

    option,args = parser.parse_args()

    return option

class gpastro(object):

    def __init__(self, u, v, du, dv, du_err, dv_err,
                 NBIN=21, MAX = 17.*60.,
                 P0=[3000., 0., 0.],
                 kernel = "15**2 * AnisotropicVonKarman(invLam=np.array([[1./3000.**2,0],[0,1./3000.**2]]))",
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id="", visit_id="", rep="", 
                 save=False):

        self.exp_id = exp_id
        self.visit_id = visit_id
        self.rep = rep
        self.save = save

        self.NBIN = NBIN
        self.MAX = MAX
        self.P0 = P0
        self.kernel = kernel

        self.u = u * arcsec
        self.v = v * arcsec
        self.coords = np.array([self.u, self.v]).T
        self.du = du * mas
        self.dv = dv * mas
        self.du_err = du_err * mas
        self.dv_err = dv_err * mas

        # split training/validation
        indice = np.linspace(0, len(self.u)-1, len(self.u)).astype(int)
        indice_train, indice_test = train_test_split(indice, test_size=0.2, random_state=42)

        self.u_train = self.u[indice_train]
        self.u_test = self.u[indice_test]

        self.v_train = self.v[indice_train]
        self.v_test = self.v[indice_test]

        self.coords_train = self.coords[indice_train]
        self.coords_test = self.coords[indice_test]

        self.du_train = self.du[indice_train]
        self.du_test = self.du[indice_test]
        
        self.dv_train = self.dv[indice_train]
        self.dv_test = self.dv[indice_test]

        self.du_err_train = self.du_err[indice_train]
        self.du_err_test = self.du_err[indice_test]
        
        self.dv_err_train = self.dv_err[indice_train]
        self.dv_err_test = self.dv_err[indice_test]

        self.logr = None
        self.xie = None
        self.xib = None

        self.xi_sep = None
        self.xi_dudu = None
        self.xi_dvdv = None
        self.xi_dudv = None

        self.u = u * arcsec
        self.v = v * arcsec
        self.coords = np.array([self.u, self.v]).T
        self.du = du * mas
        self.dv = dv * mas
        self.du_err = du_err * mas
        self.dv_err = dv_err * mas

        # split training/validation                                                                                                                             
        indice_train, indice_test

        self.dic_output = {'exp_id':self.exp_id, 
                           'input_data':{'u':self.u,
                           'v':self.v,
                           'du':self.du,
                           'dv':self.dv,
                           'du_err':self.du_err,
                           'dv_err':self.dv_err,
                           'indice_train':indice_train,
                           'indice_test':indice_test},
                           '2pcf_stat':{},
                           'gp_output':{}}
        

    def comp_eb(self):

        self.logr, xiplus, ximinus, xicross, xiz2 = vcorr(self.u/3600., self.v/3600., 
                                                          self.du, self.dv)
        self.xib = xiB(self.logr, xiplus, ximinus)
        self.xie = xiplus - self.xib

        self.dic_output['2pcf_stat'].update({'xib':self.xib,
                                             'xie':self.xie,
                                             'logr':self.logr,
                                             'xiplus':xiplus,
                                             'ximinus':ximinus,
                                             'xicross':xicross})

    def comp_xi(self):

        cat_du = treecorr.Catalog(x=self.u, y=self.v, 
                                  k=(self.du-np.mean(self.du)), w=1./self.du_err**2)
        cat_dv = treecorr.Catalog(x=self.u, y=self.v, 
                                  k=(self.dv-np.mean(self.dv)), w=1./self.dv_err**2)

        kk = treecorr.KKCorrelation(min_sep=0, max_sep=20.*60., nbins=25,
                                    bin_type='TwoD', bin_slop=0)
        kk.process(cat_du)
        self.xi_dudu = copy.deepcopy(kk.xi)

        kk.process(cat_dv)
        self.xi_dvdv = copy.deepcopy(kk.xi)

        kk.process(cat_du, cat_dv)
        self.xi_dudv = copy.deepcopy(kk.xi)

        npixels = len(kk.xi)**2
        self.xi_sep = np.array([kk.dx.reshape(npixels), 
                                kk.dy.reshape(npixels)]).T

        self.dic_output['2pcf_stat'].update({'xi_dudu':self.xi_dudu,
                                             'xi_dudv':self.xi_dudv,
                                             'xi_dvdv':self.xi_dvdv,
                                             'xi_sep':self.xi_sep})

    def gp_interp(self):

        print("start gp interp")
        gpu = treegp.GPInterpolation(kernel=self.kernel, optimizer='anisotropic', 
                                     normalize=True, nbins=self.NBIN, min_sep=0.,
                                     max_sep=self.MAX, p0=self.P0)
        gpu.initialize(self.coords_train, self.du_train, y_err=self.du_err_train)
        gpu.solve()
        self.du_test_predict = gpu.predict(self.coords_test, return_cov=False)
        self.gpu = gpu
        self.dic_output['gp_output'].update({'gpu.2pcf':gpu._optimizer._2pcf,
                                             'gpu.2pcf_weight':gpu._optimizer._2pcf_weight,
                                             'gpu.2pcf_dist':gpu._optimizer._2pcf_dist,
                                             'gpu.2pcf_fit':gpu._optimizer._2pcf_fit,
                                             'gpu.2pcf_mask':gpu._optimizer._2pcf_mask,
                                             'gpu.kernel':gpu._optimizer._kernel,
                                             'gpu.du_test_predict':self.du_test_predict,
                                             'gpu.du_test':self.du_test,
                                             'gpu.coords_test':self.coords_test})
        
        print("I did half")
        gpv = treegp.GPInterpolation(kernel=self.kernel, optimizer='anisotropic',
                                     normalize=True, nbins=self.NBIN, min_sep=0.,
                                     max_sep=self.MAX, p0=self.P0)
        gpv.initialize(self.coords_train, self.dv_train, y_err=self.dv_err_train)
        gpv.solve()
        self.dv_test_predict = gpv.predict(self.coords_test, return_cov=False)
        self.gpv = gpv

        self.dic_output['gp_output'].update({'gpv.2pcf':gpv._optimizer._2pcf,
                                             'gpv.2pcf_weight':gpv._optimizer._2pcf_weight,
                                             'gpv.2pcf_dist':gpv._optimizer._2pcf_dist,
                                             'gpv.2pcf_fit':gpv._optimizer._2pcf_fit,
                                             'gpv.2pcf_mask':gpv._optimizer._2pcf_mask,
                                             'gpv.kernel':gpv._optimizer._kernel,
                                             'gpv.dv_test_predict':self.dv_test_predict,
                                             'gpv.dv_test':self.dv_test,
                                             'gpv.coords_test':self.coords_test})

        X_valid = self.coords_test
        Y_valid = np.array([self.du_test, self.dv_test]).T
        Y_valid_interp = np.array([self.du_test_predict, self.dv_test_predict]).T

        self.logr_residuals, self.xiplus_residuals, self.ximinus_residuals, self.xicross_residuals, xiz2 = vcorr(X_valid[:,0]/3600., X_valid[:,1]/3600., 
                                                                                                                 Y_valid[:,0]-Y_valid_interp[:,0], 
                                                                                                                 Y_valid[:,1]-Y_valid_interp[:,1])
        self.xib_residuals = xiB(self.logr_residuals, self.xiplus_residuals, self.ximinus_residuals)
        self.xie_residuals = self.xiplus_residuals - self.xib_residuals

        self.logr_test, self.xiplus_test, self.ximinus_test, self.xicross_test, xiz2 = vcorr(X_valid[:,0]/3600., X_valid[:,1]/3600., 
                                                                                             Y_valid[:,0], Y_valid[:,1])
        self.xib_test = xiB(self.logr_test, self.xiplus_test, self.ximinus_test)
        self.xie_test = self.xiplus_test - self.xib_test

        self.dic_output['2pcf_stat'].update({'xib_test':self.xib_test,
                                             'xie_test':self.xie_test,
                                             'logr_test':self.logr,
                                             'xib_residuals':self.xib_residuals,
                                             'xie_residuals':self.xie_residuals,
                                             'logr_residuals':self.logr_residuals})
    def plot_gaussian_process(self):

        plotting.plot_gaussian_process(self)

    def save_output(self):

        pkl_name = os.path.join(self.rep, 'gp_output_%i.pkl'%(int(self.exp_id)))
        pkl_file = open(pkl_name, 'wb')
        pickle.dump(self.dic_output, pkl_file)
        pkl_file.close()

if __name__ == "__main__":

    option = read_option()
    INPUT = os.path.join(option.rep, 'input.pkl')

    dic = pickle.load(open(INPUT, 'rb'))
    print("gp_astro start")
    gp = gpastro(dic['u'], dic['v'], 
                 dic['du'], dic['dv'], 
                 dic['du_err'], dic['dv_err'],
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id=dic['exp_id'], visit_id="",
                 rep=option.rep, save=True)
    gp.comp_eb()
    gp.comp_xi()
    print("start gp")
    gp.gp_interp()
    print("do plot")
    gp.plot_gaussian_process()
    gp.save_output()
