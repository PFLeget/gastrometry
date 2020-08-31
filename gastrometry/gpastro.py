import numpy as np
import copy
import treecorr
import treegp
from gastrometry import vcorr, xiB, plotting
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import os
import pickle
import parser, optparse
import fitsio

def read_option():

    usage = "launch gp_job"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--rep","-r",dest="rep",help="dir input/output", default=100)

    option,args = parser.parse_args()

    return option

class get_mean(object):

    def __init__(self, fits_file):

        self.mean = fitsio.read(fits_file)
        self.params0 = self.mean['PARAMS0'][0]
        self.coord0 = self.mean['COORDS0'][0]
        self.y0 = self.mean['_AVERAGE'][0]
        self.u0 = self.mean['_U0'][0]
        self.v0 = self.mean['_V0'][0]
        
def build_average_knn(X, rep='', comp='du',ccd_name='42', n_neighbors=4):
    """Compute spatial average from meanify output for a given coordinate using KN interpolation.               
        If no average_fits was given, return array of 0.                                                            
                                                                                                                    
        :param X: Coordinates of training stars or coordinates where to interpolate. (n_samples, 2)                 
    """
    file = "mean_%s_%s_all.fits"%((comp, ccd_name))
    
    gm = get_mean(os.path.join(rep, file))

    neigh = KNeighborsRegressor(n_neighbors=n_neighbors ,weights='distance')
    neigh.fit(gm.coord0, gm.params0)
    average = neigh.predict(X)
        
    return average
        
class comp_mean_interp(object):
    
    def __init__(self, xccd, yccd, ccd_name, rep_mean=''):
        
        self.X = np.array([xccd, yccd]).T
        self.rep_mean = rep_mean
        self.ccd_name = ccd_name
        self.y0_du = np.zeros(len(xccd))
        self.y0_dv = np.zeros(len(xccd))
        
    def return_mean(self):
        
        for i in range(104):
            try:
                Filtre = (self.ccd_name == (i+1))
                self.y0_du[Filtre] = build_average_knn(self.X[Filtre], rep=self.rep_mean, comp='du',
                                                       ccd_name=str(i+1), n_neighbors=4)
                self.y0_dv[Filtre] = build_average_knn(self.X[Filtre], rep=self.rep_mean, comp='dv',
                                                       ccd_name=str(i+1), n_neighbors=4)
            except:
                print("no file exit for this CCD (name: %s)"%(str(i+1)))

class gpastro(object):

    def __init__(self, u, v, du, dv, du_err, dv_err,
                 NBIN=21, MAX = 17.*60., xccd=None, yccd=None, chipnum=None,
                 P0=[3000., 0., 0.],
                 kernel = "15**2 * AnisotropicVonKarman(invLam=np.array([[1./3000.**2,0],[0,1./3000.**2]]))",
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id="", visit_id="", rep="", 
                 save=False, rep_mean=None):

        self.exp_id = exp_id
        self.visit_id = visit_id
        self.rep = rep
        self.rep_mean = rep_mean
        self.save = save

        self.NBIN = NBIN
        self.MAX = MAX
        self.P0 = P0
        self.kernel = kernel
        
        self.u = u * arcsec
        self.v = v * arcsec
        self.coords = np.array([self.u, self.v]).T
        self.xccd = xccd
        self.yccd = yccd
        self.chipnum = chipnum
        self.du = du * mas
        self.dv = dv * mas
        self.du_err = du_err * mas
        self.dv_err = dv_err * mas

        if self.rep_mean is not None:
            print('A mean function is used, the rep used is this one: %s'%(self.rep_mean))
            if xccd is None:
                ValueError("need an xccd")
            if yccd is None:
                ValueError("need an yccd")
            if chipnum is None:
                ValueError("need a ccd_name")
            cmi = comp_mean_interp(self.xccd, self.yccd,
                                   self.chipnum, rep_mean=self.rep_mean)
            cmi.return_mean()
            self.y0_du = cmi.y0_du
            self.y0_dv = cmi.y0_dv
        else:
            print('No mean function is used')
            self.y0_du = 0.
            self.y0_dv = 0.
            
        self._arcsec = arcsec
        self._mas = mas

        # split training/validation
        indice = np.linspace(0, len(self.u)-1, len(self.u)).astype(int)
        indice_train, indice_test = train_test_split(indice, test_size=0.2, random_state=42)

        for comp in ['u', 'v']:
            exec("self.%s_train = self.%s[indice_train]"%((comp, comp)))
            exec("self.%s_test = self.%s[indice_test]"%((comp, comp)))
            exec("self.d%s_train = self.d%s[indice_train]"%((comp, comp)))
            exec("self.d%s_test = self.d%s[indice_test]"%((comp, comp)))

            if self.rep_mean is not None:
                exec("self.y0_d%s_train = self.y0_d%s[indice_train]"%((comp, comp)))
                exec("self.y0_d%s_test = self.y0_d%s[indice_test]"%((comp, comp)))
            else:
                exec("self.y0_d%s_train = 0."%(comp))
                exec("self.y0_d%s_test = 0."%(comp))

            exec("self.d%s_err_train = self.d%s_err[indice_train]"%((comp, comp)))
            exec("self.d%s_err_test = self.d%s_err[indice_test]"%((comp, comp)))

        self.coords_train = self.coords[indice_train]
        self.coords_test = self.coords[indice_test]

        self.logr = None
        self.xie = None
        self.xib = None

        self.xi_sep = None
        self.xi_dudu = None
        self.xi_dvdv = None
        self.xi_dudv = None

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

    def gp_interp(self, dic_all=None):

        if dic_all is not None:
            if self.rep_mean is not None:
                cmi = comp_mean_interp(dic_all['x'], dic_all['y'],
                                       dic_all['chip_num'], rep_mean=self.rep_mean)
                cmi.return_mean()
                self.y0_du_all = cmi.y0_du
                self.y0_dv_all = cmi.y0_dv
            
        print("start gp interp")

        for comp in ['u', 'v']:
            if comp == 'v':
                print('I did half')
            exec("gp%s = treegp.GPInterpolation(kernel=self.kernel, optimizer=\'anisotropic\', normalize=True, nbins=self.NBIN, min_sep=0., max_sep=self.MAX, p0=self.P0)"%(comp))
            # remove mean function
            exec("ygp = self.d%s_train - self.y0_d%s_train"%((comp, comp)))
            exec("gp%s.initialize(self.coords_train, ygp, y_err=self.d%s_err_train)"%((comp, comp)))
            exec("gp%s.solve()"%(comp))
            exec("self.d%s_test_predict = gp%s.predict(self.coords_test, return_cov=False)"%((comp, comp)))
            exec("self.d%s_predict = gp%s.predict(self.coords, return_cov=False)"%((comp, comp)))
            # add back mean function
            exec("self.d%s_test_predict += self.y0_d%s_test"%((comp, comp)))
            exec("self.d%s_predict += self.y0_d%s"%((comp, comp)))
            
            exec("self.gp%s = gp%s"%((comp, comp)))

            if dic_all is not None:
                self.coords_all = np.array([dic_all['u']*self._arcsec,
                                            dic_all['v']*self._arcsec]).T
                exec("self.d%s_all = dic_all['d%s'] * self._mas"%((comp, comp)))
                exec("self.d%s_err_all = dic_all['d%s_err'] * self._mas"%((comp, comp)))
                exec("self.d%s_all_predict = gp%s.predict(self.coords_all, return_cov=False)"%((comp, comp)))
                self.xccd_all = dic_all['x']
                self.yccd_all = dic_all['y']
                self.chip_num_all = dic_all['chip_num']
                if self.rep_mean is not None:
                    exec("self.d%s_all_predict += self.y0_d%s_all"%((comp, comp)))
            else:
                self.coords_all = None
                exec("self.d%s_all_predict = None"%(comp))
                self.xccd_all = None
                self.yccd_all = None
                self.chip_num_all = None

            gp_name = 'gp%s'%(comp)
        
            exec('self.dic_output[\'gp_output\'][gp_name+\'.2pcf\'] = gp%s._optimizer._2pcf'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.2pcf_weight\'] = gp%s._optimizer._2pcf_weight'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.2pcf_dist\'] = gp%s._optimizer._2pcf_dist'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.2pcf_fit\'] = gp%s._optimizer._2pcf_fit'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.2pcf_mask\'] = gp%s._optimizer._2pcf_mask'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.kernel\'] = gp%s._optimizer._kernel'%(comp))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s\'] = self.d%s'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_predict\'] = self.d%s_predict'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_test_predict\'] = self.d%s_test_predict'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_test\'] = self.d%s_test'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_all\'] = self.d%s_all'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_err_all\'] = self.d%s_err_all'%((comp, comp)))
            exec('self.dic_output[\'gp_output\'][gp_name+\'.d%s_all_predict\'] = self.d%s_all_predict'%((comp, comp)))

            self.dic_output['gp_output'][gp_name+'.coords'] = self.coords
            self.dic_output['gp_output'][gp_name+'.coords_test'] = self.coords_test
            self.dic_output['gp_output'][gp_name+'.xccd'] = self.xccd
            self.dic_output['gp_output'][gp_name+'.yccd'] = self.yccd
            self.dic_output['gp_output'][gp_name+'.chipnum'] = self.chipnum
            self.dic_output['gp_output'][gp_name+'.coords_all'] = self.coords_all
            self.dic_output['gp_output'][gp_name+'.xccd_all'] = self.xccd_all
            self.dic_output['gp_output'][gp_name+'.yccd_all'] = self.yccd_all
            self.dic_output['gp_output'][gp_name+'.chip_num_all'] = self.chip_num_all


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
                 xccd=dic['x'], yccd=dic['y'], chipnum=dic['chip_num'],
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
