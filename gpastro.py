import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
import treecorr
import treegp
from astroeb import vcorr, xiB
from sklearn.model_selection import train_test_split
import pylab as plt
import os
import cPickle
import parser, optparse


def read_option():

    usage = "launch gp_job"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--rep","-r",dest="rep",help="dir input/output", default=100)

    option,args = parser.parse_args()

    return option

def return_var_map(weight, xi):
    N = int(np.sqrt(len(xi)))
    var = np.diag(np.linalg.inv(weight))
    VAR = np.zeros(N*N)
    I = 0
    for i in range(N*N):
        if xi[i] !=0:
            VAR[i] = var[I]
            I+=1
        if I == len(var):
            break
    VAR = VAR.reshape(N,N) + np.flipud(np.fliplr(VAR.reshape(N,N)))
    if N%2 == 1:
        VAR[N/2, N/2] /= 2. 
    return VAR

def plot_correlation_function(interp, save=False, rep='', 
                              specific_name_kernel='VK', NAME='du', exp='0'):

    EXT = [np.min(interp._2pcf_dist[:,0]/60.), np.max(interp._2pcf_dist[:,0]/60.), 
           np.min(interp._2pcf_dist[:,1]/60.), np.max(interp._2pcf_dist[:,1]/60.)]
    CM = plt.cm.seismic

    MAX = np.max(interp._2pcf)
    N = int(np.sqrt(len(interp._2pcf)))
    plt.figure(figsize=(14,5) ,frameon=False)
    plt.gca().patch.set_alpha(0)
    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)
    plt.suptitle(NAME+' anisotropy 2-PCF', fontsize=16)
    plt.subplot(1,3,1)
    plt.imshow(interp._2pcf.reshape(N,N), extent=EXT, interpolation='nearest', origin='lower', 
               vmin=-MAX, vmax=MAX, cmap=CM)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\xi$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)
    plt.title('Measured 2-PCF',fontsize=16)
    
    plt.subplot(1,3,2)
    plt.imshow(interp._2pcf_fit.reshape(N,N), extent=EXT, interpolation='nearest', 
               origin='lower',vmin=-MAX,vmax=MAX, cmap=CM)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\xi\'$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)
    
    var = return_var_map(interp._2pcf_weight, interp._2pcf)
    cm_residual = plt.matplotlib.cm.get_cmap('RdBu',10)
    Res = interp._2pcf[interp._2pcf_mask] - interp._2pcf_fit[interp._2pcf_mask]
    chi2 = Res.dot(interp._2pcf_weight).dot(Res)
    dof = np.sum(interp._2pcf_mask) - 4.
    
    pull = (interp._2pcf.reshape(N,N) - interp._2pcf_fit.reshape(N,N)) / np.sqrt(var)
    
    plt.title('Fitted 2-PCF'%(chi2/dof),fontsize=16)

    plt.subplot(1,3,3)
    
    plt.imshow(pull, extent=EXT, interpolation='nearest', origin='lower', vmin=-5., vmax=+5., cmap=cm_residual)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\frac{\\xi-\\xi\'}{\sigma_{\\xi}}$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)
    plt.title('Pull',fontsize=16)
    if save:
        namefig = os.path.join(rep, '2PCF_anisotropic_'+NAME+'_'+specific_name_kernel+'_%i.pdf'%(int(exp)))
        plt.savefig(namefig,transparent=True)

def plot_gp_output(X_valid, Y_valid, Y_valid_interp, Y_err, rep='', save=False, exp='0'):

    plt.figure(figsize=(16,7))
    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.98)
    plt.subplot(1,2,1)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid[:,0], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('du (mas)', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title("%i | validation data"%int(exp), fontsize=20)
    plt.subplot(1,2,2)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid_interp[:,0], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('du (mas)', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title("%i | gp prediction"%int(exp), fontsize=20)
    if save:
        namefig = os.path.join(rep, 'scalar_du_test_validation_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)
    
    plt.figure(figsize=(16,7))
    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.98)
    plt.subplot(1,2,1)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid[:,1], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('dv (mas)', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title("%i | validation data"%int(exp), fontsize=20)

    plt.subplot(1,2,2)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid_interp[:,1], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('dv (mas) / gp prediction', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title("%i | gp prediction"%int(exp), fontsize=20)
    if save:
        namefig = os.path.join(rep, 'scalar_dv_test_validation_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)
    

    fig = plt.figure(figsize=(13, 7))
    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.98)
    quiver_dict = dict(alpha=1,
                       angles='uv',
                       headlength=1.e-10,
                       headwidth=0,
                       headaxislength=0,
                       minlength=0,
                       pivot='middle',
                       scale_units='xy',
                       width=0.003,
                       color='blue',
                       scale=0.05)
    ax1 = plt.subplot(1,2,1)
    ax1.quiver(X_valid[:,0], X_valid[:,1], 
               Y_valid[:,0], Y_valid[:,1], **quiver_dict)
    ax1.quiver([1900],[2500],[10],[0], **quiver_dict)
    ax1.text(2200,2500,"(10 mas)",fontsize=12)
    ax1.set_title("%i"%int(exp)+ ' | validation data', fontsize=20)
    ax1.set_xlabel('u (arcsec)', fontsize=20)
    ax1.set_ylabel('v (arcsec)', fontsize=20)

    ax2 = plt.subplot(1,2,2)
    ax2.quiver(X_valid[:,0], X_valid[:,1], 
               Y_valid_interp[:,0], Y_valid_interp[:,1], **quiver_dict)
    ax2.set_title("%i"%int(exp)+ ' | gp prediction', fontsize=20)
    ax2.set_xlabel('u (arcsec)', fontsize=20)
    ax2.quiver([1900],[2500],[10],[0], **quiver_dict)
    ax2.text(2200,2500,"(10 mas)",fontsize=12)
    if save:
        namefig = os.path.join(rep, 'quiver_dudv_test_validation_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)

    plt.figure(figsize=(13,6))
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    plt.subplot(1,2,1)
    hist_result = plt.hist(Y_valid[:,0], bins = np.linspace(-20,20,20), 
                           histtype='step', color='b', lw=3, label="data")
    hist_result = plt.hist(Y_valid_interp[:,0], bins = np.linspace(-20,20,20), 
                           histtype='step', color='r', lw=3, label="gp predict")
    plt.title("data (Mean = %.2f ; STD = %.2f) \n gp predict (Mean = %.2f ; STD = %.2f)"%((np.mean(Y_valid[:,0]), np.std(Y_valid[:,0]),
                                                                                           np.mean(Y_valid_interp[:,0]), np.std(Y_valid_interp[:,0]))), fontsize=18)
    plt.xlabel('du (mas)', fontsize=16)
    plt.legend(fontsize=14)

    plt.subplot(1,2,2)
    hist_result = plt.hist(Y_valid[:,1], bins = np.linspace(-20,20,20), 
                           histtype='step', color='b', lw=3, label="data")
    hist_result = plt.hist(Y_valid_interp[:,1], bins = np.linspace(-20,20,20), 
                           histtype='step', color='r', lw=3, label="gp predict")
    plt.title("data (Mean = %.2f ; STD = %.2f) \n gp predict (Mean = %.2f ; STD = %.2f)"%((np.mean(Y_valid[:,1]), np.std(Y_valid[:,1]),
                                                                                           np.mean(Y_valid_interp[:,1]), np.std(Y_valid_interp[:,1]))), fontsize=18)
    plt.xlabel('dv (mas)', fontsize=16)
    plt.legend(fontsize=14)
    if save:
        namefig = os.path.join(rep, 'distribution_param_validation_gp_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)

    plt.figure(figsize=(13,6))
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    plt.subplot(1,2,1)
    residuals = Y_valid[:,0] - Y_valid_interp[:,0]
    hist_result = plt.hist(residuals, bins = np.linspace(-20,20,20), 
                           histtype='step', color='b', lw=3, label="residuals")
    plt.title("residuals (Mean = %.2f ; STD = %.2f)"%((np.mean(residuals), np.std(residuals))), fontsize=18)
    plt.xlabel('residuals du (mas)', fontsize=16)
    plt.legend(fontsize=14)

    plt.subplot(1,2,2)
    residuals = Y_valid[:,1] - Y_valid_interp[:,1]
    hist_result = plt.hist(residuals, bins = np.linspace(-20,20,20), 
                           histtype='step', color='b', lw=3, label="residuals")
    plt.title("residuals (Mean = %.2f ; STD = %.2f)"%((np.mean(residuals), np.std(residuals))), fontsize=18)
    plt.xlabel('residuals dv (mas)', fontsize=16)
    plt.legend(fontsize=14)
    if save:
        namefig = os.path.join(rep, 'distribution_residuals_validation_gp_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)

    plt.figure(figsize=(13,6))
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    plt.subplot(1,2,1)
    pull = (Y_valid[:,0] - Y_valid_interp[:,0]) / Y_err[:,0]
    hist_result = plt.hist(pull, bins = np.linspace(-6,6,30), histtype='step', color='b', lw=3)
    plt.title("Mean = %.2f ; STD = %.2f"%((np.mean(pull), np.std(pull))), fontsize=18)
    plt.xlabel('pull du', fontsize=16) 

    plt.subplot(1,2,2)
    pull = (Y_valid[:,1] - Y_valid_interp[:,1]) / Y_err[:,1]
    hist_result = plt.hist(pull, bins = np.linspace(-6,6,30), histtype='step', color='b', lw=3)
    plt.title("Mean = %.2f ; STD = %.2f"%((np.mean(pull), np.std(pull))), fontsize=18)
    plt.xlabel('pull dv', fontsize=16)
    if save:
        namefig = os.path.join(rep, 'distribution_pull_validation_gp_%i.pdf'%int(exp))
        plt.savefig(namefig,transparent=True)

    plt.figure(figsize=(16,7))
    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.98)
    plt.subplot(1,2,1)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid[:,0]-Y_valid_interp[:,0], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('residuals du (mas)', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title(int(exp), fontsize=20)

    plt.subplot(1,2,2)
    plt.scatter(X_valid[:,0], X_valid[:,1], c=Y_valid[:,1]-Y_valid_interp[:,1], 
                s=80, cmap=plt.cm.seismic, vmin=-10, vmax=10)
    cb = plt.colorbar()
    cb.set_label('residuals dv (mas)', fontsize=20)
    plt.xlabel('u (arcsec)', fontsize=20)
    plt.ylabel('v (arcsec)', fontsize=20)
    plt.title(int(exp), fontsize=20)
    if save:
        namefig = os.path.join(rep, 'distribution_residuals_validation_fov_gp_%i.pdf'%int(exp))
        plt.savefig(namefig, transparent=True)

class gpastro(object):

    def __init__(self, u, v, du, dv, du_err, dv_err, 
                 mas=3600.*1e3, arcsec=3600., 
                 exp_id="", visit_id="", rep="", save=False):

        self.exp_id = exp_id
        self.visit_id = visit_id
        self.rep = rep
        self.save = save

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

        print "start gp interp"
        L_start = 3000.
        kernel = "%f * AnisotropicVonKarman(invLam=np.array([[1./%f**2,0],[0,1./%f**2]]))"%((np.var(self.du_train), L_start, L_start))
        gpu = treegp.GPInterpolation(kernel=kernel, optimize=True,
                                     optimizer='two-pcf', anisotropic=True,
                                     normalize=True, nbins=15, min_sep=0.,
                                     max_sep=20.*60.)
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
        
        print "I did half"
        L_start = 3000.
        kernel = "%f * AnisotropicVonKarman(invLam=np.array([[1./%f**2,0],[0,1./%f**2]]))"%((np.var(self.dv_train), L_start, L_start))
        gpv = treegp.GPInterpolation(kernel=kernel, optimize=True,
                                     optimizer='two-pcf', anisotropic=True,
                                     normalize=True, nbins=15, min_sep=0.,
                                     max_sep=20.*60.)
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
                                             'gpu.dv_test':self.dv_test,
                                             'gpu.coords_test':self.coords_test})

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

    def eb_after_gp(self, rep='', save=False, exp="0"):

        plt.figure(figsize=(12,8))
        plt.scatter(np.exp(self.logr_test), self.xie_test, c='b', label='E-mode of data (validation)')
        plt.scatter(np.exp(self.logr_test), self.xib_test, c='r', label='B-mode of data (validation)')
        plt.plot(np.exp(self.logr_test), np.zeros_like(self.logr_test), 'k--', zorder=0)
        plt.ylim(-40,60)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.title(int(self.exp_id), fontsize=20)
        plt.legend(loc=1, fontsize=16)
        if save:
            namefig = os.path.join(rep, 'eb_mode_validation_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

        plt.figure(figsize=(12,8))
        plt.scatter(np.exp(self.logr_residuals), self.xie_residuals, c='b', label='E-mode of residuals (data)')
        plt.scatter(np.exp(self.logr_residuals), self.xib_residuals, c='r', label='B-mode of residuals (data)')
        plt.plot(np.exp(self.logr_residuals), np.zeros_like(self.logr_residuals), 'k--', zorder=0)
        plt.ylim(-40,60)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.title(int(self.exp_id), fontsize=20)
        plt.legend(loc=1, fontsize=16)
        if save:
            namefig = os.path.join(rep, 'eb_mode_validation_residuals_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

    def plot_fields(self, rep='', save=False, exp="0"):

        MAX = 3.*np.std(self.du)
        plt.figure(figsize=(12,10))
        plt.subplots_adjust(left=0.14, right=0.94)
        plt.scatter(self.u, self.v, c=self.du, 
                    s=40, cmap=plt.cm.seismic, 
                    vmin=-MAX, vmax=MAX)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        cb.set_label('du (mas)', fontsize=20)
        plt.xlabel('u (arcsec)', fontsize=20)
        plt.ylabel('v (arcsec)', fontsize=20)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.title(int(exp), fontsize=20)
        if save:
            namefig = os.path.join(rep, 'du_fov_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)
    
        plt.figure(figsize=(12,10))
        plt.subplots_adjust(left=0.14, right=0.94, top=0.95)
        plt.scatter(self.u, self.v, c=self.dv, 
                    s=40, cmap=plt.cm.seismic, 
                    vmin=-MAX, vmax=MAX)
        cb.ax.tick_params(labelsize=20)
        cb = plt.colorbar()
        cb.set_label('dv (mas)', fontsize=20)
        plt.xlabel('u (arcsec)', fontsize=20)
        plt.ylabel('v (arcsec)', fontsize=20)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.title(int(exp), fontsize=20)
        if save:
            namefig = os.path.join(rep, 'dv_fov_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

        fig = plt.figure(figsize=(12,10))
        plt.subplots_adjust(left=0.16, right=0.96, top=0.98)
        ax = plt.gca()
        quiver_dict = dict(alpha=1,
                           angles='uv',
                           headlength=1.e-10,
                           headwidth=0,
                           headaxislength=0,
                           minlength=0,
                           pivot='middle',
                           scale_units='xy',
                           width=0.003,
                           color='blue',
                           scale=0.3)

        ax.quiver(self.u, self.v, 
                  self.du, self.dv, **quiver_dict)
        plt.xlabel('u (arcsec)', fontsize=20)
        plt.ylabel('v (arcsec)', fontsize=20)
        plt.xticks(size=16)
        plt.yticks(size=16)
        if save:
            namefig = os.path.join(rep, 'quiver_dudv_fov_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

    def plot_eb_mode(self, rep='', save=False, exp="0"):

        plt.figure(figsize=(10,6))
        plt.subplots_adjust(bottom=0.12, top=0.95, right=0.99)
        plt.scatter(np.exp(self.logr), self.xie, s=20, 
                    alpha=1, c='b', label='E-mode')
        plt.scatter(np.exp(self.logr), self.xib, s=20,
                    alpha=1, c='r', label='B-mode')
        plt.plot(np.exp(self.logr), np.zeros_like(self.logr), 
                 'k--', alpha=0.5, zorder=0)
        MIN = np.min([np.min(self.xie), np.min(self.xib)])
        MAX = np.max([np.max(self.xie), np.max(self.xib)])
        plt.ylim(-40,60)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.title('%s %s'%((self.visit_id, self.exp_id)), fontsize=20)
        plt.legend(loc=1, fontsize=16)
        if save:
            namefig = os.path.join(rep, 'eb_mode_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

    def plot_2pcf(self, rep='', save=False, exp="0"):

        cb_label = ['$\\xi_{du,du}$ (mas$^2$)', 
                    '$\\xi_{dv,dv}$ (mas$^2$)', 
                    '$\\xi_{du,dv}$ (mas$^2$)']
        XI = [self.xi_dudu, self.xi_dvdv, self.xi_dudv]

        I = 1
        plt.figure(figsize=(14,5))
        plt.subplots_adjust(wspace=0.4,left=0.07,right=0.95, bottom=0.15,top=0.85)
        for xi in XI:
            MAX = np.max([abs(np.min(xi)), np.max(xi)])
            plt.subplot(1,3,I)
            plt.imshow(xi, cmap=plt.cm.seismic,
                       vmin=-MAX, vmax=MAX, origin="lower", 
                       extent=[np.min(self.xi_sep[:,0])/60., np.max(self.xi_sep[:,1]/60.), 
                               np.min(self.xi_sep[:,1])/60., np.max(self.xi_sep[:,1])/60.])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=16)
            cb.set_label(cb_label[I-1], fontsize=18)
            plt.xticks(size=16)
            plt.yticks(size=16)
            plt.xlabel('$\Delta u$ (arcmin)', fontsize=18)
            if I == 1:
                plt.ylabel('$\Delta v$ (arcmin)', fontsize=18)
            I += 1
        if save:
            namefig = os.path.join(rep, '2pcf_dudu_dvdv_dudv_%i.pdf'%int(exp))
            plt.savefig(namefig, transparent=True)

    def plot_gaussian_process(self):
        self.plot_fields(rep=self.rep, save=self.save, exp=self.exp_id)
        self.plot_eb_mode(rep=self.rep, save=self.save, exp=self.exp_id)
        self.plot_2pcf(rep=self.rep, save=self.save, exp=self.exp_id)
        plot_correlation_function(self.gpu._optimizer, NAME='du', 
                                  rep=self.rep, save=self.save, exp=self.exp_id)
        plot_correlation_function(self.gpv._optimizer, NAME='dv', 
                                  rep=self.rep, save=self.save, exp=self.exp_id)

        Y_valid = np.array([self.du_test, self.dv_test]).T
        Y_valid_interp = np.array([self.du_test_predict, self.dv_test_predict]).T
        Y_valid_err =  np.array([self.du_err_test,  self.dv_err_test]).T

        plot_gp_output(self.coords_test, Y_valid, Y_valid_interp, Y_valid_err, 
                       rep=self.rep, save=self.save, exp=self.exp_id)
        self.eb_after_gp(rep=self.rep, save=self.save, exp=self.exp_id)

    def save_output(self):

        pkl_name = os.path.join(self.rep, 'gp_output_%i.pkl'%(int(self.exp_id)))
        pkl_file = open(pkl_name, 'w')
        cPickle.dump(self.dic_output, pkl_file)
        pkl_file.close()

if __name__ == "__main__":

    ## A = np.loadtxt('../../Downloads/residuals4pfl/58131-z/res-meas.list')
    ## exp_id = {}
    ## for exp in A[:,21]:
    ##    if exp not in exp_id:
    ##        print exp
    ##        exp_id.update({exp:None})

    ## exp = 137108.0
    ## Filtre = (A[:,4]<-6)
    ## Filtre &= (A[:,21] == exp)
    ##gp = gpastro(A[:,8][Filtre], A[:,9][Filtre], 
    ##             A[:,10][Filtre], A[:,11][Filtre], 
    ##             A[:,12][Filtre], A[:,13][Filtre],
    ##             mas=3600.*1e3, arcsec=3600.,
    ##             exp_id="137108", visit_id="58131-z",
    ##             rep='plot_output', save=True)

    #INPUT = "/pbs/home/l/leget/sps_lsst/HSC/gp_output/137108_z/input.pkl"

    option = read_option()
    INPUT = os.path.join(option.rep, 'input.pkl')

    dic = cPickle.load(open(INPUT))
    print "gp_astro start"
    gp = gpastro(dic['u'], dic['v'], 
                 dic['du'], dic['dv'], 
                 dic['du_err'], dic['dv_err'],
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id=dic['exp_id'], visit_id="",
                 rep=option.rep, save=True)
    gp.comp_eb()
    gp.comp_xi()
    print "start gp"
    gp.gp_interp()
    print "do plot"
    gp.plot_gaussian_process()
    gp.save_output()
