import numpy as np
import copy
import treecorr
from astroeb import vcorr, xiB
from sklearn.model_selection import train_test_split
import pylab as plt

class gpastro(object):

    def __init__(self, u, v, du, dv, du_err, dv_err, 
                 mas=3600.*1e3, arcsec=3600., 
                 exp_id="", visit_id=""):
        
        self.exp_id = exp_id
        self.visit_id = visit_id

        self.u = u * arcsec
        self.v = v * arcsec
        self.du = du * mas
        self.dv = dv * mas
        self.du_err = du_err * mas
        self.dv_err = dv_err * mas

        self.logr = None
        self.xie = None
        self.xib = None

        self.xi_sep = None
        self.xi_dudu = None
        self.xi_dvdv = None
        self.xi_dudv = None

    def comp_eb(self):
        
        self.logr, xiplus, ximinus, xicross, xiz2 = vcorr(self.u/3600., self.v/3600., 
                                                          self.du, self.dv)
        self.xib = xiB(self.logr, xiplus, ximinus)
        self.xie = xiplus - self.xib

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

    def gp_interp(self):

        #self.build_stars()
        #kernel = "20. * AnisotropicVonKarman(invLam=np.array([[1./1000**2,0],[0,1./1000**2]])) + 5."
        #self.interp = piff.GPInterp2pcf(kernel=kernel, optimize=True, anisotropic=True, 
        #                                npca=0, nbins=25, min_sep=0., max_sep=20.*60., white_noise=0.)
        #self.interp.initialize(self.stars_training)
        #print "finish loading"
        #self.interp.solve(self.stars_training)
        #print "finish pcf fit"
        #self.stars_interp_validation = self.interp.interpolateList(self.stars_validation)
        #print "finish interp"
        
    def plot_fields(self):

        MAX = 3.*np.std(self.du)
        plt.figure(figsize=(10,10))
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
    
        plt.figure(figsize=(10,10))
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

        fig = plt.figure(figsize=(8,10))
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

    def plot_eb_mode(self):

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

    def plot_2pcf(self):

        cb_label = ['$\\xi_{du,du}$ (mas$^2$)', 
                    '$\\xi_{dv,dv}$ (mas$^2$)', 
                    '$\\xi_{du,dv}$ (mas$^2$)']
        XI = [self.xi_dudu, self.xi_dvdv, self.xi_dudv]

        I = 1
        plt.figure(figsize=(14,5))
        #plt.subplots_adjust(wspace=0.5, left=0.1, right=0.95, top=0.99, bottom=0.12)
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
            #plt.axis("equal")
            plt.xticks(size=16)
            plt.yticks(size=16)
            plt.xlabel('$\Delta u$ (arcmin)', fontsize=18)
            if I == 1:
                plt.ylabel('$\Delta v$ (arcmin)', fontsize=18)
            I += 1
        #plt.suptitle(int(exp), fontsize=26)


if __name__ == "__main__":

    A = np.loadtxt('data/58131-z/res-meas.list')
    exp_id = {}
    for exp in A[:,21]:
        if exp not in exp_id:
            print exp
            exp_id.update({exp:None})

    exp = 137108.0
    Filtre = (A[:,4]<-6)
    Filtre &= (A[:,21] == exp)

    print "gastro start"
    gp = gpastro(A[:,8][Filtre], A[:,9][Filtre], 
                 A[:,10][Filtre], A[:,11][Filtre], 
                 A[:,12][Filtre], A[:,13][Filtre],
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id="137108", visit_id="58131-z")
    gp.comp_eb()
    gp.comp_xi()
    gp.gp_interp()
    print "do plot"
    gp.plot_fields()
    gp.plot_eb_mode()
    gp.plot_2pcf()
    
