import numpy as np
import pylab as plt
import copy
import cPickle
import glob
import corner
import seaborn as sns
import treegp
from sklearn.neighbors import KNeighborsRegressor
from astroeb import vcorr, xiB
from astropy.stats import median_absolute_deviation as mad_astropy

def biweight_M(sample,CSTD=6.):
    """
    average using biweight (beers 90)
    """
    M = np.median(sample)
    iterate = [copy.deepcopy(M)]
    mu = (sample-M)/(CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = (sample-M)*((1.-mu**2)**2)
    down = (1.-mu**2)**2
    M = M + np.sum(up[Filtre])/np.sum(down[Filtre])

    iterate.append(copy.deepcopy(M))
    i=1
    while abs((iterate[i-1]-iterate[i])/iterate[i])<0.001:

        mu = (sample-M)/(CSTD*mad_astropy(sample))
        Filtre = (abs(mu)<1)
        up=(sample-M)*((1.-mu**2)**2)
        down = (1.-mu**2)**2
        M = M + np.sum(up[Filtre])/np.sum(down[Filtre])
        iterate.append(copy.deepcopy(M))
        i+=1
        if i == 100 :
            print('voila voila ')
            break
    return M

def biweight_S(sample,CSTD=9.):
    """
    std from biweight
    """
    M = biweight_M(sample)
    mu = (sample-M)/(CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = ((sample-M)**2)*((1.-mu**2)**4)
    down = (1.-mu**2)*(1.-5.*mu**2)
    std = np.sqrt(len(sample))*(np.sqrt(np.sum(up[Filtre]))/abs(np.sum(down[Filtre])))
    return std

def return_median(x):
    
    median = np.zeros_like(x[0])
    for i in range(len(median)):
        Filtre = np.isfinite(x[:,i])
        if np.sum(Filtre) == 0:
            median[i] = np.nan
        else:
            median[i] = biweight_M(x[:,i][Filtre])
    return median



class load_output(object):

    def __init__(self, pkls):

        self.pkls = pkls

        self.exp_id = []

        self.logr = []
        self.pcf_dudu = []
        self.pcf_dudv = []
        self.pcf_dvdv = []
        self.pcf_sep = []
        
        self.e_mode = []
        self.e_mode_test = []
        self.e_mode_residuals = []
        self.b_mode = []
        self.b_mode_test = []
        self.b_mode_residuals = []

        self.coord_test = []
        self.du_test = []
        self.dv_test = []
        self.du_predict = []
        self.dv_predict = []

    def load_data(self):
        
        I = 1
        for pkl in self.pkls:
            print "%i/%i"%((I,len(self.pkls)))
            dic = cPickle.load(open(pkl))

            self.exp_id.append(dic['exp_id'])
            
            self.logr.append(dic['2pcf_stat']['logr'])
            self.pcf_dudu.append(dic['2pcf_stat']['xi_dudu'])
            self.pcf_dudv.append(dic['2pcf_stat']['xi_dudv'])
            self.pcf_dvdv.append(dic['2pcf_stat']['xi_dvdv'])
            self.pcf_sep.append(dic['2pcf_stat']['xi_sep'])
            self.e_mode.append(dic['2pcf_stat']['xie'])
            self.e_mode_test.append(dic['2pcf_stat']['xie_test'])
            self.e_mode_residuals.append(dic['2pcf_stat']['xie_residuals'])
            self.b_mode.append(dic['2pcf_stat']['xib'])
            self.b_mode_test.append(dic['2pcf_stat']['xib_test'])
            self.b_mode_residuals.append(dic['2pcf_stat']['xib_residuals'])

            self.coord_test.append(dic['gp_output']['gpu.coords_test'])
            self.du_test.append(dic['gp_output']['gpu.du_test'])
            self.dv_test.append(dic['gp_output']['gpv.dv_test'])
            self.du_predict.append(dic['gp_output']['gpu.du_test_predict'])
            self.dv_predict.append(dic['gp_output']['gpv.dv_test_predict'])

            I += 1

    def save_output(self, pkl_name):

        dic = {'exp_id': np.array(self.exp_id),
               'logr': np.array(self.logr),
               'pcf_dudu': np.array(self.pcf_dudu),
               'pcf_dudv': np.array(self.pcf_dudv),
               'pcf_dvdv': np.array(self.pcf_dvdv),
               'pcf_sep' : np.array(self.pcf_sep),
               'e_mode': np.array(self.e_mode),
               'e_mode_test': np.array(self.e_mode_test),
               'e_mode_residuals': np.array(self.e_mode_residuals),
               'b_mode': np.array(self.b_mode),
               'b_mode_test': np.array(self.b_mode_test),
               'b_mode_residuals': np.array(self.b_mode_residuals),
               'coord_test': np.array(self.coord_test),
               'du_test': np.array(self.du_test),
               'dv_test': np.array(self.dv_test),
               'du_predict': np.array(self.du_predict),
               'dv_predict': np.array(self.dv_predict)}

        pkl = open(pkl_name, 'w')
        cPickle.dump(dic, pkl)
        pkl.close()


class plot_output(object):

    def __init__(self, pkl_output):

        print "start load"
        dic = cPickle.load(open(pkl_output))
        print "done loading"

        # remove night when rotation was done by the PI,
        # B mode appears due to the fact that this is not
        # build in JoinCal. 
        exp = dic['exp_id'].astype(int)
        Filtre = ~((exp>= 96446) & (exp<= 96656))

        Filtre &= ~((exp>= 139220) & (exp<= 139260))

        self.exp_id = dic['exp_id'][Filtre]
        self.logr = dic['logr'][Filtre]
        self.pcf_dudu = dic['pcf_dudu'][Filtre]
        self.pcf_dudv = dic['pcf_dudv'][Filtre]
        self.pcf_dvdv = dic['pcf_dvdv'][Filtre]
        self.pcf_sep = dic['pcf_sep'][Filtre]
        self.e_mode = dic['e_mode'][Filtre]
        self.e_mode_test = dic['e_mode_test'][Filtre]
        self.e_mode_residuals = dic['e_mode_residuals'][Filtre]
        self.b_mode = dic['b_mode'][Filtre]
        self.b_mode_test = dic['b_mode_test'][Filtre]
        self.b_mode_residuals = dic['b_mode_residuals'][Filtre]
        self.coord_test = dic['coord_test'][Filtre]
        self.du_test = dic['du_test'][Filtre]
        self.dv_test = dic['dv_test'][Filtre]
        self.du_predict = dic['du_predict'][Filtre] 
        self.dv_predict = dic['dv_predict'][Filtre]


    def plot_residuals(self):
        STD_u = np.zeros(len(self.exp_id))
        STD_v = np.zeros(len(self.exp_id))

        STD_u_corr = np.zeros(len(self.exp_id))
        STD_v_corr = np.zeros(len(self.exp_id))

        plt.figure(figsize=(8,8))

        plt.subplots_adjust(wspace=0, hspace=0)

        #self.residuals_gp = []
        #self.residuals = []

        plt.subplot(2,2,3)
        #residuals = []
        for i in range(len(self.exp_id)):
        #    for j in range(len(self.du_test[i])):
        #        if np.sqrt(self.du_test[i][j]**2+self.dv_test[i][j]**2)<40:
        #            self.residuals.append([self.du_test[i][j], self.dv_test[i][j]])
        #            self.residuals_gp.append([self.du_test[i][j]-self.du_predict[i][j], 
        #                                      self.dv_test[i][j]-self.dv_predict[i][j]])
        #self.residuals = np.array(self.residuals)
        #self.residuals_gp = np.array(self.residuals_gp)
        #sns.jointplot(residuals[:,0], residuals[:,1], kind="kde", height=7, space=0)
        #fig = corner.corner(residuals, labels=['du','dv'],
        #                    levels=(0.68, 0.95), color='r')
        #corner.corner(residuals_gp, levels=(0.68, 0.95), fig=fig, color='b')
        

            plt.scatter(self.du_test[i], self.dv_test[i], s=5, alpha=0.01, c='r')

            plt.scatter(self.du_test[i]-self.du_predict[i], 
                        self.dv_test[i]-self.dv_predict[i], s=5, alpha=0.01, c='b')

            STD_u[i] = biweight_S(self.du_test[i])
            STD_v[i] = biweight_S(self.dv_test[i])

            STD_u_corr[i] = biweight_S(self.du_test[i]-self.du_predict[i])
            STD_v_corr[i] = biweight_S(self.dv_test[i]-self.dv_predict[i])

        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('du (mas)', fontsize=18)
        plt.ylabel('dv (mas)', fontsize=18)

        plt.subplot(2,2,1)
        plt.hist(self.du_test[i],bins=np.linspace(-30, 30, 30), histtype='step', color='r')
        plt.hist(self.du_test[i]-self.du_predict[i],bins=np.linspace(-30,30,30), histtype='step', color='b')
        plt.xlim(-30,30)
        plt.xticks([],[])
        plt.yticks([],[])


        plt.subplot(2,2,4)
        plt.hist(self.dv_test[i], bins=np.linspace(-30, 30, 30),
                 histtype='step', color='r', orientation='horizontal')
        plt.hist(self.dv_test[i]-self.dv_predict[i], bins=np.linspace(-30,30,30), 
                 histtype='step', color='b', orientation='horizontal')
        plt.ylim(-30,30)
        plt.xticks([],[])
        plt.yticks([],[])

        plt.figure()
        plt.hist(STD_u, bins=np.linspace(0, 30, 31), histtype='step', color='r')
        plt.hist(STD_u_corr, bins=np.linspace(0, 30, 31), histtype='step', color='b')

        plt.figure()
        plt.hist(STD_v, bins=np.linspace(0, 30, 31), histtype='step', color='r')
        plt.hist(STD_v_corr, bins=np.linspace(0, 30, 31), histtype='step', color='b')

    def meanify(self, CMAP = None, MAX = 1.):
        #self.coord_test
        #self.du_test 
        #self.dv_test 
        #self.du_predict
        #self.dv_predict

        self.mean_u = treegp.meanify(bin_spacing=30.,
                                     statistics='median')

        self.mean_v = treegp.meanify(bin_spacing=30.,
                                     statistics='median')
        for i in range(len(self.exp_id)):
            self.mean_u.add_field(self.coord_test[i], self.du_test[i]-self.du_predict[i])
            self.mean_v.add_field(self.coord_test[i], self.dv_test[i]-self.dv_predict[i])
        self.mean_u.meanify()
        self.mean_v.meanify()

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_u.coords0[:,0], self.mean_u.coords0[:,1],
                    c=self.mean_u.params0, cmap = CMAP, vmin=-MAX, vmax=MAX,
                    lw=0, s=8)
        plt.colorbar()

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_v.coords0[:,0], self.mean_v.coords0[:,1],
                    c=self.mean_v.params0, cmap = CMAP, vmin=-MAX,vmax=MAX,
                    lw=0, s=8)
        plt.colorbar()
        # moche mais besoin de sampler la fct moyenne pour avoir fct a 2 point qui resemble a quelque chose
        np.random.seed(42)
        D = max(np.sqrt(self.mean_u.coords0[:,0]**2 + self.mean_u.coords0[:,1]**2))
        x = np.random.uniform(-D, D, size=20000)
        y = np.random.uniform(-D, D, size=20000)
        Filtre = (np.sqrt(x**2 + y**2) < D)
        X = np.array([x[Filtre], y[Filtre]]).T
        neigh = KNeighborsRegressor(n_neighbors=3)
        neigh.fit(self.mean_u.coords0, self.mean_u.params0)
        average_u = neigh.predict(X)
        neigh.fit(self.mean_v.coords0, self.mean_v.params0)
        average_v = neigh.predict(X)
        
        plt.figure()
        plt.scatter(X[:,0], X[:,1],
                    c=average_u, cmap = CMAP, vmin=-MAX,vmax=MAX,
                    lw=0, s=8)
        plt.figure()
        plt.scatter(X[:,0], X[:,1],
                    c=average_v, cmap = CMAP, vmin=-MAX,vmax=MAX,
                    lw=0, s=8)

        plt.figure(figsize=(12,8))

        logr, xiplus, ximinus, xicross, xiz2 = vcorr(X[:,0]/3600., X[:,1]/3600.,
                                                     average_u, average_v)
        xib = xiB(logr, xiplus, ximinus)
        xie = xiplus - xib

        plt.scatter(np.exp(logr), xie, s=50, c='b', label='E-mode of meanify')
        plt.scatter(np.exp(logr), xib, s=50, c='r', label='B-mode of meanify')
        plt.plot(np.exp(logr), np.zeros_like(logr), 'k--', lw=3)
        plt.ylim(-20,40)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=22)
        plt.savefig('../../Desktop/4_eb_meanify.png')


            
    def plot_eb_mode(self):
        
        print "eb mode plot"
        plt.figure(figsize=(12,8))
        for i in range(len(self.exp_id)):
            plt.scatter(np.exp(self.logr[i]), self.e_mode[i], s=5, alpha=0.009, c='b')
            plt.scatter(np.exp(self.logr[i]), self.b_mode[i], s=5, alpha=0.009, c='r')

        efilter = np.isfinite(self.e_mode)
        ew = np.ones_like(self.e_mode)
        self.e_mode[~efilter] = 0
        ew[~efilter] = 0

        bfilter = np.isfinite(self.b_mode)
        bw = np.ones_like(self.b_mode)
        self.b_mode[~bfilter] = 0
        bw[~bfilter] = 0
 
        #plt.scatter(np.exp(self.logr[0]), np.average(self.e_mode, weights=ew, axis=0), s=50, c='b', label='average E-mode')
        #plt.scatter(np.exp(self.logr[0]), np.average(self.b_mode, weights=bw, axis=0), s=50, c='r', label='average B-mode')
        med_e = return_median(self.e_mode)
        med_b = return_median(self.b_mode)
        plt.scatter(np.exp(self.logr[0]), med_e, s=50, c='b', label='median E-mode')
        plt.scatter(np.exp(self.logr[0]), med_b, s=50, c='r', label='median B-mode')
        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(-20,40)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=22)


    def plot_eb_mode_test(self):
        
        print "eb mode plot"
        plt.figure(figsize=(12,8))
        for i in range(len(self.exp_id)):
            plt.scatter(np.exp(self.logr[i]), self.e_mode_test[i], s=5, alpha=0.009, c='b')
            plt.scatter(np.exp(self.logr[i]), self.b_mode_test[i], s=5, alpha=0.009, c='r')

        efilter = np.isfinite(self.e_mode_test)
        ew = np.ones_like(self.e_mode_test)
        self.e_mode_test[~efilter] = 0
        ew[~efilter] = 0

        bfilter = np.isfinite(self.b_mode_test)
        bw = np.ones_like(self.b_mode_test)
        self.b_mode_test[~bfilter] = 0
        bw[~bfilter] = 0
 
        #plt.scatter(np.exp(self.logr[0][20:]), np.average(self.e_mode_test[:,20:], weights=ew[:,20:], axis=0), 
        #            s=50, c='b', label='average E-mode (test)')
        #plt.scatter(np.exp(self.logr[0][20:]), np.average(self.b_mode_test[:,20:], weights=bw[:,20:], axis=0), 
        #            s=50, c='r', label='average B-mode (test)')
        med_e =return_median(self.e_mode_test)
        med_b =return_median(self.b_mode_test)
        plt.scatter(np.exp(self.logr[0][20:]), med_e[20:], s=50, c='b', label='median E-mode (test)')
        plt.scatter(np.exp(self.logr[0][20:]), med_b[20:], s=50, c='r', label='median B-mode (test)')
        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(-20,40)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=22)


    def plot_eb_mode_test_residuals(self):

        print "eb mode plot"
        plt.figure(figsize=(12,8))
        for i in range(len(self.exp_id)):
            plt.scatter(np.exp(self.logr[i]), self.e_mode_residuals[i], s=5, alpha=0.009, c='b')
            plt.scatter(np.exp(self.logr[i]), self.b_mode_residuals[i], s=5, alpha=0.009, c='r')

        efilter = np.isfinite(self.e_mode_residuals)
        ew = np.ones_like(self.e_mode_residuals)
        self.e_mode_residuals[~efilter] = 0
        ew[~efilter] = 0

        bfilter = np.isfinite(self.b_mode_residuals)
        bw = np.ones_like(self.b_mode_residuals)
        self.b_mode_residuals[~bfilter] = 0
        bw[~bfilter] = 0
 
        #plt.scatter(np.exp(self.logr[0][20:]), np.average(self.e_mode_residuals[:,20:], weights=ew[:,20:], axis=0), 
        #            s=50, c='b', label='average E-mode (test, after GP)')
        #plt.scatter(np.exp(self.logr[0][20:]), np.average(self.b_mode_residuals[:,20:], weights=bw[:,20:], axis=0), 
        #            s=50, c='r', label='average B-mode (test, after GP)')
        med_e =return_median(self.e_mode_residuals)
        med_b =return_median(self.b_mode_residuals)
        plt.scatter(np.exp(self.logr[0]), med_e, s=50, c='b', label='median E-mode (test, after GP)')
        plt.scatter(np.exp(self.logr[0]), med_b, s=50, c='r', label='median B-mode (test, after GP)')
        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(-20,40)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=22)


    def plot_2pcf(self):

        cb_label = ['$\\xi_{du,du}$ (mas$^2$)',
                    '$\\xi_{dv,dv}$ (mas$^2$)',
                    '$\\xi_{du,dv}$ (mas$^2$)']

        dudu = np.zeros_like(self.pcf_dudu)
        dudv = np.zeros_like(self.pcf_dudu)
        dvdv = np.zeros_like(self.pcf_dudu)

        for i in range(len(self.pcf_dudu)):
            sig_u = np.max(self.pcf_dudu[i])
            sig_v = np.max(self.pcf_dvdv[i])
            dudu[i] = copy.deepcopy(self.pcf_dudu[i]) / sig_u**2
            dvdv[i] = copy.deepcopy(self.pcf_dvdv[i]) / sig_v**2
            dudv[i] = copy.deepcopy(self.pcf_dudv[i]) / (sig_v*sig_u)

        XI = [np.mean(dudu, axis=0), 
              np.mean(dvdv, axis=0),
              np.mean(dudv, axis=0)]

        I = 1
        plt.figure(figsize=(14,5))
        plt.subplots_adjust(wspace=0.4,left=0.07,right=0.95, bottom=0.15,top=0.85)
        for xi in XI:
            MAX = 0.1
            plt.subplot(1,3,I)
            plt.imshow(xi, cmap=plt.cm.seismic,
                       vmin=-MAX, vmax=MAX, origin="lower",
                       extent=[np.min(self.pcf_sep[0][:,0])/60., np.max(self.pcf_sep[0][:,1]/60.),
                               np.min(self.pcf_sep[0][:,1])/60., np.max(self.pcf_sep[0][:,1])/60.])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=16)
            cb.set_label(cb_label[I-1], fontsize=18)
            plt.xticks(size=16)
            plt.yticks(size=16)
            plt.xlabel('$\Delta u$ (arcmin)', fontsize=18)
            if I == 1:
                plt.ylabel('$\Delta v$ (arcmin)', fontsize=18)
            I += 1



if __name__ == '__main__':

    #pkls = glob.glob('../../sps_lsst/HSC/gp_output/*_z/gp_output*.pkl')

    #lo = load_output(pkls)
    #lo.load_data()
    #lo.save_output('final_z_gp_output.pkl')

    po = plot_output('tests/final_z_gp_output.pkl')
    ##po.plot_eb_mode()
    ##plt.savefig('../../Desktop/1_eb_glob.png')
    ##po.plot_eb_mode_test()
    ##plt.savefig('../../Desktop/2_eb_glob_test.png')
    ##po.plot_eb_mode_test_residuals()
    ##plt.savefig('../../Desktop/3_eb_glob_test_afterGP.png')
    ##po.plot_residuals()
    po.meanify()
    #po.plot_2pcf()
