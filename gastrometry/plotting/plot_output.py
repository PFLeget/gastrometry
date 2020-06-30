import numpy as np
import pylab as plt
import copy
import pickle
import glob
import os
import treegp
from sklearn.neighbors import KNeighborsRegressor
from gastrometry import vcorr, xiB, biweight_median
from astropy.stats import median_absolute_deviation as mad_astropy

def median_check_finite(x):
    """
    Median using biweight_median, but remove the nan.

    :param sample: 1d numpy array. The sample where you want
                   to compute the median with outlier rejection.
    """
    median = np.zeros_like(x[0])
    for i in range(len(median)):
        Filtre = np.isfinite(x[:,i])
        if np.sum(Filtre) == 0:
            median[i] = np.nan
        else:
            median[i] = biweight_median(x[:,i][Filtre])

    return median

def mean_sigma_clipping(x, n_sigma=6):
    Filtre = np.array([True]*len(x))
    counts = [np.sum(Filtre)+1, np.sum(Filtre)]
    i = 0 
    while counts[-2] != counts[-1]:
        Filtre &= (abs(x-np.median(x)) < n_sigma * mad_astropy(x[Filtre]))
        counts.append(np.sum(Filtre))
        i += 1
        if i>10:
            break
    print('PF mean:', np.sum(Filtre), '/',  len(Filtre), float(np.sum(Filtre)) / float(len(Filtre)))
    return np.mean(x[Filtre]), np.std(x[Filtre])

def mean_check_finite(x):
    """
    Mean, but remove the nan.

    :param sample: 1d numpy array. The sample where you want
                   to compute the mean with outlier rejection.
    """
    mean = np.zeros_like(x[0])
    std = np.zeros_like(x[0])
    for i in range(len(mean)):
        Filtre = np.isfinite(x[:,i])
        if np.sum(Filtre) == 0:
            mean[i] = np.nan
            std[i] = np.nan
        else:
            mean[i], std[i] = mean_sigma_clipping(x[:,i][Filtre])
    return mean, std

class load_output(object):

    def __init__(self, rep_output):

        self.rep_output = rep_output

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

        self.x_test = []
        self.y_test = []

    def load_data(self):

        I = 1
        for rep in self.rep_output:
            print("%i/%i"%((I,len(self.rep_output))))
            try:
                pkl = glob.glob(os.path.join(rep,'gp_output*.pkl'))[0]
                dic = pickle.load(open(pkl, 'rb'))
                dic_input = pickle.load(open(os.path.join(rep,'input.pkl'), 'rb'))
            except:
                print('file do not exist')
                continue

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

            self.x_test.append(dic_input['x'][dic['input_data']['indice_test']])
            self.y_test.append(dic_input['y'][dic['input_data']['indice_test']])

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
               'dv_predict': np.array(self.dv_predict),
               'x_test': np.array(self.x_test),
               'y_test': np.array(self.y_test)}

        pkl = open(pkl_name, 'wb')
        pickle.dump(dic, pkl)
        pkl.close()


class plot_output(object):

    def __init__(self, pkl_output):

        print("start load")
        dic = pickle.load(open(pkl_output, 'rb'))
        print("done loading")

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
        self.x_test = dic['x_test'][Filtre]
        self.y_test = dic['y_test'][Filtre]


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


    def meanify_ccd(self, CMAP = None, MAX = 1., bin_spacing=5., sm = 8, stat_used='mean'):
        
        self.mean_u = treegp.meanify(bin_spacing=bin_spacing,
                                     statistics=stat_used)

        self.mean_v = treegp.meanify(bin_spacing=bin_spacing,
                                     statistics=stat_used)
        for i in range(len(self.exp_id)):
            coord = np.array([self.x_test[i], self.y_test[i]]).T
            self.mean_u.add_field(coord, self.du_test[i]-self.du_predict[i])
            self.mean_v.add_field(coord, self.dv_test[i]-self.dv_predict[i])
        self.mean_u.meanify()
        self.mean_v.meanify()

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_u.coords0[:,0], self.mean_u.coords0[:,1],
                    c=self.mean_u.params0, cmap = CMAP, vmin=-MAX, vmax=MAX,
                    lw=0, s=sm)
        plt.colorbar()

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_v.coords0[:,0], self.mean_v.coords0[:,1],
                    c=self.mean_v.params0, cmap = CMAP, vmin=-MAX,vmax=MAX,
                    lw=0, s=sm)
        plt.colorbar()

            
    def plot_eb_mode(self, YLIM=[-5,30]):
        
        print("eb mode plot")
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(bottom=0.12, top=0.98,right=0.99)
 
        mean_e, std_e = mean_check_finite(self.e_mode)
        mean_b, std_b = mean_check_finite(self.b_mode)
        

        plt.plot(np.exp(self.logr[0]), mean_e,'b', lw=3, label='mean E-mode')
        plt.fill_between(np.exp(self.logr[0]), mean_e-std_e, mean_e+std_e, color='b', alpha=0.4, label='$\pm 1 \sigma$ E-mode')
        plt.plot(np.exp(self.logr[0]), mean_b,'r', lw=3, label='mean B-mode')
        plt.fill_between(np.exp(self.logr[0]), mean_b-std_b, mean_b+std_b, color='r', alpha=0.4, label='$\pm 1 \sigma$ B-mode')

        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(YLIM[0], YLIM[1])
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=20)


    def plot_eb_mode_test(self, YLIM=[-5,30]):

        print("eb mode plot test")
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(bottom=0.12, top=0.98,right=0.99)
        
        mean_e, std_e = mean_check_finite(self.e_mode_test)
        mean_b, std_b = mean_check_finite(self.b_mode_test)

        plt.plot(np.exp(self.logr[0]), mean_e,'b', lw=3, label='mean E-mode (test)')
        plt.fill_between(np.exp(self.logr[0]), mean_e-std_e, mean_e+std_e, color='b', alpha=0.4, label='$\pm 1 \sigma$ E-mode')
        plt.plot(np.exp(self.logr[0]), mean_b,'r', lw=3, label='mean B-mode (test)')
        plt.fill_between(np.exp(self.logr[0]), mean_b-std_b, mean_b+std_b, color='r', alpha=0.4, label='$\pm 1 \sigma$ B-mode')

        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(YLIM[0], YLIM[1])
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=20)


    def plot_eb_mode_test_residuals(self, YLIM=[-5,30]):

        print("eb mode plot test residuals")
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(bottom=0.12, top=0.98,right=0.99)

        mean_e, std_e = mean_check_finite(self.e_mode_residuals)
        mean_b, std_b = mean_check_finite(self.b_mode_residuals)

        plt.plot(np.exp(self.logr[0]), mean_e,'b', lw=3, label='mean E-mode (test, after GP)')
        plt.fill_between(np.exp(self.logr[0]), mean_e-std_e, mean_e+std_e, color='b', alpha=0.4, label='$\pm 1 \sigma$ E-mode')
        plt.plot(np.exp(self.logr[0]), mean_b,'r', lw=3, label='mean B-mode (test, after GP)')
        plt.fill_between(np.exp(self.logr[0]), mean_b-std_b, mean_b+std_b, color='r', alpha=0.4, label='$\pm 1 \sigma$ B-mode')

        plt.plot(np.exp(self.logr[0]), np.zeros_like(self.logr[0]), 'k--', lw=3)
        plt.ylim(YLIM[0], YLIM[1])
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend(fontsize=20)


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
    ##rep = glob.glob('../../sps_lsst/HSC/gp_output/*')
    ##lo = load_output(rep)
    ##lo.load_data()
    ##lo.save_output('final_gp_outputs.pkl')

    po = plot_output('../../../hsc_outputs/v3/outputs/final_gp_outputs_all.pkl')
    
    po.plot_eb_mode(YLIM=[-10,60])
    plt.savefig('1_eb_glob_vk_v3.pdf')
    po.plot_eb_mode_test(YLIM=[-10,60])
    plt.savefig('2_eb_glob_test_vk_v3.pdf')
    po.plot_eb_mode_test_residuals(YLIM=[-10,60])
    plt.savefig('3_eb_glob_test_afterGP_vk_v3.pdf')

    ##po.plot_residuals()
    ##po.meanify()
    #po.plot_2pcf()
    #po.meanify_ccd(CMAP = None, MAX = 1.)
