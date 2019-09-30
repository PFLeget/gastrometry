import numpy as np
import pylab as plt
import copy
import cPickle
import glob

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

        self.exp_id = dic['exp_id']
        self.logr = dic['logr']
        self.pcf_dudu = dic['pcf_dudu']
        self.pcf_dudv = dic['pcf_dudv']
        self.pcf_dvdv = dic['pcf_dvdv']
        self.pcf_sep = dic['pcf_sep']
        self.e_mode = dic['e_mode']
        self.e_mode_test = dic['e_mode_test']
        self.e_mode_residuals = dic['e_mode_residuals']
        self.b_mode = dic['b_mode']
        self.b_mode_test = dic['b_mode_test']
        self.b_mode_residuals = dic['b_mode_residuals']
        self.coord_test = dic['coord_test']
        self.du_test = dic['du_test']
        self.dv_test = dic['dv_test']
        self.du_predict = dic['du_predict'] 
        self.dv_predict = dic['dv_predict']


    def plot_eb_mode(self):
        
        print "eb mode plot"
        plt.figure(figsize=(12,8))
        for i in range(len(self.exp_id)):
            plt.scatter(np.exp(self.logr[i]), self.e_mode[i], s=10, alpha=0.009, c='b')
            plt.scatter(np.exp(self.logr[i]), self.b_mode[i], s=10, alpha=0.009, c='r')

        efilter = np.isfinite(self.e_mode)
        ew = np.ones_like(self.e_mode)
        self.e_mode[~efilter] = 0
        ew[~efilter] = 0

        bfilter = np.isfinite(self.b_mode)
        bw = np.ones_like(self.b_mode)
        self.b_mode[~bfilter] = 0
        bw[~bfilter] = 0
 
        plt.scatter(np.exp(self.logr[0]), np.average(self.e_mode, weights=ew, axis=0), s=40, c='b', label='average E-mode')
        plt.scatter(np.exp(self.logr[0]), np.average(self.b_mode, weights=bw, axis=0), s=40, c='r', label='average B-mode')
        plt.ylim(-40,100)
        plt.xlim(0.005, 1.5)
        plt.xscale('log')
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
        plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)
        plt.legend()


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

    pkls = glob.glob('../../sps_lsst/HSC/gp_output/*_z/gp_output*.pkl')

    lo = load_output(pkls)
    lo.load_data()
    lo.save_output('final_z_gp_output.pkl')

    #po = plot_output('final_z_gp_output.pkl')
    #po.plot_eb_mode()
    #po.plot_2pcf()
