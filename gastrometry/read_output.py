import numpy as np
import copy
import cPickle
import glob
import os

class gather_input(object):

    def __init__(self, rep_output):

        self.rep_output = rep_output

        self.exp_id = []

        self.u = []
        self.v = []
        self.du = []
        self.dv = []

    def load_data(self):

        I = 1
        for rep in self.rep_output:
            print "%i/%i"%((I,len(self.rep_output)))
            try:
                dic_input = cPickle.load(open(os.path.join(rep,'input.pkl')))
            except:
                print 'file do not exist'
                continue
            self.exp_id.append(dic_input['exp_id'])
            self.u.append(dic_input['u'])
            self.v.append(dic_input['v'])
            self.du.append(dic_input['du'])
            self.dv.append(dic_input['dv'])
            I += 1

    def save_output(self, pkl_name):

        dic = {'exp_id': np.array(self.exp_id),
               'u': np.array(self.u),
               'v': np.array(self.v),
               'du': np.array(self.du),
               'dv': np.array(self.dv)}

        pkl = open(pkl_name, 'w')
        cPickle.dump(dic, pkl)
        pkl.close()


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
            print "%i/%i"%((I,len(self.rep_output)))
            try:
                pkl = glob.glob(os.path.join(rep,'gp_output*.pkl'))[0]
                dic = cPickle.load(open(pkl))
                dic_input = cPickle.load(open(os.path.join(rep,'input.pkl')))
            except:
                print 'file do not exist'
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

        pkl = open(pkl_name, 'w')
        cPickle.dump(dic, pkl)
        pkl.close()

def gather_input_all(rep_out, rep_save=''):

    filters = ['g', 'r', 'i', 'z', 'y']

    rep_all = glob.glob(os.path.join(rep_out, '*'))
    lo = gather_input(rep_all)
    lo.load_data()
    lo.save_output(os.path.join(rep_save, 'inputs_all.pkl'))

    for f in filters:
        print f
        rep_filters = glob.glob(os.path.join(rep_out, '*_%s*'%(f)))
        lo = load_output(rep_filters)
        lo.load_data()
        lo.save_output(os.path.join(rep_save, 'inputs_%s.pkl'%(f)))

def write_output(rep_out, rep_save=''):

    filters = ['g', 'r', 'i', 'z', 'y']

    rep_all = glob.glob(os.path.join(rep_out, '*'))
    lo = load_output(rep_all)
    lo.load_data()
    lo.save_output(os.path.join(rep_save, 'final_gp_outputs_all.pkl'))

    for f in filters:
        print f
        rep_filters = glob.glob(os.path.join(rep_out, '*_%s*'%(f)))
        lo = load_output(rep_filters)
        lo.load_data()
        lo.save_output(os.path.join(rep_save, 'final_gp_outputs_%s.pkl'%(f)))

if __name__ == '__main__':

    #write_output('../../../sps_lsst/HSC/gp_output/', rep_save='')
    gather_input_all('../../../sps_lsst/HSC/gp_output/', rep_save='')
