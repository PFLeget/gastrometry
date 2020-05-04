import numpy as np
import pylab as plt
import os
from gastrometry import biweight_median, biweight_mad
from gastrometry import median_check_finite
import pickle

def plot_distribution_residuals(path='../data/', stat = None,
                                filters=['g', 'r', 'i', 'z', 'y'], 
                                colors=['b', 'g', 'r', 'k', 'c'] ,mas=3600.*1e3):

    if stat is not None:
        stat = pickle.load(open(stat, 'rb'))
    else:
        stat = {'du':{'mad':[],
                      'median':[],
                      'filters':[]},
                'dv':{'mad':[],
                      'median':[],
                      'filters':[]},
                'D':{'mad':[],
                     'median':[],
                     'filters':[]}}

        for f in filters:
            pkl = os.path.join(path, 'inputs_%s.pkl'%(f))
            dic = pickle.load(open(pkl, 'rb'))
        
            for exp in range(len(dic['exp_id'])):
                print(exp)
                for comp in ['du', 'dv']:
                    stat[comp]['median'].append(biweight_median(dic[comp][exp]*mas))
                    stat[comp]['mad'].append(biweight_mad(dic[comp][exp]*mas))
                    stat[comp]['filters'].append(f)

                D = np.sqrt((dic['du'][exp]*mas)**2 + (dic['dv'][exp]*mas)**2)
                stat['D']['median'].append(biweight_median(D))
                stat['D']['mad'].append(biweight_mad(D))
                stat['D']['filters'].append(f)
                
        stat_file = open('stat.pkl', 'wb')
        pickle.dump(stat, stat_file)
        stat_file.close()

    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(top=0.99, right=0.99, left=0.14)
    I = 0
    #for f in filters:
    #    Filtre = (np.array(stat['du']['filters']) == f)
    #    plt.hist(np.array(stat['du']['mad'])[Filtre], 
    #             bins = np.linspace(0, 40, 41), density=True, 
    #             histtype='step', color= colors[I])
    #    I += 1
    plt.hist(np.array(stat['du']['mad']), 
             bins = np.linspace(0, 40, 41), #density=True, 
             histtype='step', color= 'b', lw=2, label='du MAD')
    plt.hist(np.array(stat['dv']['mad']), 
             bins = np.linspace(0, 40, 41), #density=True, 
             histtype='step', color= 'r', lw=2, label='dv MAD')
    plt.hist(np.array(stat['D']['median']),
             bins = np.linspace(0, 40, 41), #density=True,
             histtype='step', color= 'k', lw=2, label='$\left<\sqrt{du^2+dv^2}\\right>$')
    #plt.hist(np.array(stat['D']['mad']),
    #         bins = np.linspace(0, 40, 41), density=True,
    #         histtype='step', color= 'k')
    plt.xlim(0, 42)
    plt.xlabel('mas', fontsize=14)
    plt.ylabel('# of visits', fontsize=14)
    plt.legend(fontsize=12)
    return stat

def plot_eb_mode_full_survey(output_pkl):
     
    dic = pickle.load(open(output_pkl, 'rb'))
    
    plt.figure(figsize=(12,8))
    for i in range(len(dic['e_mode'][:,0])):
        plt.scatter(np.exp(dic['logr'][i]), dic['e_mode'][i], s=5, alpha=0.009, c='b')
        plt.scatter(np.exp(dic['logr'][i]), dic['b_mode'][i], s=5, alpha=0.009, c='r')

    efilter = np.isfinite(dic['e_mode'])
    ew = np.ones_like(dic['e_mode'])
    dic['e_mode'][~efilter] = 0
    ew[~efilter] = 0
        
    bfilter = np.isfinite(dic['b_mode'])
    bw = np.ones_like(dic['b_mode'])
    dic['b_mode'][~bfilter] = 0
    bw[~bfilter] = 0

    med_e =median_check_finite(dic['e_mode'])
    med_b =median_check_finite(dic['b_mode'])
    plt.scatter(np.exp(dic['logr'][0]), med_e, s=50, c='b', label='median E-mode (test, after GP)')
    plt.scatter(np.exp(dic['logr'][0]), med_b, s=50, c='r', label='median B-mode (test, after GP)')
    plt.plot(np.exp(np.exp(dic['logr'][0])), np.zeros_like(np.exp(dic['logr'][0])), 'k--', lw=3)
    plt.ylim(-20,40)
    plt.xlim(0.005, 1.5)
    plt.xscale('log')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('$\Delta \\theta$ (degree)', fontsize=22)
    plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=22)


if __name__ == "__main__":

    #stat = plot_distribution_residuals(path='../data/', stat = 'stat.pkl', mas=3600.*1e3)
    #plt.savefig('../../../../Dropbox/hsc_astro/figures/histo_stds_full_survey_no_corrections.pdf')
    
    plot_eb_mode_full_survey('../data/final_gp_outputs_all.pkl')
