import numpy as np
import pylab as plt
import os
from gastrometry import biweight_median, biweight_mad
import cPickle

def plot_distribution_residuals(path='../data/', stat = None,
                                filters=['g', 'r', 'i', 'z', 'y'], 
                                colors=['b', 'g', 'r', 'k', 'c'] ,mas=3600.*1e3):

    if stat is not None:
        stat = cPickle.load(open(stat))
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
            dic = cPickle.load(open(pkl))
        
            for exp in range(len(dic['exp_id'])):
                print exp
                for comp in ['du', 'dv']:
                    stat[comp]['median'].append(biweight_median(dic[comp][exp]*mas))
                    stat[comp]['mad'].append(biweight_mad(dic[comp][exp]*mas))
                    stat[comp]['filters'].append(f)

                D = np.sqrt((dic['du'][exp]*mas)**2 + (dic['dv'][exp]*mas)**2)
                stat['D']['median'].append(biweight_median(D))
                stat['D']['mad'].append(biweight_mad(D))
                stat['D']['filters'].append(f)
                
        stat_file = open('stat.pkl', 'w')
        cPickle.dump(stat, stat_file)
        stat_file.close()

    plt.figure(figsize=(8, 8))
    I = 0
    #for f in filters:
    #    Filtre = (np.array(stat['du']['filters']) == f)
    #    plt.hist(np.array(stat['du']['mad'])[Filtre], 
    #             bins = np.linspace(0, 40, 41), density=True, 
    #             histtype='step', color= colors[I])
    #    I += 1
    plt.hist(np.array(stat['du']['mad']), 
             bins = np.linspace(0, 40, 41), #density=True, 
             histtype='step', color= 'b')
    plt.hist(np.array(stat['dv']['mad']), 
             bins = np.linspace(0, 40, 41), #density=True, 
             histtype='step', color= 'r')
    plt.hist(np.array(stat['D']['median']),
             bins = np.linspace(0, 40, 41), #density=True,
             histtype='step', color= 'grey')
    #plt.hist(np.array(stat['D']['mad']),
    #         bins = np.linspace(0, 40, 41), density=True,
    #         histtype='step', color= 'k')
    return stat

if __name__ == "__main__":

    stat = plot_distribution_residuals(path='../data/', stat = 'stat.pkl', mas=3600.*1e3)
