import glob
import os
import numpy as np
import cPickle

output = "/pbs/home/l/leget/sps_lsst/HSC/gp_output"

#nights = glob.glob('/sps/snls13/HSC/prod.2019-04/dbimage_35UN7JY/fitastrom_ULHBNSI/data/*z/res-meas.list') 

filt_hsc = ['g', 'r', 'r2', 'i', 'i2', 'z', 'y']
#filt_hsc = ['g', 'r', 'r2', 'i', 'i2', 'y']

for f in filt_hsc:

    nights = glob.glob('/sps/snls15/HSC/prod.2019-04.dev/dbimage_JLBFUJY/fitastrom_FUINMJY/data/*%s/res-meas.list'%(f))

    N = 1
    for night in nights:

        A = np.loadtxt(night)
        exp_id = {}
        for exp in A[:,21]:
            if exp not in exp_id:
                exp_id.update({exp:None})
        E = 1
        for exp in exp_id:
            print "%i/%i "%((N, len(nights))), "%i/%i "%((E, len(exp_id))), "%s"%f
            Filtre = (A[:,4]<-6)
            Filtre &= (A[:,21] == exp)
            rep  = os.path.join(output, "%i_%s"%((int(exp), f)))
            #os.system('mkdir %s'%(rep))

            pkl_name = os.path.join(rep, 'input.pkl')
            pkl_file = open(pkl_name, 'w')
            dic = {'exp_id':'%i'%(exp),
                   'u':A[:,8][Filtre], 
                   'v':A[:,9][Filtre],
                   'x':A[:,6][Filtre],
                   'y':A[:,7][Filtre],
                   'julian_date':A[:,5][Filtre],
                   'du':A[:,10][Filtre],
                   'dv':A[:,11][Filtre],
                   'du_err':A[:,12][Filtre],
                   'dv_err':A[:,13][Filtre]}
            cPickle.dump(dic, pkl_file)
            pkl_file.close()
            E += 1
        N += 1
