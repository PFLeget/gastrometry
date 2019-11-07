import os
import glob
import numpy as np 

rep_inout = '/pbs/home/l/leget/sps_lsst/HSC/gp_output/'

os.system('tar cvzf astrometry.tar.gz ../astrometry/')
os.system('mv astrometry.tar.gz ../')

#filt_hsc = ['z', 'g', 'r', 'r2', 'i', 'i2', 'y']
filt_hsc = ['g', 'r', 'r2', 'i', 'i2', 'y']
J = 0
for f in filt_hsc:
    visits = glob.glob(rep_inout+'*%s'%f)
    for i in range(len(visits)):
        print visits[i]
        fichier=open('machine_gun_jobs_%i.sh'%(J),'w')
        fichier.write('#!/bin/bash \n')
        fichier.write('\n')
        fichier.write('home=/pbs/home/l/leget/HSC/ \n')
        fichier.write('\n')
        fichier.write('cp ${home}/astrometry.tar.gz . \n')
        fichier.write('\n')
        fichier.write('tar xzvf astrometry.tar.gz \n')
        fichier.write('\n')
        fichier.write('cd astrometry/ \n')
        fichier.write('\n')
        fichier.write('python gpastro.py --rep %s'%(visits[i]))

        fichier.close()
        o_log = os.path.join(visits[i], "output_o.log")
        e_log = os.path.join(visits[i], "output_e.log")
        #os.system('qsub -P P_lsst -q long -o %s -e %s -l s_fsize=4G -l s_vmem=16G -l sps=1,ct=20:00:00 machine_gun_jobs_%i.sh'%((o_log, e_log, i)))
        os.system('qsub -P P_lsst -pe multicores 8 -q mc_long -o %s -e %s -l sps=1 machine_gun_jobs_%i.sh'%((o_log, e_log, J)))
        os.system('rm machine_gun_jobs_%i.sh*'%(J))
        J += 1
    #if i > 10:
    #    break
