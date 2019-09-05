import os
import glob
import numpy as np 

rep_inout = '/pbs/home/l/leget/sps_lsst/HSC/gp_output/'
visits = glob.glob(rep_inout+'*z')

os.system('tar cvzf astrometry.tar.gz ../astrometry/')
os.system('mv astrometry.tar.gz ../')


for i in range(len(visits)):
    print visits[i]
    fichier=open('machine_gun_jobs_%i.sh'%(i),'w')
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
    os.system('qsub -P P_lsst -q long -o %s -e %s -l s_fsize=4G -l s_vmem=16G -l sps=1,ct=20:00:00 machine_gun_jobs_%i.sh'%((o_log, e_log, i)))
    os.system('rm machine_gun_jobs_%i.sh*'%(i))

    if i > 10:
        break
