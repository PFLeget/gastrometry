import os
import glob
import numpy as np 

def launch_jobs_ccin2p3(rep_inout='/pbs/home/l/leget/sps_lsst/HSC/gp_output/',
                        filt_telescop = ['z', 'g', 'r', 'r2', 'i', 'i2', 'y'],
                        NBIN=21, MAX = 17.*60.,
                        P0=[3000., 0., 0.],
                        queue = 'lsst',
                        cpu_time = '04:00:00',
                        kernel = "15**2 * AnisotropicVonKarman(invLam=np.array([[1./3000.**2,0],[0,1./3000.**2]]))",
                        rep_mean = None):
    J = 0
    stop = False
    for f in filt_telescop:
        visits = os.path.join(rep_inout, '*%s'%f) 
        visits = glob.glob(visits)
        for i in range(len(visits)):
            print(visits[i])

            yaml_name = os.path.join(visits[i], 'config_file.yaml')
            config_yaml = open(yaml_name,'w')
            config_yaml.write('rep: %s\n'%visits[i])
            config_yaml.write('\n')
            config_yaml.write('NBIN: %i\n'%NBIN)
            config_yaml.write('\n')
            config_yaml.write('MAX: %f\n'%MAX)
            config_yaml.write('\n')
            config_yaml.write('P0: {0}\n'.format(P0))
            config_yaml.write('\n')
            config_yaml.write('kernel: %s\n'%kernel)
            if rep_mean is not None:
                config.yaml.write('\n')
                config.yaml.write('rep_mean: %s'%rep_mean)
            config_yaml.close()

            fichier=open('machine_gun_jobs_%i.sh'%(J),'w')
            fichier.write('#!/bin/bash \n')
            fichier.write('\n')
            fichier.write('gastrify %s'%(yaml_name))

            fichier.close()

            o_log = os.path.join(visits[i], "output_o.log")
            e_log = os.path.join(visits[i], "output_e.log")
            command = 'qsub -P P_%s -pe multicores 8 -q mc_long'%(queue)
            command += ' -o %s -e %s -l h_cpu=%s -l sps=1 machine_gun_jobs_%i.sh'%((o_log, e_log, cpu_time, J))
            os.system(command)
            os.system('rm machine_gun_jobs_%i.sh*'%(J))
            J += 1
            #if i > 10:
            #    stop = True
            #    break
        #if stop:
        #    break

