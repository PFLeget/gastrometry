import numpy as np
import glob
import os

outputs = glob.glob('/pbs/home/l/leget/sps_lsst/HSC/gp_output/*z')

for output in outputs:
    logs = glob.glob(os.path.join(output, '*.log'))
    if len(logs) != 0:
        for log in logs:
            os.system('rm %s'%(log))

