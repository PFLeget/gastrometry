import numpy as np
import glob
import os

outputs = glob.glob('/pbs/home/l/leget/sps_lsst/HSC/gp_output/*')

ndone = 0
not_done = []
for output in outputs:

    pdf = glob.glob(os.path.join(output, '*.pdf'))
    pkl = glob.glob(os.path.join(output, '*.pkl'))
    log = glob.glob(os.path.join(output, '*.log'))
    
    ok = True
    
    if len(pdf) != 16:
        ok &= False

    if len(pkl) != 2:
        ok &= False

    if len(log) != 2:
        ok &= False

    if ok:
        ndone += 1
    else:
        not_done.append(output)

print "%i/%i"%((ndone, len(outputs)))


    
