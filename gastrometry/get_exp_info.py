from astropy.io import fits
import os
import glob
import pickle

def get_exp_id(filename):
    exp_id = ""
    for i in range(len(filename)):
        if  filename[-(i+1)] == '/':
            if 'p042' in exp_id:
                break
            else:
                exp_id = ""
        else:
            exp_id = filename[-(i+1)] + exp_id
    if 'p042' in exp_id:
        return exp_id[:-4]
    else:
        return None

def get_exp_info(hsc_images="/sps/lsst/HSC/prod.2020-03.calib/dbimage_UI5XG7I/data/",
                 rep_output=""):
    
    data_path = os.path.join(hsc_images, '*/*042/calibrated.fz')
    filesname = glob.glob(data_path)

    KEY = ['INR-STR', 'INR-END', 'EXPTIME', 
           'SESEEING', 'AIRMASS', 'MJD-STR']

    dic_output = {}
    
    i = 0
    nfiles = len(filesname)
    for filename in filesname:
        print("%i / %i"%((i+1, nfiles)))
        fits_file = fits.open(filename)
        exp_id = get_exp_id(filename)
        dic_output.update({exp_id:{key: fits_file[1].header[key] for key in KEY}})
        i += 1

    pkl_name = os.path.join(rep_output, 'exp_info_hsc.pkl')
    pkl_file = open(pkl_name, 'wb')
    pickle.dump(dic_output, pkl_file)
    pkl_file.close()

if __name__ == "__main__":

    DATA = "/sps/lsst/HSC/prod.2020-03.calib/dbimage_UI5XG7I/data/"
    get_exp_info(hsc_images=DATA, rep_output="")


    
