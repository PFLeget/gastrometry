import yaml

def read_config(file_name):
    """Read a configuration dict from a file.
    
    :param file_name:   The file name from which the configuration dict should be read.
    """
    with open(file_name) as fin:
        config = yaml.load(fin.read())
    return config

def gastrogp(config, read_input_only=False, 
             interp_only=False, write_output=False):
    """
    To do.
    """
    import glob
    import numpy as np
    from gastrometry import read_input

    if not read_input_only and not interp_only:
        raise ValueError("At least one option should be set to True.")

    if read_input_only and interp_only:
        raise ValueError("Just one option should be set to True.")

    if 'output' not in config:
        raise ValueError("output field is required in config dict")

    if read_input_only:
        if 'input' not in config:
            raise ValueError("input field is required in config dict")
        for key in ['directory', 'filt_telescop']:
            if key not in config['input']:
                raise ValueError("%s field is required in config dict"%key)
        read_input(input_astrometry=config['input']['directory'],
                   output=config['output']['directory'],
                   filt_telescop=config['input']['filt_telescop'])
    if interp_only:
        if 'interp' not in config:
            raise ValueError("interp field is required in config dict")
        for key in ['NBIN', 'MAX', 'P0', 'kernel']:
            raise ValueError("%s field is required in config dict"%key)
        launch_jobs_ccin2p3(rep_inout=config['output']['directory'],
                            filt_telescop = config['interp']['filt_telescop'],
                            NBIN=config['interp']['NBIN'],
                            MAX = config['interp']['MAX'],
                            P0=config['interp']['P0'],
                            kernel = config['interp']['kernel'])
        
    #for key in ['output', 'hyper']:
    #    if key not in config:
    #        raise ValueError("%s field is required in config dict"%key)
    #for key in ['file_name']:
    #    if key not in config['output']:
    #        raise ValueError("%s field is required in config dict output"%key)

    #for key in ['file_name']:
    #    if key not in config['hyper']:
    #        raise ValueError("%s field is required in config dict hyper"%key)

    #if 'dir' in config['output']:
    #    dir = config['output']['dir']
    #else:
    #    dir = None

    #if 'bin_spacing' in config['hyper']:
    #    bin_spacing = config['hyper']['bin_spacing'] #in arcsec
    #else:
    #    bin_spacing = 120. #default bin_spacing: 120 arcsec

    #if 'statistic' in config['hyper']:
    #    if config['hyper']['statistic'] not in ['mean', 'median']:
    #        raise ValueError("%s is not a suported statistic (only mean and median are currently suported)"
    #                         %config['hyper']['statistic'])
    #    else:
    #        stat_used = config['hyper']['statistic']
    #else:
    #    stat_used = 'mean' #default statistics: arithmetic mean over each bin

    #if 'params_fitted' in config['hyper']:
    #    if type(config['hyper']['params_fitted']) != list:
    #        raise TypeError('must give a list of index for params_fitted')
    #    else:
    #        params_fitted = config['hyper']['params_fitted']
    #else:
    #    params_fitted = None

    #if isinstance(config['output']['file_name'], list):
    #    psf_list = config['output']['file_name']
    #    if len(psf_list) == 0:
    #        raise ValueError("file_name may not be an empty list")
    #elif isinstance(config['output']['file_name'], str):
    #    file_name = config['output']['file_name']
    #    if dir is not None:
    #        file_name = os.path.join(dir, file_name)
    #    psf_list = sorted(glob.glob(file_name))
    #    if len(psf_list) == 0:
    #        raise ValueError("No files found corresponding to "+config['file_name'])
    #elif not isinstance(config['file_name'], dict):
    #    raise ValueError("file_name should be either a dict or a string")

    #if psf_list is not None:
    #    logger.debug('psf_list = %s',psf_list)
    #    npsfs = len(psf_list)
    #    logger.debug('npsfs = %d',npsfs)
    #    config['output']['file_name'] = psf_list


def gastrify(config):

    import cPickle
    import os
    from gastrometry import gpastro
    for key in ['rep', 'NBIN', 'MAX', 'P0', 'kernel']:
        if key not in config:
                raise ValueError("%s field is required in config dict"%key)

    INPUT = os.path.join(config['rep'], 'input.pkl')

    dic = cPickle.load(open(INPUT))
    print "gp_astro start"
    gp = gpastro(dic['u'], dic['v'],
                 dic['du'], dic['dv'],
                 dic['du_err'], dic['dv_err'],
                 NBIN=config['NBIN'], MAX = config['MAX'],
                 P0=config['P0'],
                 kernel = config['kernel'],
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id=dic['exp_id'], visit_id="",
                 rep=config['rep'], save=True)
    gp.comp_eb()
    gp.comp_xi()
    print "start gp"
    gp.gp_interp()
    print "do plot"
    gp.plot_gaussian_process()
    gp.save_output()


if __name__ == '__main__':

    config = {'input':{'directory':'/sps/snls15/HSC/prod.2019-04.dev/dbimage_JLBFUJY/fitastrom_FUINMJY/data/',
                       'filt_telescop':['g', 'r', 'r2', 'i', 'i2', 'z', 'y']},
              'interp':{'NBIN':21,
                        'MAX': 17.*60.,
                        'P0': [3000., 0., 0.],
                        'kernel': '15**2 * AnisotropicVonKarman(invLam=np.array([[1./3000.**2,0],[0,1./3000.**2]]))'},
              'output':{'directory':'/pbs/home/l/leget/sps_lsst/HSC/gastrometry_test'}}

    gastrogp(config, read_input_only=True,
             interp_only=False, write_output=False)
