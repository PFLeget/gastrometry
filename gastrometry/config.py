

def gastrogp(config, read_input_only=False, 
             interp_only=False, write_output_only=False):
    """
    To do.
    """
    import glob
    import numpy as np

    for key in ['output', 'hyper']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)
    for key in ['file_name']:
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output"%key)

    for key in ['file_name']:
        if key not in config['hyper']:
            raise ValueError("%s field is required in config dict hyper"%key)

    if 'dir' in config['output']:
        dir = config['output']['dir']
    else:
        dir = None

    if 'bin_spacing' in config['hyper']:
        bin_spacing = config['hyper']['bin_spacing'] #in arcsec
    else:
        bin_spacing = 120. #default bin_spacing: 120 arcsec

    if 'statistic' in config['hyper']:
        if config['hyper']['statistic'] not in ['mean', 'median']:
            raise ValueError("%s is not a suported statistic (only mean and median are currently suported)"
                             %config['hyper']['statistic'])
        else:
            stat_used = config['hyper']['statistic']
    else:
        stat_used = 'mean' #default statistics: arithmetic mean over each bin

    if 'params_fitted' in config['hyper']:
        if type(config['hyper']['params_fitted']) != list:
            raise TypeError('must give a list of index for params_fitted')
        else:
            params_fitted = config['hyper']['params_fitted']
    else:
        params_fitted = None

    if isinstance(config['output']['file_name'], list):
        psf_list = config['output']['file_name']
        if len(psf_list) == 0:
            raise ValueError("file_name may not be an empty list")
    elif isinstance(config['output']['file_name'], str):
        file_name = config['output']['file_name']
        if dir is not None:
            file_name = os.path.join(dir, file_name)
        psf_list = sorted(glob.glob(file_name))
        if len(psf_list) == 0:
            raise ValueError("No files found corresponding to "+config['file_name'])
    elif not isinstance(config['file_name'], dict):
        raise ValueError("file_name should be either a dict or a string")

    if psf_list is not None:
        logger.debug('psf_list = %s',psf_list)
        npsfs = len(psf_list)
        logger.debug('npsfs = %d',npsfs)
        config['output']['file_name'] = psf_list
