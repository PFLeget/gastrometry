import yaml

def read_config(file_name):
    """Read a configuration dict from a file.
    
    :param file_name:   The file name from which the configuration dict should be read.
    """
    with open(file_name) as fin:
        config = yaml.load(fin.read())
    return config

def gastrogp(config, read_input_only=False, 
             interp_only=False, write_output_only=False):
    """
    To do.
    """
    import glob
    import os
    import numpy as np
    from gastrometry import read_input
    from gastrometry import gather_input_all
    from gastrometry import write_output
    from gastrometry import launch_jobs_ccin2p3

    config_setup = [read_input_only, interp_only,
                    write_output_only]

    if sum([not i for i in config_setup]) == len(config_setup):
        raise ValueError("At least one option should be set to True.")

    if sum(config_setup) >= 2:
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
        for key in ['NBIN', 'MAX', 'P0', 'kernel', 'filt_telescop']:
            if key not in config['interp']:
                raise ValueError("%s field is required in config dict"%key)
        if 'queue' not in config['interp']:
            queue = 'lsst'
        else:
            queue = config['interp']['queue']
    
        if 'cpu_time' not in config['interp']:
            cpu_time = '04:00:00'
        else:
            cpu_time = config['interp']['cpu_time']

        launch_jobs_ccin2p3(rep_inout=config['output']['directory'],
                            filt_telescop = config['interp']['filt_telescop'],
                            NBIN=config['interp']['NBIN'],
                            MAX = config['interp']['MAX'],
                            P0=config['interp']['P0'],
                            kernel = config['interp']['kernel'],
                            queue = queue,
                            cpu_time = cpu_time)
    if write_output_only:
        if 'input' not in config:
            raise ValueError("input field is required in config dict")
        for key in ['directory', 'filt_telescop']:
            if key not in config['input']:
                raise ValueError("%s field is required in config dict"%key)

        rep_save = os.path.join(config['output']['directory'], 'outputs')
        os.system('mkdir %s'%(rep_save))
        gather_input_all(config['output']['directory'], rep_save=rep_save)
        write_output(config['output']['directory'], rep_save=rep_save) 


def gastrify(config):

    import pickle
    import os
    from gastrometry import gpastro
    for key in ['rep', 'NBIN', 'MAX', 'P0', 'kernel']:
        if key not in config:
                raise ValueError("%s field is required in config dict"%key)

    INPUT = os.path.join(config['rep'], 'input.pkl')

    dic = pickle.load(open(INPUT, 'rb'))
    print("gp_astro start")
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
    print("start gp")
    gp.gp_interp()
    print("do plot")
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
