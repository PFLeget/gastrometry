import yaml

def read_config(file_name):
    """Read a configuration dict from a file.
    
    :param file_name:   The file name from which the configuration dict should be read.
    """
    with open(file_name) as fin:
        config = yaml.safe_load(fin.read())
    return config

def gastrogp(config, read_input_only=False, 
             interp_only=False, write_output_only=False,
             comp_meanify=False, comp_plot_paper=False,
             comp_wrms_vs_mag=False):
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
    from gastrometry import run_ma_poule_mean
    from gastrometry import run_ma_poule_wrms_vs_mag
    from gastrometry.plotting import build_mean_in_tp
    from gastrometry.plotting import plot_paper

    config_setup = [read_input_only, interp_only,
                    write_output_only, comp_meanify,
                    comp_plot_paper, comp_wrms_vs_mag]

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

        if 'rep_mean' not in config['interp']:
            rep_mean = None
        else:
            rep_mean = config['interp']['rep_mean']

        launch_jobs_ccin2p3(rep_inout=config['output']['directory'],
                            filt_telescop = config['interp']['filt_telescop'],
                            NBIN=config['interp']['NBIN'],
                            MAX = config['interp']['MAX'],
                            P0=config['interp']['P0'],
                            kernel = config['interp']['kernel'],
                            queue = queue,
                            cpu_time = cpu_time,
                            rep_mean=rep_mean)
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

    if comp_wrms_vs_mag or write_output_only:
        for Bool in [False, True]:
            run_ma_poule_wrms_vs_mag(config['output']['directory'], 
                                     bin_spacing=0.2, gp_corrected=Bool)

    if comp_meanify or write_output_only:
        if 'comp_meanify' not in config:
            raise ValueError("comp_meanify field is required in config dict")
        for key in ['bin_spacing', 'statistics', 'gp_corrected']:
            if key not in config['comp_meanify']:
                raise ValueError("%s field is required in config dict"%key)
        run_ma_poule_mean(config['output']['directory'],
                          bin_spacing=config['comp_meanify']['bin_spacing'],
                          statistics=config['comp_meanify']['statistics'],
                          nccd=105,
                          gp_corrected=config['comp_meanify']['gp_corrected'])
        build_mean_in_tp(rep_mean=os.path.join(config['output']['directory'], 'mean_function/all/'),
                         file_out=os.path.join(config['output']['directory'], 'mean_function/mean_tp.pkl'))

    if comp_plot_paper or write_output_only:
        if 'correction_name' not in config['output']:
                raise ValueError("%s field is required in config dict"%('correction_name'))
        plot_paper(config['output']['directory'],
                   correction_name=config['output']['correction_name'])


def gastrify(config):

    import pickle
    import os
    from gastrometry import gpastro
    for key in ['rep', 'NBIN', 'MAX', 'P0', 'kernel']:
        if key not in config:
                raise ValueError("%s field is required in config dict"%key)

    if 'rep_mean' in config:
        rep_mean = config['rep_mean']
    else:
        rep_mean = None
            
    INPUT = os.path.join(config['rep'], 'input.pkl')

    dic = pickle.load(open(INPUT, 'rb'))
    print("gp_astro start")
    gp = gpastro(dic['u'], dic['v'],
                 dic['du'], dic['dv'],
                 dic['du_err'], dic['dv_err'],
                 xccd=dic['x'], yccd=dic['y'], chipnum=dic['chip_num'],
                 NBIN=config['NBIN'], MAX = config['MAX'],
                 P0=config['P0'],
                 kernel = config['kernel'],
                 mas=3600.*1e3, arcsec=3600.,
                 exp_id=dic['exp_id'], visit_id="",
                 rep=config['rep'],
                 rep_mean=rep_mean, save=True)
    gp.comp_eb()
    gp.comp_xi()
    print("start gp")
    gp.gp_interp(dic_all=dic['dic_all'])
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
