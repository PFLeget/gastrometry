# fig 12 (improvement E/B mode on test)
# fig 13 wrms vs mag --> je peux pas le faire car j'ai besoin des trois run, donc toujours a la main


import os
import pylab as plt
from gastrometry.plotting import plot_single_exposure, plot_eb_mode_single_visit
from gastrometry.plotting import plot_gpfit
from gastrometry.plotting import plot_mean_ccd, plot_fov_mean
from gastrometry.plotting import plot_outputeb

def plot_paper(rep_in='../../../hsc_outputs/v3.3/astro_VK/', correction_name='Von Karman kernel'):

    rep_out = os.path.join(rep_in, 'plot_paper')
    os.system('mkdir %s'%(rep_out))

    # fig 1 --> fig 5
    for exp in ['13268', '53732', '137108']:
        plot_single_exposure(os.path.join(rep_in,'%s_z/input.pkl'%(exp)),
                             subtitle3="\n(exp id: %s)"%(exp), read_python2=False)
        plt.savefig(os.path.join(rep_out,'%s_z.png'%(exp)))
        plt.close()

    plot_single_exposure(os.path.join(rep_in,'96450_z/input.pkl'),
                         MAX=39, scale_arrow=0.5, subtitle3="\n(exp id: 96450)",
                         x_arrow=1600, y_arrow=2700, size_arrow=100,
                         x_text=1750, y_text=2650)
    plt.savefig(os.path.join(rep_out,'96450_z_b_mode.png'))
    plt.close()

    plot_eb_mode_single_visit(os.path.join(rep_in,'137108_z/input.pkl'),
                              mas=3600.*1e3, arcsec=3600., YLIM=[-20,70], title="(exp id: 137108)",
                              read_python2=False)
    plt.savefig(os.path.join(rep_out, '137108_z_eb_mode.pdf'))
    plt.close()

    plot_eb_mode_single_visit(os.path.join(rep_in, '96450_z/input.pkl'),
                              mas=3600.*1e3, arcsec=3600., YLIM=[-5,300], title="(exp id: 96450)")
    plt.savefig(os.path.join(rep_out, '96450_z_b_mode_eb_mode.pdf'))
    plt.close()

    #fig 8 & 9
    plot_gpfit(os.path.join(rep_in, '137108_z/gp_output_137108.pkl'),
	       rep_out=rep_out, exp_id='137108')

    #fig 10
    plot_fov_mean(os.path.join(rep_in, 'mean_function/mean_tp.pkl'),
                  rep_fig=rep_out)
    plt.close()

    #fig 11 & 14
    for i in [7, 11, 14, 42]:
        plot_mean_ccd(os.path.join(rep_in, 'mean_function/all/mean_du_%i_all.fits'%(i)),
                      os.path.join(rep_in, 'mean_function/all/mean_dv_%i_all.fits'%(i)),
                      name= 'CCD %i'%(i), cmap=plt.cm.inferno, MAX=2,
                      name_fig=os.path.join(rep_out, 'ccd_%i_mean.png'%(i)))

    # resume eb_mode_plot fig 3 & 12
    po = plot_outputeb(os.path.join(rep_in, 'outputs/final_gp_outputs_all.pkl'))
    po.plot_eb_mode(YLIM=[-10,60], add_title=None)
    plt.savefig(os.path.join(rep_out, '0_eb_mode_all_no_correction.pdf'))
    po.plot_eb_mode_test(YLIM=[-10,60], add_title='No correction (validation sample)')
    plt.savefig(os.path.join(rep_out, '1_eb_mode_test_no_correction.pdf'))
    po.plot_eb_mode_test_residuals(YLIM=[-10,60], add_title='GP corrected with %s (validation sample)'%(correction_name))
    plt.savefig(os.path.join(rep_out, '2_eb_mode_test_gp_corrected.pdf'))
    
if __name__ == '__main__':

    plot_paper(rep_in='../../../hsc_outputs/v3.3/astro_VK/', correction_name='Von Karman kernel')
