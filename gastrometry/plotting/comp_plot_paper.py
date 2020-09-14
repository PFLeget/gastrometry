# fig 8 (137108) fit gp
# fig 9 (137108) fit gp E/B mode improvement
# fig 10 (fov tp mean)
# fig 11 (fov ccd mean)
# fig 12 (improvement E/B mode on test)
# fig 13 wrms vs mag
# fig 14 (fov ccd mean before/after taking into account)

import os
import pylab as plt
from gastrometry.plotting import plot_single_exposure, plot_eb_mode_single_visit
from gastrometry.plotting import plot_gpfit


def plot_paper(rep_in='../../../hsc_outputs/v3.3/astro_VK/'):

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
    

if __name__ == '__main__':

    plot_paper(rep_in='../../../hsc_outputs/v3.3/astro_VK/')
