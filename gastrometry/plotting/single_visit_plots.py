import numpy as np
import pylab as plt
import pickle
from sklearn.model_selection import train_test_split
from gastrometry import biweight_median, biweight_mad
from gastrometry import vcorr, xiB


def plot_single_exposure(input_pkl, CMAP=plt.cm.seismic, MAX=14, 
                         SIZE=20, mas=3600.*1e3, arcsec=3600., 
                         scale_arrow=0.2, subtitle3="", 
                         x_arrow=1900, y_arrow=2500, size_arrow=40,
                         x_text=2050, y_text=2450, read_python2=False):

    if read_python2:
        dic = pickle.load(open(input_pkl, 'rb'), encoding='latin1')
    else:
        dic = pickle.load(open(input_pkl, 'rb'))

    fig = plt.figure(figsize=(15.5,6.5))
    plt.subplots_adjust(wspace=0,top=0.85,right=0.99,left=0.07)

    ax1 = plt.subplot(1,3,1)
    s1 = ax1.scatter(dic['u']*arcsec, dic['v']*arcsec, 
                     c=dic['du']*mas, lw=0, s=SIZE,
                     cmap=CMAP, vmin=-MAX, vmax=MAX)
    ax1.set_ylabel('v (arcmin)',fontsize=18)
    plt.xticks([-40 * 60, -20 * 60 , 0, 20 * 60, 40 * 60],
               ['$-40$', '$-20$', '$0$','$20$', '$40$'], fontsize=16)
    plt.yticks([-40 * 60, -20 * 60 , 0, 20 * 60, 40 * 60],
               ['$-40$', '$-20$', '$0$','$20$', '$40$'], fontsize=16)
    plt.axis('equal')
    #ax1.set_xticks(np.linspace(-2000, 2000, 5))

    ax2 = plt.subplot(1,3,2)
    s2 = ax2.scatter(dic['u']*arcsec, dic['v']*arcsec,
                     c=dic['dv']*mas, lw=0, s=SIZE,
                     cmap=CMAP, vmin=-MAX, vmax=MAX)
    ax2.set_yticks([],[])
    ax2.set_xlabel('u (arcmin)',fontsize=18)
    plt.xticks([-40 * 60, -20 * 60 , 0, 20 * 60, 40 * 60],
               ['$-40$', '$-20$', '$0$','$20$', '$40$'], fontsize=16)
    plt.axis('equal')
    #ax2.set_xticks(np.linspace(-2000, 2000, 5))

    ax3 = plt.subplot(1,3,3)
    quiver_dict = dict(alpha=1,
                       angles='uv',
                       headlength=5,
                       headwidth=5,
                       headaxislength=3,
                       minlength=0,
                       pivot='middle',
                       scale_units='xy',
                       width=0.003,
                       color='blue',
                       scale=scale_arrow)
    indice = np.linspace(0, len(dic['u'])-1, len(dic['u'])).astype(int)
    indice_train, indice_test = train_test_split(indice, test_size=0.5, random_state=42)
    ax3.quiver(dic['u'][indice_train]*arcsec, dic['v'][indice_train]*arcsec,
               dic['du'][indice_train]*mas, dic['dv'][indice_train]*mas,**quiver_dict)
    ax3.set_yticks([],[])
    ax3.quiver([x_arrow],[y_arrow],[size_arrow],[0], **quiver_dict)
    ax3.text(x_text, y_text, "%i mas"%(size_arrow), fontsize=14)
    plt.xticks([-40 * 60, -20 * 60 , 0, 20 * 60, 40 * 60],
               ['$-40$', '$-20$', '$0$','$20$', '$40$'], fontsize=16)
    plt.axis('equal')
    #ax3.set_xticks(np.linspace(-2000, 2000, 5))
    ax3.set_title('astrometric residuals'+subtitle3, fontsize=18)

    ax_cbar1 = fig.add_axes([0.08, 0.9, 0.29, 0.025])
    ax_cbar2 = fig.add_axes([0.385, 0.9, 0.29, 0.025])

    cb1 = plt.colorbar(s1, cax=ax_cbar1, orientation='horizontal')
    cb2 = plt.colorbar(s2, cax=ax_cbar2, orientation='horizontal')

    cb1.set_label('du (mas)', fontsize='18',labelpad=-50)
    cb1.ax.tick_params(labelsize=14)
    cb2.set_label('dv (mas)', fontsize='18',labelpad=-52)
    cb2.ax.tick_params(labelsize=14)
       
def plot_single_exposure_hist(input_pkl, MAX=30., NBIN=30, mas=3600.*1e3, arcsec=3600.):

    dic = pickle.load(open(input_pkl, 'rb'))

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(wspace=0, hspace=0, top=0.99, right=0.99)
    plt.subplot(2,2,3)

    plt.scatter(dic['du']*mas, dic['dv']*mas, 
                s=25, lw=0, alpha=0.1, c='b')

    median_u = biweight_median(dic['du']*mas)
    mad_u = biweight_mad(dic['du']*mas)

    median_v = biweight_median(dic['dv']*mas)
    mad_v = biweight_mad(dic['dv']*mas)

    plt.xlim(-MAX, MAX)
    plt.ylim(-MAX, MAX)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('du (mas)', fontsize=18)
    plt.ylabel('dv (mas)', fontsize=18)

    plt.subplot(2,2,1)
    plt.hist(dic['du']*mas, bins=np.linspace(-MAX, MAX, NBIN), 
             histtype='step', color='b', lw=2,
             label = 'Median = %.2f mas\n MAD = %.2f mas'%((median_u, mad_u)))
    plt.xlim(-MAX, MAX)
    ylim = plt.ylim()
    plt.ylim(ylim[0], ylim[1] + ylim[1]*.2)
    plt.legend()
    plt.xticks([],[])
    plt.yticks([],[])


    plt.subplot(2,2,4)
    plt.hist(dic['dv']*mas, bins=np.linspace(-MAX, MAX, NBIN),
             histtype='step', color='b', lw=2, orientation='horizontal',
             label = 'Median = %.2f mas\n MAD = %.2f mas'%((median_v, mad_v)))
    plt.ylim(-MAX, MAX)
    xlim = plt.xlim()
    plt.xlim(xlim[0], xlim[1] + xlim[1]*.2)
    plt.legend()
    plt.xticks([],[])
    plt.yticks([],[])

def plot_eb_mode_single_visit(input_pkl,mas=3600.*1e3, arcsec=3600., YLIM=[-20,60], title="",read_python2=False):

    if read_python2:
        dic = pickle.load(open(input_pkl, 'rb'), encoding='latin1')
    else:
        dic = pickle.load(open(input_pkl, 'rb'))

    logr, xiplus, ximinus, xicross, xiz2 = vcorr(dic['u'], dic['v'],
                                                 dic['du']*mas, dic['dv']*mas)
    xib = xiB(logr, xiplus, ximinus)
    xie = xiplus - xib

    plt.figure(figsize=(10,6))
    plt.subplots_adjust(bottom=0.13, top=0.93, left=0.12, right=0.99)
    plt.scatter(np.exp(logr) * 60, xie, c='b', label='E-mode')
    plt.scatter(np.exp(logr) * 60, xib, c='r', label='B-mode')
    plt.plot(np.exp(logr) * 60, np.zeros_like(logr), 'k--', zorder=0)
    plt.ylim(YLIM[0], YLIM[1])
    plt.xlim(0.005 * 60, 1.5 * 60)
    plt.xscale('log')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('$\Delta \\theta$ (arcmin)', fontsize=24)
    plt.ylabel('$\\xi_{E/B}$ (mas$^2$)', fontsize=24)
    #plt.title(int(self.exp_id), fontsize=20)
    plt.title(title, fontsize=22)
    plt.legend(loc=1, fontsize=20)


if __name__ == '__main__':

    #for exp in ['13268', '53732', '137108']:
    #    plot_single_exposure('../../../hsc_outputs/v3.3/astro_VK/%s_z/input.pkl'%(exp),
    #                         subtitle3="\n(exp id: %s)"%(exp), read_python2=False,
    #                         x_arrow=1800, y_arrow=2500, scale_arrow=0.15)
    #    plt.savefig('/Users/leget/Dropbox/hsc_astro/figures/%s_z_1.png'%(exp))

    #plot_single_exposure('../../../hsc_outputs/v3.3/astro_VK/96450_z/input.pkl',
    #                     MAX=39, scale_arrow=0.5, subtitle3="\n(exp id: 96450)",
    #                     x_arrow=1600, y_arrow=2700, size_arrow=100,
    #                     x_text=1750, y_text=2650)
    #plt.savefig('/Users/leget/Dropbox/hsc_astro/figures/96450_z_b_mode_1.png')
    

    plot_eb_mode_single_visit('../../../hsc_outputs/v3.3/astro_VK/137108_z/input.pkl', 
                              mas=3600.*1e3, arcsec=3600., YLIM=[-20,70], title="(exp id: 137108)",
                              read_python2=False)
    plt.savefig('/Users/leget/Dropbox/hsc_astro/figures/137108_z_eb_mode1.pdf')

    plot_eb_mode_single_visit('../../../hsc_outputs/v3.3/astro_VK/96450_z/input.pkl',
                              mas=3600.*1e3, arcsec=3600., YLIM=[-5,300], title="(exp id: 96450)")
    plt.savefig('/Users/leget/Dropbox/hsc_astro/figures/96450_z_b_mode_eb_mode1.pdf')

    #plt.show()
    
