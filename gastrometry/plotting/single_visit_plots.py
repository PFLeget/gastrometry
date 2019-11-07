import numpy as np
import pylab as plt
import cPickle
from sklearn.model_selection import train_test_split


def plot_single_exposure(input_pkl, CMAP=plt.cm.seismic, MAX=14, 
                         SIZE=20, mas=3600.*1e3, arcsec=3600.):

    dic = cPickle.load(open(input_pkl))

    fig = plt.figure(figsize=(15.5,6.5))
    plt.subplots_adjust(wspace=0,top=0.85,right=0.99,left=0.07)

    ax1 = plt.subplot(1,3,1)
    s1 = ax1.scatter(dic['u']*arcsec, dic['v']*arcsec, 
                     c=dic['du']*mas, lw=0, s=SIZE,
                     cmap=CMAP, vmin=-MAX, vmax=MAX)
    ax1.set_ylabel('v (arcsec)',fontsize=16)
    ax1.set_xticks(np.linspace(-2000, 2000, 5))

    ax2 = plt.subplot(1,3,2)
    s2 = ax2.scatter(dic['u']*arcsec, dic['v']*arcsec, 
                     c=dic['dv']*mas, lw=0, s=SIZE,
                     cmap=CMAP, vmin=-MAX, vmax=MAX)
    ax2.set_yticks([],[])
    ax2.set_xlabel('u (arcsec)',fontsize=16)
    ax2.set_xticks(np.linspace(-2000, 2000, 5))

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
                       scale=0.2)
    indice = np.linspace(0, len(dic['u'])-1, len(dic['u'])).astype(int)
    indice_train, indice_test = train_test_split(indice, test_size=0.5, random_state=42)
    ax3.quiver(dic['u'][indice_train]*arcsec, dic['v'][indice_train]*arcsec,
               dic['du'][indice_train]*mas, dic['dv'][indice_train]*mas,**quiver_dict)
    ax3.set_yticks([],[])
    ax3.quiver([1900],[2500],[40],[0], **quiver_dict)
    ax3.text(2050,2450,"(40 mas)",fontsize=11)
    ax3.set_xticks(np.linspace(-2000, 2000, 5))
    ax3.set_title('astrometric residuals', fontsize=16)

    ax_cbar1 = fig.add_axes([0.08, 0.9, 0.29, 0.025])
    ax_cbar2 = fig.add_axes([0.385, 0.9, 0.29, 0.025])

    cb1 = plt.colorbar(s1, cax=ax_cbar1, orientation='horizontal')
    cb2 = plt.colorbar(s2, cax=ax_cbar2, orientation='horizontal')

    cb1.set_label('du (mas)', fontsize='16',labelpad=-50)
    cb2.set_label('dv (mas)', fontsize='16',labelpad=-52)
       
def plot_single_exposure_hist(input_pkl, mas=3600.*1e3, arcsec=3600.):

    dic = cPickle.load(open(input_pkl))

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(2,2,3)

    plt.scatter(dic['du']*mas, dic['dv']*mas, s=15, alpha=0.1, c='b')

    #STD_u[i] = biweight_S(self.du_test[i])
    #STD_v[i] = biweight_S(self.dv_test[i])

    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('du (mas)', fontsize=18)
    plt.ylabel('dv (mas)', fontsize=18)

    plt.subplot(2,2,1)
    plt.hist(dic['du']*mas, bins=np.linspace(-30, 30, 30), histtype='step', color='b')
    plt.xlim(-30,30)
    plt.xticks([],[])
    plt.yticks([],[])


    plt.subplot(2,2,4)
    plt.hist(dic['dv']*mas, bins=np.linspace(-30, 30, 30),
             histtype='step', color='b', orientation='horizontal')
    plt.ylim(-30,30)
    plt.xticks([],[])
    plt.yticks([],[])


if __name__ == '__main__':

    #plot_single_exposure('../tests/before_spline/137108_z/input.pkl')
    #plt.savefig('../../../Dropbox/hsc_astro/figures/137108_z.pdf')
    plot_single_exposure_hist('../tests/before_spline/137108_z/input.pkl')
    plt.show()

    
