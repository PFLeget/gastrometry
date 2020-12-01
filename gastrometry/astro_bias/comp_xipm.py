import numpy as np
import pylab as plt
import pyccl as ccl
import copy
from astropy.io import fits
import warnings
import shear_subaru
import os
import pickle
from scipy.stats import binned_statistic
path = os.path.dirname(shear_subaru.__file__)

class comp_shear_cl(object):

    def __init__(self, Omega_ch2=0.1, Omega_bh2=0.023, AS=None, S8=None,
                 Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, alpha=0.45,
                 matter_power_spectrum = 'halofit', A0=0, eta=0,
                 delta_m = 0., delta_z=[0., 0., 0., 0.],
                 ell = np.arange(20, 3000)):

        self._matter_power_spectrum = matter_power_spectrum

        self.update_mbias(delta_m=delta_m)
        self.update_IA(A0=A0, eta=eta)
        self.update_cosmology(Omega_ch2=Omega_ch2, Omega_bh2=Omega_bh2,
                              AS=AS, S8=S8, alpha=alpha,
                              Omega_nu_h2=Omega_nu_h2,
                              H0=H0, ns=ns, w0=w0)
        self.load_photo_z()
        self.update_redshift_bias(delta_z)
        self.ell = ell

    def update_mbias(self, delta_m=0):
        self.delta_m = delta_m

    def update_IA(self, A0=1, eta=1):
        self.A0 = A0
        self.eta = eta

    def update_redshift_bias(self, delta_z=[0., 0., 0., 0.]):
        self.delta_z = delta_z
        dz = copy.deepcopy(np.array(delta_z) / 100.)
        self.redshifts_bias = []
        for i in range(len(dz)):
            new_z = self.redshifts[i] - dz[i]
            self.redshifts_bias.append(copy.deepcopy(self.redshifts[i]) - dz[i])

    def update_cosmology(self, Omega_ch2=0.1, Omega_bh2=0.023, AS=None, S8=None,
                         Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, alpha=0.45):

        self.Omega_ch2 = Omega_ch2
        self.Omega_bh2 = Omega_bh2
        self.H0 = H0
        self._h = self.H0 / 100.

        self._Omega_b = self.Omega_bh2 / self._h**2
        self._Omega_c = self.Omega_ch2 / self._h**2
        self._Omega_m = self._Omega_b + self._Omega_c
        self.Omega_nu_h2 = Omega_nu_h2
        self._m_nu = (self.Omega_nu_h2 / self._h**2) * 93.14

        if S8 is None and AS is None:
            raise ValueError('S8 or AS should be given')
        if S8 is not None and AS is not None:
            raise ValueError('Just S8 or AS should be given')

        if S8 is not None:
            self.AS = None
            self._A_s = None
            self._sigma8 = S8 * (1./ (self._Omega_m/0.3)**alpha)
        if AS is not None:
            self.AS = AS
            self._A_s = np.exp(self.AS) * 1e-10
            self._sigma8 = None

        self.n_s = ns
        self.w0 = w0

        self.cosmology = ccl.Cosmology(Omega_c=self._Omega_c, Omega_b=self._Omega_b,
                                       h=self._h, n_s=self.n_s, sigma8=self._sigma8, A_s=self._A_s,
                                       w0=self.w0, m_nu=self._m_nu,
                                       matter_power_spectrum=self._matter_power_spectrum)

    def load_photo_z(self, survey = 'DES', plot=False):

        # NEED TO UPDATE IN ORDER TO BE LIKE DES
        # OR SUBARU need to think about that
        self.redshifts = []
        self.nz = []
        if survey == 'DES':
            filename = 'y1_redshift_distributions_v1.fits'
            redshifts_data = fits.open(filename, memmap=True)
            for i in range(4):
                self.redshifts.append(redshifts_data[1].data['Z_MID'])
                self.nz.append(redshifts_data[1].data['BIN%i'%(i+1)]*redshifts_data[1].header['NGAL_%i'%(i+1)])
            self.redshifts = np.array(self.redshifts)
            self.nz = np.array(self.nz)
        else:
            for i in range(4):
                file_bin = np.loadtxt(os.path.join(path, 'data/photo-z/bin%i'%(i+1))+key+'.dat', comments='#')
                self.redshifts.append(file_bin[:,0])
                self.nz.append(file_bin[:,1])

            self.redshifts = np.array(self.redshifts)
            self.nz = np.array(self.nz)

        if plot:
            C = ['k', 'b', 'y', 'r']
            plt.figure(figsize=(12,4))
            for i in range(4):
                plt.plot(self.redshifts[i], self.nz[i], C[i], lw=3)

            plt.plot([0,2.6], [0,0], 'k--')
            plt.xlim(0,2.6)
            #plt.ylim(-0.1,3.8)
            plt.show()

    def intrinsic_al(self, redshift, A0=1, eta=1, z0=0.62):
        AI = A0 * ((1+redshift) / (1+z0))**eta
        return AI

    def multiplicatif_bias(self, delta_m=0.01*100.):

        dm = delta_m / 100.
        return (1+dm)**2

    def comp_xipm(self, theta):
        self.wl_bin_shear = []
        nell = len(self.ell)
        self.theta = theta
        self.Cl = np.zeros(len(self.ell), dtype={'names':('11', '12', '13', '14',
                                                          '22', '23', '24',
                                                          '33', '34',
                                                          '44'),
                                                 'formats':('f8', 'f8', 'f8', 'f8',
                                                            'f8', 'f8','f8',
                                                            'f8', 'f8',
                                                            'f8')})

        self.xip = np.zeros(len(theta), dtype={'names':('11', '12', '13', '14',
                                                        '22', '23', '24',
                                                        '33', '34',
                                                        '44'),
                                               'formats':('f8', 'f8', 'f8', 'f8',
                                                          'f8', 'f8','f8',
                                                          'f8', 'f8',
                                                          'f8')})

        self.xim = np.zeros(len(theta), dtype={'names':('11', '12', '13', '14',
                                                        '22', '23', '24',
                                                        '33', '34',
                                                        '44'),
                                               'formats':('f8', 'f8', 'f8', 'f8',
                                                          'f8', 'f8','f8',
                                                          'f8', 'f8',
                                                          'f8')})
        for i in range(4):
            filtre = (self.redshifts_bias[i] > 0)
            # je ne sais plus si c'est important que le redshift de l ai soit le meme que
            # celui de la distribution des galaxies 
            #AI = self.intrinsic_al(self.redshifts[i][filtre], A0=self.A0, eta=self.eta, z0=0.62)
            AI = self.intrinsic_al(self.redshifts[i], A0=self.A0, eta=self.eta, z0=0.62)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.wl_bin_shear.append(ccl.WeakLensingTracer(self.cosmology,
                                                               dndz=(self.redshifts_bias[i][filtre], self.nz[i][filtre]),
                                                               has_shear=True,
                                                               ia_bias=(self.redshifts[i], AI)))
                                                               #ia_bias=(self.redshifts[i][filtre], AI)))

        for i in range(4):
            for j in range(4):
                if j>=i:
                    key = "%i%i"%((i+1, j+1))
                    cl = ccl.angular_cl(self.cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], self.ell)
                    m_ij = self.multiplicatif_bias(delta_m=self.delta_m)
                    cl *= m_ij
                    self.Cl[key] = cl
                    self.xip[key] = ccl.correlation(self.cosmology,
                                                    self.ell,
                                                    self.Cl[key],
                                                    theta,
                                                    type='GG+',
                                                    method='fftlog')
                    self.xim[key] = ccl.correlation(self.cosmology,
                                                    self.ell,
                                                    self.Cl[key],
                                                    theta,
                                                    type='GG-',
                                                    method='fftlog')

    def plots(self):

        # plot xiplus, ximoins

        xip = self.xip
        xim = self.xim

        plt.figure(figsize=(10,8))
        plt.subplots_adjust(wspace=0.0, hspace=0., top=0.99)

        XLIM = [2, 300]

        SUBPLOTS_plus = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13]
        KEY_plus = ['14', '13', '12', '11',
                    '24', '23', '22',
                    '34', '33',
                    '44']


        #YLIM = [[-0.49,2.5], [-0.49,2.5], [-0.49,2.5], [-0.49,2.5],
        #        [-0.49, 2.8], [-0.49, 2.8], [-0.49, 2.8],
        #        [-0.49, 4.9], [-0.49, 4.9],
        #        [-0.49, 5.4]]

        YLIM = [[-0.49, 7], [-0.49, 7], [-0.49, 7], [-0.49, 7],
                [-0.49, 7], [-0.49, 7], [-0.49, 7],
                [-0.49, 7], [-0.49, 7],
                [-0.49, 7]]

        SUBPLOTS_minus = [24, 23, 22, 21, 20, 19, 18, 16, 15, 12]

        KEY_minus = ['14', '13', '12', '11',
                     '24', '23', '22',
                     '34', '33',
                     '44']

        LEGEND = ['1,4', '1,3', '1,2', '1,1', '2,4', 
                  '2,3', '2,2', '3,4', '3,3', '4,4']

        for i in range(len(SUBPLOTS_plus)):
            key = KEY_plus[i]
            plt.subplot(6, 4, SUBPLOTS_plus[i])
            plt.plot(self.theta*60., self.theta * 60. * xip[key] * 1e4, 'b', label=LEGEND[i])

            shear_bias_xipm = 1e-5
            shear_bias_xipm_des = 1e-5 / 25.
            plt.plot(self.theta*60.,
                     np.ones_like(self.theta) * 60. * shear_bias_xipm * 1e4, 'k--')
            plt.plot(self.theta*60.,
                     np.ones_like(self.theta) * 60. * shear_bias_xipm_des * 1e4, 'r--')
            
            plt.plot(XLIM, np.zeros(2), 'k')
            plt.xlim(XLIM[0], XLIM[1])
            plt.ylim(YLIM[i][0], YLIM[i][1])
            plt.xscale('log')
            plt.xticks([],[])
            if SUBPLOTS_plus[i] not in [1, 5, 9, 13]:
                plt.yticks([],[])
            else:
                plt.ylabel('$\\theta \\xi_{+}$ / $10^{-4}$', fontsize=10)

            leg = plt.legend(handlelength=0, handletextpad=0, 
                         loc=2, fancybox=True, fontsize=8)
            for item in leg.legendHandles:
                item.set_visible(False)

            key = KEY_minus[i]
            ax = plt.subplot(6, 4, SUBPLOTS_minus[i])
            plt.plot(self.theta*60., self.theta * 60. * xim[key] * 1e4, 'r', label=LEGEND[i])
            plt.plot(XLIM, np.zeros(2), 'k')
            plt.xlim(XLIM[0], XLIM[1])
            plt.ylim(YLIM[i][0], YLIM[i][1])
            plt.xscale('log')

            if SUBPLOTS_minus[i] in [21, 22, 23, 24]:
                plt.xlabel('$\\theta$ (arcmin)', fontsize=12)
            else:
                plt.xticks([],[])

            plt.yticks([],[])

            leg = plt.legend(handlelength=0, handletextpad=0, 
                         loc=2, fancybox=True, fontsize=8)
            for item in leg.legendHandles:
                item.set_visible(False)

            ax2 = ax.twinx()
            ax2.set_ylim(YLIM[i][0], YLIM[i][1])

            if SUBPLOTS_minus[i] not in [12, 16, 20, 24]:
                ax2.set_yticks([],[])
            else:
                ax2.set_ylabel('$\\theta \\xi_{-}$ / $10^{-4}$', fontsize=10)



def plots_hsc_paper(theta, xip, xip_s8p, xip_s8m):
    
    # plot xiplus, ximoins
    shear_color = 'k'
    atmo_color = 'b'
    non_linear_color = 'r'

    xip_dic = pickle.load(open('xip_30_sec.pkl', 'rb'))
    #print(xip_dic['xip_atm'])
    g = 0.2 * 1e3
    theta_atm = xip_dic['r'] * 60.
    #xip_atmo = (1./8.) * (xip_dic['xip_atm']**2 / g**4)
    xip_atmo = (1./8.) * (xip_dic['xip_atm_squared'] / g**4)
    #plt.plot(theta_atm, xip_atmo * theta_atm * 1e4, )
    #plt.show()

    #francis = (xip_dic['xip_atm']  / (200 * 200) )**2 /16

    #print(francis - xip_atmo)
    
    #plt.plot(theta_atm, theta_atm * francis * 1e4)
    #plt.xscale('log')
    #plt.show() 

    plt.figure(figsize=(12,7))
    plt.subplots_adjust(wspace=0.0, hspace=0., top=0.99, left=0.05, right=0.99)

    XLIM = [2, 300]
    
    SUBPLOTS_plus = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13]
    KEY_plus = ['14', '13', '12', '11',
                '24', '23', '22',
                '34', '33',
                '44']
    
    HMAX = 4.5
    HMIN = -0.1
    #YLIM = [[-0.49, 2.5], [-0.49, 2.5], [-0.49, 2.5], [-0.49, 2.5],
    #        [-0.49, 2.8], [-0.49, 2.8], [-0.49, 2.8],
    #        [-0.49, 4.9], [-0.49, 4.9],
    #        [-0.49, 5.4]]
    YLIM = [[HMIN, HMAX], [HMIN, HMAX], [HMIN, HMAX], [HMIN, HMAX],
            [HMIN, HMAX], [HMIN, HMAX], [HMIN, HMAX],
            [HMIN, HMAX], [HMIN, HMAX],
            [HMIN, HMAX]]

    LEGEND = ['1,4', '1,3', '1,2', '1,1', '2,4', 
              '2,3', '2,2', '3,4', '3,3', '4,4']

    for i in range(len(SUBPLOTS_plus)):
        key = KEY_plus[i]
        plt.subplot(4, 4, SUBPLOTS_plus[i])
        if i == 1:
            LABEL = 'Contribution from astrometric residuals (simulation LSST like)'
        else:
            LABEL = None
        plt.plot(theta_atm, xip_atmo * theta_atm * 1e4, atmo_color, lw=2, label=LABEL)
        plt.text(165, 3.8, LEGEND[i], color='black', 
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'), fontsize=10)
        if i == 1:
            LABEL = 'Cosmic shear signal with DES Y1 cosmology'
        else:
            LABEL = None
        plt.plot(theta*60., theta * 60. * xip[key] * 1e4, shear_color, label=LABEL)
        if i == 1:
            LABEL = '$\pm$ 1$\sigma$ on $S_8$ from cosmic shear DES Y1 analysis'
        else:
            LABEL = None
        plt.fill_between(theta*60, theta * 60. * xip_s8m[key] * 1e4,
                         theta * 60. * xip_s8p[key] * 1e4, alpha=0.5, color=shear_color, label=LABEL)
                    
        
        #shear_bias_xipm = 1e-5
        #shear_bias_xipm_des = 1e-5 / 25.
        #plt.plot(theta*60.,
        #         theta * 60. * shear_bias_xipm * 1e4, atmo_color+'--')
        #plt.plot(theta*60.,
        #         theta * 60. * shear_bias_xipm_des * 1e4, atmo_color+'-.')

        # non linear scale
        dic = pickle.load(open('xip_xim_real.pkl', 'rb'))
        ang = dic['xip'][(dic['xip']['BIN1'] == int(key[0])) & (dic['xip']['BIN2'] == int(key[1]))]['ANG']
        mask = dic['xip'][(dic['xip']['BIN1'] == int(key[0])) & (dic['xip']['BIN2'] == int(key[1]))]['USED']
        nlscale = ang[~mask]
        if i == 1:
            LABEL = 'Scale removed from DES Y1 cosmic shear analysis'
        else:
            LABEL = None
        plt.fill_betweenx(YLIM[i], [0,0], [nlscale[-1], nlscale[-1]], color=non_linear_color, alpha=0.2, label=LABEL)
            

        #atmo scale:
        #plt.fill_betweenx(YLIM[i], [nlscale[-1],nlscale[-1]], [50, 50], color=atmo_color, alpha=0.2)
        
        if HMIN<=0:
            plt.plot(XLIM, np.zeros(2), 'k', alpha=0.5)
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(YLIM[i][0], YLIM[i][1])
        plt.xscale('log')
        if SUBPLOTS_plus[i] not in [4, 7, 10, 13]:
            plt.xticks([],[])
        else:
            plt.xlabel('$\\theta$ (arcmin)', fontsize=12)
        if SUBPLOTS_plus[i] not in [1, 5, 9, 13]:
            plt.yticks([],[])
        else:
            plt.ylabel('$\\theta \\xi_{+}$ / $10^{-4}$', fontsize=12)

        #leg = plt.legend(handlelength=0, handletextpad=0,
        #                 loc=1, fancybox=True, fontsize=8)
        #for item in leg.legendHandles:
        #    item.set_visible(False)
        if i == 1:
            plt.legend(bbox_to_anchor=(0.9, -2.3), loc=2, borderaxespad=0.,fontsize=12)
       


if __name__ == '__main__':

    if False:
        csc = comp_shear_cl(Omega_ch2=0.1195, Omega_bh2=0.0267, AS=None, S8=0.782,
                            Omega_nu_h2=4.5*1e-3, H0=75., ns=0.99, w0=-1, alpha=0.5,
                            matter_power_spectrum = 'halofit', A0=1.0, eta=2.8,
                            delta_m = 0., delta_z=[0., 0., 0., 0.],
                            ell=np.arange(20, 6000))
    
        theta = np.linspace(1./60, 5., 500)
        csc.comp_xipm(theta)

        cscp = comp_shear_cl(Omega_ch2=0.1195, Omega_bh2=0.0267, AS=None, S8=0.782 + 0.027,
                             Omega_nu_h2=4.5*1e-3, H0=75., ns=0.99, w0=-1, alpha=0.5,
                             matter_power_spectrum = 'halofit', A0=1.0, eta=2.8,
                             delta_m = 0., delta_z=[0., 0., 0., 0.],
                             ell=np.arange(20, 6000))
        cscp.comp_xipm(theta)

        cscm = comp_shear_cl(Omega_ch2=0.1195, Omega_bh2=0.0267, AS=None, S8=0.782 - 0.027,
                             Omega_nu_h2=4.5*1e-3, H0=75., ns=0.99, w0=-1, alpha=0.5,
                             matter_power_spectrum = 'halofit', A0=1.0, eta=2.8,
                             delta_m = 0., delta_z=[0., 0., 0., 0.],
                             ell=np.arange(20, 6000))
        cscm.comp_xipm(theta)

        dic = {'theta':theta,
               'xip':csc.xip,
               'xim':csc.xim,
               'xip_s8p':cscp.xip,
               'xip_s8m':cscm.xip}
        f = open('out_cos_cs.pkl','wb')
        pickle.dump(dic, f)
        f.close
    
    dic = pickle.load(open('out_cos_cs.pkl', 'rb'))
    plots_hsc_paper(dic['theta'], dic['xip'], dic['xip_s8p'], dic['xip_s8m'])
    plt.savefig('../../../../../Dropbox/hsc_astro/figures/xipm_effects.pdf')
    plt.show()
