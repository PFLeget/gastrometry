import numpy as np
import treegp
import pylab as plt

class comp_mean(object):

    def __init__(self, nights, mas=3600.*1e3, arcsec=3600.,
                 bin_spacing=30., statistics='mean'):

        self.nights = nights
        self.mas = mas
        self.arcsec = arcsec
        self.bin_spacing = bin_spacing
        self.statistics = statistics

        self.mean_u = treegp.meanify(bin_spacing=self.bin_spacing, 
                                     statistics=self.statistics)
        self.mean_v = treegp.meanify(bin_spacing=self.bin_spacing, 
                                     statistics=self.statistics)
        self.nstars = 0
        self.nvisits = 0

    def load(self):
        for night in self.nights:
            A = np.loadtxt(night)
            Filtre = (A[:,4]<-6)

            exp_id = {}
            for exp in A[:,21]:
                if exp not in exp_id:
                    exp_id.update({exp:None})
            self.nvisits += len(exp_id.keys())
            self.nstars += np.sum(Filtre)
            print night
            print "total visit: ", self.nvisits
            print "total object: ", self.nstars
            print ""

            coords = np.array([A[:,8][Filtre] * self.arcsec, A[:,9][Filtre] * self.arcsec]).T
            du = A[:,10][Filtre] * self.mas
            dv = A[:,11][Filtre] * self.mas

            self.mean_u.add_field(coords, du)
            self.mean_v.add_field(coords, dv)
            
    def comp_mean(self):
        self.mean_u.meanify()
        self.mean_v.meanify()

    def plot(self, cmap=plt.cm.seismic, MAX=2.):

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_u.coords0[:,0], self.mean_u.coords0[:,1], 
                    c=self.mean_u.params0, cmap = cmap, vmin=-MAX, vmax=MAX,
                    lw=0, s=8)
        plt.colorbar()

        plt.figure(figsize=(12,10))
        plt.scatter(self.mean_v.coords0[:,0], self.mean_v.coords0[:,1], 
                    c=self.mean_v.params0, cmap = cmap, vmin=-MAX,vmax=MAX,
                    lw=0, s=8)
        plt.colorbar()

    def save_mean(self, directory='', name_outputs=['mean_gp_du.fits','mean_gp_dv.fits']):
        self.mean_u.save_results(directory=directory, name_output=name_outputs[0])
        self.mean_v.save_results(directory=directory, name_output=name_outputs[1])

if __name__ == "__main__":

    #nights = ['../../Downloads/residuals4pfl/57402-z/res-meas.list',
    #          '../../Downloads/residuals4pfl/57755-z/res-meas.list',
    #          '../../Downloads/residuals4pfl/58131-z/res-meas.list']

    import glob
    nights = glob.glob('/sps/snls13/HSC/prod.2019-04/dbimage_35UN7JY/fitastrom_ULHBNSI/data/*z/res-meas.list')

    cm = comp_mean(nights, mas=3600.*1e3, arcsec=3600.,
                   bin_spacing=10., statistics='mean')
    cm.load()
    cm.comp_mean()
    cm.plot(cmap=None)
    cm.save_mean(directory='', name_outputs=['mean_gp_du_z.fits',
                                             'mean_gp_dv_z.fits'])
