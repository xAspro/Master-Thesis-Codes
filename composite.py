# Checked once. Runs without error. No implementation of this file in this file.
print("In composite.py")

import numpy as np
import scipy.optimize as op
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import corner
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}
from numpy.polynomial import Chebyshev as T
from numpy.polynomial.polynomial import polyval

def getselfn(selfile):
    """
    Reads selection map.

    Parameters
    ----------
    selfile : str
        File path to the selection map.

    Returns
    -------
    z : numpy.ndarray
        Array of redshift values.

    mag : numpy.ndarray
        Array of magnitude values.

    p : numpy.ndarray
        Array of selection probability values.

    dz : numpy.ndarray
        Array of redshift bin widths.

    dm : numpy.ndarray
        Array of magnitude bin widths.
    """

    with open(selfile,'r') as f: 
        z, mag, p, dz, dm = np.loadtxt(f, usecols=(1,2,3,4,5), unpack=True)
    
    return z, mag, p, dz, dm 

def getqlums(lumfile):

    """Read quasar luminosities."""

    z, mag, p, area, sample_id = np.loadtxt(lumfile, usecols=(1,2,3,4,5),
                                            unpack=True)

    sample_id = np.atleast_1d(sample_id)

    select = None 

    if sample_id[0] == 13:
        # Restrict Richards (SDSS) sample.
        select = (((z>=0.6) & (z<0.8) & (mag<=-23.1)) | 
                  ((z>=0.8) & (z<1.0) & (mag<=-23.7)) |
                  ((z>=1.0) & (z<1.2)) |
                  ((z>=1.2) & (z<1.4) & (mag<=-24.3)) |
                  ((z>=1.4) & (z<1.6)) |
                  ((z>=1.6) & (z<1.8) & (mag<=-24.9)) |
                  ((z>=1.8) & (z<2.2)) |
                  ((z>=3.5) & (z<4.7) & (mag<=-26.1)))

    if sample_id[0] == 15:
        # Restrict Croom (2SLAQ) sample.
        select = (((z>=0.6) & (z<0.8) & (mag<=-20.7)) | 
                  ((z>=0.8) & (z<1.2) & (mag<=-21.9)) |
                  ((z>=1.2) & (z<1.8) & (mag<=-22.5)) |
                  ((z>=1.8) & (z<2.2) & (mag<=-23.1)))
        
    if sample_id[0] == 1:
        # Restrict BOSS sample.
        select = ((z < 2.2) | (z >= 2.8))
    
    if sample_id[0] == 8:
        # Restrict McGreer's samples to faint quasars to avoid
        # overlap with Yang.
        select = (mag>-26.73)

    z = z[select]
    mag = mag[select]
    p = p[select]

    return z, mag, p 

def volume(z, area, cosmo=cosmo):
    """
    Calculates the comoving volume element.
    Equation 5 in Kulkarni 2019.

    Parameters
    ----------
    z : numpy.ndarray
        Array of redshift values.

    area : float
        Area of survey.

    cosmo : dict, optional
        Dictionary containing cosmological parameters. Defaults to cosmo.

    Returns
    -------
    volperstr : numpy.ndarray
        Array of comoving volume elements.
    """

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1 

class selmap:
    """
    A class to represent a selection map object.

    Attributes
    ----------
    z : numpy.ndarray
        Array of redshift values.

    m : numpy.ndarray
        Array of magnitude values.

    p : numpy.ndarray
        Array of selection probability values.

    dz : numpy.ndarray
        Array of redshift bin widths.

    dm : numpy.ndarray
        Array of magnitude bin widths.

    area : float
        Area of survey for the selection map.

    sid : int
        Identifier or index for the selection map.

    volume : numpy.ndarray
        Array of comoving volume elements.
    
    """

    def __init__(self, selection_map_file, area, sample_id):
        """
        Constructor - Initialises the selection map object with 
                      selection map file, area, and sample id.

        Parameters
        ----------
        - selection_map_file : str
            File path to the selection map.
        - area : float
            Area of survey for the selection map.
        - sample_id : int
            Identifier or index for the selection map.

        Returns
        -------
        None

        """

        self.z, self.m, self.p, self.dz, self.dm  = getselfn(selection_map_file)

        # self.dz = dz
        # self.dm = dm 
        # print( 'dz={:.3f}, dm={:.3f}, sample_id={:d}'.format(dz, dm, sample_id))
        # print( 'sample_id={:d}'.format(sample_id))
        # print( 'size of z: ', self.z.size, '\n')

        self.sid = sample_id 

        ### Removing certain redshift ranges for certain samples.
        ### Explained in Kulkarni 2019.

        if sample_id == 7:
            # Giallongo's sample needs special treatment due to
            # non-uniform selection map grid.  
            self.dz = np.array([0.5, 0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 1.5, 1.5, 1.5])
            self.dm = np.array([1.0, 1.0, 1.0, 2.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0])

            # Correct Giallongo's p values to match published LF.  See
            # comments in giallongo15_sel_correction.dat.
            with open('Data_new/giallongo15_sel_correction.dat', 'r') as f: 
                corr = np.loadtxt(f, usecols=(4,), unpack=True)
            self.p = self.p/corr

        if sample_id == 1:
            # Restrict BOSS sample 
            select = ((self.z<2.2) | (self.z>=2.8))
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            self.dz = self.dz[select]
            self.dm = self.dm[select]
            
        if sample_id == 13:
            # Restrict Richards sample
            select = (((self.z>=0.6) & (self.z<0.8) & (self.m<=-23.1)) | 
                      ((self.z>=0.8) & (self.z<1.0) & (self.m<=-23.7)) |
                      ((self.z>=1.0) & (self.z<1.2)) |
                      ((self.z>=1.2) & (self.z<1.4) & (self.m<=-24.3)) |
                      ((self.z>=1.4) & (self.z<1.6)) |
                      ((self.z>=1.6) & (self.z<1.8) & (self.m<=-24.9)) |
                      ((self.z>=1.8) & (self.z<2.2)) |
                      ((self.z>=3.5) & (self.z<4.7) & (self.m<=-26.1)))

            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            self.dz = self.dz[select]
            self.dm = self.dm[select]

        if sample_id == 15:
            # Restrict Croom sample
            select = (((self.z>=0.6) & (self.z<0.8) & (self.m<=-20.7)) | 
                      ((self.z>=0.8) & (self.z<1.2) & (self.m<=-21.9)) |
                      ((self.z>=1.2) & (self.z<1.8) & (self.m<=-22.5)) |
                      ((self.z>=1.8) & (self.z<2.2) & (self.m<=-23.1)))

            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            self.dz = self.dz[select]
            self.dm = self.dm[select]
            
        if sample_id == 8:
            # Restrict McGreer's samples to faint quasars to avoid
            # overlap with Yang.
            select = (self.m>-26.73)
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            self.dz = self.dz[select]
            self.dm = self.dm[select]

        if self.z.size == 0:
            return 

        self.area = area
        self.volume = volume(self.z, self.area) # cMpc^3 dz^-1 

        return

    def nqso(self, lumfn, theta):
        """
        Calculates the total number of quasar in the survey volume, predicted by 
        the QLF model, while accounting for the selection probability.

        Parameters
        ----------
        lumfn : lf
            Luminosity function object.

        theta : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (float) Total number of quasars in the survey volume.
        """

        phi = 10.0**lumfn.log10phi(theta, self.m, self.z)
        tot = phi*self.p*self.volume*self.dz*self.dm 
        return np.sum(tot) 
            
class lf:
    """5-parameter model for beta; for polynomial model see lf_polyb below.

    """


    """
    
    FILL THIS!!!
    """

    def __init__(self, quasar_files=None, selection_maps=None, pnum=np.array([2,2,1,1])):
        """
        Constructor - Initialises the luminosity function object with quaasar data, 
                      selection maps, and number of parameters for the double power law.

        Parameters
        ----------
        quasar_files : (list) str
            List of file paths containing sample data.

        selection_maps : (list) list of tuples
            Each tuple contains three elements:
            - A string representing the file path to the selection map.
            - A float representing the area of survey for the selection map.
            - An integer representing the identifier or index for the selection map.

        pnum : int or numpy.ndarray, optional
            Specifies the number of parameters for the double power law.
            - If an integer, all groups of parameters have the same size.
            - If an array, each element specifies the size of a group of parameters.
            Defaults to np.array([2, 2, 1, 1]).
        Returns
        -------
        None
        """

        self.pnum = pnum 
        
        for datafile in quasar_files:
            z, m, p = getqlums(datafile)
            try:
                self.z=np.append(self.z,z)
                # print('\n\n\n\t\tsize of z: ', z.size)
                # print('\n\n\n\t\tshape of z: ', z.shape)
                self.M1450=np.append(self.M1450,m)
                self.p=np.append(self.p,p)
            except(AttributeError):
                self.z=z
                # print('\n\n\n\t\t\tsize of single z: ', z.size)
                # print('\n\n\n\t\t\tshape of single z: ', z.shape)
                self.M1450=m
                self.p=p

        self.maps = [selmap(*x) for x in selection_maps]

        return

    def atz(self, z, p):
        """
        Creates a Chebyshev polynomial for redshift evolution of 
        QLF parameters (p), and evaluates it at '1+z'.
        Equation 16 of Kulkarni 2019.

        Parameters
        ----------
        z : numpy.ndarray
            Array of redshift values.

        p : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (numpy.ndarray) Array of QLF parameters.
        """
        
        return T(p)(1+z)
    
    def atz_beta(self, z, p):
        """
        Creates a Double Power Law for redshift evolution of beta,
        and calculated the value at the given data points.
        Equation 17 of Kulkarni 2019.

        Parameters
        ----------
        z : numpy.ndarray
            Array of redshift values.

        p : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (numpy.ndarray) Array of parameters for beta.
        """

        h, f0, z0, a, b = p 
        zeta = np.log10((1.0+z)/(1.0+z0))
        return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))
    
    def atz_beta2(self, z, p):

        """
        Creates a Polynomial Model for redshift evolution of beta.

        Parameters
        ----------
        z : numpy.ndarray
            Array of redshift values.

        p : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (numpy.ndarray) Array of parameters for beta.
        """

        a, b, c, d, e, *_ = p 
        return a + b*z + c*z**2 + d*z**3 + e*z**4

    def getparams(self, theta):
        """
            Splits the parameter array 'theta' into individual LF parameters,
            based on 'self.pnum'.

            Parameters
            ----------
            theta : numpy.ndarray
                Array of parameters.

            Returns
            -------
            (numpy.ndarray) Array of LF parameters.
        """

        if isinstance(self.pnum, int):
            # Case 1: `self.pnum` is a single integer.
            # Each parameter group has the same number of parameters (`self.pnum`).
            splitlocs = self.pnum*np.array([1,2,3])
        else:
            # Case 2: `self.pnum` is an array or list.
            # Each parameter group has a different number of parameters.
            # The number of parameters for each group is given by `self.pnum[i]`.
            splitlocs = np.cumsum(self.pnum)

        return np.split(theta,splitlocs)

    def log10phi(self, theta, mag, z):
        """
        Calculates the log10 of the quasar luminosity function 
        for the given parameters, and on the given data points.

        Parameters
        ----------
        theta : numpy.ndarray
            Array of parameters.

        mag : numpy.ndarray
            Array of magnitude values.

        z : numpy.ndarray
            Array of redshift values.

        Returns
        -------
        (numpy.ndarray) log10 of the quasar luminosity function 
                        at the given data points.
        """

        # print("self: ", self)
        # print("\ntheta: ", theta,"\n")

        params = self.getparams(theta)
        # print("\nparams: ", params,"\n")

        log10phi_star = self.atz(z, params[0])
        # print("\nlog10phi_star: ", log10phi_star,"\n")
        # print('\nlen(log10phi_star): ', len(log10phi_star),'\n')
        M_star = self.atz(z, params[1])
        # print("\nM_star: ", M_star,"\n")
        # print('\nlen(M_star): ', len(M_star),'\n')
        alpha = self.atz(z, params[2])
        # print("\nalpha: ", alpha,"\n")
        # print('\nlen(alpha): ', len(alpha),'\n')
        # print('\nz: ', z,'\n')
        # print('\nlen(z): ', len(z),'\n')
        # print('\nparams[3]: ', params[3],'\n')
        # print('\nlen(params[3]): ', len(params[3]),'\n')
        # beta = self.atz(z, params[3])
        beta = self.atz_beta(z, params[3])
        # beta = self.atz_beta2(z, params[3])
        # print("\nbeta: ", beta,"\n")
        # print('\nlen(beta): ', len(beta),'\n')
        
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        # print("\nphi: ", phi,"\n")
        return np.log10(phi)

    def lfnorm(self, theta):
        """
        Calculates the total number of quasar in each survey volume, predicted by 
        the QLF model, while accounting for their selection probability.

        Parameters
        ----------
        theta : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (float) Total number of quasars in the survey volume.
        """

        ns = np.array([x.nqso(self, theta) for x in self.maps])
        return np.sum(ns) 
        
    def neglnlike(self, theta):
        """
        Negative log likelihood function.
        Equation 10 in Kulkarni 2019.

        Parameters
        ----------
        theta : numpy.ndarray
            Array of parameters.

        Returns
        -------
        (float) Negative log likelihood.
        """

        logphi = self.log10phi(theta, self.M1450, self.z) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 
        # print('\n\nlogphi: ', logphi)
        # print('size of logphi: ', logphi.size)
        # print('shape of logphi: ', logphi.shape)

        # print('\n\nsize of self.z: ', self.z.size)
        # print('shape of self.z: ', self.z.shape)


        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        """
        Finds the best-fit parameters that minimize the negative log
        likelihood function - self.neglnlike.

        Parameters
        ----------
        guess : numpy.ndarray
            Initial guess for the parameters.

        method : str, optional
            Optimization method. Defaults to 'Nelder-Mead'.

        Returns
        -------
        result : scipy.optimize.optimize.OptimizeResult
            Result of the optimization.
        """

        result = op.minimize(self.neglnlike,
                             guess,
                             method=method, options={'maxfev': 20000,
                                                     'maxiter': 20000,
                                                     'disp': True})

        if not result.success:
            print( 'Likelihood optimisation did not converge.')

        self.bf = result 
        return result
    
    def create_param_range(self):
        print("In composite.py class-lf create_param_range")

        half = self.bf.x/2.0
        double = 2.0*self.bf.x
        self.prior_min_values = np.where(half < double, half, double) 
        self.prior_max_values = np.where(half > double, half, double)
        assert(np.all(self.prior_min_values < self.prior_max_values))

        return

    def lnprior(self, theta):
        # print("In composite.py class-lf lnprior")
        """
        Set up uniform priors.

        """

        params = self.getparams(theta)
        alpha = params[2]
        alpha_atz6 = self.atz(6.0, alpha) 
        
        if (np.all(theta < self.prior_max_values) and
            np.all(theta > self.prior_min_values) and
            alpha_atz6 < -4.0):
            return 0.0 

        return -np.inf

    def lnprob(self, theta):
        # print("In composite.py class-lf lnprob")

        lp = self.lnprior(theta)
        
        if not np.isfinite(lp):
            return -np.inf

        return lp - self.neglnlike(theta)

    def run_mcmc(self):
        print("In composite.py class-lf run_mcmc")
        """
        Run emcee.

        """
        self.ndim, self.nwalkers = self.bf.x.size, 100
        self.mcmc_start = self.bf.x 
        pos = [self.mcmc_start + 1e-4*np.random.randn(self.ndim) for i
               in range(self.nwalkers)]
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                             self.lnprob)

        self.sampler.run_mcmc(pos, 1000)
        self.samples = self.sampler.chain[:, 500:, :].reshape((-1, self.ndim))

        return

    def corner_plot(self, labels=None, dirname=''):
        print("In composite.py class-lf corner_plot")

        mpl.rcParams['font.size'] = '14'
        f = corner.corner(self.samples, labels=labels, truths=self.bf.x)
        plotfile = dirname+'triangle.png'
        f.savefig(plotfile)
        mpl.rcParams['font.size'] = '22'

        return

    def plot_chains(self, fig, param, ylabel):
        print("In composite.py class-lf plot_chains")
        ax = fig.add_subplot(self.bf.x.size, 1, param+1)
        for i in range(self.nwalkers): 
            ax.plot(self.sampler.chain[i,:,param], c='k', alpha=0.1)
        ax.axhline(self.bf.x[param], c='#CC9966', dashes=[7,2], lw=2) 
        ax.set_ylabel(ylabel)
        if param+1 != self.bf.x.size:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('step')
            
        return 

    def chains(self, labels=None, dirname=''):
        print("In composite.py class-lf chains")

        mpl.rcParams['font.size'] = '10'
        nparams = self.bf.x.size
        plot_number = 0 

        fig = plt.figure(figsize=(12, 2*nparams), dpi=100)
        for i in range(nparams): 
            self.plot_chains(fig, i, ylabel=labels[i])
            
        plotfile = dirname+'chains.pdf' 
        plt.savefig(plotfile,bbox_inches='tight')
        plt.close('all')
        mpl.rcParams['font.size'] = '22'
        
        return

class lf_polyb:
    print("In composite.py class-lf_polyb")

    """Same as lf above, except this has polynomial model for beta. 

    """
    

    def __init__(self, quasar_files=None, selection_maps=None, pnum=np.array([2,2,1,1])):
        print("In composite.py class-lf_polyb __init__")

        self.pnum = pnum 
        
        for datafile in quasar_files:
            z, m, p = getqlums(datafile)
            try:
                self.z=np.append(self.z,z)
                self.M1450=np.append(self.M1450,m)
                self.p=np.append(self.p,p)
            except(AttributeError):
                self.z=z
                self.M1450=m
                self.p=p

        self.maps = [selmap(*x) for x in selection_maps]

        return

    def atz(self, z, p):
        # print("In composite.py class-lf_polyb atz")

        """Redshift evolution of QLF parameters."""
        
        return T(p)(1+z)
    
    def getparams(self, theta):
        # print("In composite.py class-lf_polyb getparams")

        if isinstance(self.pnum, int):
            # Evolution of each LF parameter described by 'atz' using same
            # number 'self.pnum' of parameters.
            splitlocs = self.pnum*np.array([1,2,3])
        else:
            # Evolution of each LF parameter described by 'atz' using
            # different number 'self.pnum[i]' of parameters.
            splitlocs = np.cumsum(self.pnum)

        return np.split(theta,splitlocs)

    def log10phi(self, theta, mag, z):
        # print("In composite.py class-lf_polyb log10phi")

        params = self.getparams(theta)

        log10phi_star = self.atz(z, params[0])
        M_star = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz(z, params[3])
        
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log10(phi)

    def lfnorm(self, theta):
        print("In composite.py class-lf_polyb lfnorm")

        ns = np.array([x.nqso(self, theta) for x in self.maps])
        return np.sum(ns) 
        
    def neglnlike(self, theta):
        print("In composite.py class-lf_polyb neglnlike")

        logphi = self.log10phi(theta, self.M1450, self.z) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 

        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        print("In composite.py class-lf_polyb bestfit")
        result = op.minimize(self.neglnlike,
                             guess,
                             method=method, options={'maxfev': 20000,
                                                     'maxiter': 20000,
                                                     'disp': True})

        if not result.success:
            print( 'Likelihood optimisation did not converge.')

        self.bf = result 
        return result
    
    def create_param_range(self):
        print("In composite.py class-lf_polyb create_param_range")

        half = self.bf.x/2.0
        double = 2.0*self.bf.x
        self.prior_min_values = np.where(half < double, half, double) 
        self.prior_max_values = np.where(half > double, half, double)
        assert(np.all(self.prior_min_values < self.prior_max_values))

        return

    def lnprior(self, theta):
        print("In composite.py class-lf_polyb lnprior")
        """
        Set up uniform priors.

        """

        params = self.getparams(theta)
        alpha = params[2]
        alpha_atz6 = self.atz(6.0, alpha) 
        
        if (np.all(theta < self.prior_max_values) and
            np.all(theta > self.prior_min_values) and
            alpha_atz6 < -4.0):
            return 0.0 

        return -np.inf

    def lnprob(self, theta):
        print("In composite.py class-lf_polyb lnprob")

        lp = self.lnprior(theta)
        
        if not np.isfinite(lp):
            return -np.inf

        return lp - self.neglnlike(theta)

    def run_mcmc(self):
        print("In composite.py class-lf_polyb run_mcmc")
        """
        Run emcee.

        """
        self.ndim, self.nwalkers = self.bf.x.size, 100
        self.mcmc_start = self.bf.x 
        pos = [self.mcmc_start + 1e-4*np.random.randn(self.ndim) for i
               in range(self.nwalkers)]
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                             self.lnprob)

        self.sampler.run_mcmc(pos, 1000)
        self.samples = self.sampler.chain[:, 500:, :].reshape((-1, self.ndim))

        return

    def corner_plot(self, labels=None, dirname=''):
        print("In composite.py class-lf_polyb corner_plot")

        mpl.rcParams['font.size'] = '14'
        f = corner.corner(self.samples, labels=labels, truths=self.bf.x)
        plotfile = dirname+'triangle.png'
        f.savefig(plotfile)
        mpl.rcParams['font.size'] = '22'

        return

    def plot_chains(self, fig, param, ylabel):
        print("In composite.py class-lf_polyb plot_chains")

        ax = fig.add_subplot(self.bf.x.size, 1, param+1)
        for i in range(self.nwalkers): 
            ax.plot(self.sampler.chain[i,:,param], c='k', alpha=0.1)
        ax.axhline(self.bf.x[param], c='#CC9966', dashes=[7,2], lw=2) 
        ax.set_ylabel(ylabel)
        if param+1 != self.bf.x.size:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('step')
            
        return 

    def chains(self, labels=None, dirname=''):
        print("In composite.py class-lf_polyb chains")

        mpl.rcParams['font.size'] = '10'
        nparams = self.bf.x.size
        plot_number = 0 

        fig = plt.figure(figsize=(12, 2*nparams), dpi=100)
        for i in range(nparams): 
            self.plot_chains(fig, i, ylabel=labels[i])
            
        plotfile = dirname+'chains.pdf' 
        plt.savefig(plotfile,bbox_inches='tight')
        plt.close('all')
        mpl.rcParams['font.size'] = '22'
        
        return
    
    

    
