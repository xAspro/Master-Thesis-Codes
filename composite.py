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

from pathos.multiprocessing import ProcessingPool as Pool
import dill
import re

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

        params = self.getparams(theta)

        log10phi_star = self.atz(z, params[0])
        M_star = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))

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

    # def find_Pb_Yb_Vb(self):
    #     Pb = np.random.uniform(0.0, 1.0, size=self.nwalkers_with_bp)
    #     mean = self.bf.x[0]
    #     print("mean = ", mean)
    #     # log10_Yb = np.random.normal(loc=mean, scale=1, size=self.nwalkers_with_bp)
    #     # Yb = 10.0**log10_Yb
    #     # log10_Vb = np.random.normal(loc=mean, scale=5, size=self.nwalkers_with_bp)
    #     # Vb = 10.0**log10_Vb

    #     Yb = np.random.normal(loc=mean, scale=5, size=self.nwalkers_with_bp)

    #     # Hand picking alpha and beta for gamma distribution
    #     # to get mode around 10 and mean around 20
    #     alpha, beta = 2, 10
    #     Vb = np.random.gamma(shape=alpha, scale=beta, size=self.nwalkers_with_bp)

    #     # print("\nStatistics of Yb and Vb for testing purposes")
    #     # print("Yb mean = ", np.mean(Yb), "\tYb std = ", np.std(Yb))
    #     # print("Vb mean = ", np.mean(Vb), "\tVb std = ", np.std(Vb))
    #     # print("Pb mean = ", np.mean(Pb), "\tPb std = ", np.std(Pb))
    #     # print("\n\n")
    #     # print("log10_Yb mean = ", np.mean(log10_Yb),
    #     #       "\tlog10_Yb std = ", np.std(log10_Yb))
    #     # print("log10_Vb mean = ", np.mean(log10_Vb),
    #     #       "\tlog10_Vb std = ", np.std(log10_Vb))

    #     # print("Mean = ", mean),
        
    #     # import sys

    #     # sys.exit("\nQuiting for testing purposes\n")

    #     print("shape = ", np.array([Pb, Yb, Vb]).T.shape)

    #     return np.array([Pb, Yb, Vb]).T
    
    def _lnprior_with_bad_points(self, params):
        logphi, M_star, alpha, beta = params[:4]
        Pb, Yb, Vb = params[4:]

        if self.prior_tag == 1:
            if Pb < 0 or Pb > 1:
                return -np.inf
            if Vb <= 0 or Vb > 1e10:
                return -np.inf
            if Yb < -1e10 or Yb > 1e10:
                return -np.inf
            
            if logphi < -20 or logphi > 0:
                return -np.inf
            
            if M_star < -50 or M_star > 0:
                return -np.inf
            
            if alpha < -7 or alpha > beta:
                return -np.inf
            
            if beta > 0:
                return -np.inf
            # print("Returning 0 for prior tag 1")
            return 0
        
        elif self.prior_tag == 2:
            if Pb < 0 or Pb > 1:
                return -np.inf
            if Vb <= 0:
                return -np.inf
            
            if logphi < -20 or logphi > 0:
                return -np.inf
            
            if M_star < -50 or M_star > 0:
                return -np.inf
            
            if alpha < -7 or alpha > beta:
                return -np.inf
            
            if beta > 0:
                return -np.inf
            # print("Returning 0 for prior tag 2")
            return - np.log(1 + Pb) - np.log(1 + Vb) 
        
    
    def _lnlikelihood_with_bad_points(self, params):
        func_params = params[:4]
        Pb, Yb, Vb = params[4:]

        epsilon = 1e-10  # Small value to prevent division by zero
        safe_sig2 = self.data.log_err**2 + epsilon
        safe_Vb = Vb + epsilon

        logforeground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((self.data.logphi - self.log10phi(func_params, self.data.mag)))**2 / safe_sig2, -1e10, 1e10))
        logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((self.data.logphi - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

        a = np.log(1 - Pb) + logforeground_model
        b = np.log(Pb) + logbackground_model

        lnL = np.sum(np.logaddexp(a, b))
        
        return lnL
    
    def _lnposterior_with_bad_points(self, params):
        lp = self._lnprior_with_bad_points(params)
        if not np.isfinite(lp):
            return -np.inf

        ll = self._lnlikelihood_with_bad_points(params)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll
    
    def run_mcmc_with_bad_points(self, ncores):
        self.prior_tag = 1
        self.function_tag = 1

        self.ndim_with_bp, self.nwalkers_with_bp = self.bf.x.size + 3, 100
        self.mcmc_start = self.bf.x 
        
        pos_with_bp = np.hstack((np.array([self.bf.x 
                    + 1e-2*np.random.randn(self.bf.x.size) for i 
                    in range(self.nwalkers_with_bp)]), self.find_Pb_Yb_Vb()))
        
        if ncores <= 1:
            pool = None
        else:
            pool = Pool(ncores)
        print("Using multiprocessing pool with {} cores...\n\n\n".format(ncores))

        lnposterior_pickable = dill.loads(dill.dumps(self._lnposterior_with_bad_points))
        self.sampler_with_bp = emcee.EnsembleSampler(self.nwalkers_with_bp, self.ndim_with_bp,
                                            lnposterior_pickable, pool=pool)
        print("Running MCMC with {} walkers and {} dimensions...".format(self.nwalkers_with_bp, self.ndim_with_bp))


        DISCARD = 1000

        self.sampler_with_bp.run_mcmc(pos_with_bp, DISCARD, progress=True)
        self.sampler_with_bp.reset()
        self.sampler_with_bp.run_mcmc(None, 3000, progress=True)

        self.samples_with_bp = self.sampler_with_bp.get_chain(flat=True)

        


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

    def sample_data_set(self):
        """
        This is a function which does nothing but provide a sample data set,
        which was obtained from running mcmc in `individual.py` file.
        """

        zmean = [0.305, 0.500, 0.699, 0.902, 1.102, 1.298, 1.502, 1.701, 1.976, 
                 2.302, 2.446, 2.548, 2.645, 2.746, 2.850, 2.950, 3.050, 3.148, 
                 3.246, 3.342, 3.442, 3.870, 4.343, 4.922, 5.999]

        logphi = [
            -5.8123, -6.4152, -7.0860, -6.4479, -6.2338, -6.2530, -6.2179, -6.2102, 
            -6.1750, -5.8939, -6.1222, -5.7050, -5.7862, -6.0597, -6.0751, -6.5549, 
            -6.5404, -6.2448, -7.0072, -7.5788, -6.7057, -7.9790, -7.6661, -7.2109, 
            -11.7806
        ]

        logphi_l = [
            -7.2440, -6.5481, -7.1345, -6.7162, -6.3641, -6.3556, -6.3520, -6.2861, 
            -6.5757, -6.0534, -6.6768, -6.7663, -5.8448, -6.2410, -6.3852, -6.8366, 
            -6.7352, -6.9824, -8.3627, -8.9081, -7.2521, -8.0520, -9.1543, -7.5857, 
            -11.9955
        ]

        logphi_u = [
            -5.6829, -6.0554, -6.2760, -6.3228, -6.1095, -6.0637, -6.0871, -6.1273, 
            -6.0574, -5.7132, -6.0204, -5.7455, -5.7271, -5.9955, -5.9023, -6.1506, 
            -6.4709, -5.9880, -6.1737, -5.4836, -6.3671, -7.7915, -6.1355, -6.9908, 
            -11.0626
        ]

        M_star = [
            -21.3815, -23.3261, -25.0078, -24.5790, -24.7391, -25.0499, -25.3398, -25.4468, 
            -25.5517, -24.9456, -25.2789, -24.4237, -24.6506, -25.5248, -25.3310, -26.2386, 
            -26.1322, -25.1820, -26.9001, -27.5000, -26.3003, -27.3448, -26.3596, -25.7025, 
            -31.5716
        ]

        M_star_l = [
            -24.0021, -23.6584, -25.0226, -24.9778, -24.9200, -25.1955, -25.4669, -25.5810, 
            -26.1027, -25.3222, -26.3185, -26.3243, -24.8320, -25.8943, -26.1352, -26.7247, 
            -26.4682, -26.8006, -29.7789, -30.4433, -27.7482, -27.4777, -28.3320, -26.0888, 
            -31.6762
        ]

        M_star_u = [
            -21.2237, -22.7327, -23.8103, -24.4039, -24.5331, -24.7403, -25.0790, -25.3398, 
            -25.3735, -24.5729, -25.1411, -24.3793, -24.4275, -25.3274, -24.8381, -25.3667, 
            -26.0178, -24.5800, -24.8789, -23.0703, -25.5200, -27.1652, -24.2435, -25.1100, 
            -30.1533
        ]

        alpha = [
            -2.7308, -3.3047, -4.3388, -3.6924, -3.6904, -3.6084, -3.7419, -3.7697, 
            -3.5928, -2.9482, -3.0561, -2.7177, -2.7851, -3.2676, -2.7138, -3.4523, 
            -3.7326, -2.7462, -3.2712, -4.3261, -3.0391, -5.3042, -3.4402, -3.2905, 
            -5.5750
        ]

        alpha_l = [
            -3.8568, -3.5979, -4.4419, -3.9769, -3.8309, -3.7400, -3.9089, -3.9005, 
            -3.9318, -3.1459, -4.0794, -4.0619, -2.9288, -3.7850, -3.3733, -4.5393, 
            -4.5487, -3.6286, -5.1750, -5.5801, -4.5576, -5.6212, -4.6919, -3.5510, 
            -6.5226
        ]

        alpha_u = [
            -2.6798, -2.9562, -3.3041, -3.4870, -3.5195, -3.4259, -3.5884, -3.6576, 
            -3.4955, -2.7391, -2.9227, -2.7022, -2.6457, -3.0078, -2.4341, -2.5965, 
            -3.4697, -2.4101, -2.1692, -2.2083, -2.2095, -4.5489, -2.9652, -3.0701, 
            -4.4757
        ]

        beta = [
            -0.4567, -1.5267, -2.0849, -1.7867, -1.6928, -1.7600, -1.7503, -1.6327, 
            -1.6167, -1.4378, -1.3308, -0.7445, -0.5643, -1.0570, -1.1204, -1.5443, 
            -1.5193, -1.1523, -1.8045, -1.9982, -1.1488, -2.0012, -2.0946, -1.1883, 
            -2.2936
        ]

        beta_l = [
            -2.0993, -1.7210, -2.1517, -1.9898, -1.7547, -1.8074, -1.7895, -1.6953, 
            -1.9931, -1.5304, -1.7993, -1.9127, -0.7383, -1.2601, -1.3947, -1.7377, 
            -1.6440, -1.7539, -1.9838, -2.1391, -1.6121, -2.0423, -2.6479, -1.6477, 
            -2.4503
        ]

        beta_u = [
            -0.2429, -1.3134, -1.7548, -1.7032, -1.5781, -1.4974, -1.5952, -1.5839, 
            -1.5375, -1.2912, -1.2700, -0.7224, -0.3381, -0.9771, -0.9211, -1.3649, 
            -1.4442, -0.9194, -1.6695, -1.5854, -0.7169, -1.7931, -0.8642, -0.4580, 
            -2.2205
        ]

        return [[logphi, logphi_l, logphi_u],
                [M_star, M_star_l, M_star_u],
                [alpha, alpha_l, alpha_u],
                [beta, beta_l, beta_u]], np.array(zmean)
    
    def read_parameters_with_bp(self):
        """
        Reads the parameters_with_bp.dat file and returns arrays for zmean, values, and credibility intervals.

        Parameters
        ----------
        None

        Returns
        -------
        ret : list of lists
            Each inner list contains the values for a single redshift bin:
            [log10phi, log10phi_lower, log10phi_upper, M_star, M_star_lower, M_star_upper,
             alpha, alpha_lower, alpha_upper, beta, beta_lower, beta_upper]
        zmean : numpy.ndarray
            Array of mean redshift values corresponding to the parameters.
        """
        filename = 'parameters_with_bp.dat'
        zmean = []
        values = []
        intervals = []
        with open(filename, 'r') as f:
            print(f"Reading parameters from {filename}...")
            for line in f:
                print(line)
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                print(f"Processing line: {line}")
                # Split by the two '|' separators
                if line.count('|') == 2:
                    print(f"Line has 2 '|' separators: {line}")
                    parts = line.split('|')
                    left = parts[0].strip()
                    middle = parts[1].strip()
                    right = parts[2].strip()
                    # left: zmean
                    z = float(left)
                    # middle: 4 parameter values
                    vals = [float(x) for x in middle.split()]
                    # right: 4 pairs of intervals (lower upper for each param)
                    right_parts = right.split()
                    interval_pairs = []
                    for i in range(0, 8, 2):
                        interval_pairs.append([float(right_parts[i]), float(right_parts[i+1])])
                    zmean.append(z)
                    values.append(vals)
                    intervals.append(interval_pairs)
        
        print(f"Read {len(zmean)} redshift bins from {filename}.")
        zmean = np.array(zmean)
        values = np.array(values)
        intervals = np.array(intervals)  # shape (N, 4, 2)
        print(f"zmean shape: {zmean.shape}, values shape: {values.shape}, intervals shape: {intervals.shape}")

        ret = []
        for i in range(len(zmean)):
            ret.append([[values[i][0], intervals[i][0][0], intervals[i][0][1]],
                        [values[i][1], intervals[i][1][0], intervals[i][1][1]],
                        [values[i][2], intervals[i][2][0], intervals[i][2][1]],
                        [values[i][3], intervals[i][3][0], intervals[i][3][1]]])

        ret = np.transpose(ret, (1, 2, 0))  # shape (4, 3, N)
        return ret, zmean
    
    def fit_polynomial_curve(self, data, err, label, degree=3):
        """
        Fit a polynomial of given degree to data with errors. Additionally,
        plots the data for visual checks too!

        Parameters
        ----------
        data : ndarray
            ndarray of (x, y), where x is the independent variable and y is the dependent variable.
        err : array-like
            Symmetric errors for y values.
        degree : int, optional
            Degree of the polynomial to fit. Default is 3.

        Returns
        -------
        coeffs : ndarray
            Polynomial coefficients (highest power first).
        poly_fn : function
            Function to evaluate the fitted polynomial.
        """
        x, y = data
        # Fit using weighted least squares
        
        # # Fit Chebyshev polynomial (automatically scales x to [-1, 1])
        # cheb_fit = T.fit(x, y, degree, w=1.0/err)
        # # cheb_fit is a callable polynomial function
        # poly_fn = cheb_fit
        # coeffs = cheb_fit.coef  # Chebyshev coefficients

        # # Plot the data points with error bars and the best-fit polynomial curve
        # plt.figure(figsize=(8, 5))
        # plt.errorbar(x, y, yerr=err, fmt='o', label='Data', capsize=3)
        # x_fit = np.linspace(np.min(x), np.max(x), 200)
        # y_fit = poly_fn(x_fit)
        # plt.plot(x_fit, y_fit, 'r-', label='Best-fit polynomial')

        # # Estimate 1-sigma error band for the fit using covariance matrix
        # # (np.polyfit does not return cov by default, so we use a simple MC approach)
        # n_mc = 1000
        # mc_curves = []
        # for _ in range(n_mc):
        #     y_mc = y + np.random.normal(0, err)
        #     cheb_mc = T.fit(x, y_mc, degree, w=1.0/err)
        #     mc_curves.append(cheb_mc(x_fit))  # Use the Chebyshev object directly
        # mc_curves = np.array(mc_curves)
        # y_std = np.std(mc_curves, axis=0)

        # Fit normal polynomial (monomial basis)
        coeffs = np.polyfit(x, y, degree, w=1.0/err)
        poly_fn = np.poly1d(coeffs)

        # Plot the data points with error bars and the best-fit polynomial curve
        plt.figure(figsize=(8, 5))
        plt.errorbar(x, y, yerr=err, fmt='o', label='Data', capsize=3)
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = poly_fn(x_fit)
        plt.plot(x_fit, y_fit, 'r-', label='Best-fit polynomial')

        # Estimate 1-sigma error band for the fit using covariance matrix
        n_mc = 1000
        mc_curves = []
        for _ in range(n_mc):
            y_mc = y + np.random.normal(0, err)
            coeffs_mc = np.polyfit(x, y_mc, degree, w=1.0/err)
            poly_mc = np.poly1d(coeffs_mc)
            mc_curves.append(poly_mc(x_fit))
        mc_curves = np.array(mc_curves)
        y_std = np.std(mc_curves, axis=0)

        plt.fill_between(x_fit, y_fit - y_std, y_fit + y_std, color='r', alpha=0.2, label=r'1$\sigma$ error band')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim()  # Adjust y-limits as needed
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'polynomial_fit_{label}_with_errors.png')
        plt.close()
        return coeffs, poly_fn
    
    def logprior_for_1_param(self, theta):
        arr = theta[:-3]
        Pb, Yb, Vb = theta[-3:]
        # if np.any((-50 > arr) | (arr > 50)):
        #     return -np.inf
        # x2 = arr[0]
        # x1 = arr[1]
        # x0 = arr[2]
        # if not ((-50 < x2 < 50) and (-50 < x1 < 50) and (-50 < x0 < 50)):
        #     return -np.inf
        # if (np.any((-50 > arr) | (arr > 50))):
        if (np.any((-50 > arr) | (arr > 7))): # for beta and alpha
            return -np.inf
        # if (np.any((-40 > arr) | (arr > 5))): ## FOR m_star
        #     return -np.inf
        if not ((0 < Pb < 1) and (0 < Vb < 1e2) and (-1e2 < Yb < 1e2)):
            return -np.inf
        rnge = self.rnge
        # print("\nPrior coeff =", self.coeff, "Range =", rnge)
        # print("arr =", arr)
        # print("self.coeff - rnge =", self.coeff - rnge, "self.coeff + rnge =", self.coeff + rnge)

        # print("arr < self.coeff - rnge =", np.any(arr < self.coeff - rnge))
        # print("arr > self.coeff + rnge =", np.any(arr > self.coeff + rnge))

        # print("np.any((arr < self.coeff - rnge) | (arr > self.coeff + rnge)) =", np.any((arr < self.coeff - rnge) | (arr > self.coeff + rnge)))

        # Check if any coefficient is outside the range of [coeff - rnge, coeff + rnge]
        # if np.any((arr < self.coeff - rnge) | (arr >
        if np.any((arr < self.coeff - rnge) | (arr > self.coeff + rnge)):
            return -np.inf
        
        # print("Prior tag =", self.prior_tag)
        if self.prior_tag == 1:
            # print("Returning 0 for prior tag 1")
            return 0.0
        if self.prior_tag == 2:
            return -Pb
    
    def loglike_for_1_param(self, theta, x, y, err, degree=3):
        arr = theta[:-3]
        Pb, Yb, Vb = theta[-3:]

        poly_fn = np.poly1d(arr)(x)
        # poly_fn = self.atz(x, arr)

        epsilon = 1e-10  # Small value to prevent division by zero
        safe_sig2 = err**2 + epsilon
        safe_Vb = Vb + epsilon

        logforeground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((y - poly_fn))**2 / safe_sig2, -1e10, 1e10))
        logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((y - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

        a = np.log(1 - Pb) + logforeground_model
        b = np.log(Pb) + logbackground_model

        lnL = np.sum(np.logaddexp(a, b))

        if np.isnan(lnL):
            print("\nNaN in loglike_for_1_param")
            print("Pb =", Pb)
            print("Yb =", Yb)
            print("Vb =", Vb)
            print("safe_sig2 =", safe_sig2)
            print("safe_Vb =", safe_Vb)
            print("logforeground_model =", logforeground_model)
            print("logbackground_model =", logbackground_model)
            print("a =", a)
            print("b =", b)
            print("lnL =", lnL)
            print()
            return -np.inf

        return lnL


    def logpos_for_1_param(self, theta, x, y, err, degree=3):
        lp = self.logprior_for_1_param(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglike_for_1_param(theta, x, y, err, degree)
        if np.isnan(ll):
            print("NaN in loglike_for_1_param")
        if np.isnan(lp):
            print("NaN in logprior_for_1_param")
        return lp + ll
    
    def find_Pb_Yb_Vb(self, y, walkers=1):
        Pb = np.random.uniform(0.0, 1.0, size=walkers)

        mean = np.mean(y)
        Yb = np.random.normal(loc=mean, scale=10, size=walkers)

        # # Hand picking alpha and beta for gamma distribution
        # # to get mean around 10 and Variance around 20
        # alpha, beta = 5, 2

        # Hand picking alpha and beta for gamma distribution
        # to get mean around 2 and Variance around 2
        alpha, beta = 2, 1
        Vb = np.random.gamma(shape=alpha, scale=beta, size=walkers)

        # print("shape = ", np.array([Pb, Yb, Vb]).T.shape)

        return np.array([Pb, Yb, Vb]).T
    

    def mcmc_for_1_param(self, x, y, err, label, coeff, degree=3):
        x = np.array(x)
        y = np.array(y)
        err = np.array(err)
        self.prior_tag = 1  # Set prior tag for the MCMC run

        nburns = 20000
        nprod = 100000

        walkers = 50
        ndim = degree + 4


        oversample_factor = 1000
        n_visualize = walkers * oversample_factor


        self.coeff = coeff

        if label == 'logphi':
            self.rnge = 1
        elif label == 'M_star':
            self.rnge = 5.5
        elif label == 'alpha':
            self.rnge = 1
        elif label == 'beta':
            self.rnge = 1

        pos_all = []
        for _ in range(n_visualize):
            walker_pos = np.empty(ndim)
            walker_pos[:degree+1] = coeff + np.random.uniform(-self.rnge, self.rnge, degree+1)
            walker_pos[degree+1:] = self.find_Pb_Yb_Vb(y)
            pos_all.append(walker_pos)
        pos_all = np.array(pos_all)


        figure = corner.corner(
            pos_all,  # All coefficients: polynomial plus nuisance parameters
            labels=[f"param_{i}" for i in range(degree+1)] + ['Pb', 'Yb', 'Vb'],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12}
        )
        plt.suptitle("Initial Walker Distribution (Poly Coeffs)", fontsize=14)
        figure.tight_layout()
        figure.savefig(f'initial_pos_cornerstyle_{label}.png')
        plt.close()

        pos = pos_all[:walkers]

        ##################################################################


        print("pos shape:", pos.shape)

        sampler = emcee.EnsembleSampler(walkers, ndim, self.logpos_for_1_param,
                                        args=(x, y, err, degree))
        sampler.run_mcmc(pos, nburns, progress=True)
        sampler.reset()
        sampler.run_mcmc(None, nprod, progress=True)

        samples = sampler.get_chain(flat=True)
        print("MCMC sampling completed.")

        labels = [f'$x^{degree - i}$' for i in range(degree + 1)] + ['Pb', 'Yb', 'Vb']

        # Compute bounds that cover a central q percentile of the samples
        def central_bounds(samples, q=0.95):
            """
            Compute the lower and upper bounds that enclose a central q percentile of the samples.
            The bounds are at (1-q)/2 and 1-(1-q)/2 percentiles.

            Parameters
            ----------
            samples : ndarray
                MCMC samples, shape (nsamples, ndim)
            q : float
                Central percentile to cover (e.g., 0.95 for 95% interval)

            Returns
            -------
            bounds : ndarray
                Array of shape (ndim, 2): lower and upper bounds for each parameter
            """
            lower = 100 * (1 - q) / 2
            upper = 100 * (1 + q) / 2
            return np.percentile(samples, [lower, upper], axis=0).T

        # Example usage: print 95% credible intervals for each parameter
        bounds = central_bounds(samples, q=0.90)
        for i, (lo, hi) in enumerate(bounds):
            print(f"Param {i}: {lo:.4f} to {hi:.4f} (central 95%)")

        corner.corner(samples, labels=labels, bounds=bounds,
                      quantiles=[0.16, 0.5, 0.84], bins=20, 
                      levels=[0.1175, 0.393, 0.676, 0.865, 0.955, 0.989],
                    #   levels=[0.1175, 0.393, 0.676, 0.865],
                      smooth=True,
                      fill_contours=True,
                      plot_datapoints=False,
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.suptitle(f'Prior tag: {self.prior_tag}', fontsize=14)
        plt.savefig(f'mcmc-{label}_results_{self.prior_tag}_Orignal.png')
        plt.close()

        corner.corner(samples, labels=labels, bounds=bounds,
                      quantiles=[0.16, 0.5, 0.84], bins=20, 
                      levels=[0.1175, 0.393, 0.676],
                    #   levels=[0.1175, 0.393, 0.676, 0.865],
                      smooth=True,
                      fill_contours=True,
                      plot_datapoints=False,
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.suptitle(f'Prior tag: {self.prior_tag}', fontsize=14)
        plt.savefig(f'mcmc-{label}_results_{self.prior_tag}_Clean.png')
        plt.close()

        # Plot chains for each parameter
        mpl.rcParams['font.size'] = '10'
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
        for i in range(ndim):
            ax = axes[i] if ndim > 1 else axes
            for j in range(walkers):
                ax.plot(sampler.chain[j, :, i], color='k', alpha=0.1)
            ax.set_ylabel(f'param_{i}')
        axes[-1].set_xlabel('step')
        plt.tight_layout()
        plt.savefig(f'mcmc-{label}_chains_{self.prior_tag}.png')
        plt.close()

        # Print autocorrelation time and acceptance rate
        try:
            tau = sampler.get_autocorr_time()
            print("Autocorrelation time for each parameter:", tau)
        except Exception as e:
            print("Could not compute autocorrelation time:", e)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

        taus = []
        x_tau = []
        for i in range(nprod // 50, nprod, nprod // 50):
            tau = emcee.autocorr.integrated_time(sampler.get_chain()[:i], tol=0)
            taus.append(tau)
            x_tau.append(i)

        # Plot autocorrelation time as a function of steps
        taus = np.array(taus)
        plt.figure(figsize=(8, 4))
        for i in range(ndim):
            plt.plot(x_tau, taus[:, i], label=f'param_{i}')
        plt.xlabel('Number of steps')
        plt.ylabel('Autocorrelation time')
        plt.title('Autocorrelation time vs steps')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'mcmc-{label}_autocorr_{self.prior_tag}.png')
        plt.close()



        marginalised_samples = samples[:, :-3]

        bins = 20
        hist, edges = np.histogramdd(marginalised_samples, bins=bins)

        max_idx = np.unravel_index(np.argmax(hist), hist.shape)

        map_params = []
        for i in range(degree+1):
            # Bin edges for this dimension
            bin_edges = edges[i]
            # Center of the bin
            center = 0.5 * (bin_edges[max_idx[i]] + bin_edges[max_idx[i]+1])
            map_params.append(center)
        map_params = np.array(map_params)
        print(f"MAP (marginalized over nuisance) for {label}:", map_params)

        median_Pb = np.median(samples[:, -3])
        median_Yb = np.median(samples[:, -2])
        median_Vb = np.median(samples[:, -1])

        is_bad = self.find_bad_points(np.poly1d, x, y, err, map_params,
                                      [median_Pb, median_Yb, median_Vb], degree=degree)
        print(f"Bad points for {label}:", is_bad)

        self.plot_bad_points(np.poly1d, x, y, err, map_params, is_bad, label, degree=degree)

        # import sys
        # sys.exit("Testing")

        # Compute 16th, 50th, and 84th percentiles for each parameter (excluding nuisance)
        percentiles = np.percentile(marginalised_samples, [16, 50, 84], axis=0)
        minus_sigma = percentiles[1] - percentiles[0]
        plus_sigma = percentiles[2] - percentiles[1]

        return [map_params, minus_sigma, plus_sigma]
    
    def find_bad_points(self, func, x, y, err, map_params, nuisance_params, degree=3):
        """
        Find bad points in the data based on the fitted polynomial and the MAP parameters.

        Parameters
        ----------
        x : array-like
            Independent variable values.
        y : array-like
            Dependent variable values.
        err : array-like
            Errors in the dependent variable.
        map_params : array-like
            MAP parameters from MCMC.
        degree : int, optional
            Degree of the polynomial fit. Default is 3.

        Returns
        -------
        is_bad : array-like
            Boolean array indicating which points are considered "bad".
        """
        # poly_fn = T(map_params[:degree+1])
        poly_fn = func(map_params[:degree+1])
        poly_fnx = poly_fn(x)
        residuals = y - poly_fnx

        safe_sig2 = err**2 + 1e-10  # Small value to prevent division by zero
        safe_Vb = nuisance_params[2] + 1e-10

        Pb = nuisance_params[0]
        Yb = nuisance_params[1]


        log_p_fg = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip((residuals)**2 / safe_sig2, -1e10, 1e10))
        log_p_bg = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((poly_fnx - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

        log_numerator = np.log(Pb) + log_p_bg
        log_denominator = np.logaddexp(np.log(1 - Pb) + log_p_fg, np.log(Pb) + log_p_bg)

        log_bad_prob = log_numerator - log_denominator

        # is_bad = log_bad_prob > -0.69  # This is equivalent to 50% in linear scale
        is_bad = log_bad_prob > -0.1  # This is equivalent to 90% in linear scale
        print("log_bad_prob= ", log_bad_prob, "\texp(log_bad_prob)= ", np.exp(log_bad_prob))

        return is_bad


    def plot_bad_points(self, func, x, y, err, map_params, is_bad, label, degree=3):
        """
        Plot the data points, fitted polynomial, and highlight bad points.

        Parameters
        ----------
        x : array-like
            Independent variable values.
        y : array-like
            Dependent variable values.
        err : array-like
            Errors in the dependent variable.
        map_params : array-like
            MAP parameters from MCMC.
        is_bad : array-like
            Boolean array indicating which points are considered "bad".
        label : str
            Label for the plot title.
        degree : int, optional
            Degree of the polynomial fit. Default is 3.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        err = np.asarray(err)

        # poly_fn = T(map_params[:degree+1])
        poly_fn = func(map_params[:degree+1])
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = poly_fn(x_fit)

        plt.figure(figsize=(8, 5))
        plt.errorbar(x, y, yerr=err, fmt='o', label='Data', capsize=3)
        plt.plot(x_fit, y_fit, 'r-', label='Best-fit polynomial')

        print("x= ", x, "\ty= ", y, "\terr= ", err)
        print("is_bad= ", is_bad)

        print("len(x)= ", len(x), "\tlen(y)= ", len(y), "\tlen(err)= ", len(err))
        print("len(is_bad)= ", len(is_bad))

        # Highlight bad points
        bad_x = x[is_bad]
        bad_y = y[is_bad]
        bad_err = err[is_bad]
        plt.errorbar(bad_x, bad_y, yerr=bad_err, fmt='o', color='orange', label='Bad points', capsize=3)

        if func == np.poly1d:
            title = 'Polynomial Fit'
        else:
            title = 'Full Param T fit'
        plt.xlabel('z')
        plt.ylabel(label)
        plt.ylim(np.min(y) - 1, np.max(y) + 1)  # Adjust y-limits as needed
        plt.title(f'{title} with Bad Points Highlighted: {label}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'bad_points_plot_{label}_{self.prior_tag}.png')
        plt.close()


    
    def run_mcmc_for_for_1_param(self, pnum=np.array([3,4,2,5])):
        pnum = np.array([3, 4, 4, 2])
        # data_full = self.sample_data_set()
        data_full = self.read_parameters_with_bp()

        zmean = data_full[1]
        data = np.array(data_full[0])

        params = ["logphi", "M_star", "alpha", "beta"]
        coeff_list = []
        map_param = []
        for i in range(len(data)):
            # if i < 3:
            #     continue
            # if i != 0:
            #     continue
            print(f"(x, y): {(zmean, data[i][0])}")
            print(f"Errors: {(data[i][2] - data[i][1])/2.0}")
            print(f"Degree of polynomial for {params[i]}: {pnum[i]}")
            print(f"Label for {params[i]}: {params[i]}")
            d = (zmean, data[i][0])
            err = ((data[i][2] - data[i][1])/2.0)
            coeffs, poly_fn = self.fit_polynomial_curve(
                d, err,
                params[i], degree=(pnum[i]-1)
            )
            print(f"Fitted coefficients for {params[i]}: {coeffs}")
            print(f"Polynomial function for {params[i]}: \n{poly_fn}")
            coeff_list.append(coeffs)

            map_param.append(self.mcmc_for_1_param(
                zmean, data[i][0], (data[i][2] - data[i][1])/2.0,
                params[i], coeffs, degree=(pnum[i]-1)
            ))

        # Save the MAP parameter estimates to a file, one row per line
        with open("composite_map_param_estimate.dat", "w") as f:
            for row in map_param:
                f.write(" ".join(str(x) for x in row) + "\n")

        print("MAP: ", map_param)
        initial_pos = map_param[:, 0]
        uncertainties = np.array([row[1:] for row in map_param])

        self.mcmc_all_params(
            data_full, initial_pos, uncertainties, pnum=pnum
        )


    def log_prior_full(self, theta):
        """
        Set up uniform priors for the full dataset.
        """
        params = self.getparams(theta)
        z = np.linspace(0.1, 6.0, 20) 
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])

        # print("theta = ", theta)

        # print("alpha = ", alpha)
        # print("beta = ", beta)
        alpha_atz6 = self.atz(6.0, alpha) 

        Pb, Yb, Vb = theta[-3:]

        # print("Pb = ", Pb, "\tYb = ", Yb, "\tVb = ", Vb)
        # print("params = ", params)
        # print("theta = ", theta)

        # print(f"self.prior_min_values = {self.prior_min_values}")
        # print(f"self.prior_max_values = {self.prior_max_values}")

        # print(f"size of self.prior_min_values = {self.prior_min_values.size}")
        # print(f"size of self.prior_max_values = {self.prior_max_values.size}")
        # print(f"size of theta = {theta.size}")



        # print(f"theta[:-3] = {theta[:-3]}")
        # print(f"self.prior_max_values[:-3] = {self.prior_max_values[:-3]}")
        # print(f"self.prior_min_values[:-3] = {self.prior_min_values[:-3]}")
        # print(f"alpha_atz6 = {alpha_atz6}")
        # print(f"theta[:-3] < self.prior_max_values[:-3] = {(theta[:-3] < self.prior_max_values[:-3])}")
        # print(f"theta[:-3] > self.prior_min_values[:-3] = {(theta[:-3] > self.prior_min_values[:-3])}")
        # print(f"np.all(theta[:-3] < self.prior_max_values[:-3]) = {np.all(theta[:-3] < self.prior_max_values[:-3])}")
        # print(f"np.all(theta[:-3] > self.prior_min_values[:-3]) = {np.all(theta[:-3] > self.prior_min_values[:-3])}")
        # print(f"alpha_atz6 < -4.0 = {alpha_atz6 < -4.0}")
        # import sys; sys.exit("Testing log_prior_full")


        if (np.all(theta[:-3] < self.prior_max_values[:-3]) and
            np.all(theta[:-3] > self.prior_min_values[:-3]) and
            # alpha_atz6 < -4.0):
            1):

            if np.any(alpha > beta):
                return -np.inf
            
            if np.any(beta > 0):
                return -np.inf
            
            if Pb < 0 or Pb > 1:
                return -np.inf
            
            if Yb < -100 or Yb > 100:
                return -np.inf
            
            if Vb < 0 or Vb > 100:
                return -np.inf
            
            # print("Returning 0!!")
            return 0.0 
        
        return -np.inf
        
    def log10phi_full(self, theta, mag, z):
        """
        Calculate the log10 of the QLF for the full dataset.
        """
        # print("theta = ", theta)
        params = self.getparams(theta)

        # print("params = ", params)

        log10phi_star = self.atz(z, params[0])
        M_star = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])
        # Already removed nuisance parameters from theta
        # Pb, Yb, Vb = params[-3:]



        # print("\nlog10phi_star = ", log10phi_star)
        # print("M_star = ", M_star)
        # print("alpha = ", alpha)
        # print("beta = ", beta)
        

        # print(f"log10phi_star = {log10phi_star}, M_star = {M_star}, alpha = {alpha}, beta = {beta}")
        # # print(f"mag = {mag}, z = {z}")
        # print(f"shape of mag = {mag.shape}, shape of z = {z.shape}")

        # print(f"shape of log10phi_star = {log10phi_star.shape}, shape of M_star = {M_star.shape}")
        # print(f"shape of alpha = {alpha.shape}, shape of beta = {beta.shape}")
        # print(f"shape of M_star = {M_star.shape}, shape of mag = {mag.shape}")

        # import sys; sys.exit("Testing log10phi_full")

        ln10 = np.log(10)

        log10_num = log10phi_star
        log10_den = np.logaddexp(
            0.4 * ln10 * (alpha + 1) * (mag - M_star),
            0.4 * ln10 * (beta + 1) * (mag - M_star)
        ) / ln10

        log10phi = log10_num - log10_den

        # print(f"log10phi shape = {log10phi.shape}, mag shape = {mag.shape}, z shape = {z.shape}")
        
        # import sys; sys.exit("Testing log10phi_full")
        return log10phi

    def neg_log_like_full(self, theta, data_full):
        """
        Calculate the log-likelihood for the full dataset.
        """
        # print("In composite.py class-lf neg_log_like_full")

        zmean = data_full[1]
        data = np.array(data_full[0])

        # print(f"zmean = {zmean}, data shape = {data.shape}")

        mag = data[3, :]
        y = data[0, :]
        # print(f"mag shape = {mag.shape}")
        # print(f"mag = {mag}")
        # print(f"data[:, 0] shape = {data[:, 0].shape}")
        # print(f"data[:, 0] = {data[:, 0]}")
        # print(f"data[:, 1] shape = {data[:, 1].shape}")
        # print(f"data[:, 1] = {data[:, 1]}")
        # print(f"data[:, 2] shape = {data[:, 2].shape}")
        # print(f"data[:, 2] = {data[:, 2]}")
        # print()
        # print("data = ", data)
        # print(f"data shape = {data.shape}")
        # print()
        # import sys; sys.exit("Testing neg_log_like_full")


        sig = (data[2, :] - data[1, :]) / 2.0

        Pb, Yb, Vb = theta[-3:]

        logphi = self.log10phi_full(theta[:-3], mag, zmean)
        logphi /= np.log10(np.e)  # Convert to base e


        epsilon = 1e-10  # Small value to prevent division by zero
        safe_sig2 = sig**2 + epsilon
        safe_Vb = Vb + epsilon

        log_foreground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((y - logphi))**2 / safe_sig2, -1e10, 1e10))
        log_background_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((y - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

        a = np.log(1 - Pb) + log_foreground_model
        b = np.log(Pb) + log_background_model

        lnL = np.sum(np.logaddexp(a, b))

        if np.isnan(lnL):
            # print("\nNaN in neg_log_like_full")
            # print("Pb =", Pb)
            # print("Yb =", Yb)
            # print("Vb =", Vb)
            # print("safe_sig2 =", safe_sig2)
            # print("safe_Vb =", safe_Vb)
            # print("log_foreground_model =", log_foreground_model)
            # print("log_background_model =", log_background_model)
            # print("a =", a)
            # print("b =", b)
            # print("lnL =", lnL)
            # print()

            return np.inf
        
        return -lnL  # Return negative log-likelihood for minimization


    def log_prob_full(self, theta, data_full):
        """
        Calculate the log-probability for the full dataset.
        """
        lp = self.log_prior_full(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.neg_log_like_full(theta, data_full)
        if np.isnan(ll):
            print("NaN in neg_log_like_full")
        if np.isnan(lp):
            print("NaN in log_prior_full")
        return lp - ll
    
    def find_best_fit_full(self, data_full, guess, method='Nelder-Mead'):
        """
        Find the best fit parameters for the full dataset using optimization.
        """
        print("In composite.py class-lf find_best_fit_full")
        print("Initial guess for parameters:", guess)
        # import sys; sys.exit("Testing find_best_fit_full")

        main_bounds = [(None, None) for _ in range(len(guess) - 3)]
        nuisance_bounds = [(0, 1), (-100, 100), (0, 100)]
        bounds = main_bounds + nuisance_bounds

        
        result = op.minimize(self.neg_log_like_full,
                             guess,
                             args=(data_full,),
                             bounds=bounds,
                             method=method, options={'maxfev': 20000,
                                                     'maxiter': 20000,
                                                     'disp': True})

        if not result.success:
            print('Likelihood optimisation did not converge.')

        self.bf_full = result
        print("Best fit parameters for full dataset:", result.x)
        # import sys; sys.exit("Testing find_best_fit_full")
        return result

    def mcmc_all_params(self, data_full, guess, pnum=np.array([3,4,2,5])):
        """
        Run MCMC to fit for all data, with all the parameters together.
        The parameters follows polynomial form, 
        where the degree of polynomial is given by pnum.
        """

        print("In composite.py class-lf mcmc_all_params")

        zmean = data_full[1]
        data = np.array(data_full[0])

        ndim = np.sum(pnum) + 3
        nwalkers = 100

        initial_guess = np.zeros(ndim)
        splitlocs = np.cumsum(pnum)
        initial_guess[:splitlocs[0]] = guess[:splitlocs[0]]         # logphi
        initial_guess[splitlocs[0]:splitlocs[1]] = guess[splitlocs[0]:splitlocs[1]] # M_star
        initial_guess[splitlocs[1]:splitlocs[2]] = guess[splitlocs[1]:splitlocs[2]] # alpha
        initial_guess[splitlocs[2]:splitlocs[3]] = guess[splitlocs[2]:splitlocs[3]] # beta
        initial_guess[-3:] = guess[-3:]  # Pb, Yb, Vb

        pos = np.zeros((nwalkers, ndim))
        for i in range(ndim):
            pos[:, i] = initial_guess[i] - 0.0001 * np.random.uniform(0, 1, nwalkers)



        # Run MCMC for all parameters
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_full, args=(data_full,))
        print("Running MCMC for all parameters...")
        sampler.run_mcmc(pos, 2000, progress=True)
        sampler.reset()
        sampler.run_mcmc(None, 20000, progress=True)
        samples = sampler.get_chain(flat=True)

        # Plot corner plot (same format as previous)
        labels = [f'param_{i}' for i in range(ndim - 3)] + ['Pb', 'Yb', 'Vb']
        corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], bins=20, 
                      levels=[0.1175, 0.393, 0.676, 0.865, 0.955, 0.989],
                      smooth=True, fill_contours=True, plot_datapoints=False,
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.suptitle('Full MCMC fit', fontsize=14)
        plt.savefig('mcmc_full_corner.png')
        plt.close()

        # Plot chains for each parameter
        mpl.rcParams['font.size'] = '10'
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
        for i in range(ndim):
            ax = axes[i] if ndim > 1 else axes
            for j in range(nwalkers):
                ax.plot(sampler.chain[j, :, i], alpha=0.1)
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel('step')
        plt.tight_layout()
        plt.savefig('mcmc_full_chains.png')
        plt.close()

        # Print autocorrelation time and acceptance rate
        try:
            tau = sampler.get_autocorr_time()
            print("Autocorrelation time for each parameter:", tau)
        except Exception as e:
            print("Could not compute autocorrelation time:", e)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

        import sys; sys.exit("Testing mcmc_all_params")


        
        # FOR THIS 19 DIMENSIONAL PLOT AND HISTOGRAM, SAVE EACH DIMENSION SEPARATELY
        # THEN YOU CAN DO THE HISTOGRAM WITHOUT RUNNING OUT OF RAM
        

        # Find MAP (maximum a posteriori) estimate
        bins = 2
        marginalised_samples = samples[:, :-3]
        hist, edges = np.histogramdd(marginalised_samples, bins=bins)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        map_params = []
        for i in range(marginalised_samples.shape[1]):
            bin_edges = edges[i]
            center = 0.5 * (bin_edges[max_idx[i]] + bin_edges[max_idx[i]+1])
            map_params.append(center)
        map_params = np.array(map_params)
        print("MAP (marginalized over nuisance):", map_params)

        # Median nuisance parameters
        median_Pb = np.median(samples[:, -3])
        median_Yb = np.median(samples[:, -2])
        median_Vb = np.median(samples[:, -1])

        # Find bad points
        is_bad = self.find_bad_points(self.log10phi_full,
            data_full[1], data_full[0][0], (data_full[0][2] - data_full[0][1]) / 2.0,
            map_params, [median_Pb, median_Yb, median_Vb], degree=(len(map_params)-1)
        )
        print("Bad points:", is_bad)

        # Plot final output with bad points
        self.plot_bad_points(self.log10phi_full,
            data_full[1], data_full[0][0], (data_full[0][2] - data_full[0][1]) / 2.0,
            map_params, is_bad, 'logphi', degree=(len(map_params)-1)
        )

    def read_datapoints_with_bp(self, filename):
        """
        Reads the datapoints_with_bp.dat file and returns a structured numpy array.
        Skips comment lines starting with '#' or empty lines.

        Parameters
        ----------
        filename : str
            Path to the datapoints_with_bp.dat file.

        Returns
        -------
        np.ndarray
            Structured numpy array with the following fields:
            - 'zmin': Minimum redshift (float)
            - 'zmax': Maximum redshift (float)
            - 'label': Label (string)
            - 'badness': Badness score (int)
            - 'bad_prob': Probability of being bad (float)
            - 'mag': Magnitude (float)
            - 'mag_err': Magnitude error (float)
            - 'logphi': Logarithm of the luminosity function (float)
            - 'map_log_phi': MAP log luminosity function (float)
            - 'uperr': Upper error (float)
            - 'downerr': Lower error (float)
            - 'log_err': Logarithm of the error (float)
            - 'residual': Residual (float)
            - 'sig_dif': Significance difference (float)
            - 'prob': Probability of residual (float)
        """
        print("In composite.py class-lf read_datapoints_with_bp")

        dtype = [
            ('zmin', float),
            ('zmax', float),
            ('label', 'U40'),
            ('badness', int),
            ('bad_prob', float),
            ('mag', float),
            ('mag_err', float),
            ('logphi', float),
            ('map_log_phi', float),
            ('uperr', float),
            ('downerr', float),
            ('log_err', float),
            ('residual', float),
            ('sig_dif', float),
            ('prob', float)
        ]

        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Split using two or more spaces as delimiter
                parts = re.split(r'\s{2,}', line)
                zmin = float(parts[0])
                zmax = float(parts[1])
                label = parts[2]
                rest = parts[3:]
                row = [zmin, zmax, label]
                row += [int(rest[0])]
                row += [float(x) for x in rest[1:]]
                data.append(tuple(row))
        return np.array(data, dtype=dtype)


    def call_mcmc(self, pnum=np.array([3,4,4,5])):
        """
        Call the MCMC function to fit all parameters.
        """
        print("In composite.py class-lf call_mcmc")

        self.pnum = pnum

        guess = np.array([-0.23, 0.95, -7.08, -0.29, 2.11, -5.06, -21.44, -0.14, 1.05, -1.94, 2.65, 0.06, -0.68, 2.27, -2.60, -0.52])

        database = self.read_datapoints_with_bp("datapoints_with_bp.dat")

        # Extract the data from the structured array
        zmin = database['zmin']
        zmax = database['zmax']
        zmean = self.sample_data_set()[1]
        mag = database['mag']

        # fitting z min for this size and format
        new_zmean = [zmean[0]]
        cnt = 0
        for i in range(1, len(zmin)):
                
            if zmin[i] != zmin[i-1]:
                cnt += 1
                print(f"\nzmin[{i}] = {zmin[i]:.3f}, zmin[{i-1}] = {zmin[i-1]:.3f}")
                print(f"cnt = {cnt}")
                print(f"zmean[{cnt}] = {zmean[cnt]:.3f}")
                print(f"zmean[{cnt-1}] = {zmean[cnt-1]:.3f}\n")
                new_zmean.append(zmean[cnt])
            else:
                new_zmean.append(new_zmean[-1])
        zmean = np.array(new_zmean)

        print(f"zmean: {zmean}")
        print(f"shape of zmean: {zmean.shape}, shape of zmin: {zmin.shape}, shape of zmax: {zmax.shape}")
        
        logphi = database['logphi']
        mag = database['mag']
        logphi_err = database['log_err']

        data_full = [
            np.array([
                database['logphi'].tolist(),
                (database['logphi'] - database['downerr']).tolist(),
                (database['logphi'] + database['uperr']).tolist(),
                mag.tolist()
            ]),
            zmean
        ]

        print("\nData for full fit:")
        print(data_full)
        print(f"shape of data_full[0]: {data_full[0].shape}, shape of data_full[1]: {data_full[1].shape}")


        nuisance_param_guess = np.array([0.0, -5, 2])

        guess = np.concatenate((guess, nuisance_param_guess))

        self.find_best_fit_full(data_full, guess)

        # Plot each parameter (logphi, M_star, alpha, beta) separately with its data points
        params = ["logphi", "M_star", "alpha", "beta"]
        splitlocs = np.cumsum(pnum)
        param_indices = [0] + splitlocs.tolist()
        zmean = data_full[1]
        data = np.array(data_full[0])

        param_data = self.sample_data_set()

        for i, param in enumerate(params):
            # Use sample_data_set for plotting the parameter evolution
            sample_data = param_data[0]
            sample_zmean = param_data[1]
            y_data = sample_data[i][0]
            yerr = (np.array(sample_data[i][2]) - np.array(sample_data[i][1])) / 2.0
            yerr = np.abs(yerr)
            coeffs = self.bf_full.x[param_indices[i]:param_indices[i+1]]
            # degree = len(coeffs) - 1

            # Use the correct function for each parameter
            z_fit = np.linspace(np.min(sample_zmean), np.max(sample_zmean), 200)
            params_split = self.getparams(self.bf_full.x[:-3])
            if param == "logphi":
                y_fit = self.atz(z_fit, params_split[0])
            elif param == "M_star":
                y_fit = self.atz(z_fit, params_split[1])
            elif param == "alpha":
                y_fit = self.atz(z_fit, params_split[2])
            elif param == "beta":
                y_fit = self.atz_beta(z_fit, params_split[3])

            print(f"Plotting {param} with coefficients: {coeffs}")


            plt.figure(figsize=(8, 5))
            plt.errorbar(sample_zmean, y_data, yerr=yerr, fmt='o', label='Sample Data', capsize=3)
            plt.plot(z_fit, y_fit, 'r-', label='Best-fit function')
            plt.xlabel('z')
            plt.ylabel(param)
            plt.title(f'{param} vs z')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'best_fit_{param}_with_sample_data.png')
            plt.close()

        # import sys; sys.exit("Testing call_mcmc")

        # Set the best fit line to be 0.1 in all its parameters with alternating signs, starting with -0.1 for each parameter group
        # Determine the split locations for each parameter group
        splitlocs = np.cumsum(pnum)
        max_full_x = np.zeros_like(self.bf_full.x[:-3])
        start = 0
        for end in splitlocs:
            # if end == splitlocs[-1]:
            #     break
            length = end - start
            # Alternating signs: -0.1, +0.1, -0.1, ...
            vals = np.array([50 * 0.1 ** (length - i) for i in range(length)])
            max_full_x[start:end] = vals
            start = end 

        print("max_full_x:", max_full_x)

        # # Manually set prior ranges for the full fit using bf_full.x
        # half = self.bf_full.x / 20
        # double = 20 * self.bf_full.x
        # self.min_prior_full = np.where(half < double, half, double)
        # self.max_prior_full = np.where(half > double, half, double)
        # assert np.all(self.min_prior_full < self.max_prior_full)

        self.min_prior_full = np.concatenate([np.array(-1 * max_full_x), np.array([0, -100, 0])])
        self.max_prior_full = np.concatenate([np.array(max_full_x), np.array([1, 100, 100])])

        # print("Prior min values for full fit:", self.min_prior_full)
        # print("Prior max values for full fit:", self.max_prior_full)

        self.prior_min_values = self.min_prior_full
        self.prior_max_values = self.max_prior_full


        # print("Size of guess:", guess.size)
        # print("Size of prior min values:", self.prior_min_values.size)
        # print("Size of prior max values:", self.prior_max_values.size)
        # print("splitlocs:", splitlocs)
        # print("splitlocs[:-1]:", splitlocs[:-1])
        # print("size of self.bf_full.x:", self.bf_full.x.size)

        # import sys; sys.exit("Testing call_mcmc")

        guess = np.zeros_like(self.prior_max_values)
        guess[-3:] = np.array([0.1, -5, 2])  # Pb, Yb, Vb

        print("Guess for MCMC:", guess)
        print("self.prior_min_values:", self.prior_min_values)
        print("self.prior_max_values:", self.prior_max_values)

        print("min < guess < max:",
              np.all(self.prior_min_values < guess) and
              np.all(self.prior_max_values > guess))
        
        # FOR THIS 19 DIMENSIONAL PLOT AND HISTOGRAM, SAVE EACH DIMENSION SEPARATELY
        # THEN YOU CAN DO THE HISTOGRAM WITHOUT RUNNING OUT OF RAM

        self.mcmc_all_params(data_full, guess, pnum=pnum)

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
    
