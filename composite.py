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

    def find_Pb_Yb_Vb(self):
        Pb = np.random.uniform(0.0, 1.0, size=self.nwalkers_with_bp)
        mean = self.bf.x[0]
        print("mean = ", mean)
        # log10_Yb = np.random.normal(loc=mean, scale=1, size=self.nwalkers_with_bp)
        # Yb = 10.0**log10_Yb
        # log10_Vb = np.random.normal(loc=mean, scale=5, size=self.nwalkers_with_bp)
        # Vb = 10.0**log10_Vb

        Yb = np.random.normal(loc=mean, scale=5, size=self.nwalkers_with_bp)

        # Hand picking alpha and beta for gamma distribution
        # to get mode around 10 and mean around 20
        alpha, beta = 2, 10
        Vb = np.random.gamma(shape=alpha, scale=beta, size=self.nwalkers_with_bp)

        # print("\nStatistics of Yb and Vb for testing purposes")
        # print("Yb mean = ", np.mean(Yb), "\tYb std = ", np.std(Yb))
        # print("Vb mean = ", np.mean(Vb), "\tVb std = ", np.std(Vb))
        # print("Pb mean = ", np.mean(Pb), "\tPb std = ", np.std(Pb))
        # print("\n\n")
        # print("log10_Yb mean = ", np.mean(log10_Yb),
        #       "\tlog10_Yb std = ", np.std(log10_Yb))
        # print("log10_Vb mean = ", np.mean(log10_Vb),
        #       "\tlog10_Vb std = ", np.std(log10_Vb))

        # print("Mean = ", mean),
        
        # import sys

        # sys.exit("\nQuiting for testing purposes\n")

        print("shape = ", np.array([Pb, Yb, Vb]).T.shape)

        return np.array([Pb, Yb, Vb]).T
    
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
                [beta, beta_l, beta_u]], zmean
    
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
        coeffs = np.polyfit(x, y, degree, w=1.0/err)
        poly_fn = np.poly1d(coeffs)
        # Plot the data points with error bars and the best-fit polynomial curve
        plt.figure(figsize=(8, 5))
        plt.errorbar(x, y, yerr=err, fmt='o', label='Data', capsize=3)
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = poly_fn(x_fit)
        plt.plot(x_fit, y_fit, 'r-', label='Best-fit polynomial')

        # Estimate 1-sigma error band for the fit using covariance matrix
        # (np.polyfit does not return cov by default, so we use a simple MC approach)
        n_mc = 1000
        mc_curves = []
        for _ in range(n_mc):
            y_mc = y + np.random.normal(0, err)
            coeffs_mc = np.polyfit(x, y_mc, degree, w=1.0/err)
            mc_curves.append(np.poly1d(coeffs_mc)(x_fit))
        mc_curves = np.array(mc_curves)
        y_std = np.std(mc_curves, axis=0)

        plt.fill_between(x_fit, y_fit - y_std, y_fit + y_std, color='r', alpha=0.2, label=r'1$\sigma$ error band')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'polynomial_fit_{label}_with_errors.png')
        plt.close()
        return coeffs, poly_fn
    
    def logprior_for_1_param(self, theta):
        arr = theta[:-3]
        Pb, Yb, Vb = theta[-3:]
        if not np.any((-30 < arr) & (arr < 10)):
            return -np.inf
        if not ((0 < Pb < 1) and (0 < Vb < 1e10) and (-1e10 < Yb < 1e10)):
            return -np.inf
        return 0.0
    
    def loglike_for_1_param(self, theta, x, y, err, degree=3):
        arr = theta[:-3]
        Pb, Yb, Vb = theta[-3:]

        poly_fn = np.poly1d(arr)(x)

        epsilon = 1e-10  # Small value to prevent division by zero
        safe_sig2 = err**2 + epsilon
        safe_Vb = Vb + epsilon

        logforeground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((y - poly_fn))**2 / safe_sig2, -1e10, 1e10))
        logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((y - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

        a = np.log(1 - Pb) + logforeground_model
        b = np.log(Pb) + logbackground_model

        lnL = np.sum(np.logaddexp(a, b))

        return lnL


    def logpos_for_1_param(self, theta, x, y, err, degree=3):
        lp = self.logprior_for_1_param(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglike_for_1_param(theta, x, y, err, degree)
        return lp + ll

    def mcmc_for_1_param(self, x, y, err, label, coeff, degree=3):
        walkers = 20
        ndim = degree + 4
        # Each walker position: [poly_coeffs..., Pb, Yb, Vb]
        pos = []
        for _ in range(walkers):
            walker_pos = np.empty(ndim)
            walker_pos[:degree+1] = coeff + 1e-2 * np.random.randn(degree+1)
            walker_pos[degree+1] = np.random.uniform(0, 1)
            walker_pos[degree+2] = np.random.uniform(-1e2, 1e2)
            walker_pos[degree+3] = np.random.uniform(0, 1e2)
            pos.append(walker_pos)
        pos = np.array(pos)

        print("pos shape:", pos.shape)

        sampler = emcee.EnsembleSampler(walkers, ndim, self.logpos_for_1_param,
                                        args=(x, y, err, degree))
        sampler.run_mcmc(pos, 5000, progress=True)
        sampler.reset()
        sampler.run_mcmc(None, 100000, progress=True)

        samples = sampler.get_chain(flat=True)
        print("MCMC sampling completed.")

        corner.corner(samples, labels=[f'param_{i}' for i in range(degree + 4)],
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(f'mcmc-{label}_results.png')
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
        plt.savefig(f'mcmc-{label}_chains.png')
        plt.close()

        # Print autocorrelation time and acceptance rate
        try:
            tau = sampler.get_autocorr_time()
            print("Autocorrelation time for each parameter:", tau)
        except Exception as e:
            print("Could not compute autocorrelation time:", e)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

        marginalised_samples = sampler.get_chain(flat=True)[:, :-3]  # Exclude Pb, Yb, Vb

        # Find the index of the maximum posterior sample
        max_idx = np.argmax(marginalised_samples, axis=0)
        print("Index of maximum posterior sample:", max_idx)
        map_params = sampler.get_chain(flat=True)[max_idx]

        print(f"MAP estimate for {label}: {map_params}")

        # Optionally, print each parameter's MAP value
        for i, val in enumerate(map_params):
            print(f"param_{i} (MAP): {val}")

        return map_params

        TOMORROW!!! CREATE A NEW LF SAVE FILE WHICH CONTAINS BASICALLY A VERY SMALL BIN
        ITS TAKING TOO MUCH TIME BECAUSE NP.SAVE AND NP.LOAD IS MAKING IT SLOW

    
    def run_mcmc_for_for_1_param(self, pnum=np.array([3,4,2,5])):
        data_full = self.sample_data_set()
        zmean = data_full[1]
        data = np.array(data_full[0])

        params = ["logphi", "M_star", "alpha", "beta"]
        coeff_list = []
        map_param = []
        for i in range(len(data)):
            print(f"(x, y): {(zmean, data[i][0])}")
            print(f"Errors: {(data[i][2] - data[i][1])/2.0}")
            print(f"Degree of polynomial for {params[i]}: {pnum[i]}")
            print(f"Label for {params[i]}: {params[i]}")
            d = (zmean, data[i][0])
            err = ((data[i][2] - data[i][1])/2.0)
            coeffs, poly_fn = self.fit_polynomial_curve(
                d, err,
                params[i], degree=pnum[i]
            )
            print(f"Fitted coefficients for {params[i]}: {coeffs}")
            print(f"Polynomial function for {params[i]}: \n{poly_fn}")
            coeff_list.append(coeffs)

            map_param.append(self.mcmc_for_1_param(
                zmean, data[i][0], (data[i][2] - data[i][1])/2.0,
                params[i], coeffs, degree=pnum[i]
            ))

        # Save the MAP parameter estimates to a file, one row per line
        with open("composite_map_param_estimate.dat", "w") as f:
            for row in map_param:
                f.write(" ".join(str(x) for x in row) + "\n")


        

















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
    
