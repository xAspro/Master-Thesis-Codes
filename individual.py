# Checked once. No implementation of this code directly in here.
print("In individual.py")

import numpy as np
import scipy.optimize as op
from scipy.stats import beta as beta_dist
from scipy.stats import lognorm
import emcee
import matplotlib as mpl
mpl.use('Agg') 
# mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '16'
import matplotlib.pyplot as plt
import triangle 
from astropy.stats import poisson_conf_interval as pci
from astropy.stats import knuth_bin_width  as kbw
from scipy.integrate import quad 
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}
import gammapi
import rtg
import corner
import drawlf

import time

from pathos.multiprocessing import ProcessingPool as Pool
import dill
from collections import namedtuple


def getqlums(lumfile, zlims=None):

    """

    Read quasar luminosities.

    Parameters
    ----------
    lumfile : str
        The file containing the quasar luminosities.
    zlims : list of float, optional
        The redshift limits for the quasar luminosities. 
        The default is None.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - z : ndarray
            The redshifts of the selected quasars.
        - mag : ndarray
            The magnitudes of the selected quasars.
        - p : ndarray
            The probabilities of the selected quasars.
        - area : ndarray
            The areas of the selected quasars.
        - sample_id : ndarray
            The sample IDs of the quasars.
        - z_all : ndarray
            All redshifts of the quasars within the redshift limits.
        - mag_all : ndarray
            All magnitudes of the quasars within the redshift limits.
        - p_all : ndarray
            All probabilities of the quasars within the redshift limits.
        - area_all : ndarray
            All areas of the quasars within the redshift limits.
        - sample_id_all : ndarray
            All sample IDs of the quasars within the redshift limits.

    Notes
    -----
    The selected values are based on the redshift limits and the sample ID.
    It is possible that the select value is None. In that case, all values
    are returned. The array sizes are not guaranteed to be the same.

    CHECK THE FUNCTION ONCE MORE, LATER ON!!!
    FOR getselfunc(), WE HAVE np.squeeze IN SELFILE CONSTRUCTOR!!!
    NEED TO CHECK IF FOR QLUMS ALSO DO WE HAVE SOMETHING LIKE THAT TO HANDLE MULTIPLE DIMENSIONS!!!

    """

    with open(lumfile,'r') as f: 
        z, mag, p, area, sample_id = np.loadtxt(lumfile,
                                                usecols=(1,2,3,4,5),
                                                unpack=True)
    if zlims is None:
        # print("In getqlums")
        # print("In lumfile = ", lumfile)
        # print("zlims is None")
        # print("select1 is None")
        select = None
    else:
        z_min, z_max = zlims 
        select = ((z>=z_min) & (z<z_max))

    z_all = z[select]
    mag_all = mag[select]
    p_all = p[select]
    area_all = area[select]
    sample_id_all = sample_id[select]

    # print("z = ", z, "\t\tsample_id = ", sample_id)
    # print("z_all = ", z_all)

    try:
        sid = sample_id[0]
    except(IndexError):
        sid = sample_id

    select = None

    # if sid == 13: 
    #     select = ((((z_all>=0.0) & (z_all<0.2) & (mag_all<=-20.7)) |
    #                ((z_all>=0.2) & (z_all<0.4) & (mag_all<=-20.3)) |
    #                ((z_all>=0.4) & (z_all<0.6) & (mag_all<=-21.3)) |
    #                ((z_all>=0.6) & (z_all<0.8) & (mag_all<=-23.1)) | 
    #                ((z_all>=0.8) & (z_all<1.0) & (mag_all<=-23.7)) |
    #                ((z_all>=1.0) & (z_all<1.2)) |
    #                ((z_all>=1.2) & (z_all<1.4) & (mag_all<=-24.3)) |
    #                ((z_all>=1.4) & (z_all<1.6)) |
    #                ((z_all>=1.6) & (z_all<1.8) & (mag_all<=-24.9)) |
    #                ((z_all>=1.8) & (z_all<2.2))) |
    #               ((z_all>=3.5) & (z_all<4.7) & (mag_all<=-26.1)))
        
    # if sid == 15: 
    #     select = (((z_all>=0.4) & (z_all<0.6)) |
    #               ((z_all>=0.6) & (z_all<0.8) & (mag_all<=-20.7)) | 
    #               ((z_all>=0.8) & (z_all<1.2) & (mag_all<=-21.9)) |
    #               ((z_all>=1.2) & (z_all<1.8) & (mag_all<=-22.5)) |
    #               ((z_all>=1.8) & (z_all<2.2) & (mag_all<=-23.1)))

    if sid == 8:
        select = (mag_all > -26.73)




    z = z_all[select]
    mag = mag_all[select]
    p = p_all[select]
    area = area_all[select]
    sample_id = sample_id_all[select]

    # if select is None:
    #     print("In getqlums")
    #     print("In lumfile = ", lumfile)
    #     print("select2 is None")
    #     print("z: ", z)
    #     print("size of z: ", z.size)
    # if z.size != 0:
    #     print("\n\n\n\n\n\t\t\t\tz[0]: ", z[0])
    #     # CHECK THIS!!!!
    #     # Sometimes z will be a number and sometimes it will be an array of a number(at least while calling bins.py)
    #     # Because bins.py always gives zlims, so None case will occur only when select2 is None!

    return (z, mag, p, area, sample_id, z_all, mag_all, p_all,
            area_all, sample_id_all)
        
def getselfn(selfile, zlims=None):

    """

    Read selection map.

    Parameters
    ----------
    selfile : str
        The file containing the selection map.
    zlims : list of float, optional
        The redshift limits for the quasar luminosities. 
        The default is None.

    Returns
    -------
    - z : ndarray
        The redshifts of the selected quasars.
    - mag : ndarray
        The magnitudes of the selected quasars.
    - p : ndarray
        The probabilities of the selected quasars.
    - dz : ndarray
        The dz of the selected quasars.
    - dm : ndarray
        The dm of the selected quasars.

    Notes
    -----
    The selected values are based on the redshift limits and the sample ID.
    It is possible that the select value is None. In that case, all values
    are returned. The array sizes are not guaranteed to be the same.

    CHECK THE FUNCTION ONCE MORE, LATER ON!!!
    FOR getselfunc(), WE HAVE np.squeeze IN SELFILE CONSTRUCTOR!!!
    NEED TO CHECK IF FOR QLUMS ALSO DO WE HAVE SOMETHING LIKE THAT TO HANDLE MULTIPLE DIMENSIONS!!!

    """

    # print("In getselfn")
    # print("selfile: ", selfile)

    with open(selfile,'r') as f: 
        # print("f: ", f)
        z, mag, p, dz, dm = np.loadtxt(f, usecols=(1,2,3,4,5), unpack=True)

    if zlims is None:
        # Can afford to do this because of np.squeeze in selmap constructor
        select = None
    else:
        z_min, z_max = zlims 
        select = ((z>=z_min) & (z<z_max))

    return z[select], mag[select], p[select], dz[select], dm[select]


def volume(z, area, cosmo=cosmo):
    """
    Calculate the comoving volume in cMpc^3 for a given redshift and area.

    Parameters
    ----------
    - z : array_like
        Redshift values.
    - area : float
        Area in square degrees.
    - cosmo : dict, optional
        Cosmological parameters. Default is a flat LambdaCDM cosmology.

    Returns
    -------
    float
        Comoving volume in cMpc^3 per unit redshift.
    """

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1

def percentiles(x):
    """
    Calculate the 1-sigma percentiles for a given array.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    list
        List containing the upper, lower, and central percentiles.
    """
    
    u = np.percentile(x, 15.87) 
    l = np.percentile(x, 84.13)
    c = np.median(x) 

    return [u, l, c] 

LFData = namedtuple("LFData", ["sid","mag", "mag_err", "logphi", "uperr", 
                               "downerr", "log_err", "bad_prob", "is_bad"])

class selmap:
    """
    Selection Map Class.

    This class represents a selection map for quasars and provides methods to read,
    process, and analyze the selection map data. It is used in conjunction with the
    Quasar Luminosity Function (QLF) class to compute the number of quasars in a given
    redshift range and area.

    Methods
    -------
    - __init__(self, x, zlims=None)
        Constructor - Initializes the selection map object with selection map file, area, and sample id.

    - nqso(self, lumfn, theta)
        Computes the number of quasars in the selection map for a given luminosity function and parameters.

    Attributes
    ----------
    - label : str
        Label of the selection map.
    - sid : int
        Sample ID of the selection map.
    - area : float
        Area of the selection map in square degrees.
    - z : ndarray
        Redshift values of quasars in the selection map.
    - m : ndarray
        Magnitudes of quasars in the selection map.
    - p : ndarray
        Selection probabilities of quasars in the selection map.
    - volarr : ndarray
        Comoving volume of the selection map in cMpc^3.
    - dz : ndarray
        Redshift intervals for quasars in the selection map.
    - dm : ndarray
        Magnitude intervals for quasars in the selection map.
    """

    def __init__(self, x, zlims=None):
        """
        Constructor - Initialises the selection map object with 
                      selection map file, area, and sample id.

        Parameters
        ----------
        - x : tuple
            A tuple containing the selection map file, area, sample id, and label.
        - zlims : list of float, optional
            The redshift limits for the selection map. 
            The default is None.

        Returns
        -------
        None

        Notes
        ------
        CHECK!!! Understand the purpose of dz and dm in the context of the selection map.
        CHECK!!! Understand the purpose of z_all, m_all, p_all, dz_all_array, and dm_all_array.
        CHECK!!! Understand the if sample_id == 7 condition and its implications. It seems like it will reset.
        CHECK!!! Understand each cases separately, again, thoroughly!!!

        """

        selection_map_file, area, sample_id, label = x

        self.label = label
        self.sid = sample_id
        
        if sample_id == 7:
            # Set dz and dm for Giallongo's sample.  This sample needs
            # special treatment due to non-uniform selection map grid.
            # Here, we are assuming that np.diff(zlims) is smaller
            # than the delta-z values in Giallongo's selection maps.
            self.dz = np.diff(zlims)
            self.dm = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            with open(selection_map_file, 'r') as f: 
                z, mag, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)
            z_min, z_max = zlims 
            select = ((z>=z_min) & (z<z_max))
            self.dm = self.dm[select]
            
        self.z_all, self.m_all, self.p_all, self.dz_all_array, self.dm_all_array = getselfn(selection_map_file, zlims=zlims)

        self.z = self.z_all
        self.m = self.m_all
        self.p = self.p_all
        self.dz_array = self.dz_all_array
        self.dm_array = self.dm_all_array
    
        select = None

        if sample_id == 7:
            # Correct Giallongo's p values to match published LF.  See
            # comments in giallongo15_sel_correction.dat.
            with open(selection_map_file, 'r') as f: 
                z, mag, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)
            z_min, z_max = zlims 
            select = ((z>=z_min) & (z<z_max))
            with open('Data_new/giallongo15_sel_correction.dat', 'r') as f: 
                corr = np.loadtxt(f, usecols=(4,), unpack=True)
            corr = corr[select]
            self.p_all = self.p_all/corr
            select = ((self.z_all>=z_min) & (self.z_all<z_max))

        # if sample_id == 13: 
        #     select = ((((self.z_all>=0.0) & (self.z_all<0.2) & (self.m_all<=-20.7)) | 
        #                ((self.z_all>=0.2) & (self.z_all<0.4) & (self.m_all<=-20.4)) | 
        #                ((self.z_all>=0.4) & (self.z_all<0.6) & (self.m_all<=-21.3)) | 
        #                ((self.z_all>=0.6) & (self.z_all<0.8) & (self.m_all<=-23.1)) | 
        #                ((self.z_all>=0.8) & (self.z_all<1.0) & (self.m_all<=-23.7)) |
        #                ((self.z_all>=1.0) & (self.z_all<1.2)) |
        #                ((self.z_all>=1.2) & (self.z_all<1.4) & (self.m_all<=-24.3)) |
        #                ((self.z_all>=1.4) & (self.z_all<1.6)) |
        #                ((self.z_all>=1.6) & (self.z_all<1.8) & (self.m_all<=-24.9)) |
        #                ((self.z_all>=1.8) & (self.z_all<2.2))) |
        #               ((self.z_all>=3.5) & (self.z_all<4.7) & (self.m_all<=-26.1)))

        # if sample_id == 15: 
        #     select = (((self.z_all>=0.4) & (self.z_all<0.6)) | 
        #               ((self.z_all>=0.6) & (self.z_all<0.8) & (self.m_all<=-20.7)) | 
        #               ((self.z_all>=0.8) & (self.z_all<1.2) & (self.m_all<=-21.9)) |
        #               ((self.z_all>=1.2) & (self.z_all<1.8) & (self.m_all<=-22.5)) |
        #               ((self.z_all>=1.8) & (self.z_all<2.2) & (self.m_all<=-23.1)))
            
        if sample_id == 8:
            # Restrict McGreer's samples to faint quasars to avoid
            # overlap with Yang.
            select = (self.m_all>-26.73)

        if self.z_all.size == 0:
            return # This selmap has no points in zlims

        self.z = np.squeeze(self.z_all[select])
        self.m = np.squeeze(self.m_all[select])
        self.p = np.squeeze(self.p_all[select])
        self.dz_array = np.squeeze(self.dz_all_array[select])
        self.dm_array = np.squeeze(self.dm_all_array[select])

        # Just two aliases for older parts of the code 
        self.dz = self.dz_array
        self.dm = self.dm_array

        if self.z.size == 0:
            return # This selmap has no points in zlims

        self.area = area
        self.volarr = volume(self.z, self.area)*self.dz_array
        self.volarr_all = volume(self.z_all, self.area)*self.dz_all_array

        # self.z = self.z_all
        # self.m = self.m_all
        # self.p = self.p_all
        # self.dz_array = self.dz_all_array
        # self.dm_array = self.dm_all_array

        return

    def nqso(self, lumfn, theta):
        """
        Compute the number of quasars in the selection map for a given luminosity function and parameters.

        Parameters
        ----------
        - lumfn : object
            The luminosity function object.
        - theta : ndarray
            The parameter vector containing the QLF parameters.

        Returns
        -------
        - n : float
            The number of quasars in the selection map.
        """

        try: 
            psi = 10.0**lumfn.log10phi(theta, self.m)
            # Except for Giallongo's sample, self.dm is assumed to be
            # constant here; may not be true.
            tot = psi*self.p*self.volarr*self.dm_array
            return np.sum(tot)
        except(AttributeError):
            # For Checking Purposes
            import sys
            sys.exit("AttributeError in nqso: " + str(lumfn) + " " + str(theta) + "\n" +
                     "self.m = " + str(self.m) + "\n" +
                     "self.p = " + str(self.p) + "\n" +
                     "self.volarr = " + str(self.volarr) + "\n" +
                     "self.dm_array = " + str(self.dm_array))
            return 0 
            
class lf:
    """
    Quasar Luminosity Function (QLF) Class.

    This class represents the Quasar Luminosity Function (QLF) and provides methods 
    to compute, fit, and analyze the QLF using quasar data and selection maps. 
    It supports both parametric modeling and binned analysis of the QLF.

    Methods
    -------
    - __init__(self, quasar_files=None, selection_maps=None, zlims=None)
        Constructor - Initializes the QLF object with quasar data files, selection maps, 
        and optional redshift limits.

    - log10phi(self, theta, mag)
        Computes the logarithm (base 10) of the QLF for given parameters and magnitudes.
        The QLF is modeled using a double power law.

    - lfnorm(self, theta)
        Computes the total number of quasars predicted by the QLF model in the survey volume.

    - neglnlike(self, theta)
        Computes the negative log-likelihood for the QLF model.

    - bestfit(self, guess, method='Nelder-Mead')
        Finds the best-fit parameters for the QLF model using optimization.

    - run_mcmc(self)
        Runs Markov Chain Monte Carlo (MCMC) sampling to estimate the posterior distribution 
        of the QLF parameters.

    - get_percentiles(self)
        Computes 1-sigma percentiles for the QLF parameters from the MCMC samples.

    - draw(self, z_plot, composite=None, dirname='', plotlit=False)
        Plots the QLF data, best-fit model, and posterior samples.

    - get_lf(self, sid, z_plot)
        Computes the binned QLF for a specific sample ID and redshift.

    - plot_literature(self, ax, z_plot)
        Plots QLF data from the literature for comparison.

    Attributes
    ----------
    - z : ndarray
        Redshift of selected quasars in the current redshift bin.
    - M1450 : ndarray
        Absolute selected magnitudes of quasars in the current redshift bin.
    - p : ndarray
        Selection selected probabilities of quasars in the current redshift bin.
    - area : ndarray
        Survey area corresponding to the selected quasars in the current redshift bin.
    - sid : ndarray
        Sample IDs of selected quasars in the current redshift bin.
    - z_all : ndarray
        Redshift of all quasars in the current redshift bin.
    - M1450_all : ndarray
        Absolute magnitudes of all quasars in the current redshift bin.
    - p_all : ndarray
        Selection probabilities of all quasars in the current redshift bin.
    - area_all : ndarray
        Survey area corresponding to all quasars in the current redshift bin.
    - sid_all : ndarray
        Sample IDs of all quasars in the current redshift bin.
    - zlims : list of float, optional
        The redshift limits for the quasar luminosities. If provided, it is used to filter data and set the `dz` attribute.
    - dz : float
        Redshift interval for the current redshift bin. This attribute is only set if 
        `zlims` is provided during initialization.
    - maps : list
        List of selection map objects corresponding to the surveys used in the analysis.
    - logphi_data : ndarray
        Logarithm (base 10) of the quasar luminosity function for each Quasar data, 
        computed from the selection maps.

    Notes
    -----
    - This class is designed to work with quasar data and selection maps to compute 
      the QLF in specific redshift bins.
    - The QLF is modeled using a double power law, with parameters such as `phi_star`, 
      `M_star`, `alpha`, and `beta`.
    """

    def __init__(self, quasar_files=None, selection_maps=None, zlims=None):
        """
        Constructor - Initializes the QLF object with quasar data files, selection maps, 
        and optional redshift limits.

        Parameters
        ----------
        - quasar_files : list of str, optional
            List of quasar files to be read. The default is None.
        - selection_maps : list of str, optional
            List of selection map files to be read. The default is None.
        - zlims : list of float, optional
            The redshift limits for the quasar luminosities.

        Returns
        -------
        None

        Notes
        -----
        - If `zlims` is provided, the `dz` attribute is set to the difference between 
          the upper and lower redshift limits (`zlims[1] - zlims[0]`).
        - If `quasar_files` or `selection_maps` is None, the object will not contain 
          any data or selection maps.

        CHECK!!! How can quasar_files be None?
        CHECK!!! zlims and dz too!!!
        """

        self.zlims = zlims

        for datafile in quasar_files:
            (z, m, p, area, sid,
             z_all, m_all, p_all,
             area_all, sid_all) = getqlums(datafile, zlims=zlims)

            try:
                self.z=np.append(self.z,z)
                self.M1450=np.append(self.M1450,m)
                self.p=np.append(self.p,p)
                self.area=np.append(self.area,area)
                self.sid=np.append(self.sid,sid)
            except(AttributeError):
                self.z=z
                self.M1450=m
                self.p=p
                self.area=area
                self.sid=sid

            try:
                self.z_all=np.append(self.z_all,z_all)
                self.M1450_all=np.append(self.M1450_all,m_all)
                self.p_all=np.append(self.p_all,p_all)
                self.area_all=np.append(self.area_all,area_all)
                self.sid_all=np.append(self.sid_all,sid_all)
            except(AttributeError):
                self.z_all=z_all
                self.M1450_all=m_all
                self.p_all=p_all
                self.area_all=area_all
                self.sid_all=sid_all
                
        if zlims is not None:
            self.dz = zlims[1]-zlims[0]

        self.maps = [selmap(x, zlims) for x in selection_maps]

        # Remove selection maps that lie outside our redshift range.
        self.maps = [x for x in self.maps if x.z.size > 0]

        # Don't want a selmap if there are no qsos from that survey in
        # this redshift bin.  
        samples = set(np.unique(self.sid))
        self.maps = [x for x in self.maps if x.sid in samples]


        # self.sid = self.sid_all
        # self.z = self.z_all
        # self.M1450 = self.M1450_all
        # self.p = self.p_all
        # self.area = self.area_all

        return

    def log10phi(self, theta, mag):
        """
        Compute the logarithm (base 10) of the QLF for given parameters and magnitudes.
        The QLF is modeled using a double power law.
        
        Parameters
        ----------
        - theta : ndarray
            The parameter vector containing the QLF parameters.
        - mag : ndarray
            The absolute magnitudes of the quasars.
            
        Returns
        -------
        - logphi : ndarray
            The logarithm (base 10) of the QLF for the given parameters and magnitudes.
        """

        log10phi_star, M_star, alpha, beta = theta 

        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log10(phi)

    def lfnorm(self, theta):
        """
        Compute the total number of quasars predicted by the QLF model in the survey volume.
        This is done by integrating the QLF over the survey volume and the selection maps.

        Parameters
        ----------
        - theta : ndarray
            The parameter vector containing the QLF parameters.

        Returns
        -------
        - ns : float
            The total number of quasars predicted by the QLF model in the survey volume.

        """

        ns = np.array([x.nqso(self, theta) for x in self.maps])
        return sum(ns) 

    def neglnlike(self, theta):
        """
        Compute the negative log-likelihood for the QLF model by calculating the 
        log-likelihood of the observed data given the QLF model.

        Parameters
        ----------
        - theta : ndarray
            The parameter vector containing the QLF parameters.
        Returns
        -------
        - nll : float
            The negative log-likelihood for the QLF model.
        """

        logphi = self.log10phi(theta, self.M1450) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 

        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        """
        Find the best-fit parameters for the QLF model using optimization.
        This function uses the `scipy.optimize.minimize` function to minimize the
        negative log-likelihood function.

        Parameters
        ----------
        - guess : ndarray
            Initial guess for the QLF parameters.
        - method : str, optional
            The optimization method to use. The default is 'Nelder-Mead'.
        
        Returns
        -------
        - result : OptimizeResult
            The optimization result represented as a `scipy.optimize.OptimizeResult` object.
            The `x` attribute of the result contains the best-fit parameters.
        """

        result = op.minimize(self.neglnlike,
                             guess,
                             method=method,
                             options={'maxfev': 8000,
                                      'maxiter': 8000,
                                      'disp': False})

        if not result.success:
            print( 'Likelihood optimisation did not converge.')

        self.bf = result 
        return result
    
    def create_param_range(self):
        """
        Create parameter ranges for optimization.

        This function calculates the minimum and maximum values for the parameters
        based on the best-fit parameters (`self.bf.x`). The minimum values are set
        to half of the best-fit values, and the maximum values are set to twice the
        best-fit values. A redundant assertion is included to ensure that the 
        minimum values are always less than the maximum values.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes
        ----------
        self.prior_min_values : ndarray
            The minimum values for the parameters, calculated as half of the best-fit values.
        self.prior_max_values : ndarray
            The maximum values for the parameters, calculated as twice the best-fit values.

        """

        half = self.bf.x/2.0
        double = 2.0*self.bf.x
        self.prior_min_values = np.where(half < double, half, double) 
        self.prior_max_values = np.where(half > double, half, double)
        assert(np.all(self.prior_min_values < self.prior_max_values))

        # MAP value for zls = (0.4, 0.6)
        # self.prior_min_values = np.array([-7.9247152, -8.90048553, -1.24158352, -0.30304404]) - 0.2
        # self.prior_max_values = np.array([-7.9247152, -8.90048553, -1.24158352, -0.30304404]) + 0.2

        return

    def _lnprior(self, theta):
        """
        Checks if the parameters are within the prior bounds.

        Parameters
        ----------
        theta : ndarray
            The parameter vector to be evaluated.

        Returns
        -------
        float
            The log prior probability. Returns 0.0 if theta is within the prior
            bounds, and -np.inf if theta is outside the prior bounds.
        """

        if (np.all(theta < self.prior_max_values) and
            np.all(theta > self.prior_min_values)):
            return 0.0 

        return -np.inf
    
    def _lnprob(self, theta):
        """
        Compute the log-probability of the QLF model given the parameters.

        Parameters
        ----------
        theta : ndarray
            The parameter vector to be evaluated.

        Returns
        -------
        float
            The log-probability of the QLF model given the parameters. Returns -np.inf
            if the log prior is not finite.
        """
        time.sleep(0.01)  # Simulate some CPU time for testing purposes
        lp = self._lnprior(theta)
        
        if not np.isfinite(lp):
            return -np.inf

        return lp - self.neglnlike(theta)


    def run_mcmc(self, ncores=1):
        """
        Run Markov Chain Monte Carlo (MCMC) sampling to estimate the posterior distribution
        of the QLF parameters.
        This function initializes the MCMC sampler, runs the sampling process, and stores
        the samples in the `self.samples` attribute.

        Parameters
        ----------
        - ncores : int
            The number of CPU cores to use for parallel processing. This allows the MCMC
            sampling to be performed in parallel, speeding up the computation.

        Returns
        -------
        None

        """
        
        self.ndim, self.nwalkers = self.bf.x.size, 50
        self.mcmc_start = self.bf.x 
        pos = [self.mcmc_start + 1e-4*np.random.randn(self.ndim) for i
               in range(self.nwalkers)]

        print("\n\n\tNumber of cores available for MCMC: ", ncores)

        pool = Pool(ncores)
        print("Using multiprocessing pool with {} cores...\n\n\n".format(ncores))

        lnprob_pickable = dill.loads(dill.dumps(self._lnprob))
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                            lnprob_pickable, pool=None)
        print("Running MCMC with {} walkers and {} dimensions...".format(self.nwalkers, self.ndim))
        self.sampler.run_mcmc(pos, 1500, progress=True)

        self.samples = self.sampler.chain[:, 500:, :].reshape((-1, self.ndim))
        
        return
    
    def find_Pb_Yb_Vb(self):
        Pb = np.random.uniform(0.0, 1.0, size=self.nwalkers_with_bp)

        mean = self.bf.x[0]
        Yb = np.random.normal(loc=mean, scale=2, size=self.nwalkers_with_bp)

        # # Hand picking alpha and beta for gamma distribution
        # # to get mean around 10 and Variance around 20
        # alpha, beta = 5, 2

        # Hand picking alpha and beta for gamma distribution
        # to get mean around 2 and Variance around 2
        alpha, beta = 2, 1
        Vb = np.random.gamma(shape=alpha, scale=beta, size=self.nwalkers_with_bp)

        print("shape = ", np.array([Pb, Yb, Vb]).T.shape)

        return np.array([Pb, Yb, Vb]).T
    
    def get_qlf_data(self):

        data = []
        for sid in np.unique(self.sid):
            if sid==6:
                # Glikman's sample needs wider bins.
                mbins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
            elif sid == 7:
                mbins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
            elif sid == 10 or sid == 18:
                mbins = np.arange(-30.9, -17.3, 1.8)
            else:
                mbins = np.arange(-30.9, -13.3, 0.6)

            

            m = self.M1450[self.sid == sid]
            selmaps = [x for x in self.maps if x.sid == sid]
            
            Veff = np.array([drawlf.totBinVol(self, mi, mbins, selmaps) for mi in m])
            mask = Veff > 0

            Veff = Veff[mask]
            m = m[mask]

            # Estimates the number density of quasars in the given magnitude range
            # using the effective volume (Veff) for each bin.
            h, edges = np.histogram(m, bins=mbins, weights=1.0/Veff)

            logphi = np.log10(h)
            mag = 0.5 * (edges[1:] + edges[:-1])
            mag_err = 0.5 * (edges[1:] - edges[:-1])


            # Calculate errorbars on our binned LF.  These have been estimated
            # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
            # implemented in astropy.stats.poisson_conf_interval.  The
            # interval='frequentist-confidence' option to that astropy function is
            # exactly equal to the Gehrels formulas, although the documentation
            # does not say so.

            # Estimates the number density of quasars in the given magnitude range
            # without using the effective volume (Veff) for each bin.
            n, _ = np.histogram(m, bins=mbins)
            nlims = pci(n,interval='frequentist-confidence')
            nlims *= h/n 
            uperr = np.log10(nlims[1]) - logphi 
            downerr = logphi - np.log10(nlims[0])

            log_err = 0.5 * (uperr + downerr)

            mask = np.isfinite(logphi)
            logphi = logphi[mask]
            uperr = uperr[mask]
            downerr = downerr[mask]
            log_err = log_err[mask]
            mag = mag[mask]
            mag_err = mag_err[mask]

            sid_arr = np.full_like(logphi, sid, dtype=int)

            data.append(LFData(sid=sid_arr, logphi=logphi, uperr=uperr, downerr=downerr, log_err=log_err,
                           mag=mag, mag_err=mag_err, is_bad=np.zeros_like(sid_arr, dtype=bool), 
                           bad_prob=np.zeros_like(sid_arr, dtype=float)))


        all_sid = np.concatenate([d.sid for d in data])
        all_logphi = np.concatenate([d.logphi for d in data])
        all_uperr = np.concatenate([d.uperr for d in data])
        all_downerr = np.concatenate([d.downerr for d in data])
        all_log_err = np.concatenate([d.log_err for d in data])
        all_mag = np.concatenate([d.mag for d in data])
        all_mag_err = np.concatenate([d.mag_err for d in data])

        print("\n\nlen(all_sid) = ", len(all_sid))

        self.data = LFData(
            sid=all_sid,
            logphi=all_logphi,
            uperr=all_uperr,
            downerr=all_downerr,
            log_err=all_log_err,
            mag=all_mag,
            mag_err=all_mag_err,
            is_bad=np.zeros_like(all_sid, dtype=bool),  # Initialize is_bad as False
            bad_prob=np.zeros_like(all_sid, dtype=float)  # Initialize bad_prob as 0.0
        )

        return
    
    def _lnprior_with_bad_points(self, params):
        logphi, M_star, alpha, beta = params[:4]
        Pb, Yb, Vb = params[4:]

        if np.any(params[:4] < self.prior_min_values) or np.any(params[:4] > self.prior_max_values):
            return -np.inf

        if self.prior_tag == 1:
            if Pb < 0 or Pb > 1:
                return -np.inf
            if Vb <= 0 or Vb > 10:
                return -np.inf
            if Yb < -50 or Yb > 20:
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
    
    def plot_chains_and_corner(self, dirname='', prior_tag=1, run_counter=0):
        # Print parameter medians and 1-sigma intervals
        param_names = [r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$', r'$P_b$', r'$Y_b$', r'$V_b$']
        self.cred_interval = []
        for i, name in enumerate(param_names):
            vals = percentiles(self.samples_with_bp[:, i])
            print(f"{name}: median = {vals[2]:.4f}, -1σ = {vals[0]:.4f}, +1σ = {vals[1]:.4f}")
            self.cred_interval.append(vals[:2])
        self.cred_interval = np.array(self.cred_interval)


        fig = self.corner_quantile(
            self.samples_with_bp,
            central_fraction=0.8,
            labels=[r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$', r'$P_b$', r'$Y_b$', r'$V_b$'],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],
        )
        fig.suptitle(f"Prior Model: {prior_tag}\nrun_counter: {run_counter}", fontsize=16)
        # fig.savefig(f"{dirname}mcmc_{np.mean(self.z)}_{self.prior_tag}_checking_corner_with_bad_points_{np.mean(self.z):.3f}.png")
        plt.savefig(f"{dirname}Corner_{np.mean(self.z):.4f}_{self.prior_tag}_with_bad_points_{run_counter}.png")

        # Plot MCMC chains for all parameters with bad points
        fig, axes = plt.subplots(self.ndim_with_bp, 1, figsize=(12, 2 * self.ndim_with_bp), sharex=True)
        param_names = [r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$', r'$P_b$', r'$Y_b$', r'$V_b$']
        for i, name in enumerate(param_names):
            ax = axes[i]
            for walker in range(self.nwalkers_with_bp):
                ax.plot(self.sampler_with_bp.chain[walker, :, i], alpha=0.1)
            median = np.median(self.samples_with_bp[:, i])
            ax.axhline(median, color='red', linestyle='--', label='Median')
            ax.set_ylabel(name)
            if i == 0:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel('step')
        plt.tight_layout()
        plt.savefig(f"{dirname}Chains_{np.mean(self.z):.4f}_{self.prior_tag}_with_bad_points.png")



    def run_mcmc_with_bad_points(self, ncores, dirname='', prior_tag=1):

        self.prior_tag = prior_tag

        self.ndim_with_bp, self.nwalkers_with_bp = self.bf.x.size + 3, 25
        self.mcmc_start = self.bf.x 
        # If due to one bad point in the faint end, if the best fit has beta > 0,
        # manually setting the guess for beta to be negative.
        if self.bf.x[3] > 0:
            self.mcmc_start[3] = -2.0

        # self.mcmc_start = [-5.7, -21.3, -2.74, -1.0]

        pos_with_bp = np.hstack((np.array([self.bf.x 
                    + 1e-2*np.random.randn(self.bf.x.size) for i 
                    in range(self.nwalkers_with_bp)]), self.find_Pb_Yb_Vb()))
        
        self.get_qlf_data()

        # print("Len of self.data = ", len(self.data.logphi))
        # import sys
        # sys.exit()

        print("\n\n\tNumber of cores available for MCMC: ", ncores)

        # print("len(self.sid) = ", len(self.sid))
        # print("len(sid_all) = ", len(self.sid_all))
        # print("self.sid==self.sid_all = ", np.all(self.sid==self.sid_all))

        if ncores <= 1:
            pool = None
            print("Not using parallel processing...\n\n\n")
        else:
            pool = Pool(ncores)
            print("Using parallel processing with {} cores...\n\n\n".format(ncores))


        lnposterior_pickable = dill.loads(dill.dumps(self._lnposterior_with_bad_points))
        self.sampler_with_bp = emcee.EnsembleSampler(self.nwalkers_with_bp, self.ndim_with_bp,
                                            lnposterior_pickable, pool=pool)
        print("Running MCMC with {} walkers and {} dimensions...".format(self.nwalkers_with_bp, self.ndim_with_bp))


        DISCARD = 5000

        self.sampler_with_bp.run_mcmc(pos_with_bp, DISCARD, progress=True)
        self.sampler_with_bp.reset()
        self.sampler_with_bp.run_mcmc(None, 100000, progress=True)
        
        self.samples_with_bp = self.sampler_with_bp.get_chain(flat=True)


        # Print autocorrelation time and acceptance rate for the sampler with bad points

        from emcee.autocorr import AutocorrError
        try:
            tau = self.sampler_with_bp.get_autocorr_time()
        except AutocorrError as e:
            print("Could not compute reliable autocorrelation time:", e)
            tau = e.tau 
            warning_msg = str(e)

        print("tau:", tau)
        acceptance_fraction = self.sampler_with_bp.acceptance_fraction
        print("Mean acceptance fraction:", np.mean(acceptance_fraction))
        print("Acceptance fraction per walker:", acceptance_fraction)

        self.plot_chains_and_corner(dirname=dirname, prior_tag=prior_tag)

        # # Plot MCMC chains for each parameter separately and save
        # param_names = [r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$', r'$P_b$', r'$Y_b$', r'$V_b$']
        # for i, name in enumerate(param_names):
        #     fig, ax = plt.subplots(figsize=(12, 3))
        #     for walker in range(self.nwalkers_with_bp):
        #         ax.plot(self.sampler_with_bp.chain[walker, :, i], alpha=0.1)
        #     median = np.median(self.samples_with_bp[:, i])
        #     ax.axhline(median, color='red', linestyle='--', label='Median')
        #     ax.set_ylabel(name)
        #     ax.set_xlabel('step')
        #     ax.legend(fontsize=8)
        #     plt.tight_layout()
        #     # plt.savefig(f"{dirname}mcmc_{np.mean(self.z)}_{self.prior_tag}_checking_chains_{i}_with_bad_points_{np.mean(self.z):.3f}.png")
        #     plt.savefig(f"{dirname}Chains_{np.mean(self.z)}_{self.prior_tag}_Individual_{i}_with_bad_points.png")
        #     plt.close(fig)


        if np.any(np.isnan(tau)):
            print("Warning: Autocorrelation time contains NaN values. This may indicate convergence issues.")
            with open(f"{dirname}nan_flagged", "w") as f:
                f.write("Autocorrelation time contains NaN values. MCMC flagged as problematic.\n")
            import sys
            sys.exit("Exiting due to NaN in autocorrelation time.")

        # print("\n\ntau:", tau)
        # print("self.sampler_with_bp.get_chain().shape[0] / 50:", self.sampler_with_bp.get_chain().shape[0] / 50)
        # print("tau > self.sampler_with_bp.get_chain().shape[0] / 50:", tau > self.sampler_with_bp.get_chain().shape[0] / 50)
        # if np.any(tau > self.sampler_with_bp.get_chain().shape[0] / 50):
        #     print("\n\n\nRecomputing MCMC with more steps...\n\n\n")
        #     self.sampler_with_bp.reset()
        #     self.sampler_with_bp.run_mcmc(None, 60000, progress=True)
        #     self.samples_with_bp = self.sampler_with_bp.get_chain(flat=True)

        #     self.plot_chains_and_corner(dirname=dirname, prior_tag=prior_tag, run_counter=1)

        #     # Print autocorrelation time and acceptance rate for the sampler with bad points
        #     try:
        #         tau = self.sampler_with_bp.get_autocorr_time()
        #     except AutocorrError as e:
        #         print("Could not compute reliable autocorrelation time:", e)
        #         tau = e.tau 
        #         warning_msg = str(e)

        #     print("Autocorrelation time (per parameter):", tau)
        #     acceptance_fraction = self.sampler_with_bp.acceptance_fraction
        #     print("Mean acceptance fraction:", np.mean(acceptance_fraction))
        #     print("Acceptance fraction per walker:", acceptance_fraction)



        samples_4d = self.samples_with_bp[:, :4]

        bins = 20
        hist, edges = np.histogramdd(samples_4d, bins=bins)

        max_idx = np.unravel_index(np.argmax(hist), hist.shape)

        map_params = []
        for i in range(4):
            # Bin edges for this dimension
            bin_edges = edges[i]
            # Center of the bin
            center = 0.5 * (bin_edges[max_idx[i]] + bin_edges[max_idx[i]+1])
            map_params.append(center)
        map_params = np.array(map_params)
        print("MAP (marginalized over nuisance):", map_params)

        # import sys
        # from datetime import datetime
        # sys.exit(f"\nQuitting for testing purposes\nprior tag = {self.prior_tag}\nTime right now = {datetime.now()}\n")
        
        self.samples = self.samples_with_bp[:, :4]
        self.marginalise_and_find_MAP()
        self.savedata_with_bp()

        return
    
    def clear_samples(self):
        """
        Clear the stored samples and reset the sampler.
        
        This method clears the `samples` attribute and resets the `sampler` attribute
        to None, effectively removing any previously stored MCMC samples.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.samples = None
        self.sampler = None
        self.sampler_with_bp = None
        self.samples_with_bp = None
        return
    

    def corner_quantile(self, samples, central_fraction=0.8, **kwargs):
        lower_q = (1 - central_fraction) / 2
        upper_q = 1 - lower_q
        bounds = [
            (np.quantile(samples[:, i], lower_q), np.quantile(samples[:, i], upper_q))
            for i in range(samples.shape[1])
        ]
        # This is for checking purpose.
        # Variance alone is checked from the left tail.
        bounds[-1] = (np.min(samples[:, -1]), np.quantile(samples[:, -1], upper_q))
        return corner.corner(samples, range=bounds, **kwargs)
    
    def marginalise_and_find_MAP(self):
        samples_4d = self.samples_with_bp[:, :4]

        bins = 20
        hist, edges = np.histogramdd(samples_4d, bins=bins)

        max_idx = np.unravel_index(np.argmax(hist), hist.shape)

        map_params = []
        for i in range(4):
            # Bin edges for this dimension
            bin_edges = edges[i]
            # Center of the bin
            center = 0.5 * (bin_edges[max_idx[i]] + bin_edges[max_idx[i]+1])
            map_params.append(center)
        self.map_params = np.array(map_params)
        print("MAP (marginalized over nuisance):", self.map_params)



    def savedata_with_bp(self):
        if self.zlims == (0.1, 0.4):
            with open("parameters_with_bp.dat", "w") as f:
                f.write("# The parameters of the QLF with bad points are given here.\n")
                f.write("# The columns are as follows:\n")
                f.write("#                   Values                    |                                   Credibility Interval\n")
                f.write("# phi_star     M_star     alpha     beta      |           phi_star              M_star             alpha             beta    \n")

        with open("parameters_with_bp.dat", "a") as f:
            f.write("{:9.4f}   {:11.4f}   {:7.4f}   {:7.4f}   |       {:7.4f} {:7.4f}     {:7.4f} {:7.4f}   {:7.4f} {:7.4f}   {:7.4f} {:7.4f}\n".format(
                self.map_params[0], self.map_params[1], self.map_params[2], self.map_params[3],
                self.cred_interval[0][0], self.cred_interval[0][1], self.cred_interval[1][0], self.cred_interval[1][1],
                self.cred_interval[2][0], self.cred_interval[2][1], self.cred_interval[3][0], self.cred_interval[3][1]
            ))
        return




    def get_percentiles(self):
        """
        Calculate the 1-sigma percentile errors for the luminosity function (LF) parameters.

        This method computes the 1-sigma percentiles for the LF parameters 
        (phi_star, M_star, alpha, and beta) based on the provided samples. 
        The computed percentiles are stored as attributes of the object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.phi_star = percentiles(self.samples[:,0])
        self.M_star = percentiles(self.samples[:,1])
        self.alpha = percentiles(self.samples[:,2])
        self.beta = percentiles(self.samples[:,3])

        return 

    def corner_plot(self, labels=[r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$'], dirname=''):
        """
        Create a corner plot of the posterior samples.

        Parameters
        ----------
        - labels : list of str, optional
            Labels for the parameters. The default is [r'$\\phi_*$', r'$M_*$', r'$\\alpha$', r'$\\beta$'].
        - dirname : str, optional
            Directory name to save the plot. The default is ''.

        Returns
        -------
        None
        """

        mpl.rcParams['font.size'] = '14'
        self.medians = np.median(self.samples_with_bp, axis=0)
        f = corner.corner(self.samples_with_bp, labels=labels, truths=self.medians)
        plotfile = dirname+'triangle.png'
        f.savefig(plotfile)
        mpl.rcParams['font.size'] = '22'

        return
    
    def plot_chains(self, fig, param, ylabel):
        """
        Plot the MCMC chains for a given parameter.
        
        Parameters
        ----------
        - fig : matplotlib.figure.Figure
            The figure object to plot on.
        - param : int
            The index of the parameter to plot.
        - ylabel : str
            The label for the y-axis.

        Returns
        -------
        None
        """

        ax = fig.add_subplot(self.bf.x.size, 1, param+1)
        for i in range(self.nwalkers): 
            ax.plot(self.sampler.chain[i,:,param], c='k', alpha=0.1)
        self.medians = np.median(self.samples, axis=0)
        ax.axhline(self.medians[param], c='#CC9966', dashes=[7,2], lw=2) 
        ax.set_ylabel(ylabel)
        if param+1 != self.bf.x.size:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('step')
            
        return 

    def chains(self, labels=[r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$'], dirname=''):
        """
        Plot the MCMC chains for all parameters.

        Parameters
        ----------
        - labels : list of str, optional
            Labels for the parameters. The default is [r'$\\phi_*$', r'$M_*$', r'$\\alpha$', r'$\\beta$'].
        - dirname : str, optional
            Directory name to save the plot. The default is ''.

        Returns
        -------
        None
        """

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
    
    def plot_posterior_sample_lfs(self, ax, mags, **kwargs):
        """
        Plot posterior sample luminosity functions.
        
        Parameters
        ----------
        - ax : matplotlib.axes.Axes
            The axes object to plot on.
        - mags : ndarray
            The absolute magnitudes for which to compute the luminosity function.
        - kwargs : keyword arguments
            Additional keyword arguments for plotting (e.g., color, linestyle).

        Returns
        -------
        None    
        """

        random_thetas = self.samples[np.random.randint(len(self.samples), size=300)]
        for theta in random_thetas:
            phi_fit = self.log10phi(theta, mags)
            ax.plot(mags, phi_fit, **kwargs)

        return

    def plot_bestfit_lf(self, ax, mags, **kwargs):
        """
        Plot the best-fit luminosity function.
        
        Parameters
        ----------
        - ax : matplotlib.axes.Axes
            The axes object to plot on.
        - mags : ndarray
            The absolute magnitudes for which to compute the luminosity function.
        - kwargs : keyword arguments
            Additional keyword arguments for plotting (e.g., color, linestyle).
            
        Returns
        -------
        None
        """

        phi_fit = self.log10phi(self.bf.x, mags)
        ax.plot(mags, phi_fit, **kwargs)
        ax.plot(mags, phi_fit, lw=1, c='k', zorder=kwargs['zorder'])

        return

    def quasar_volume(self, sample_id):
        """
        Calculate the volume of a quasar sample.

        Parameters
        ----------
        - sample_id : int
            The sample ID for which to calculate the volume.
        
        Returns
        -------
        - volume : float
            The volume of the quasar sample in cMpc^3.

        Notes
        -----
        CHECK!!! This function is not used in the code.
        """

        smap = [x for x in self.maps if x.sid == sample_id]

        return smap[0].volume # cMpc^3

    def binVol(self, selmap, mrange, zrange):
        """
        Calculate volume in an M-z bin for *one* selmap.

        Parameters
        ----------
        - selmap : selmap object
            The selection map object for which to calculate the volume.
        - mrange : tuple of float
            The magnitude range for the bin.
        - zrange : tuple of float
            The redshift range for the bin.

        Returns
        -------
        - v : float
            The volume in the M-z bin for the selection map.
        """

        v = 0.0
        for i in range(selmap.m.size):
            if (selmap.m[i] >= mrange[0]) and (selmap.m[i] < mrange[1]):
                if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                    if selmap.sid == 7:
                        v += selmap.volarr[i]*selmap.p[i]*selmap.dm[i]
                    else:
                        v += selmap.volarr[i]*selmap.p[i]*selmap.dm
        print("v = ", v, " for selmap sid = ", selmap.sid, " mrange = ", mrange, " zrange = ", zrange)
        print("v.shape = ", np.shape(v))
        return v

    def totBinVol(self, m, mbins, selmaps):
        """
        Calculate the total volume in an M-z bin for *all* selmaps.
        This is done by summing the volumes for each selection map.

        Parameters
        ----------
        - m : float
            The magnitude for which to calculate the volume.
        - mbins : ndarray
            The magnitude bins for the binning.
        - selmaps : list of selmap objects
            The selection map objects for which to calculate the volume.

        Returns
        -------
        - total_vol : float
            The total volume in the M-z bin for all selection maps.
        """

        idx = np.searchsorted(mbins, m)
        mlow = mbins[idx-1]
        mhigh = mbins[idx]
        mrange = (mlow, mhigh)

        v = np.array([self.binVol(x, mrange, self.zlims) for x in selmaps])
        total_vol = v.sum() 

        return total_vol
    
    def get_lf(self, sid, z_plot):
        """
        Calculate the binned luminosity function for a given sample ID and redshift.
        This function computes the binned luminosity function using the selection maps
        and the quasar data. It returns the binned magnitudes, left and right errors,
        logarithm of the luminosity function, and upper and lower errors.

        Parameters
        ----------
        - sid : int
            The sample ID for which to calculate the binned luminosity function.
        - z_plot : float
            The redshift at which to calculate the binned luminosity function.

        Returns
        -------
        - mags : ndarray
            The binned magnitudes.
        - left : ndarray
            The left error bars for the binned luminosity function.
        - right : ndarray
            The right error bars for the binned luminosity function.
        - logphi : ndarray
            The logarithm of the binned luminosity function.
        - uperr : ndarray
            The upper error bars for the binned luminosity function.
        - downerr : ndarray
            The lower error bars for the binned luminosity function.
        """

        # Bin data.  This is only for visualisation and to compare
        # with reported binned values.  

        z = self.z[self.sid==sid]
        m = self.M1450[self.sid==sid]
        p = self.p[self.sid==sid]

        selmaps = [x for x in self.maps if x.sid == sid]

        if sid==6:
            # Glikman's sample needs wider bins.
            bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
        else:
            bins = np.arange(-30.9, -17.3, 0.6)
        
        v1 = np.array([self.totBinVol(x, bins, selmaps) for x in m])

        v1_nonzero = v1[np.where(v1>0.0)]
        m = m[np.where(v1>0.0)]

        h = np.histogram(m, bins=bins, weights=1.0/(v1_nonzero))

        nums = h[0]
        mags = (h[1][:-1] + h[1][1:])*0.5
        dmags = np.diff(h[1])*0.5

        left = mags - h[1][:-1]
        right = h[1][1:] - mags
        
        phi = nums
        logphi = np.log10(phi) # cMpc^-3 mag^-1

        # Calculate errorbars on our binned LF.  These have been estimated
        # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
        # implemented in astropy.stats.poisson_conf_interval.  The
        # interval='frequentist-confidence' option to that astropy function is
        # exactly equal to the Gehrels formulas, although the documentation
        # does not say so.
        n = np.histogram(m, bins=bins)[0]
        nlims = pci(n,interval='frequentist-confidence')
        nlims *= phi/n 
        uperr = np.log10(nlims[1]) - logphi 
        downerr = logphi - np.log10(nlims[0])
        
        return mags, left, right, logphi, uperr, downerr

    def plot_literature(self, ax, z_plot):
        """
        Magic number warning: the selection function below is set by hand! 

        Parameters
        ----------
        - ax : matplotlib.axes.Axes
            The axes object to plot on.
        - z_plot : float
            The redshift at which to plot the literature data.

        Returns
        -------
        None

        Notes
        -----
        CHECK!!! This function is called only if `draw` is called with `plotlit=True`.
        """
        
        qlf_file = 'Data/allqlfs.dat'
        (counter, sample, z_bin, z_min, z_max, z_mean, M1450, left, right,
         logphi, uperr, downerr, nqso, Veff, P) = np.loadtxt(qlf_file, unpack=True)
        
        selection = ((z_min<=z_plot) & (z_max>z_plot)) 
        
        def select(a):
            return a[selection]

        z_bin   = select(z_bin)
        z_min   = select(z_min)
        z_max   = select(z_max)
        z_mean  = select(z_mean)
        M1450   = select(M1450)
        left    = select(left)
        right   = select(right) 
        logphi  = select(logphi)
        uperr   = select(uperr)
        downerr = select(downerr)
        nqso    = select(nqso)
        Veff    = select(Veff)
        P       = select(P)
        sample  = select(sample)

        ax.scatter(M1450, logphi, c='#d7191c',
                   edgecolor='None', zorder=301,
                   label=r'reported values')
        
        ax.errorbar(M1450, logphi, ecolor='#d7191c', capsize=0,
                    xerr=np.vstack((left, right)),
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=302)

        return 
        
    def plot_hopkins(self, ax, filename):
        """
        Plot Hopkins et al. (2007) QLF data.
        This function reads the data from the specified file and plots it on the given axes.

        Parameters
        ----------
        - ax : matplotlib.axes.Axes
            The axes object to plot on.
        - filename : str
            The name of the file containing the QLF data.

        Returns
        -------
        None

        Notes
        -----
        CHECK!!! This function is not called in the code.
        """

        with open(filename, 'r') as f:
            M1450, phi = np.loadtxt(f, usecols=(1,4), unpack=True)

        # 0.4 changes from per units log_10(L) to per unit M
        phi = np.log10(phi) - np.log10(0.4)
        ax.plot(M1450, phi, lw=2, c='k', label='Hopkins')

        return

    def draw(self, z_plot, composite=None, dirname='', plotlit=False):
        """
        Plot data, best fit LF, and posterior LFs.
        This function creates a plot of the luminosity function (LF) for a given redshift
        and saves it to a file.

        Parameters
        ----------
        - z_plot : float
            The redshift at which to plot the LF.
        - composite : str, optional
            The name of the composite LF to plot. The default is None.
        - dirname : str, optional
            The directory name to save the plot. The default is ''.
        - plotlit : bool, optional
            Whether to plot literature data. The default is False.

        Returns
        -------
        None
        """
        mpl.rcParams['font.size'] = '22'

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', which='major', length=7, width=1)
        ax.tick_params('both', which='minor', length=3, width=1)

        mag_plot = np.linspace(-30.0,-20.0,num=100) 
        self.plot_posterior_sample_lfs(ax, mag_plot, lw=1,
                                       c='#ffbf00', alpha=0.1, zorder=2) 
        self.plot_bestfit_lf(ax, mag_plot, lw=2,
                             c='#ffbf00', label='fit', zorder=3)

        cs = {13: 'r', 15:'g', 1:'b', 17:'m', 8:'c', 6:'#ff7f0e',
              7:'#8c564b', 18:'#7f7f7f', 10:'#17becf', 11:'r'}

        def dsl(i):
            for x in self.maps:
                if x.sid == i:
                    return x.label
            return

        sids = np.unique(self.sid)
        for i in sids:
            mags, left, right, logphi, uperr, downerr = self.get_lf(i, z_plot)
            ax.scatter(mags, logphi, c=cs[i], edgecolor='None', zorder=4, s=35, label=dsl(i))
            ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)

        if plotlit: 
            self.plot_literature(ax, z_plot) 
            # self.plot_hopkins(ax, 'hopkins_bol_z3.8.dat')
            # self.plot_hopkins(ax, 'hopkins2.dat')
            
        ax.set_xlim(-17.0, -31.0)
        ax.set_ylim(-12.0, -5.0)
        ax.set_xticks(np.arange(-31,-16, 2))

        ax.set_xlabel(r'$M_{1450}$')
        ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

        legend_title = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
        plt.legend(loc='lower left', fontsize=12, handlelength=3,
                   frameon=False, framealpha=0.0, labelspacing=.1,
                   handletextpad=0.4, borderpad=0.2, scatterpoints=1, title=legend_title)

        plottitle = r'${:g}\leq z<{:g}$'.format(self.zlims[0], self.zlims[1]) 
        plt.title(plottitle, size='medium', y=1.01)

        plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)
        plt.savefig(plotfile, bbox_inches='tight')

        plt.close('all') 

        return 

    def get_gammapi_percentiles(self, z, rt=True):
        """
        Calculate photoionization rate posterior mean value and 1-sigma
        percentile.

        """
        if rt:
            rindices = np.random.randint(len(self.samples), size=100)
            g = np.array([np.log10(rtg.gamma_HI(z, self.log10phi, theta,
                                                individual=True))
                          for theta
                          in self.samples[rindices]])
            u = np.percentile(g, 15.87) 
            l = np.percentile(g, 84.13)
            c = np.mean(g)
            self.gammapi = [u, l, c]

            gammafile = 'gammahi_z{0:.3f}'.format(z)
            np.savez(gammafile, z=z, g=g, ulc=self.gammapi) 
            
        else:
            rindices = np.random.randint(len(self.samples), size=300)
            g = np.array([np.log10(gammapi.Gamma_HI(self.log10phi, theta, z,
                                                    fit='individual'))
                          for theta
                          in self.samples[rindices]])
            u = np.percentile(g, 15.87) 
            l = np.percentile(g, 84.13)
            c = np.mean(g)
            self.gammapi = [u, l, c]
            
        return 