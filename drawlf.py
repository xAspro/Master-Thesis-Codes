# Checked once. No implementation of this code directly in here.
print("In drawlf.py")

import numpy as np
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '16'
import matplotlib.pyplot as plt
from astropy.stats import knuth_bin_width  as kbw
from astropy.stats import poisson_conf_interval as pci
from scipy.stats import binned_statistic as bs
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}

xlim = (-12.0, -34.0)
ylim = (-16.0, 0.0)

# xlim = (125, -100)
# ylim = (-125, 125)

# xlim = (25, -50)
# ylim = (-45, 25)

# xlim = (-30, -40)
# ylim = (-5.0, 5.0)

"""Makes LF plots at particular redshifts.  Shows data with
individual and composite models.  This is similar to the draw()
function in individual.py but is more flexible.

"""

def lfsample(theta, n, mlims):
    # print("In drawlf.py lfsample")

    """
    Return n qso magnitudes between mlims[0] and mlims[1] when the LF
    is described by parameters theta.

    """

    mmin = mlims[0]
    mmax = mlims[1]

    def lnprob(x, theta):
        # print("In drawlf.py lfsample lnprob")

        if x < mmax and x > mmin: 
            mag = x 
            log10phi_star, M_star, alpha, beta = theta 
            phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                         10.0**(0.4*(beta+1)*(mag-M_star)))
            return np.log(phi)
        else:
            return -np.inf 
    
    ndim = 1 
    nwalkers = 250
    dm = np.abs(mmin-mmax)
    p0 = (np.random.rand(ndim*nwalkers)*dm + mmin).reshape((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    sample = sampler.flatchain[:,0]

    return np.random.choice(sample, n)
    
def plot_posterior_sample_lfs(lf, ax, maglims, **kwargs):
    """
    Plot the posterior sample LFs in the given magnitude range.

    Parameters
    ----------
    - lf : object
        The luminosity function object containing the samples and methods.
    - ax : matplotlib.axes.Axes
        The axes on which to plot the posterior sample LFs.
    - maglims : tuple
        A tuple containing the minimum and maximum magnitudes for the plot.
    - kwargs : dict
        Additional keyword arguments for plotting, such as color and line width.
    
    Returns
    -------
    - f : matplotlib.collections.PolyCollection
        The filled area representing the posterior sample LFs.

    Notes
    -----
    - This function samples the luminosity function using the provided samples
      and computes the log10 of the phi values for the given magnitudes.
    - It fills the area between the 15.87th and 84.13th percentiles of the
      log10phi values, providing a visual representation of the uncertainty
      in the luminosity function.
    """

    nmags = 100
    mags = np.linspace(*maglims, num=nmags)
    print("\n\n\n\nmags= ", mags,'\n\n\n\n')
    print('\n\n\n\nc= ', kwargs['c'], '\n\n\n\n')
    nsample = 1000
    rsample = lf.samples[np.random.randint(len(lf.samples), size=nsample)]
    rsample = rsample[:, :4]  # Ensure we only take the first 4 parameters
    # print("rsample= ", rsample)
    phi = np.zeros((nsample, nmags))
    
    for i, theta in enumerate(rsample):
        phi[i] = lf.log10phi(theta, mags)
        # phi[i] = lf.log10phi(lf, theta, mags)

    up = np.percentile(phi, 15.87, axis=0)
    down = np.percentile(phi, 84.13, axis=0)
    # print("\n\nlf.__dict__= ", lf.__dict__)
    # Assuming `lf` is your object
    # for key, value in lf.__dict__.items():
    #     print(f"\n'{key}' : \t{value}")
    # print('\n\n\n\nlf.samples= ', lf.samples)
    # print("\n\n\n\nphi= ", phi)

    # L1 = int(len(phi)/10)
    # L2 = int(len(phi[0])/10)

    # cnt1, cnt2 = L1, L2
    # for i in range(len(phi)):
    #     cnt1 -= 1
    #     if cnt1 == 0:
    #         cnt1 = L1
    #         cnt2 = L2
    #         print("[", end=' ')
    #         for j in range(len(phi[i])):
    #             cnt2 -= 1
    #             if cnt2 == 0:
    #                 cnt2 = L2
    #                 print(phi[i][j], end=' ')
    #         print("]")
        
    print("\n\n\n\nup= ", up)
    print("\n\n\n\ndown= ", down)
    f = ax.fill_between(mags, down, y2=up, color='#ffbf00', alpha=0.7)
    # f = ax.fill_between(mags, down, y2=up, color=kwargs['c'], alpha=0.7)

    return f

def plot_bestfit_lf(lf, ax, mags, **kwargs):
    """
    Plot the best fit luminosity function (LF) on the given axes.

    Parameters
    ----------
    - lf : object
        The luminosity function object containing the samples and methods.
    - ax : matplotlib.axes.Axes
        The axes on which to plot the best fit LF.
    - mags : array-like
        An array of magnitudes at which to evaluate the LF.
    - kwargs : dict
        Additional keyword arguments for plotting, such as line width,
        color, and zorder.

    Returns
    -------
    - bf : matplotlib.lines.Line2D
        The line object representing the best fit LF.
    """

    bf = np.median(lf.samples, axis=0)
    phi_fit = lf.log10phi(bf, mags)
    bf, = ax.plot(mags, phi_fit, lw=1.5, c='#ffbf00', zorder=kwargs['zorder'])
    return bf 

def binVol(self, selmap, mrange, zrange):
    # print("In drawlf.py binVol")

    """

    Calculate volume in an M-z bin for *one* selmap.

    """

    v = 0.0
    for i in range(selmap.m.size):
        if (selmap.m[i] >= mrange[0]) and (selmap.m[i] < mrange[1]):
            if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                if selmap.sid == 7: # Giallongo 
                    v += selmap.volarr[i]*selmap.p[i]*selmap.dm[i]
                else:
                    v += selmap.volarr[i]*selmap.p[i]*selmap.dm_array[i]

    return v


def binVol_all(self, selmap, mrange, zrange):
    # print("In drawlf.py binVol_all")

    """

    Calculate volume in an M-z bin for *one* selmap.

    """

    v = 0.0
    for i in range(selmap.m_all.size):
        if (selmap.m_all[i] >= mrange[0]) and (selmap.m_all[i] < mrange[1]):
            if (selmap.z_all[i] >= zrange[0]) and (selmap.z_all[i] < zrange[1]):
                if selmap.sid == 7: # Giallongo 
                    v += selmap.volarr_all[i]*selmap.p_all[i]*selmap.dm[i]
                else:
                    v += selmap.volarr_all[i]*selmap.p_all[i]*selmap.dm_all_array[i]

    return v


def totBinVol(lf, m, mbins, selmaps):
    # print("In drawlf.py totBinVol")

    """

    Given magnitude bins mbins and a list of selection maps
    selmaps, compute the volume for an object with magnitude m.

    """
    # print("In drawlf.py totBinVol")
    # print("m= ", m)
    # print("mbins= ", mbins)
    idx = np.searchsorted(mbins, m)
    mlow = mbins[idx-1]
    mhigh = mbins[idx]
    mrange = (mlow, mhigh)

    v = np.array([binVol(lf, x, mrange, lf.zlims) for x in selmaps])
    total_vol = v.sum() 

    return total_vol


def totBinVol_all(lf, m, mbins, selmaps):
    # print("In drawlf.py totBinVol_all")

    """

    Given magnitude bins mbins and a list of selection maps
    selmaps, compute the volume for an object with magnitude m.

    """

    idx = np.searchsorted(mbins, m)
    mlow = mbins[idx-1]
    mhigh = mbins[idx]
    mrange = (mlow, mhigh)

    v = np.array([binVol_all(lf, x, mrange, lf.zlims) for x in selmaps])
    total_vol = v.sum() 

    return total_vol


def get_lf(lf, sid, z_plot, special='None'):
    # print("In drawlf.py get_lf")
    
    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  
    m = lf.M1450[lf.sid==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10 or sid == 18:
        bins = np.arange(-30.9, -17.3, 1.8)
    elif special == 'croom_comparison':
        # These M1450 bins result in the Mgz2 bins of Croom09.  The
        # 1.23 converts between the two magnitudes (Eqn B8 of Ross13).
        bins = np.arange(-30,-19.5,0.5)+1.23
    elif special == 'croom_comparison_Mgz2':
        # These Mgz2 bins of Croom09.  
        bins = np.arange(-30,-19.5,0.5)
    else:
        bins = np.arange(-30.9, -13.3, 0.6)

    v1 = np.array([totBinVol(lf, x, bins, selmaps) for x in m])

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

    # print( 'sid=', sid )
    # print( 'mags=', mags)
    # print( 'nums=', nums)
    # print( 'total=', np.sum(nums))

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


def get_lf_all(lf, sid, z_plot, special='None'):
    # print("In drawlf.py get_lf_all")

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  

    m = lf.M1450_all[lf.sid_all==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10 or sid == 18:
        bins = np.arange(-30.9, -17.3, 1.8)
    elif special == 'croom_comparison':
        # These M1450 bins result in the Mgz2 bins of Croom09.  The
        # 1.23 converts between the two magnitudes (Eqn B8 of Ross13).
        bins = np.arange(-30,-19.5,0.5)+1.23
    elif special == 'croom_comparison_Mgz2':
        # These Mgz2 bins of Croom09.  
        bins = np.arange(-30,-19.5,0.5)
    else:
        bins = np.arange(-30.9, -13.3, 0.6)

    v1 = np.array([totBinVol_all(lf, x, bins, selmaps) for x in m])

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


def get_lf_sample(lf, sid, z_plot):
    # print("In drawlf.py get_lf_sample")

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  

    m = lf.M1450[lf.sid==sid]
    n = m.size
    mlims = (m.min(), m.max())
    theta = np.median(lf.samples, axis=0)
    m = lfsample(theta, n, mlims)

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10:
        bins = np.arange(-30.9, -17.3, 1.8)
    else:
        bins = np.arange(-30.9, -17.3, 0.6)

    v1 = np.array([totBinVol(lf, x, bins, selmaps) for x in m])

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


def plot_giallongo_z5p75(lf, ax, mags):
    # print("In drawlf.py plot_giallongo_z5p75")

    M_star_giallongo = -23.4
    log10phi_star_giallongo = -5.8
    beta = -1.66 # Giallongo et al. call this -beta
    alpha = -3.35 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al. 2015 fit at $z=5.75$', dashes=[7,2])

    return 


def plot_giallongo_z4p75(lf, ax, mags):
    # print("In drawlf.py plot_giallongo_z4p75")

    M_star_giallongo = -23.6
    log10phi_star_giallongo = -5.7
    beta = -1.81 # Giallongo et al. call this -beta
    alpha = -3.14 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al. 2015 fit at $z=4.75$', dashes=[7,2])

    return 


def plot_giallongo_z4p25(lf, ax, mags):
    # print("In drawlf.py plot_giallongo_z4p25")

    M_star_giallongo = -23.2
    log10phi_star_giallongo = -5.2
    beta = -1.52 # Giallongo et al. call this -beta
    alpha = -3.13 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al. 2015 fit at $z=4.25$', dashes=[7,2])

    return 

def savedata(data):
    # print("\n\nIn drawlf.py savedata\n")
    # print("data= ", data)
    zlims = data[0]
    if zlims == (0.1, 0.4):
    # if zlims == (5.5, 6.5):
        with open('datapoints.dat', 'w') as f:
            f.write('# The data points for the QLF at different redshift bins are given here.\n')
            f.write('# status 1 points are the selected points and status 0 points are the rejected points. \n')
            f.write('# The columns are as follows:\n')
            f.write('# zmin   zmax   label                      status  mags      logphi      right    left     uperr    downerr\n')


    with open('datapoints.dat', 'a') as f:
        # f.write('{:6.2f} {:6.2f}   '.format(zlims[0], zlims[1]))
        # for cnt in range(1, len(data)):
        #     for d in data[cnt]:
        #         if isinstance(d, str):
        #             if cnt == 1:
        #                 f.write('{:20s}'.format(d))
        #             else:
        #                 f.write('{:25s}'.format(d))
        #             continue
        #         print()
        #         print("d= ", d)
        #         f.write('{:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f}\n{:20s}'.format(d[0], d[1], d[2], d[3], d[4], d[5],''))
        for cnt in range(1, len(data)):
            for d in data[cnt]:
                if isinstance(d, str):
                    continue
                f.write('{:6.2f} {:6.2f}   {:25s} {:7d}  {:7.3f}   {:<7.3f}   {:7.3f}  {:7.3f}  {:7.3f}  {:7.3f}\n'.format(zlims[0], zlims[1], data[cnt][0], d[0], d[1], d[2], d[3], d[4], d[5], d[6]))


    return

def render(ax, lf, composite=None, showMockSample=False, show_individual_fit=True, c2=None, c3=None, includes_bad_points=False):
    # print("In drawlf.py render")
    # show_individual_fit, lf
    # Everything else is not given this time

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 
    show_individual_fit = True
    showMockSample = False

    if show_individual_fit: 
        mag_plot = np.linspace(-34.0, -12.0, num=200) 
        # indf = plot_posterior_sample_lfs(lf, ax, (-34.0, -12.0), lw=1,
                                    #    c='#ffbf00', alpha=0.1, zorder=2) 
        indf = plot_posterior_sample_lfs(lf, ax, xlim, lw=1,
                                        alpha=0.1, zorder=2) 
        # plot_bestfit_lf(lf, ax, mag_plot, lw=2,
        #                      c='#ffbf00', zorder=3, label='This work')

        # indbf = plot_bestfit_lf(lf, ax, mag_plot, lw=2,
                            #  c='#ffbf00', zorder=3)
        indbf = plot_bestfit_lf(lf, ax, np.linspace(*xlim, num=200), lw=2,
                              zorder=3)
        
        # if z_plot < 4.5:
        #     plot_giallongo_z4p25(lf, ax, mag_plot)

        # if z_plot < 5.5 and z_plot > 4.7:
        #     plot_giallongo_z4p75(lf, ax, mag_plot)

        # if z_plot > 5.5:
        #     plot_giallongo_z5p75(lf, ax, mag_plot)
            

    if composite is not None:

        nmags = 200 
        mags = np.linspace(-34.0, -12.0, num=nmags)
        bf = np.median(composite.samples, axis=0)
        nsample = 1000
        rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = composite.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c1f = ax.fill_between(mags, down, y2=up, color='grey', zorder=6, alpha=0.7)

        p = np.median(phi, axis=0)
        c1bf, = ax.plot(mags, p, color='k', zorder=6, lw=1)


    if c2 is not None: 
        nmags = 200 
        mags = np.linspace(-34.0, -12.0, num=nmags)
        bf = np.median(c2.samples, axis=0)
        nsample = 1000
        rsample = c2.samples[np.random.randint(len(c2.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = c2.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c2f = ax.fill_between(mags, down, y2=up, color='forestgreen', zorder=4, alpha=0.7)

        p = np.median(phi, axis=0)
        c2bf, = ax.plot(mags, p, color='forestgreen', zorder=4, lw=1)


    if c3 is not None: 
        nmags = 200 
        mags = np.linspace(-34.0, -12.0, num=nmags)
        bf = np.median(c3.samples, axis=0)
        nsample = 1000
        rsample = c3.samples[np.random.randint(len(c3.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = c3.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c3f = ax.fill_between(mags, down, y2=up, color='peru', zorder=5, alpha=0.7)

        p = np.median(phi, axis=0)
        c3bf, = ax.plot(mags, p, color='brown', zorder=5, lw=1)
        
        
    cs = { 1 : '#1f77b4', # "blue"
           6 : '#17becf', # "cyan"
           7 : '#9467bd', # "purple"
           8 : '#8c564b', # "brown"
           10 : '#ff7f0e', # "orange"
           11 : '#7f7f7f', # "grey"
           13 : '#d62728', # "red"
           15 : '#2ca02c', # "green"
           17 : '#bcbd22', # "yellow"
           18 : '#e377c2' # "pink"
    }

    cs = { 
        1: '#1f77b4',  # blue
        2: '#ff7f0e',  # orange
        3: '#2ca02c',  # green
        4: '#d62728',  # red
        5: '#9467bd',  # purple
        6: '#8c564b',  # brown
        7: '#e377c2',  # pink
        8: '#7f7f7f',  # grey
        9: '#bcbd22',  # yellow
        10: '#17becf', # cyan
        11: '#aec7e8', # light blue
        12: '#ffbb78', # light orange
        13: '#98df8a', # light green
        14: '#ff9896', # light red
        15: '#c5b0d5', # light purple
        16: '#c49c94', # light brown
        17: '#f7b6d2', # light pink
        18: '#c7c7c7', # light grey
        19: '#dbdb8d', # light yellow
        20: '#9edae5', # light cyan
        21: '#393b79', # dark blue
        22: '#637939', # dark green
        23: '#8c6d31', # dark brown
        24: '#843c39', # dark red
        25: '#7b4173', # dark purple
        26: '#5254a3',  # dark blue
        105: '#000000', # black
        106: '#444444', # dark grey
        107: '#888888', # light grey
        108: '#b09349', # light grey
        109: '#43497b', # light grey
        110: '#ab3239',
        111: '#193472',
        112: '#d95f02',
        113: '#7570b3',
        114: '#e7298a',
        115: '#66a61e',
        116: '#e6ab02',
        117: '#a6761d',
        150: '#ff0000', # red
        151: '#00ff00', # green
        152: '#0000ff', # blue
        153: '#ffff00', # yellow
        154: '#ff00ff', # pink
        155: '#00ffff', # cyan
        156: '#000000', # black
        157: '#ffffff', # white
    }
    
    # By now best fit LFs have been plotted. Now plot the data.

    def dsl(i):
        for x in lf.maps:
            if x.sid == i:
                return x.label
        return

    sids = np.unique(lf.sid)
    
    bad_data_set = False
    if bad_data_set:
        for i in sids: 
            mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z_plot)
            print( mags[logphi>-100.0])
            print( logphi[logphi>-100.0])
            ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)
            ax.scatter(mags, logphi, c='#ffffff', edgecolor=cs[i], zorder=4, s=16, label=dsl(i)+' (rejected bins)')
        return
    
    data = [lf.zlims]
    cnt = 0
    # This segment plots the data points in the graph.
    for sid in sids[::-1]:
        i = int(sid)
        cnt += 1

        mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z_plot)

        # print( mags[logphi>-100.0])
        # print( logphi[logphi>-100.0])

        mask = logphi > -100.0
        data.append([])
        data[cnt].append(dsl(i))
        for j in range(len(mags[mask])):
            data[cnt].append([1, mags[mask][j], logphi[mask][j], right[mask][j], left[mask][j], uperr[mask][j], downerr[mask][j]])


        # For some reason, the omitted data are not actually omitted. 
        # Since the data points dont make any sense, they dont show up.
        # But they are still there in the background.
        # Theplot could have been done after masking the data points.
        # mask = logphi > -100.0
        # mags = mags[mask]
        # logphi = logphi[mask]
        # and so on.
        # If I do this, I need to do the same for rejected bins as well.
        ax.scatter(mags, logphi, c=cs[i], edgecolor='None', zorder=4, s=20, label=dsl(i))
        ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=4)

        if i == 8:
            # No need to plot rejected bins for McGreer's data because
            # they were rejected for overlap with Yang, not due to
            # incompleteness.
            continue 
        
        # For the rejected bins!
        mags_all, left_all, right_all, logphi_all, uperr_all, downerr_all = get_lf_all(lf, i, z_plot)
        # print( mags_all[logphi_all!=logphi])
        # print( logphi_all[logphi_all!=logphi])

        select = (logphi_all!=logphi)
        mags_all = mags_all[select]
        left_all = left_all[select]
        right_all = right_all[select]
        logphi_all = logphi_all[select]
        uperr_all  = uperr_all[select]
        downerr_all = downerr_all[select]

        mask = logphi_all > -100.0

        for j in range(len(mags_all[mask])):
            data[cnt].append([0, mags_all[mask][j], logphi_all[mask][j], right_all[mask][j], left_all[mask][j], uperr_all[mask][j], downerr_all[mask][j]])

        if mags_all.any(): 
            ax.errorbar(mags_all, logphi_all, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left_all, right_all)), 
                        yerr=np.vstack((uperr_all, downerr_all)),
                        fmt='None', zorder=4)
            ax.scatter(mags_all, logphi_all, c='#ffffff', edgecolor=cs[i],
                       zorder=4, s=16, label=dsl(i)+' (rejected bin)')

    savedata(data)

    if showMockSample:
        for i in sids:
            mags, left, right, logphi, uperr, downerr = get_lf_sample(lf, i, z_plot)
            ax.scatter(mags, logphi, c='k', edgecolor='None', zorder=4, s=16, label=dsl(i))
            ax.errorbar(mags, logphi, ecolor='k', capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)

    if c2 is not None: 
        return (indf, indbf), (c1f, c1bf), (c2f, c2bf), (c3f, c3bf)
    else:
        return 

def draw(lf, composite=None, dirname='', showMockSample=False, show_individual_fit=True, includes_bad_points=False):
    # print("In drawlf.py draw")

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 
    
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    render(ax, lf, composite=composite, showMockSample=showMockSample,
           show_individual_fit=show_individual_fit, includes_bad_points=includes_bad_points)

    # ax.set_xlim(-12.0, -34.0)
    # ax.set_ylim(-16.0, 0.0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_xticks(np.arange(-34,-11, 2))

    ax.set_xlabel(r'$M_{1450}$')
    ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

    legend_title = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.2, scatterpoints=1,
               title=legend_title)

    plottitle = r'${:g}\leq z<{:g}$'.format(lf.zlims[0], lf.zlims[1]) 
    plt.title(plottitle, size='medium', y=1.01)

    plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)

    plt.savefig(plotfile, bbox_inches='tight')
    plt.savefig(plotfile.replace('.pdf', '.png'), bbox_inches='tight')

    plt.close('all') 

    # letting the user know where the plot was saved
    print("saved the figure in ", plotfile)
    print("saved the figure in ", plotfile.replace('.pdf', '.png'))
    print("command executed in drawlf.py")

    return 
    
