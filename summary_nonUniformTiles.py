import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial import Chebyshev as T
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

clr = {'masters12': 'dodgerblue',
       'jiang16': 'dodgerblue',
       'glikman11': 'dodgerblue'}

def croom09(ax, param):

    if param == 0:

        z, p, pdownerr, puperr = np.loadtxt('Scratch/croom_phistar.txt', unpack=True)
        pdownerr = -pdownerr 
        
    if param == 1:

        z, p, pdownerr, puperr = np.loadtxt('Scratch/croom_Mgstar.txt', unpack=True)
        pdownerr = -pdownerr

        # Convert from Mgz2 to M1450 using equation B8 of Ross et
        # al. 2013.
        p = p + 1.23
        
    if param == 2:

        z, p, pdownerr, puperr = np.loadtxt('Scratch/croom_alpha.txt', unpack=True)
        pdownerr = -pdownerr 
        
    if param == 3:

        z, p, pdownerr, puperr = np.loadtxt('Scratch/croom_beta.txt', unpack=True)
        pdownerr = -pdownerr 

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                yerr=np.vstack((pdownerr, puperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, label='published values from Croom et al. 2009')

    return


# These redshift bins are labelled "bad" and are plotted differently.
reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

colors = ['tomato', 'tomato', 'tomato', 'tomato']
colors2 = ['k', 'k', 'k', 'k']
colors3 = ['g', 'g', 'g', 'g']
colors4 = ['m', 'm', 'm', 'm'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,7.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=500)
cfit = False

def plot_model(composite, param_number, ax):

    if param_number == 0:
        print( 'phi')

    nsample = 10000
    np.random.seed()
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
    nzs = len(z) 
    beta = np.zeros((nsample, nzs))

    for i, theta in enumerate(rsample):
        params = composite.getparams(theta)
        if param_number < 3: 
            beta[i] = composite.atz(z, params[param_number])
        else:
            beta[i] = composite.atz_beta(z, params[3]) 

    up = np.percentile(beta, 15.87, axis=0)
    down = np.percentile(beta, 84.13, axis=0)
    
    ax.fill_between(z, down, y2=up, color='turquoise', zorder=1, label='Model 2')

    beta = np.median(beta, axis=0)
    ax.plot(z, beta, color='turquoise', zorder=2, lw=1)
    
    return

def plot_model_polyb(composite, param_number, ax):

    nsample = 10000
    np.random.seed()
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
    nzs = len(z) 
    beta = np.zeros((nsample, nzs))

    for i, theta in enumerate(rsample):
        params = composite.getparams(theta)
        beta[i] = composite.atz(z, params[param_number])

    up = np.percentile(beta, 15.87, axis=0)
    down = np.percentile(beta, 84.13, axis=0)
    
    ax.fill_between(z, down, y2=up, color='peru', zorder=1, label='Model 3')

    beta = np.median(beta, axis=0)
    ax.plot(z, beta, color='peru', zorder=2, lw=1)
    
    return


def getParam(individuals, param, which='old', dtype='good'):

    if which == 'nonUniformTiles':

        zmean, zl, zu, u, l, c = np.loadtxt('bins_nonUniformTiles.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)

    elif which == 'all2SLAQandSDSS':

        zmean, zl, zu, u, l, c = np.loadtxt('bins_noExclusion2SLAQSDSS.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)

    elif which == 'all2SLAQsomeSDSS':

        zmean, zl, zu, u, l, c = np.loadtxt('bins_all2SLAQsomeSDSS.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)
        
    elif individuals is not None: 
    
        zmean = np.array([x.z.mean() for x in individuals])
        zl = np.array([x.zlims[0] for x in individuals])
        zu = np.array([x.zlims[1] for x in individuals])

        if param == 0: 
            u = np.array([x.phi_star[0] for x in individuals])
            l = np.array([x.phi_star[1] for x in individuals])
            c = np.array([x.phi_star[2] for x in individuals])
        elif param == 1:
            u = np.array([x.M_star[0] for x in individuals])
            l = np.array([x.M_star[1] for x in individuals])
            c = np.array([x.M_star[2] for x in individuals])
        elif param == 2:
            u = np.array([x.alpha[0] for x in individuals])
            l = np.array([x.alpha[1] for x in individuals])
            c = np.array([x.alpha[2] for x in individuals])
        elif param == 3:
            u = np.array([x.beta[0] for x in individuals])
            l = np.array([x.beta[1] for x in individuals])
            c = np.array([x.beta[2] for x in individuals])

    else:

        zmean, zl, zu, u, l, c = np.loadtxt('bins.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)

    m = np.ones_like(zmean, dtype=bool)
    m[reject] = False
    minv = np.logical_not(m) 

    if dtype == 'good': 
        return zmean[m], zl[m], zu[m], u[m], l[m], c[m]
    else:
        return zmean[minv], zl[minv], zu[minv], u[minv], l[minv], c[minv]

        
def plot_phi_star(fig, composite, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-12, -5)
    ax.set_yticks(np.arange(-12, -4, 1))

    if compOpt is not None: 
        phi = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[0])
        ax.plot(z, phi, color='g', zorder=2, dashes=[7,2])

    if composite is not None: 
        bf = np.median(composite.samples, axis=0)
        if sample:

            nsample = 1000
            rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
            nzs = len(z) 
            phi = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = composite.getparams(theta)
                phi[i] = composite.atz(z, params[0]) 
                
            up = np.percentile(phi, 15.87, axis=0)
            down = np.percentile(phi, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=1)
            
        phi = np.median(phi, axis=0)
        ax.plot(z, phi, color='k', zorder=2, lw=1)


    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=20)


    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='nonUniformTiles', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors2[0], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='nonUniformTiles', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='all2SLAQandSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[0], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='all2SLAQandSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[0], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='all2SLAQsomeSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[0], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='all2SLAQsomeSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[0], edgecolor='None', zorder=4, s=20)

    
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 2)
        print( coeffs )
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2],
        #          label='Least-square Chebyshev French curve', zorder=3)

        def func(z, p0, p1, p2):
            return T([p0, p1, p2])(z)

        sigma = uperr + downerr 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print( popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-12.2, 6.6, 4.6, 4.9, -0.1])
        print( popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,1,2,3), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((uperr, downerr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=20)

    croom09(ax, 0)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}'+
                  r'\mathrm{cMpc}^{-3}\right)$')
    ax.yaxis.labelpad = 8
    ax.set_xticklabels('')

    return

def plot_m_star(fig, composite, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+2)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-32, -20)
    ax.set_yticks(np.arange(-32, -19, 2))

    if compOpt is not None:
        M = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[1])
        print( M)
        ax.plot(z, M, color='g', zorder=2, dashes=[7,2])

    if composite is not None:
        bf = np.median(composite.samples, axis=0)
        if sample:

            nsample = 1000
            rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
            nzs = len(z) 
            M = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = composite.getparams(theta)
                M[i] = composite.atz(z, params[1]) 
                
            up = np.percentile(M, 15.87, axis=0)
            down = np.percentile(M, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=1)
            
        M = np.median(M, axis=0)
        ax.plot(z, M, color='k', zorder=2, lw=1)

        
    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=6, s=20)


    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='nonUniformTiles', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors2[1], edgecolor='None', zorder=4, s=20)


    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='all2SLAQandSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[1], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='all2SLAQsomeSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[1], edgecolor='None', zorder=4, s=20)
    
    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,4,5,6), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((uperr, downerr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=20)

    cfit = False
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 1)
        print( 'cm=', coeffs)
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3) 

        def func(z, p0, p1):
            return T([p0, p1])(z)

        sigma = np.abs(u-l)
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print( popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=20)


    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='nonUniformTiles', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='all2SLAQandSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='all2SLAQsomeSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[1], edgecolor='None', zorder=4, s=20)
    
    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        # def func(z, h, f0, z0, a, b):
        #     zeta = np.log10((1.0+z)/(1.0+z0))
        #     return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        # def func(z, p0, p1, p2):
        #     zeta = np.log10((1.0+z)/(1.0+3.5))
        #     return T([p0, p1, p2])(zeta)

        def func(z, p0, p1, p2):
            zeta = np.log10((1.0+z)/(1.0+3.5))
            return T([p0, p1, p2])(1+z)

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-22.,1,1])
        print( popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])

    croom09(ax, 1)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.yaxis.labelpad = 12
    ax.set_xticklabels('')

    return

def plot_alpha(fig, composite, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-6, 0)
    ax.set_yticks(np.arange(-7, -0.9, 1))


    if compOpt is not None: 
        alpha = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[2])
        ax.plot(z, alpha, color='g', zorder=2, dashes=[7,2],
                label='likelihood maximum')

    if composite is not None:
        bf = np.median(composite.samples, axis=0)
        if sample:

            nsample = 1000
            rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
            nzs = len(z) 
            alpha = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = composite.getparams(theta)
                alpha[i] = composite.atz(z, params[2]) 
                
            up = np.percentile(alpha, 15.87, axis=0)
            down = np.percentile(alpha, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=1, label='Model 1')

        
        alpha = np.median(alpha, axis=0) 
        ax.plot(z, alpha, color='k', zorder=2, lw=1)


    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=6, s=20, label='assuming uniform tiles')

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor=colors[2], zorder=6, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='nonUniformTiles', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors2[2], edgecolor='None', zorder=4, s=20, label='allowing for non-uniform tiles')

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='nonUniformTiles', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='all2SLAQandSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[2], edgecolor='None', zorder=4, s=20, label='non-uniform tiles; no selection')

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='all2SLAQandSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[2], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='all2SLAQsomeSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[2], edgecolor='None', zorder=4, s=20, label='non-uniform tiles; old seln for SDSS')

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='all2SLAQsomeSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[2], edgecolor='None', zorder=4, s=20)
    
    cfit = False
    if cfit: 
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1.0, c, 2)
        print( 'c=', coeffs)

        def func(z, p0, p1, p2):
            return T([p0, p1, p2])(z)

        sigma = u-l
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print( popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2],
                 label=r'French curve')

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,2.0,1.4,-0.7])
        print( popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])
        
    croom09(ax, 2)

    
    plt.legend(loc='upper left', fontsize=8, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=-0.1, borderpad=0.01, scatterpoints=1)

    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright-end slope)')
    ax.set_xlabel('$z$')

    return

def plot_beta(fig, composite, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+4)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')

    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-3, 0)
    ax.set_yticks(np.arange(-3, 0.2, 0.5))
    
    if compOpt is not None:
        beta = compOpt.atz_beta(z, compOpt.getparams(compOpt.bf.x)[3])
        ax.plot(z, beta, color='g', zorder=2, dashes=[7,2])

    if composite is not None:


        # bf = np.median(composite.samples, axis=0)
        # beta = composite.atz_beta(z, composite.getparams(bf)[3])
        # ax.plot(z, beta, color='k', zorder=2, lw=1)
        
        # bf16 = np.percentile(composite.samples, 15.87, axis=0)
        # beta16 = composite.atz_beta(z, composite.getparams(bf16)[3])

        # bf84 = np.percentile(composite.samples, 84.13, axis=0)
        # beta84 = composite.atz_beta(z, composite.getparams(bf84)[3])

        # ax.fill_between(z, beta84, y2=beta16, color='grey', zorder=1)

        
        if sample:

            nsample = 10000
            np.random.seed()
            rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
            nzs = len(z) 
            beta = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = composite.getparams(theta)
                beta[i] = composite.atz_beta(z, params[3]) 
                
            up = np.percentile(beta, 15.87, axis=0)
            down = np.percentile(beta, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=1)

        # bfs = np.median(rsample, axis=0) 
        # print( 'median beta (beta):', composite.getparams(bfs)[3])
        # print( 'median beta (samples):', composite.getparams(bf)[3])

        beta = np.median(beta, axis=0)
        ax.plot(z, beta, color='k', zorder=2, lw=1)

        # beta = composite.atz_beta(z, composite.getparams(bf)[3])
        # ax.plot(z, beta, color='k', zorder=2, lw=1)


    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=6, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='nonUniformTiles', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors2[3], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='all2SLAQandSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[3], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='all2SLAQsomeSDSS', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[3], edgecolor='None', zorder=4, s=20)
    
    cfit = False
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 3)
        print( coeffs)
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3)

        def func(z, p0, p1, p2, p3):
                return T([p0, p1, p2, p3])(z)

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print( 'cb=', popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='r', dashes=[7,2])

    polyfit = False
    if polyfit:
        zc = np.linspace(0, 7, 500)
        p = np.polyfit(np.log10(zmean+10), c, 2)
        print( p)
        # plt.plot(zc, np.polyval(p, np.log10((zc+1))), lw=1, c='k', dashes=[7,2], zorder=3)
        plt.plot(zc, np.polyval(p, np.log10((zc+10))), lw=1, c='k', dashes=[7,2], zorder=3)

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,2.0,1.4,-0.7])
        print( popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='k', dashes=[7,2])

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor=colors[3], zorder=6, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='nonUniformTiles', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors2[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='all2SLAQandSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors3[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors3[3], edgecolor='None', zorder=4, s=20)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='all2SLAQsomeSDSS', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors4[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zmean, c, color=colors4[3], edgecolor='None', zorder=4, s=20)
    
    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,7,8,9), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((uperr, downerr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=20)

    croom09(ax, 3)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint-end slope)')
    ax.set_xlabel('$z$')

    return 

def summary_plot(composite=None, individuals=None, compOpt=None, sample=False):

    mpl.rcParams['font.size'] = '14'
    
    fig = plt.figure(figsize=(6, 6), dpi=100)

    print( 'laying out figure')

    K = 4
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.1         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    print( 'plotting now')
    
    plot_phi_star(fig, composite, individuals=individuals, compOpt=compOpt)
    plot_m_star(fig, composite, individuals=individuals, compOpt=compOpt)
    plot_alpha(fig, composite, individuals=individuals, compOpt=compOpt)
    plot_beta(fig, composite, individuals=individuals, compOpt=compOpt)

    plt.savefig('evolution_nonUniformTiles.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

summary_plot()

