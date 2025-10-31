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

# These redshift bins are labelled "bad" and are plotted differently.
# reject = [0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
reject = []

colors = ['tomato', 'forestgreen', 'goldenrod', 'saddlebrown']
# colors = ['k', 'k', 'k', 'k'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,8.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=500)
cfit = False

def plot_model(composite, param_number, ax):

    if param_number == 0:
        print('phi')

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
    
    mf = ax.fill_between(z, down, y2=up, color='forestgreen', zorder=1, alpha=0.7)

    beta = np.median(beta, axis=0)
    m, = ax.plot(z, beta, color='forestgreen', zorder=2, lw=1)
    
    return mf, m

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
    
    mf = ax.fill_between(z, down, y2=up, color='peru', zorder=1, alpha=0.7)

    beta = np.median(beta, axis=0)
    m, = ax.plot(z, beta, color='brown', zorder=2, lw=1)
    
    return mf, m 

def getParam(individuals, param, which='old', dtype='good'):

    # # setting dtype to be always good
    # dtype = 'good'

    if individuals is not None: 
    
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
        print("Inside plot_phi_star, compOpt is not None")
        phi = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[0])
        ax.plot(z, phi, color='g', zorder=2, dashes=[7,2])

    if composite is not None: 
        print("Inside plot_phi_star, composite is not None")
        # bf = np.median(composite.samples, axis=0)
        bf = composite.map_params
        if sample:

            nsample = 1000
            rsample = composite.samples[0][np.random.randint(len(composite.samples[0]), size=nsample)]
            nzs = len(z) 
            phi = np.zeros((nsample, nzs))
            # print("rsample.shape:", rsample.shape)
            # import sys; sys.exit(0)
            # phi = composite.atz(z, rsample[0])

            for i, theta in enumerate(rsample):
                # print("theta:", theta)
                # print("theta.shape:", theta.shape)
                # params = composite.getparams(theta)
                # print("params:", params)
                params = theta[:-3]
                # print("params:", params)
                # print("params.shape:", params.shape)
                phi[i] = composite.atz(z, params) 
                
            up = np.percentile(phi, 15.87, axis=0)
            down = np.percentile(phi, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=5, alpha=0.7)
            
        # phi = np.median(phi, axis=0)
        print("bf:", bf)
        print("bf[0]:", bf[0])
        phi = composite.atz(z, bf[0])
        ax.plot(z, phi, color='k', zorder=5, lw=1)

    if lfg_break is not None: 
        print("Inside plot_phi_star, lfg_break is not None")
        plot_model(lfg_break, 0, ax)

    if lfg_polyb is not None: 
        print("Inside plot_phi_star, lfg_polyb is not None")
        plot_model_polyb(lfg_polyb, 0, ax)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=30)

    for i in range(len(zmean)):
        plt.annotate(str(i), (zmean[i], c[i]), fontsize=8, color='k')

    with open('plot_phi_star.txt', 'w') as f:
        for i in range(len(zmean)):
            f.write(f'{zmean[i]:.2f} {c[i]:.2f}\n')

    zm, zl, zu, u, l, centre = getParam(individuals, 0, which='new', dtype='bad')
    left = zm-zl
    right = zu-zm
    uperr = abs(u-centre)
    downerr = abs(centre-l)
    ax.errorbar(zm, centre, ecolor='grey', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zm, centre, color='#ffffff', edgecolor='grey', zorder=6, s=27)


    # # print('zmean:', zmean)
    # # print('zm:', zm)
    # # print('len(zmean):', len(zmean))
    # # print('len(zm):', len(zm))
    # # print('zm[:10]:', zm[:10])
    # # print('zmean[:10]:', zmean[:10])
    # print('c:', c)
    # print('centre:', centre)
    # # print('min(zmean[:10]):', np.min(zmean[:10]))
    # # print('max(zmean[:10]):', np.max(zmean[:10]))
    # # print('min(zm):', np.min(zm))
    # # print('max(zm):', np.max(zm))
    # print('min(c):', np.min(c))
    # print('max(c):', np.max(c))
    
    
    # print()
    # print("Good data:")
    # for i in range(len(zmean)):
    #     print(f"{i}: {zmean[i]}, {c[i]}")
    # print()
    # print()


    # xlim_l = min(np.min(zmean[:10]),np.min(zm[:10]))-1
    # xlim_r = max(np.max(zmean[:10]),np.max(zm[:10]))+1
    xlim_l = zmin
    xlim_r = zmax
    ax.set_xlim(xlim_l, xlim_r)
    ax.set_xticks(np.arange(xlim_l, xlim_r, 1))

    ylim_l = np.min(c) - 1
    ylim_r = np.max(c) + 1
    ylim_r = min(4, ylim_r)
    ax.set_ylim(ylim_l, ylim_r)
    ax.set_yticks(np.arange(ylim_l, ylim_r, abs(ylim_r-ylim_l)/10))
    
    
    if cfit:
        print("Using Chebyshev fit for phi*")
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 2)
        print(coeffs )
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2],
        #          label='Least-square Chebyshev French curve', zorder=3)

        def func(z, p0, p1, p2):
            return T([p0, p1, p2])(z)

        sigma = uperr + downerr 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print(popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    curvefit = False
    if curvefit:
        print("Using curve_fit for phi*")
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-12.2, 6.6, 4.6, 4.9, -0.1])
        print(popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,1,2,3), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)

    # ax.set_xticks((0,1,2,3,4,5,6,7))
    # ax.set_xticks((0,1,2,3,4,5,6,7,8,9,10,11,12))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}'+
                  r'\mathrm{cMpc}^{-3}\right)$')
    ax.yaxis.labelpad = 8
    ax.set_xticklabels('')

    # handles = [plt.Line2D([0], [0], marker='', color='w', markerfacecolor='blue', markersize=10, label=f'{i}: {round(zmean[i], 3)}') for i in range(len(zmean))]
    # # handles = [plt.Text(0, 0, f'{i}: {round(zmean[i], 3)}') for i in range(len(zmean))]

    # ax.legend(handles=handles, title='Redshift bins', bbox_to_anchor=(0.05, 0.05), loc='upper left', fontsize=6, title_fontsize=6)

    # fig.tight_layout()

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
    # ax.set_xlim(zmin, zmax)
    # ax.set_ylim(-32, -20)
    # ax.set_yticks(np.arange(-32, -19, 2))

    if compOpt is not None:
        M = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[1])
        print(M)
        ax.plot(z, M, color='g', zorder=2, dashes=[7,2])

    if composite is not None:
        # bf = np.median(composite.samples, axis=0)
        bf = composite.map_params
        if sample:

            nsample = 1000
            rsample = composite.samples[1][np.random.randint(len(composite.samples[1]), size=nsample)]
            nzs = len(z) 
            M = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                # print('theta:', theta)
                # print("theta.shape:", theta.shape)
                # params = composite.getparams(theta)
                params = theta[:-3]
                print('params:', params)
                M[i] = composite.atz(z, params) 
                
            up = np.percentile(M, 15.87, axis=0)
            down = np.percentile(M, 84.13, axis=0)
            print("\n\n\n\n\n\n")
            print('up:', up)
            print('down:', down)
            print("M:", M)
            print("\n\n\n\n\n\n")
            ax.fill_between(z, down, y2=up, color='grey', zorder=5, alpha=0.7)

            
        # M = np.median(M, axis=0)
        M = composite.atz(z, bf[1])
        ax.plot(z, M, color='k', zorder=5, lw=1)
        

    if lfg_break is not None: 
        plot_model(lfg_break, 1, ax)

    if lfg_polyb is not None: 
        plot_model_polyb(lfg_polyb, 1, ax)
        
    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=6, s=30, label='included data')

    for i in range(len(zmean)):
        plt.annotate(str(i), (zmean[i], c[i]), fontsize=8, color='k')

    with open('plot_m_star.txt', 'w') as f:
        for i in range(len(zmean)):
            f.write(f'{zmean[i]:.2f} {c[i]:.2f}\n')

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,4,5,6), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)

    cfit = False
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 1)
        print('cm=', coeffs)
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3) 

        def func(z, p0, p1):
            return T([p0, p1])(z)

        sigma = np.abs(u-l)
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print(popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    zm, zl, zu, u, l, centre = getParam(individuals, 1, which='new', dtype='bad')
    left = zm-zl
    right = zu-zm
    uperr = abs(u-centre)
    downerr = abs(centre-l)
    ax.errorbar(zm, centre, ecolor='grey', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zm, centre, color='#ffffff', edgecolor='grey', zorder=6, s=27, label='excluded data')
        

    # print('zmean:', zmean)
    # print('zm:', zm)
    # print('len(zmean):', len(zmean))
    # print('len(zm):', len(zm))
    # print('zm[:10]:', zm[:10])
    # print('zmean[:10]:', zmean[:10])
    # print('c:', c)
    # print('centre:', centre)
    # # print('min(zmean[:10]):', np.min(zmean[:10]))
    # # print('max(zmean[:10]):', np.max(zmean[:10]))
    # # print('min(zm):', np.min(zm))
    # # print('max(zm):', np.max(zm))
    # print('min(c):', np.min(c))
    # print('max(c):', np.max(c))
    
    
    # print()
    # print("Good data:")
    # for i in range(len(zmean)):
    #     print(f"{i}: {zmean[i]}, {c[i]}")
    # print()
    # print()


    # xlim_l = min(np.min(zmean[:10]),np.min(zm[:10]))-1
    # xlim_r = max(np.max(zmean[:10]),np.max(zm[:10]))+1
    xlim_l = zmin
    xlim_r = zmax
    ax.set_xlim(xlim_l, xlim_r)
    ax.set_xticks(np.arange(xlim_l, xlim_r, 1))

    ylim_l = np.min(c) - 1
    ylim_r = np.max(c) + 1
    ylim_r = min(14, ylim_r)
    ax.set_ylim(ylim_l, ylim_r)
    ax.set_yticks(np.arange(ylim_l, ylim_r, abs(ylim_r-ylim_l)/10))
    
    
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
        print(popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])
        
    # ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.yaxis.labelpad = 12
    ax.set_xticklabels('')

    plt.legend(loc='upper right', fontsize=10,
               handlelength=3, frameon=False, framealpha=0.0,
               labelspacing=.1, handletextpad=-0.3, borderpad=0.1,
               scatterpoints=1)
    

    return

def plot_alpha(fig, composite, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-7, -1)
    ax.set_yticks(np.arange(-7, -0.9, 1))

    m1f, m1, m2f, m2, m3f, m3= None, None, None, None, None, None

    if compOpt is not None: 
        alpha = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[2])
        ax.plot(z, alpha, color='g', zorder=2, dashes=[7,2],
                label='likelihood maximum')

    if composite is not None:
        # bf = np.median(composite.samples, axis=0)
        bf = composite.map_params
        if sample:

            nsample = 1000
            rsample = composite.samples[2][np.random.randint(len(composite.samples[2]), size=nsample)]
            nzs = len(z) 
            alpha = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = theta[:-3]
                alpha[i] = composite.atz(z, params) 
                
            up = np.percentile(alpha, 15.87, axis=0)
            down = np.percentile(alpha, 84.13, axis=0)
            m1f = ax.fill_between(z, down, y2=up, color='grey', zorder=5, label='Model 1', alpha=0.7)

        
        # alpha = np.median(alpha, axis=0) 
        alpha = composite.atz(z, bf[2])
        m1, = ax.plot(z, alpha, color='k', zorder=5, lw=1)

    if lfg_break is not None: 
        m2f, m2 = plot_model(lfg_break, 2, ax)

    if lfg_polyb is not None: 
        m3f, m3 = plot_model_polyb(lfg_polyb, 2, ax)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=6, s=30)

    for i in range(len(zmean)):
        plt.annotate(str(i), (zmean[i], c[i]), fontsize=8, color='k')

    with open('plot_alpha.txt', 'w') as f:
        for i in range(len(zmean)):
            f.write(f'{zmean[i]:.2f} {c[i]:.2f}\n')

    cfit = False
    if cfit: 
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1.0, c, 3)
        print('c=', coeffs)

        def func(z, p0, p1, p2, p3):
            return T([p0, p1, p2, p3])(z)

        sigma = u-l
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print(popt)
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
        print(popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])

    zm, zl, zu, u, l, centre = getParam(individuals, 2, which='new', dtype='bad')
    left = zm-zl
    right = zu-zm
    uperr = abs(u-centre)
    downerr = abs(centre-l)
    ax.errorbar(zm, centre, ecolor='grey', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zm, centre, color='#ffffff', edgecolor='grey', zorder=6, s=27)


    # print('zmean:', zmean)
    # print('zm:', zm)
    # print('len(zmean):', len(zmean))
    # print('len(zm):', len(zm))
    # print('zm[:10]:', zm[:10])
    # print('zmean[:10]:', zmean[:10])
    # print('c:', c)
    # print('centre:', centre)
    # # print('min(zmean[:10]):', np.min(zmean[:10]))
    # # print('max(zmean[:10]):', np.max(zmean[:10]))
    # # print('min(zm):', np.min(zm))
    # # print('max(zm):', np.max(zm))
    # print('min(c):', np.min(c))
    # print('max(c):', np.max(c))
    
    
    # print()
    # print("Good data:")
    # for i in range(len(zmean)):
    #     print(f"{i}: {zmean[i]}, {c[i]}")
    # print()
    # print()


    # xlim_l = min(np.min(zmean[:10]),np.min(zm[:10]))-1
    # xlim_r = max(np.max(zmean[:10]),np.max(zm[:10]))+1
    xlim_l = zmin
    xlim_r = zmax
    ax.set_xlim(xlim_l, xlim_r)
    ax.set_xticks(np.arange(xlim_l, xlim_r, 1))

    ylim_l = np.min(c) - 1
    ylim_r = np.max(c) + 1
    ylim_r = min(14, ylim_r)
    ax.set_ylim(ylim_l, ylim_r)
    ax.set_yticks(np.arange(ylim_l, ylim_r, abs(ylim_r-ylim_l)/10))
    
        

    handles, labels = [], []

    handles.append((m1f,m1))
    labels.append('Model 1')

    handles.append((m2f,m2))
    labels.append('Model 2')

    handles.append((m3f,m3))
    labels.append('Model 3')

    # Print the initial handles and labels
    print("Initial handles:", handles)
    print("Initial labels:", labels)

    # Filter out (None, None) entries
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if h != (None, None)]
    handles, labels = zip(*filtered_handles_labels) if filtered_handles_labels else ([], [])

    # Convert single-element tuples to lists
    handles = list(handles)
    labels = list(labels)

    # Print the filtered handles and labels
    print("\nFiltered handles:", handles)
    print("Filtered labels:", labels)

    plt.legend(handles, labels, loc='upper right', fontsize=10,
               handlelength=3, frameon=False, framealpha=0.0,
               labelspacing=.1, handletextpad=0.3, borderpad=0.1,
               scatterpoints=1)

    # ax.set_xticks((0,1,2,3,4,5,6,7))
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
    # ax.set_xlim(zmin, zmax)
    # ax.set_ylim(-3, 0)
    # ax.set_ylim(min(-3, np.min(individuals[0].beta[1]), np.min(individuals[0].beta[2])),
                # max(0, np.max(individuals[0].beta[1]), np.max(individuals[0].beta[2])))
    # ax.set_yticks(np.arange(-3, 0.2, 0.5))
    
    if compOpt is not None:
        print("Inside plot_beta, compOpt is not None")
        beta = compOpt.atz_beta(z, compOpt.getparams(compOpt.bf.x)[3])
        ax.plot(z, beta, color='g', zorder=2, dashes=[7,2])

    if composite is not None:
        print("Inside plot_beta, composite is not None")


        # bf = np.median(composite.samples, axis=0)
        # beta = composite.atz_beta(z, composite.getparams(bf)[3])
        # ax.plot(z, beta, color='k', zorder=2, lw=1)
        
        # bf16 = np.percentile(composite.samples, 15.87, axis=0)
        # beta16 = composite.atz_beta(z, composite.getparams(bf16)[3])

        # bf84 = np.percentile(composite.samples, 84.13, axis=0)
        # beta84 = composite.atz_beta(z, composite.getparams(bf84)[3])

        # ax.fill_between(z, beta84, y2=beta16, color='grey', zorder=1)

        
        if sample:
            print("Inside plot_beta, sample is True")
            nsample = 10000
            np.random.seed()
            rsample = composite.samples[3][np.random.randint(len(composite.samples[3]), size=nsample)]
            nzs = len(z) 
            beta = np.zeros((nsample, nzs))

            for i, theta in enumerate(rsample):
                params = theta[:-3]
                beta[i] = composite.atz_beta(z, params) 
                
            up = np.percentile(beta, 15.87, axis=0)
            down = np.percentile(beta, 84.13, axis=0)
            ax.fill_between(z, down, y2=up, color='grey', zorder=5, alpha=0.7)

        # bfs = np.median(rsample, axis=0) 
        # print('median beta (beta):', composite.getparams(bfs)[3])
        # print('median beta (samples):', composite.getparams(bf)[3])

        # beta = np.median(beta, axis=0)
        beta = composite.atz_beta(z, composite.map_params[3])
        ax.plot(z, beta, color='k', zorder=5, lw=1)

        # beta = composite.atz_beta(z, composite.getparams(bf)[3])
        # ax.plot(z, beta, color='k', zorder=2, lw=1)

    if lfg_break is not None: 
        print("Inside plot_beta, lfg_break is not None")
        plot_model(lfg_break, 3, ax)

    if lfg_polyb is not None: 
        print("Inside plot_beta, lfg_polyb is not None")
        plot_model_polyb(lfg_polyb, 3, ax)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = abs(u-c)
    downerr = abs(c-l)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=6, s=30)

    for i in range(len(zmean)):
        plt.annotate(str(i), (zmean[i], c[i]), fontsize=8, color='k')

    with open('plot_beta.txt', 'w') as f:
        for i in range(len(zmean)):
            f.write(f'{zmean[i]:.2f} {c[i]:.2f}\n')

    cfit = False
    if cfit:
        print("Inside plot_beta, cfit is True")
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 3)
        print(coeffs)
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3)

        def func(z, p0, p1, p2, p3):
                return T([p0, p1, p2, p3])(z)

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print('cb=', popt)
        plt.plot(zc, func(zc+1, *popt), lw=1, c='r', dashes=[7,2])

    polyfit = False
    if polyfit:
        print("Inside plot_beta, polyfit is True")
        zc = np.linspace(0, 7, 500)
        p = np.polyfit(np.log10(zmean+10), c, 2)
        print(p)
        # plt.plot(zc, np.polyval(p, np.log10((zc+1))), lw=1, c='k', dashes=[7,2], zorder=3)
        plt.plot(zc, np.polyval(p, np.log10((zc+10))), lw=1, c='k', dashes=[7,2], zorder=3)

    curvefit = False
    if curvefit:
        print("Inside plot_beta, curvefit is True")
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        #popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,2.0,1.4,-0.7])
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,4.0,1.4,-0.7])
        print(popt)
        plt.plot(zc, func(zc, *popt), lw=1, c='k', dashes=[7,2])

    zm, zl, zu, u, l, centre = getParam(individuals, 3, which='new', dtype='bad')
    left = zm-zl
    right = zu-zm
    uperr = abs(u-centre)
    downerr = abs(centre-l)
    ax.errorbar(zm, centre, ecolor='grey', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zm, centre, color='#ffffff', edgecolor='grey', zorder=6, s=27)

    # print('zmean:', zmean)
    # print('zm:', zm)
    # print('len(zmean):', len(zmean))
    # print('len(zm):', len(zm))
    # print('zm[:10]:', zm[:10])
    # print('zmean[:10]:', zmean[:10])
    # print('c:', c)
    # print('centre:', centre)
    # # print('min(zmean[:10]):', np.min(zmean[:10]))
    # # print('max(zmean[:10]):', np.max(zmean[:10]))
    # # print('min(zm):', np.min(zm))
    # # print('max(zm):', np.max(zm))
    # print('min(c):', np.min(c))
    # print('max(c):', np.max(c))
    
    
    # print()
    # print("Good data:")
    # for i in range(len(zmean)):
    #     print(f"{i}: {zmean[i]}, {c[i]}")
    # print()
    # print()


    # xlim_l = min(np.min(zmean[:10]),np.min(zm[:10]))-1
    # xlim_r = max(np.max(zmean[:10]),np.max(zm[:10]))+1
    xlim_l = zmin
    xlim_r = zmax
    ax.set_xlim(xlim_l, xlim_r)
    ax.set_xticks(np.arange(xlim_l, xlim_r, 1))

    ylim_l = np.min(c) - 1
    ylim_r = np.max(c) + 1
    ylim_r = min(10, ylim_r)
    ax.set_ylim(ylim_l, ylim_r)
    ax.set_yticks(np.arange(ylim_l, ylim_r, abs(ylim_r-ylim_l)/10))
    
        
    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,7,8,9), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)
    
    # ax.set_xticks((0,1,2,3,4,5,6,7))
    # ax.set_xticks((0,1,2,3,4,5,6,7,8,9,10,11,12))
    ax.set_ylabel(r'$\beta$ (faint-end slope)')
    ax.set_xlabel('$z$')

    return 

def summary_plot(composite=None, individuals=None, compOpt=None, sample=False, lfg_break=None, lfg_polyb=None, output_file_name='evolution2.pdf'):

    mpl.rcParams['font.size'] = '14'
    
    fig = plt.figure(figsize=(6, 6), dpi=300)

    print('laying out figure')

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

    print('plotting now')
    
    plot_phi_star(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample, lfg_break=lfg_break, lfg_polyb=lfg_polyb)
    plot_m_star(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample, lfg_break=lfg_break, lfg_polyb=lfg_polyb)
    plot_alpha(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample, lfg_break=lfg_break, lfg_polyb=lfg_polyb)
    plot_beta(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample, lfg_break=lfg_break, lfg_polyb=lfg_polyb)

    print("\n\n Saved result in " + output_file_name + "\n\n")
    plt.savefig(output_file_name,bbox_inches='tight')

    print("\n\n Saved result in " + output_file_name[:-4] + ".png\n\n")
    plt.savefig((output_file_name[:-4]+'.png'),bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

# summary_plot()

