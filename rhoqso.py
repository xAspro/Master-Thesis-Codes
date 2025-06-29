# Checked once. No implementation of this code in here. No implementation of this code in project.
# Have multiple savefigs. need to just call the functions.
# All rhoqso files are for fig 7 and more.
print("In rhoqso.py")

import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
# import fit_emissivity
import sounddevice as sd
import time
from composite import lf

output_data_file = 'rhoqso_output_data_2.txt'

with open(output_data_file, 'w') as f:
    f.write("# File: rhoqso_output_data\n")
    f.write("# Storing the data points and its uncertainities\n")
    f.write("# z, uzerr, lzerr, rho, rho_up, rho_low\n")

cnt = 0

# cnt changes at 7500, 465000, 922500, 1380000, and goes to the end :/

last_fit = None
pnum = [3, 4, 3, 2]

def f(loglf, theta, m, z, fit='individual'):
    global cnt, last_fit
    cnt += 1
    if fit == 'composite':
        # print('\ntheta:', theta)
        # print('m:', m)
        # print('z:', z)

        
        return 10.0**loglf(theta, m, z)

    return 10.0**loglf(theta, m)

def f2(loglf, theta, m, z, fit='individual'):
    global cnt, last_fit
    cnt += 1
    if fit == 'composite':
        # print('\ntheta:', theta)
        # print('m:', m)
        # print('z:', z)
        # QUICK SHORT HAND. NEED TO CHANGE THIS THOUGH!!!

        def getparams(theta):
            if isinstance(pnum, int):
                # Case 1: `self.pnum` is a single integer.
                # Each parameter group has the same number of parameters (`self.pnum`).
                splitlocs = pnum*np.array([1,2,3])
            else:
                # Case 2: `self.pnum` is an array or list.
                # Each parameter group has a different number of parameters.
                # The number of parameters for each group is given by `self.pnum[i]`.
                splitlocs = np.cumsum(pnum)

            return np.split(theta,splitlocs)
        params = getparams(theta)

        def atz(z, p):
            return np.polyval(p, z)

        log10phi_star = atz(z, params[0])
        M_star = atz(z, params[1])
        alpha = atz(z, params[2])
        beta = atz(z, params[3])
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(m-M_star)) +
                                     10.0**(0.4*(beta+1)*(m-M_star)))
        return phi
        

    return 10.0**loglf(theta, m)


def rhoqso(loglf, theta, mlim, z, fit='individual', mbright=-35.0):

    m = np.linspace(mbright, mlim, num=1000)
    if fit == 'composite':
        farr = f2(loglf, theta, m, z, fit='composite')
    else:
        farr = f(loglf, theta, m, z, fit='individual')
    
    return np.trapezoid(farr, m) # cMpc^-3

# def read_parameters_with_bp():
#     """
#     Copied from composite.py
#     Reads the parameters from the file 'parameters_with_bp.dat'.
#     """
#     filename = 'parameters_with_bp.dat'
#     zmean = []
#     values = []
#     intervals = []
#     with open(filename, 'r') as f:
#         print(f"Reading parameters from {filename}...")
#         for line in f:
#             line = line.strip()
#             # Skip comments and empty lines
#             if not line or line.startswith('#'):
#                 continue
#             # Split by the two '|' separators
#             if line.count('|') == 2:
#                 parts = line.split('|')
#                 left = parts[0].strip()
#                 middle = parts[1].strip()
#                 right = parts[2].strip()
#                 # left: zmean
#                 z = float(left)
#                 # middle: 4 parameter values
#                 vals = [float(x) for x in middle.split()]
#                 # right: 4 pairs of intervals (lower upper for each param)
#                 right_parts = right.split()
#                 interval_pairs = []
#                 for i in range(0, 8, 2):
#                     interval_pairs.append([float(right_parts[i]), float(right_parts[i+1])])
#                 zmean.append(z)
#                 values.append(vals)
#                 intervals.append(interval_pairs)
    
#     print(f"Read {len(zmean)} redshift bins from {filename}.")
#     zmean = np.array(zmean)
#     values = np.array(values)
#     intervals = np.array(intervals)  # shape (N, 4, 2)
#     ret = []
#     for i in range(len(zmean)):
#         ret.append([[values[i][0], intervals[i][0][0], intervals[i][0][1]],
#                     [values[i][1], intervals[i][1][0], intervals[i][1][1]],
#                     [values[i][2], intervals[i][2][0], intervals[i][2][1]],
#                     [values[i][3], intervals[i][3][0], intervals[i][3][1]]])

#     ret = np.transpose(ret, (1, 2, 0))  # shape (4, 3, N)
#     return ret, zmean


def read_parameters_with_bp(filename="parameters_with_bp_new.dat"):
    # Read the parameter file, skipping comments
    zlist = []
    params = []
    with open(filename) as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split('|')
            zmean = float(parts[0].split()[0])
            vals = [float(x) for x in parts[1].split()]
            zlist.append(zmean)
            params.append(vals)
    return np.array(zlist), np.array(params)

def get_rhoqso_from_file(mlim, z, mbright=-35.0, filename="parameters_with_bp.dat"):
    zlist, params = read_parameters_with_bp(filename)
    print(f"shape of zlist: {zlist.shape}, shape of params: {params.shape}")
    print(f"params: {params}")
    print(f"zlist: {zlist}")
    # Find the closest z in the file
    idx = np.abs(zlist - z).argmin()
    param = params[idx]  # [phi_star, M_star, alpha, beta]
    # You may need to adjust this depending on your QLF/rhoqso function signature
    # Example: call your QLF function with these params
    m = np.linspace(mbright, mlim, num=1000)
    # Example QLF calculation (replace with your actual function):
    phi_star, M_star, alpha, beta = param
    # Example: double power law (replace with your actual formula)
    phi = 10.0**phi_star / (10.0**(0.4*(alpha+1)*(m-M_star)) + 10.0**(0.4*(beta+1)*(m-M_star)))
    rho = np.trapezoid(phi, m)
    return rho, param, zlist[idx]


def get_rhoqso(lfi, mlim, z, fit='individual', mbright=-35.0):
    # lfi is just a placeholder, not used here
    # Use get_rhoqso_from_file to get rhoqso and parameter uncertainties
    rho, param, z_actual = get_rhoqso_from_file(mlim, z, mbright=mbright)
    # Set dummy uncertainties (0) since we don't have samples
    u = rho
    l = rho
    c = rho
    # Attach to lfi for compatibility with rest of code
    lfi.rhoqso = [u, l, c]

    return

def get_rhoqso2(lfi, mlim, z, fit='individual', mbright=-35.0):

    rindices = np.random.randint(len(lfi.samples), size=300)
    n = np.array([rhoqso(lfi.log10phi, theta, mlim, z, mbright=mbright) 
                  for theta
                  in lfi.samples[rindices]])
    u = np.percentile(n, 15.87) 
    l = np.percentile(n, 84.13)
    c = np.mean(n)
    lfi.rhoqso = [u, l, c]

    return

# def draw(individuals, zlims, select=False):

#     """
#     Calculates and plots LyC emissivity.

#     """

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 1.0e-3)

#     # Plot 1 
    
#     mlim = -18

#     if select: 
#         selected = [x for x in individuals if x.z.mean() < 2.0 or x.z.mean() > 2.8]
#     else:
#         selected = individuals

#     for x in selected:
#         get_rhoqso(x, mlim, x.z.mean())
    
#     c = np.array([x.rhoqso[2] for x in selected])
#     u = np.array([x.rhoqso[0] for x in selected])
#     l = np.array([x.rhoqso[1] for x in selected])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in selected])
#     uz = np.array([x.zlims[0] for x in selected])
#     lz = np.array([x.zlims[1] for x in selected])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c='tomato', edgecolor='None',
#                label='$M < -18$',
#                s=72, zorder=4, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor='tomato', capsize=0, fmt='None', elinewidth=2,
#                 yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), 
#                 zorder=3, mew=1, ms=5)

#     # Plot 2 
    
#     mlim = -21

#     for x in selected:
#         get_rhoqso(x, mlim, x.z.mean())
    
#     c = np.array([x.rhoqso[2] for x in selected])
#     u = np.array([x.rhoqso[0] for x in selected])
#     l = np.array([x.rhoqso[1] for x in selected])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in selected])
#     uz = np.array([x.zlims[0] for x in selected])
#     lz = np.array([x.zlims[1] for x in selected])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c='forestgreen', edgecolor='None',
#                label='$M<-21$',
#                s=72, zorder=4, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor='forestgreen', capsize=0, fmt='None',
#                 elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)

#     # Plot 3
    
#     mlim = -24

#     for x in selected:
#         get_rhoqso(x, mlim, x.z.mean())
    
#     c = np.array([x.rhoqso[2] for x in selected])
#     u = np.array([x.rhoqso[0] for x in selected])
#     l = np.array([x.rhoqso[1] for x in selected])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in selected])
#     uz = np.array([x.zlims[0] for x in selected])
#     lz = np.array([x.zlims[1] for x in selected])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c='goldenrod', edgecolor='None',
#                label='$M<-24$',
#                s=72, zorder=4, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor='goldenrod', capsize=0, fmt='None',
#                 elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)

#     # Plot 4
    
#     mlim = -27

#     for x in selected:
#         get_rhoqso(x, mlim, x.z.mean())
    
#     c = np.array([x.rhoqso[2] for x in selected])
#     u = np.array([x.rhoqso[0] for x in selected])
#     l = np.array([x.rhoqso[1] for x in selected])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in selected])
#     uz = np.array([x.zlims[0] for x in selected])
#     lz = np.array([x.zlims[1] for x in selected])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c='saddlebrown', edgecolor='None',
#                label='$M<-27$',
#                s=72, zorder=4, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor='saddlebrown', capsize=0, fmt='None',
#                 elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
#     plt.savefig('rhoqso.pdf',bbox_inches='tight')
#     plt.close('all')

#     return

def global_cumulative(ax, theta, mlim, color, **kwargs):
    # In this composite just is just a set of n parameter values
    nzs = 5000
    z = np.linspace(0, 30, nzs)
    

    c = np.array([rhoqso(lf.log10phi,theta, mlim, zi, fit='composite') for zi in z])
    # Set f as a very thin band around c (±0.0001)
    f = ax.fill_between(z, c - 1e-15, c + 1e-15, color=color, alpha=0.3, zorder=6)

    if color == 'forestgreen':
        p, = ax.plot(z, c, color=color, zorder=7)
    
    if color == 'peru': 
        p, = ax.plot(z, c, color='brown', zorder=7)

    if color == 'grey': 
        p, = ax.plot(z, c, color='k', zorder=7)
        
    return f, p 


def global_cumulative2(ax, composite, mlim, color, **kwargs):

    nzs = 500
    z = np.linspace(0, 7, nzs)
    nsample = 300
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]

    # bf = np.median(composite.samples, axis=0)
    # r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    # ax.plot(z, r, color='k', zorder=7)

    r = np.zeros((nsample, nzs))
    for i, theta in enumerate(rsample):
        r[i] = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])

    up = np.percentile(r, 15.87, axis=0)
    down = np.percentile(r, 84.13, axis=0)
    f = ax.fill_between(z, down, y2=up, color=color, zorder=6, alpha=0.7, **kwargs)

    c = np.median(r, axis=0)

    if color == 'forestgreen':
        p, = ax.plot(z, c, color=color, zorder=7)
    
    if color == 'peru': 
        p, = ax.plot(z, c, color='brown', zorder=7)

    if color == 'grey': 
        p, = ax.plot(z, c, color='k', zorder=7)
        
    return f, p 


# def global_differential(ax, composite, mbright, mfaint, color):

#     nzs = 50 
#     z = np.linspace(0, 7, nzs)
#     nsample = 300
#     rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]

#     bf = np.median(composite.samples, axis=0)
#     r = np.array([rhoqso(composite.log10phi, bf, mfaint, x, mbright=mbright, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     r = np.zeros((nsample, nzs))
#     for i, theta in enumerate(rsample):
#         r[i] = np.array([rhoqso(composite.log10phi, theta, mfaint, x, mbright=mbright, fit='composite') for x in z])

#     up = np.percentile(r, 15.87, axis=0)
#     down = np.percentile(r, 84.13, axis=0)

#     label = '${:d}>M>{:d}$'.format(mfaint, mbright) 
#     ax.fill_between(z, down, y2=up, color=color, zorder=6, alpha=0.5, label=label, linewidth=0)

#     return


# def global_optimum_differential(ax, composite, mbright, mfaint, color):

#     nzs = 50 
#     z = np.linspace(0, 7, nzs)

#     bf = composite.bf.x 
#     r = np.array([rhoqso(composite.log10phi, bf, mfaint, x, mbright=mbright, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     return


# def individuals_differential(ax, individuals, mbright, mfaint, color):

#     # These redshift bins are labelled "bad" and are plotted differently.
#     # reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

#     reject = []

#     # reject = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

#     m = np.ones(len(individuals), dtype=bool)
#     m[reject] = False
#     minv = np.logical_not(m)

#     individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
#     individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]
    
#     for x in individuals:
#         get_rhoqso(x, mfaint, x.z.mean(), mbright=mbright)
    
#     c = np.array([x.rhoqso[2] for x in individuals_good])
#     u = np.array([x.rhoqso[0] for x in individuals_good])
#     l = np.array([x.rhoqso[1] for x in individuals_good])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in individuals_good])
#     uz = np.array([x.zlims[0] for x in individuals_good])
#     lz = np.array([x.zlims[1] for x in individuals_good])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c=color, edgecolor='None',
#                s=72, zorder=10, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=2,
#                 yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), 
#                 zorder=10, mew=1, ms=5)

#     c = np.array([x.rhoqso[2] for x in individuals_bad])
#     u = np.array([x.rhoqso[0] for x in individuals_bad])
#     l = np.array([x.rhoqso[1] for x in individuals_bad])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in individuals_bad])
#     uz = np.array([x.zlims[0] for x in individuals_bad])
#     lz = np.array([x.zlims[1] for x in individuals_bad])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
#                 yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), 
#                 zorder=10, mew=1, ms=5)

#     ax.scatter(zs, rho, c='#ffffff', edgecolor=color,
#                s=72, zorder=10, linewidths=1) 

    
#     return 

# def individuals_cumulative(ax, individuals, mlim, color, label):

#     # These redshift bins are labelled "bad" and are plotted differently.
#     # reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

#     reject = []

#     # reject = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

#     m = np.ones(len(individuals), dtype=bool)
#     m[reject] = False
#     minv = np.logical_not(m)

#     individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
#     individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]
    
#     for x in individuals:
#         get_rhoqso(x, mlim, x.z.mean())
    
#     c = np.array([x.rhoqso[2] for x in individuals_good])
#     u = np.array([x.rhoqso[0] for x in individuals_good])
#     l = np.array([x.rhoqso[1] for x in individuals_good])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in individuals_good])
#     uz = np.array([x.zlims[0] for x in individuals_good])
#     lz = np.array([x.zlims[1] for x in individuals_good])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.scatter(zs, rho, c=color, edgecolor='None',
#                label=label,
#                s=72, zorder=10, linewidths=2) 

#     ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=2,
#                 yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), 
#                 zorder=10, mew=1, ms=5)

#     c = np.array([x.rhoqso[2] for x in individuals_bad])
#     u = np.array([x.rhoqso[0] for x in individuals_bad])
#     l = np.array([x.rhoqso[1] for x in individuals_bad])

#     rho = c
#     rho_up = np.abs(u - c)
#     rho_low = np.abs(c - l)
    
#     zs = np.array([x.z.mean() for x in individuals_bad])
#     uz = np.array([x.zlims[0] for x in individuals_bad])
#     lz = np.array([x.zlims[1] for x in individuals_bad])
    
#     uzerr = np.abs(uz - zs)
#     lzerr = np.abs(zs - lz)

#     ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
#                 yerr=np.vstack((rho_low, rho_up)),
#                 xerr=np.vstack((lzerr, uzerr)), 
#                 zorder=10, mew=1, ms=5)

#     ax.scatter(zs, rho, c='#ffffff', edgecolor=color,
#                s=72, zorder=10, linewidths=1) 

    
#     return 

def individuals_cumulative_multiple(ax, individuals, mlim, color, label):

    """A copy of individuals_cumulative for rhoqso_withGlobal_multiple.

    Some changes.

    """

    # These redshift bins are labelled "bad" and are plotted differently.
    # reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # print("\n\nIn individuals_cumulative_multiple")
    # print("individuals=", individuals)
    # print("mlim=", mlim)
    # print("len(individuals)=", len(individuals))

    reject = []

    # reject = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    # print("reject=", reject)
    m = np.ones(len(individuals), dtype=bool)
    # print("m=", m)
    # print("m[reject]=", m[reject])
    m[reject] = False
    
    minv = np.logical_not(m)

    # print("reject=", reject)
    # print("m=", m)
    # print("len(m)=", len(m))

    # for i, x in enumerate(individuals):
    #     print("i=", i)
    #     print("x=", x)

    #     print("i in set(reject)=", i in set(reject))

    individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
    # print("individuals_good=", individuals_good)
    individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]
    # print("individuals_bad=", individuals_bad)
    
    for x in individuals:
        get_rhoqso(x, mlim, x.z.mean())
        # print("x=", x)
    
    c = np.array([x.rhoqso[2] for x in individuals_good])
    u = np.array([x.rhoqso[0] for x in individuals_good])
    l = np.array([x.rhoqso[1] for x in individuals_good])

    print("\n\n\n\n\n\t[u, l, c]=", [u, l, c])

    rho = c
    rho_up = np.abs(u - c)
    rho_low = np.abs(c - l)

    print("[rho_up, rho_low]=", [rho_up, rho_low])
    print("rho=", rho)

    with open("check.dat", "a") as fcheck:
        for i in range(len(c)):
            fcheck.write(f"{u[i]:.6g} {l[i]:.6g} {c[i]:.6g} {rho_up[i]:.6g} {rho_low[i]:.6g}\n")
    import sys; sys.exit(0)

    zs = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])

    print("[uz, zs, lz]=", [uz, zs, lz])
    
    uzerr = np.abs(uz - zs)
    lzerr = np.abs(zs - lz)

    print("[uzerr, lzerr]=", [uzerr, lzerr])
    print("\n*************************************************************\n")

    with open(output_data_file, 'a') as f:
        f.write("#\n")
        f.write("# Storing the data points and its uncertainities\n")
        for i in range(len(zs)):
            f.write(f"{zs[i]:.4g}, {uzerr[i]:.4g}, {lzerr[i]:.4g}, {rho[i]:.4g}, {rho_up[i]:.4g}, {rho_low[i]:.4g}, {mlim:.3g}\n")

    ax.scatter(zs, rho, c=color, edgecolor='None',
               s=42, zorder=10, linewidths=2) 
    

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    # def func(z, a, b, c, d, e):
    #     e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
    #     return e 
    
    # sigma = u-l 
    # z = np.linspace(0.0, 7, num=1000)
    # samples = fit_emissivity.fit(zs, rho, sigma)
    # nsample = 300
    # rsample = samples[np.random.randint(len(samples), size=nsample)]
    # nzs = len(z) 
    # e = np.zeros((nsample, nzs))
    # for i, theta in enumerate(rsample):
    #     e[i] = np.array(func(z, *theta))
    #     plt.plot(z, e[i], lw=2, c='red', zorder=5, alpha=0.1)

    # up = np.percentile(e, 15.87, axis=0)
    # print( 'up=', up )
    # down = np.percentile(e, 84.13, axis=0)
    # print( 'down=', down)
    # tw18f = ax.fill_between(z, down, y2=up, color='red', zorder=5, alpha=0.6, edgecolor='None')
    
    # b = np.median(e, axis=0)
    # tw18, = plt.plot(z, b, lw=2, c='red', zorder=5)

    c = np.array([x.rhoqso[2] for x in individuals_bad])
    u = np.array([x.rhoqso[0] for x in individuals_bad])
    l = np.array([x.rhoqso[1] for x in individuals_bad])

    rho = c
    rho_up = np.abs(u - c)
    rho_low = np.abs(c - l)
    
    zs = np.array([x.z.mean() for x in individuals_bad])
    uz = np.array([x.zlims[0] for x in individuals_bad])
    lz = np.array([x.zlims[1] for x in individuals_bad])
    
    uzerr = np.abs(uz - zs)
    lzerr = np.abs(zs - lz)

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    ax.scatter(zs, rho, c='#ffffff', edgecolor=color,
               s=42, zorder=10, linewidths=1) 

    
    return 
    

# def draw_withGlobal_dense(composite, individuals, zlims, select=False):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 4.0e-3)

#     mlim = -18
#     individuals_cumulative(ax, individuals, mlim, 'tomato', '$M<-18$')
#     global_cumulative(ax, composite, mlim, 'tomato')

#     mlim = -19
#     c='#ff7f0e'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-19$')
#     global_cumulative(ax, composite, mlim, c)
    
#     mlim = -20
#     c='#1f77b4'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-20$')
#     global_cumulative(ax, composite, mlim, c)
    
#     mlim = -21
#     individuals_cumulative(ax, individuals, mlim, 'forestgreen', '$M<-21$')
#     global_cumulative(ax, composite, mlim, 'forestgreen')

#     mlim = -22
#     c='#9467bd'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-22$')
#     global_cumulative(ax, composite, mlim, c)
    
#     mlim = -23
#     c='#8c564b'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-23$')
#     global_cumulative(ax, composite, mlim, c)
    
#     mlim = -24
#     individuals_cumulative(ax, individuals, mlim, 'goldenrod', '$M<-24$')
#     global_cumulative(ax, composite, mlim, 'goldenrod')

#     mlim = -25
#     c='#bcbd22'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-25$')
#     global_cumulative(ax, composite, mlim, c)

#     mlim = -26
#     c = '#7f7f7f'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-26$')
#     global_cumulative(ax, composite, mlim, c)
    
#     mlim = -27
#     c = '#17becf'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-27$')
#     global_cumulative(ax, composite, mlim, c)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01,
#                scatterpoints=1, ncol=2)
    
#     plt.savefig('rhoqso_withGlobal.pdf',bbox_inches='tight')
#     plt.close('all')

#     return

# def draw_withGlobal(composite, individuals, zlims, select=False):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 1.0e-3)

#     mlim = -18
#     individuals_cumulative(ax, individuals, mlim, 'tomato', '$M<-18$')
#     global_cumulative(ax, composite, mlim, 'tomato')

#     mlim = -21
#     individuals_cumulative(ax, individuals, mlim, 'forestgreen', '$M<-21$')
#     global_cumulative(ax, composite, mlim, 'forestgreen')

#     mlim = -24
#     individuals_cumulative(ax, individuals, mlim, 'goldenrod', '$M<-24$')
#     global_cumulative(ax, composite, mlim, 'goldenrod')

#     mlim = -27
#     c = '#17becf'
#     individuals_cumulative(ax, individuals, mlim, c, '$M<-27$')
#     global_cumulative(ax, composite, mlim, c)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01,
#                scatterpoints=1)
    
#     plt.savefig('rhoqso_withGlobal.pdf',bbox_inches='tight')
#     plt.close('all')

#     return

def draw_withGlobal_multiple(c1, c2, c3, individuals, select=False, filename='rhoqso_withGlobal_multiple.pdf'):
    # bins.py data is individual
    # c1, c2, c3 are composite models
    # I will use jst c1 and c2

    fig = plt.figure(figsize=(7, 11), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,30)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-11, 1000000000.0)

    mlim = -18
    individuals_cumulative_multiple(ax, individuals, mlim, 'k', '$M<-18$')
    global_cumulative(ax, c1, mlim, 'grey', label='Model 1')
    global_cumulative(ax, c2, mlim, 'forestgreen', label='Model 2')
    global_cumulative(ax, c3, mlim, 'peru', label='Model 3')

    plt.text(0.7, 1.2e-4, '$M_{1450}<-18$', rotation=52, fontsize=14, ha='center')

    mlim = -21
    individuals_cumulative_multiple(ax, individuals, mlim, 'k', '$M<-21$')
    global_cumulative(ax, c1, mlim, 'grey')
    global_cumulative(ax, c2, mlim, 'forestgreen')
    global_cumulative(ax, c3, mlim, 'peru')

    plt.text(0.7, 1.2e-5, '$M_{1450}<-21$', rotation=55, fontsize=14, ha='center')
        
    mlim = -24
    individuals_cumulative_multiple(ax, individuals, mlim, 'k', '$M<-24$')
    global_cumulative(ax, c1, mlim, 'grey')
    global_cumulative(ax, c2, mlim, 'forestgreen')
    global_cumulative(ax, c3, mlim, 'peru')

    plt.text(0.5, 2e-7, '$M_{1450}<-24$', rotation=72, fontsize=14, ha='center')

    mlim = -27
    c = '#17becf'
    individuals_cumulative_multiple(ax, individuals, mlim, 'k', '$M<-27$')
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')

    plt.text(1, 2.0e-9, '$M_{1450}<-27$', rotation=67, fontsize=14, ha='center')

    handles, labels = [], []
    handles.append((m1f, m1))
    labels.append('Model 1')

    handles.append((m2f, m2))
    labels.append('Model 2')

    handles.append((m3f, m3))
    labels.append('Model 3')

    plt.legend(handles, labels, loc='upper left', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.3, borderpad=0.1,
               scatterpoints=1)
    
    plt.savefig(filename,bbox_inches='tight')
    plt.close('all')

    print(f'Saved the plot in {filename}')

    return


# def draw_differential_dense(composite, individuals, zlims, select=False):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 4.0e-3)

#     mf = -18
#     mb = -19
#     c = 'tomato'
#     global_differential(ax, composite, mb, mf, c)

#     mf = -19
#     mb = -20
#     c='#ff7f0e'
#     global_differential(ax, composite, mf, mf, c)
    
#     mf = -20
#     mb = -21
#     c='#1f77b4'
#     global_differential(ax, composite, mb, mf, c)
    
#     mf = -21
#     mb = -22
#     c = 'forestgreen'
#     global_differential(ax, composite, mb, mf, c)

#     mf = -22
#     mb = -23
#     c='#9467bd'
#     global_differential(ax, composite, mb, mf, c)
    
#     mf = -23
#     mb = -24
#     c='#8c564b'
#     global_differential(ax, composite, mb, mf, c)
    
#     mf = -24
#     mb = -25
#     c = 'goldenrod'
#     global_differential(ax, composite, mb, mf, c)

#     mf = -25
#     mb = -26
#     c='#bcbd22'
#     global_differential(ax, composite, mb, mf, c)

#     mf = -26
#     mb = -27
#     c = '#7f7f7f'
#     global_differential(ax, composite, mb, mf, c)
    
#     mf = -27
#     mb = -28
#     c = '#17becf'
#     global_differential(ax, composite, mb, mf, c)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01,
#                scatterpoints=1, ncol=2)
    
#     plt.savefig('rhoqso_diff.pdf',bbox_inches='tight')
#     plt.close('all')

#     return


# def draw_differential_dense_optimum(composite, individuals, zlims, select=False):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 4.0e-3)

#     mf = -18
#     mb = -19
#     c = 'tomato'
#     global_optimum_differential(ax, composite, mb, mf, c)

#     mf = -19
#     mb = -20
#     c='#ff7f0e'
#     global_optimum_differential(ax, composite, mf, mf, c)
    
#     mf = -20
#     mb = -21
#     c='#1f77b4'
#     global_optimum_differential(ax, composite, mb, mf, c)
    
#     mf = -21
#     mb = -22
#     c = 'forestgreen'
#     global_optimum_differential(ax, composite, mb, mf, c)

#     mf = -22
#     mb = -23
#     c='#9467bd'
#     global_optimum_differential(ax, composite, mb, mf, c)
    
#     mf = -23
#     mb = -24
#     c='#8c564b'
#     global_optimum_differential(ax, composite, mb, mf, c)
    
#     mf = -24
#     mb = -25
#     c = 'goldenrod'
#     global_optimum_differential(ax, composite, mb, mf, c)

#     mf = -25
#     mb = -26
#     c='#bcbd22'
#     global_optimum_differential(ax, composite, mb, mf, c)

#     mf = -26
#     mb = -27
#     c = '#7f7f7f'
#     global_optimum_differential(ax, composite, mb, mf, c)
    
#     mf = -27
#     mb = -28
#     c = '#17becf'
#     global_optimum_differential(ax, composite, mb, mf, c)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01,
#                scatterpoints=1, ncol=2)
    
#     plt.savefig('rhoqso_diff.pdf',bbox_inches='tight')
#     plt.close('all')

#     return


# def draw_differential(composite, individuals, zlims, select=False):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 4.0e-3)

#     mf = -18
#     mb = -21
#     c = 'tomato'
#     individuals_differential(ax, individuals, mb, mf, c)
#     global_differential(ax, composite, mb, mf, c)

#     mf = -21
#     mb = -24
#     c = 'forestgreen'
#     individuals_differential(ax, individuals, mb, mf, c)
#     global_differential(ax, composite, mb, mf, c)

#     mf = -24
#     mb = -27
#     c = 'goldenrod'
#     individuals_differential(ax, individuals, mb, mf, c)
#     global_differential(ax, composite, mb, mf, c)

#     mf = -27
#     mb = -30
#     c = 'saddlebrown'
#     individuals_differential(ax, individuals, mb, mf, c)
#     global_differential(ax, composite, mb, mf, c)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01,
#                scatterpoints=1, ncol=2)
    
#     plt.savefig('rhoqso_diff_ind.pdf',bbox_inches='tight')
#     plt.close('all')

#     return

# def draw_onlyGlobal(composite):

#     fig = plt.figure(figsize=(7, 10), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)

#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=5, width=1)

#     ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
#     ax.set_xlabel('$z$')
#     ax.set_xlim(0.,7)

#     ax.set_yscale('log')
#     ax.set_ylim(1.0e-10, 1.0e-3)

#     # Plot 1 
    
#     mlim = -18

#     z = np.linspace(0, 7, 50)
#     bf = np.median(composite.samples, axis=0)
#     r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
#         r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
#         ax.plot(z, r, color='tomato', zorder=6, alpha=0.02)
    

#     # Plot 2 
    
#     mlim = -21

#     z = np.linspace(0, 7, 50)
#     bf = np.median(composite.samples, axis=0)
#     r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
#         r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
#         ax.plot(z, r, color='forestgreen', zorder=6, alpha=0.02)
    

#     # Plot 3
    
#     mlim = -24

#     z = np.linspace(0, 7, 50)
#     bf = np.median(composite.samples, axis=0)
#     r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
#         r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
#         ax.plot(z, r, color='goldenrod', zorder=6, alpha=0.02)

    
#     # Plot 4
    
#     mlim = -27

#     z = np.linspace(0, 7, 50)
#     bf = np.median(composite.samples, axis=0)
#     r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
#     ax.plot(z, r, color='k', zorder=7)

#     for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
#         r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
#         ax.plot(z, r, color='saddlebrown', zorder=6, alpha=0.02)
    
#     plt.legend(loc='upper left', fontsize=14, handlelength=1,
#                frameon=False, framealpha=0.0, labelspacing=.1,
#                handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
#     plt.savefig('rhoqso_onlyGlobal.pdf',bbox_inches='tight')
#     plt.close('all')

#     return





# if __name__ == '__main__':
#     import time

#     start_time = time.time()
#     print("Start time:", start_time)

#     lfg1 = np.load('lfg1_old_data.npy', allow_pickle=True)
#     lfg2 = np.load('lfg2_old_data.npy', allow_pickle=True)
#     lfg3 = np.load('lfg3_old_data.npy', allow_pickle=True)
#     bins_lfs = np.load('bins_lfs_old_data.npy', allow_pickle=True)

#     lfg1 = lfg1.tolist()
#     lfg2 = lfg2.tolist()
#     lfg3 = lfg3.tolist()
#     bins_lfs = bins_lfs.tolist()

#     print("\n\n\n")
#     print(type(lfg1))
#     print(dir(lfg1))
#     print("\n\n\n")
#     print(type(bins_lfs))
#     print(dir(bins_lfs))
#     print("\n\n\n")

#     # lfg3 = None

#     # import sys
#     # sys.exit(0)
#     draw_withGlobal_multiple(lfg1, lfg2, lfg3, bins_lfs, select=False, filename='rhoqso_withGlobal_multiple2.pdf')

#     duration = 3  # seconds
#     frequency = 440  # Hz, the frequency of the beep sound (440Hz is standard A note)

#     # Generate sound wave (440Hz sine wave)
#     sample_rate = 44100  # samples per second
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     wave = 0.5 * np.sin(2 * np.pi * frequency * t)

#     # Play the generated sound wave
#     sd.play(wave, samplerate=sample_rate)

#     sd.wait()  # Wait until the sound is finished playing

#     end_time = time.time()
#     print("End time:", end_time)
#     print("Duration:", end_time - start_time)


if __name__ == '__main__':
    import time

    start_time = time.time()
    print("Start time:", start_time)

    # lfg1 = np.load('lfg1_old_data.npy', allow_pickle=True)
    # lfg2 = np.load('lfg2_old_data.npy', allow_pickle=True)
    # lfg3 = np.load('lfg3_old_data.npy', allow_pickle=True)
    bins_lfs = np.load('bins_lfs_old_data.npy', allow_pickle=True)

    # lfg1 = lfg1.tolist()
    # lfg2 = lfg2.tolist()
    # lfg3 = lfg3.tolist()
    bins_lfs = bins_lfs.tolist()

    lfg1 = [0.04689, -0.17632, 0.23130, -0.00477, -0.04919, -0.49998, -4.94620, -0.02253, 0.07626, -1.43122, 0.23785, -1.44757, 0.59961, -7.14684, 1.42234]
    lfg1 = lfg1[:-3]
    lfg2 = lfg1
    lfg3 = lfg1


    draw_withGlobal_multiple(lfg1, lfg2, lfg3, bins_lfs, select=False, filename='rhoqso_withGlobal_multiple.pdf')

    end_time = time.time()
    print("End time:", end_time)
    print("Duration:", end_time - start_time)