import numpy as np

def phi(N, Vbin):
    return N/Vbin

def Vbin(dVdz, dz, dm):
    return dVdz * 2 * dz * 2 * dm

z = np.array([5.15,5.24])
dz = 0.5

m = -19.4
dm = 0.5

import cosmolopy.distance as cd
import astropy.stats
import matplotlib.pyplot as plt

cosmo = {'omega_M_0': 0.3,
         'omega_lambda_0': 0.7,
         'omega_k_0': 0,
         'h': 0.7}

A= 0.009583 # in deg^2, but in arcmin^2 it is 34.5 arcmin^2
def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1
    # print("volperstr at z = ", z, " is ", volperstr)

    return omega*volperstr # cMpc^3 dz^-1


dVdz = volume(z, A, cosmo)

phi_val = phi(1, Vbin(dVdz, dz, dm))
print()
print("phi is ", phi_val)
print("ratio between phi is ", phi_val[1]/phi_val[0])
print("ratio between required phi is ", 1.075/1.025, " to ", 1.065/1.035)
print()

dm_values = np.arange(0.5, 0.55, 0.0005)
phi_values = []

for dm_val in dm_values:
    phi_val = phi(1, Vbin(dVdz, dz, dm_val))
    phi_values.append(phi_val)


# Assuming phi_values and dm_values are already defined
phi_values = np.array(phi_values)
print("phi_values are ", phi_values)

# Plot phi with respect to dm
plt.plot(dm_values, phi_values[:, 0], label='Phi for z = 5.15')
plt.plot(dm_values, phi_values[:, 1], label='Phi for z = 5.24')

# Plot the given values
given_values = np.array([1.025, 1.03, 1.035, 1.065, 1.07, 1.075]) * 10**-5
plt.plot(dm_values, np.full_like(dm_values, given_values[1]), 'r-', label='Main Line 1.03')
plt.fill_between(dm_values, given_values[0], given_values[2], color='red', alpha=0.3)
plt.plot(dm_values, np.full_like(dm_values, given_values[4]), 'g-', label='Main Line 1.07')
plt.fill_between(dm_values, given_values[3], given_values[5], color='green', alpha=0.3)



def range(phi, value):
    return (phi > value)

def index(phi, value):
    return np.where(range(phi, value))[0][-1]

print("range is ", range(phi_values[:, 0], 1.03 * 10**-5))
print("index is ", index(phi_values[:, 0], 1.03 * 10**-5))
print("range is ", range(phi_values[:, 0], 1.025 * 10**-5))
print("index is ", index(phi_values[:, 0], 1.025 * 10**-5))
print("range is ", range(phi_values[:, 0], 1.035 * 10**-5))
print("index is ", index(phi_values[:, 0], 1.035 * 10**-5))

print("range is ", range(phi_values[:, 1], 1.07 * 10**-5))
print("index is ", index(phi_values[:, 1], 1.07 * 10**-5))
print("range is ", range(phi_values[:, 1], 1.065 * 10**-5))
print("index is ", index(phi_values[:, 1], 1.065 * 10**-5))
print("range is ", range(phi_values[:, 1], 1.075 * 10**-5))
print("index is ", index(phi_values[:, 1], 1.075 * 10**-5))

index_5_15_1_03 = index(phi_values[:, 0], 1.03 * 10**-5)
index_5_15_1_025 = index(phi_values[:, 0], 1.025 * 10**-5)
index_5_15_1_035 = index(phi_values[:, 0], 1.035 * 10**-5)

index_5_24_1_07 = index(phi_values[:, 1], 1.07 * 10**-5)
index_5_24_1_065 = index(phi_values[:, 1], 1.065 * 10**-5)
index_5_24_1_075 = index(phi_values[:, 1], 1.075 * 10**-5)

# # Manually check for the range where phi crosses the given values
# range_5_15 = (phi_values[:, 0] > 1.03 * 10**-5)
# range_5_24 = (phi_values[:, 1] > 1.07 * 10**-5)

# index_5_15 = np.where(range_5_15)[0][-1]
# index_5_24 = np.where(range_5_24)[0][-1]

# print("range_5_15 is ", range_5_15)
# print("range_5_24 is ", range_5_24)

# print("index_5_15 is ", index_5_15)
# print("index_5_24 is ", index_5_24)

# Fit a straight line between the two values and interpolate
def linear_interpolation(y, x, target):
    print("x is ", x)
    print("y is ", y)
    print("target is ", target)
    res = y[0] + (target - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    return res

# print("dm_values[index_5_15] is ", dm_values[index_5_15])
# print("dm_values[index_5_15 + 1] is ", dm_values[index_5_15 + 1])
# print("phi_values[index_5_15, 0] is ", phi_values[index_5_15, 0])
# print("phi_values[index_5_15 + 1, 0] is ", phi_values[index_5_15 + 1, 0])

# print("dm_values[index_5_24] is ", dm_values[index_5_24])
# print("dm_values[index_5_24 + 1] is ", dm_values[index_5_24 + 1])
# print("phi_values[index_5_24, 1] is ", phi_values[index_5_24, 1])
# print("phi_values[index_5_24 + 1, 1] is ", phi_values[index_5_24 + 1, 1])

intersection_5_15_1_025 = linear_interpolation(
    [dm_values[index_5_15_1_025], dm_values[index_5_15_1_025 + 1]], 
    [phi_values[index_5_15_1_025, 0], phi_values[index_5_15_1_025 + 1, 0]], 
    1.025 * 10**-5
)
intersection_5_15_1_03 = linear_interpolation(
    [dm_values[index_5_15_1_03], dm_values[index_5_15_1_03 + 1]],
    [phi_values[index_5_15_1_03, 0], phi_values[index_5_15_1_03 + 1, 0]],
    1.03 * 10**-5
)
intersection_5_15_1_035 = linear_interpolation(
    [dm_values[index_5_15_1_035], dm_values[index_5_15_1_035 + 1]],
    [phi_values[index_5_15_1_035, 0], phi_values[index_5_15_1_035 + 1, 0]],
    1.035 * 10**-5
)

intersection_5_24_1_065 = linear_interpolation(
    [dm_values[index_5_24_1_065], dm_values[index_5_24_1_065 + 1]],
    [phi_values[index_5_24_1_065, 1], phi_values[index_5_24_1_065 + 1, 1]],
    1.065 * 10**-5
)
intersection_5_24_1_07 = linear_interpolation(
    [dm_values[index_5_24_1_07], dm_values[index_5_24_1_07 + 1]],
    [phi_values[index_5_24_1_07, 1], phi_values[index_5_24_1_07 + 1, 1]],
    1.07 * 10**-5
)
intersection_5_24_1_075 = linear_interpolation(
    [dm_values[index_5_24_1_075], dm_values[index_5_24_1_075 + 1]],
    [phi_values[index_5_24_1_075, 1], phi_values[index_5_24_1_075 + 1, 1]],
    1.075 * 10**-5
)

print()
print("intersection_5_15_1_025 is ", intersection_5_15_1_025)
# print("dm_values[index_5_15_1_025] is ", dm_values[index_5_15_1_025])
# print("dm_values[index_5_15_1_025 + 1] is ", dm_values[index_5_15_1_025 + 1])
# print("phi_values[index_5_15_1_025, 0] is ", phi_values[index_5_15_1_025, 0])
# print("phi_values[index_5_15_1_025 + 1, 0] is ", phi_values[index_5_15_1_025 + 1, 0])
print("intersection_5_15_1_03 is ", intersection_5_15_1_03)
# print("dm_values[index_5_15_1_03] is ", dm_values[index_5_15_1_03])
# print("dm_values[index_5_15_1_03 + 1] is ", dm_values[index_5_15_1_03 + 1])
# print("phi_values[index_5_15_1_03, 0] is ", phi_values[index_5_15_1_03, 0])
# print("phi_values[index_5_15_1_03 + 1, 0] is ", phi_values[index_5_15_1_03 + 1, 0])
print("intersection_5_15_1_035 is ", intersection_5_15_1_035)
# print("dm_values[index_5_15_1_035] is ", dm_values[index_5_15_1_035])
# print("dm_values[index_5_15_1_035 + 1] is ", dm_values[index_5_15_1_035 + 1])
# print("phi_values[index_5_15_1_035, 0] is ", phi_values[index_5_15_1_035, 0])
# print("phi_values[index_5_15_1_035 + 1, 0] is ", phi_values[index_5_15_1_035 + 1, 0])
print()
print("intersection_5_24_1_065 is ", intersection_5_24_1_065)
# print("dm_values[index_5_24_1_065] is ", dm_values[index_5_24_1_065])
# print("dm_values[index_5_24_1_065 + 1] is ", dm_values[index_5_24_1_065 + 1])
# print("phi_values[index_5_24_1_065, 1] is ", phi_values[index_5_24_1_065, 1])
# print("phi_values[index_5_24_1_065 + 1, 1] is ", phi_values[index_5_24_1_065 + 1, 1])
print("intersection_5_24_1_07 is ", intersection_5_24_1_07)
# print("dm_values[index_5_24_1_07] is ", dm_values[index_5_24_1_07])
# print("dm_values[index_5_24_1_07 + 1] is ", dm_values[index_5_24_1_07 + 1])
# print("phi_values[index_5_24_1_07, 1] is ", phi_values[index_5_24_1_07, 1])
# print("phi_values[index_5_24_1_07 + 1, 1] is ", phi_values[index_5_24_1_07 + 1, 1])
print("intersection_5_24_1_075 is ", intersection_5_24_1_075)
# print("dm_values[index_5_24_1_075] is ", dm_values[index_5_24_1_075])
# print("dm_values[index_5_24_1_075 + 1] is ", dm_values[index_5_24_1_075 + 1])
# print("phi_values[index_5_24_1_075, 1] is ", phi_values[index_5_24_1_075, 1])
# print("phi_values[index_5_24_1_075 + 1, 1] is ", phi_values[index_5_24_1_075 + 1, 1])
print()

# Plot the intersection points
plt.plot([intersection_5_15_1_025, intersection_5_15_1_03, intersection_5_15_1_035], [1.025 * 10**-5, 1.03 * 10**-5, 1.035 * 10**-5], 'ro')
plt.plot([intersection_5_24_1_065, intersection_5_24_1_07, intersection_5_24_1_075], [1.065 * 10**-5, 1.07 * 10**-5, 1.075 * 10**-5], 'go')


# Add circles around the intersection points
# plt.gca().add_patch(plt.Circle((intersection_5_15, 1.03 * 10**-5), 0.5e-6, color='red', fill=False))
# plt.gca().add_patch(plt.Circle((intersection_5_24, 1.07 * 10**-5), 0.5e-6, color='green', fill=False))

# Set plot labels and title
plt.xlabel('dm')
plt.ylabel('Phi')
plt.title('Phi vs dm')
plt.legend()
plt.grid(True)
plt.show()

import sys
sys.exit()

bins = np.array([-30.9, -30.3, -29.7, -29.1, -28.5, -27.9, -27.3, -26.7, -26.1, -25.5, -24.9, -24.3, -23.7, -23.1, -22.5, -21.9, -21.3, -20.7, -20.1, -19.5, -18.9, -18.3, -17.7])
print("bins are ", bins)

h = np.histogram(m, bins=bins)[0]
print("h is ", h)
nlims = astropy.stats.poisson_conf_interval(h,interval='frequentist-confidence')
print("nlims is ", nlims)
print("phi_val is ", phi_val)
phival1, phival2 = phi_val

nlims1 = nlims * phival1/h
nlims2 = nlims * phival2/h

print("nlims1 is ", nlims1)
print("nlims2 is ", nlims2)

import matplotlib.pyplot as plt

print("x size is ", bins[:-1].shape)
print("y size is ", (h * phival1).shape)
print("yerr size is ", [(phival1 - nlims1[0]).shape, (nlims1[1] - phival1).shape])

logphi1 = np.log10(h * phival1)
logphi2 = np.log10(h * phival2)

uppererr1 = np.log10(nlims1[1]) - logphi1
lowererr1 = logphi1 - np.log10(nlims1[0])

uppererr2 = np.log10(nlims2[1]) - logphi2
lowererr2 = logphi2 - np.log10(nlims2[0])

# Plot the luminosity function
plt.errorbar(bins[:-1], logphi1, yerr=[lowererr1, uppererr1], capsize=3, capthick=2, fmt='o', label='Luminosity Function for z = 5.15')
plt.errorbar(bins[:-1], logphi2, yerr=[lowererr2, uppererr2], capsize=3, capthick=2, fmt='o', label='Luminosity Function for z = 5.24')

# Set plot labels and title
plt.xlabel('Magnitude')
plt.ylabel('Log Phi')
plt.ylim(-10, -2)
plt.title('Luminosity Function with Error Bars')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()





logphi1 = np.log10(h * phival1)
logphi2 = np.log10(h * phival2)

uppererr1 = np.log10(nlims1[1]) - logphi1
lowererr1 = logphi1 - np.log10(nlims1[0])
lowererr1 = np.inf

print("uppererr1 is ", uppererr1)
print("lowererr1 is ", lowererr1)

uppererr2 = np.log10(nlims2[1]) - logphi2
lowererr2 = logphi2 - np.log10(nlims2[0])
lowererr2 = np.inf

print("uppererr2 is ", uppererr2)
print("lowererr2 is ", lowererr2)

# Convert to linear scale for plotting
phi1 = h * phival1
phi2 = h * phival2

uppererr1_linear = phi1 * (10**uppererr1 - 1)
lowererr1_linear = phi1 * (1 - 10**-lowererr1)

print("uppererr1_linear is ", uppererr1_linear)
print("lowererr1_linear is ", lowererr1_linear)

uppererr2_linear = phi2 * (10**uppererr2 - 1)
lowererr2_linear = phi2 * (1 - 10**-lowererr2)

print("uppererr2_linear is ", uppererr2_linear)
print("lowererr2_linear is ", lowererr2_linear)

# Plot the luminosity function using semilogy
plt.semilogy(bins[:-1], phi1, 'o', label='Luminosity Function for z = 5.15')
plt.errorbar(bins[:-1], phi1, yerr=[lowererr1_linear, uppererr1_linear], capsize=3, capthick=2, fmt='none')

plt.semilogy(bins[:-1], phi2, 'o', label='Luminosity Function for z = 5.24')
plt.errorbar(bins[:-1], phi2, yerr=[lowererr2_linear, uppererr2_linear], capsize=3, capthick=2, fmt='none')

# Set plot labels and title
plt.xlabel('Magnitude')
plt.ylabel('Phi')
plt.ylim(1e-10, 1e-2)
plt.title('Luminosity Function with Error Bars')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.gca().invert_xaxis()
plt.show()


# From https://www.astro.ucla.edu/~wright/CosmoCalc.html
# H0 = 70
# OmegaM = 0.3
# OmegaL = 0.7

# z = 5.24
# comoving radius = 7896.5 Mpc
# r^2 = 62354712.25
# comoving volume = 2062.381 Gpc3


# z = 5.15
# comoving radius = 7851.1 Mpc
# r^2 = 61639771.21
# comoving volume = 2027.029 Gpc3


# FOR GETTING THE UNCERTAINITY
# FROM drawlf line 117 to 240

# def get_lf(lf, sid, z_plot, special='None'):
#     # print("In drawlf.py get_lf")
    
#     # Bin data.  This is only for visualisation and to compare
#     # with reported binned values.  
#     m = lf.M1450[lf.sid==sid]

#     selmaps = [x for x in lf.maps if x.sid == sid]

#     if sid==6:
#         # Glikman's sample needs wider bins.
#         bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
#     elif sid == 7:
#         bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
#     elif sid == 10 or sid == 18:
#         bins = np.arange(-30.9, -17.3, 1.8)
#     elif special == 'croom_comparison':
#         # These M1450 bins result in the Mgz2 bins of Croom09.  The
#         # 1.23 converts between the two magnitudes (Eqn B8 of Ross13).
#         bins = np.arange(-30,-19.5,0.5) + 1.23
#     elif special == 'croom_comparison_Mgz2':
#         # These Mgz2 bins of Croom09.  
#         bins = np.arange(-30,-19.5,0.5)
#     else:
#         bins = np.arange(-30.9, -17.3, 0.6)

#     v1 = np.array([totBinVol(lf, x, bins, selmaps) for x in m])

#     v1_nonzero = v1[np.where(v1>0.0)]
#     m = m[np.where(v1>0.0)]

#     h = np.histogram(m, bins=bins, weights=1.0/(v1_nonzero))

#     nums = h[0]
#     mags = (h[1][:-1] + h[1][1:])*0.5
#     dmags = np.diff(h[1])*0.5

#     left = mags - h[1][:-1]
#     right = h[1][1:] - mags

#     phi = nums
#     logphi = np.log10(phi) # cMpc^-3 mag^-1

#     # print( 'sid=', sid )
#     # print( 'mags=', mags)
#     # print( 'nums=', nums)
#     # print( 'total=', np.sum(nums))

#     # Calculate errorbars on our binned LF.  These have been estimated
#     # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
#     # implemented in astropy.stats.poisson_conf_interval.  The
#     # interval='frequentist-confidence' option to that astropy function is
#     # exactly equal to the Gehrels formulas, although the documentation
#     # does not say so.
#     n = np.histogram(m, bins=bins)[0]
#     nlims = pci(n,interval='frequentist-confidence')
#     nlims *= phi/n 
#     uperr = np.log10(nlims[1]) - logphi 
#     downerr = logphi - np.log10(nlims[0])

#     return mags, left, right, logphi, uperr, downerr





# 5. ‘frequentist-confidence’ These are frequentist central confidence intervals:

# where 
#  is the quantile of the chi-square distribution with the indicated number of degrees of freedom and 
#  is the one-tailed probability of the normal distribution (at the point given by the parameter ‘sigma’). See Maxwell (2011) for further details.

#####################################################################################################################################################################################################################################################################################################################################################################################################################
# BUT NOTE THIS!!!!!
# THE DOCUMENT SAYS THAT 
# "
# 4. ‘sherpagehrels’ This rule is used by default in the fitting package ‘sherpa’. The documentation claims it is based on a numerical approximation published in Gehrels (1986) but it does not actually appear there. It is symmetrical, and while the upper limits are within about 1% of those given by ‘frequentist-confidence’, the lower limits can be badly wrong. The interval is
# "
#####################################################################################################################################################################################################################################################################################################################################################################################################################