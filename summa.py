import numpy as np

"""
Checks for the discrepancy in the phi values from Kocevski (2023) and Onoue (2023)
"""

# Note: Kocevski paper found out M_1450 = -19.44 +- 0.05
# whereas Onoue paper found out M_1450 = -19.5 +- 0.3
# In QLF calculation -19.4 and -19.5 are used for Kocevski and Onoue respectively

def phi(N, Vbin):
    return N/Vbin

def Vbin(dVdz, dz, dm):
    return dVdz * 2 * dz * 2 * dm

z = np.array([5.15,5.24])
dz = 0.5

m = -19.5
dm = 0.5

import cosmolopy.distance as cd
import astropy.stats
x = 0.3
cosmo = {'omega_M_0':x,
         'omega_lambda_0':1-x,
         'omega_k_0':0,
         'h':0.7}

A= 0.0095833333 # in deg^2, but in arcmin^2 it is 34.5 arcmin^2
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
import matplotlib.pyplot as plt

x_values = np.arange(0.25, 0.35, 0.01)
y_values = np.arange(-0.005, 0.005, 0.001)
phi_ratios = []

for i in range(len(y_values)):
    phi_ratios.append([])
    y = y_values[i]
    for x in x_values:
        cosmo = {'omega_M_0': x,
                'omega_lambda_0': 1 - x - y,
                'omega_k_0': y,
                'h': 0.8}
        dVdz = volume(z, A, cosmo)
        phi_val = phi(1, Vbin(dVdz, dz, dm))
        phi_ratios[i].append(phi_val[1] / phi_val[0])
    plt.plot(x_values, phi_ratios[i], label='phi ratio for y = ' + str(y))

plt.axhline(y=1.075/1.025, color='r', linestyle='--', label='upper limit = 1.075/1.025')
plt.axhline(y=1.065/1.035, color='g', linestyle='--', label='lower limit = 1.065/1.035')
plt.xlabel('x')
plt.ylabel('phi ratio')
plt.legend()
plt.show()