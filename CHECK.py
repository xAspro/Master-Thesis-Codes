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
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0,
         'h':0.7}

A= 0.009583 # in deg^2, but in arcmin^2 it is 34.5 arcmin^2
def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1
    print("volperstr at z = ", z, " is ", volperstr)

    return omega*volperstr # cMpc^3 dz^-1


dVdz = volume(z, A, cosmo)


print("phi is ", phi(1, Vbin(dVdz, dz, dm)) )




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
