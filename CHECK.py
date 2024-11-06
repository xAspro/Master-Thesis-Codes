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










from scipy.integrate import quad

# Integrate over z = 5.15 ± 0.5 or z = 5.24 ± 0.5
def differential_volume(z, cosmo=cosmo):
    omega = (A / 41253.0) * 4.0 * np.pi
    return cd.diff_comoving_volume(z, **cosmo) * omega

# Numerical integration for Vbin over the redshift range
z_min = 5.15 - 0.5
z_max = 5.15 + 0.5
Vbin_z1, _ = quad(differential_volume, z_min, z_max)

z_min = 5.24 - 0.5
z_max = 5.24 + 0.5
Vbin_z2, _ = quad(differential_volume, z_min, z_max)

print()

print("phi is ", phi(1, Vbin_z1) )
print("phi is ", phi(1, Vbin_z2) )


print()










############################################################################################################
############################################################################################################


from scipy.integrate import quad

def differential_volume(z, area, cosmo=cosmo):
    omega = (area / 41253.0) * 4.0 * np.pi
    return cd.diff_comoving_volume(z, **cosmo) * omega

# Integrate over the redshift ranges for both papers
zrange_onoue = (5.15 - 0.5, 5.15 + 0.5)
Vbin_onoue, _ = quad(differential_volume, zrange_onoue[0], zrange_onoue[1], args=(A, cosmo))

zrange_kocevski = (5.24 - 0.5, 5.24 + 0.5)
Vbin_kocevski, _ = quad(differential_volume, zrange_kocevski[0], zrange_kocevski[1], args=(A, cosmo))

print("phi for Onoue:", phi(1, Vbin_onoue))
print("phi for Kocevski:", phi(1, Vbin_kocevski))








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


# # From cosmolopy.distance

# z = 5.24
# comoving volume per steradian = 31147524593.58081

# z = 5.15
# comoving volume per steradian = 31462036093.976078
