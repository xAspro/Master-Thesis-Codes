##### Checked once.. didnt run

# In lfg1.py
# In composite.py
# Traceback (most recent call last):
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/lfg1.py", line 53, in <module>
#     lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/composite.py", line 203, in __init__
#     self.maps = [selmap(*x) for x in selection_maps]
#                  ^^^^^^^^^^
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/composite.py", line 92, in __init__
#     self.z, self.m, self.p, self.dz, self.dm  = getselfn(selection_map_file)
#                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/composite.py", line 29, in getselfn
#     z, mag, p, dz, dm = np.loadtxt(f, usecols=(1,2,3,4,5), unpack=True)
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/xr/.pyenv/versions/3.12.4/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py", line 1381, in loadtxt
#     arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/xr/.pyenv/versions/3.12.4/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py", line 1021, in _read
#     arr = _load_from_filelike(
#           ^^^^^^^^^^^^^^^^^^^^
# ValueError: invalid column index 4 at row 1 with 4 columns

print("In lfg1.py")

import sys
import numpy as np 
from composite import lf
from composite import lf_polyb
from summary_fromFile import summary_plot as sp

# Model 1 

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
             'Data_new/dr7z3p7_sample.dat',
             'Data_new/glikman11debug.dat',
             'Data_new/yang16_sample.dat',
             'Data_new/mcgreer13_dr7sample.dat',
             'Data_new/mcgreer13_s82sample.dat',
             'Data_new/mcgreer13_dr7extend.dat',
             'Data_new/mcgreer13_s82extend.dat',
             'Data_new/jiang16main_sample.dat',
             'Data_new/jiang16overlap_sample.dat',
             'Data_new/jiang16s82_sample.dat',
             'Data_new/willott10_cfhqsdeepsample.dat',
             'Data_new/willott10_cfhqsvwsample.dat',
             'Data_new/kashikawa15_sample.dat',
             'Data_new/giallongo15_sample.dat',
             'Data_new/ukidss_sample.dat',
             'Data_new/banados_sample.dat']

selnfiles_original = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 15),
             ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/glikman11_selfunc_ndwfs.dat', 0.05, 0.02, 1.71, 6),
             ('Data_new/glikman11_selfunc_dls.dat', 0.05, 0.02, 2.05, 6),
             ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0, 17),
             ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0, 8),
             ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0, 8),
             ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0, 18),
             ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0, 18),
             ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0, 18),
             ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47, 10),
             ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0, 10),
             ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5, 11),
             ('Data_new/giallongo15_sel.dat', 0.0, 0.0, 0.047, 7),
             ('Data_new/ukidss_sel_4.dat', 0.1, 0.1, 3370.0, 19),
             ('Data_new/banados_sel_4.dat', 0.1, 0.1, 2500.0, 20)]

selnfiles = [(A[0],) + A[3:] for A in selnfiles_original]

lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg1.bestfit(g, method=method)

lfg1.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg1.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 10.0, 200.0, 2.0])

assert(np.all(lfg1.prior_min_values < lfg1.prior_max_values))
assert(np.all(lfg1.bf.x < lfg1.prior_max_values))
assert(np.all(lfg1.prior_min_values < lfg1.bf.x))

lfg1.run_mcmc()

