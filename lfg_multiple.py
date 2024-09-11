# Checked once. Didnt run. 
# Also check bins.py, importing it might cause this to run bins fully. that should be avoided in general.
# But that is for a later time.

# OUTPUT:
# Model 1 commence run_mcmc
# Time taken:  817.4496459960938  seconds
# Model 1 over
# laying out figure
# plotting now
# Traceback (most recent call last):
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/lfg_multiple.py", line 90, in <module>
#     end_time = time.time()
#     ^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/summary_fromFile.py", line 641, in summary_plot
#     plot_alpha(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample, lfg_break=lfg_break, lfg_polyb=lfg_polyb)
#   File "/Users/xr/Desktop/Old Laptop/Desktop/TIFR College/Semesters/Master's Thesis/Master's Thesis Codes/QLF/summary_fromFile.py", line 459, in plot_alpha
#     handles.append((m2f,m2))
#                     ^^^
# UnboundLocalError: cannot access local variable 'm2f' where it is not associated with a value
# python lfg_multiple.py  1201.58s user 11.13s system 99% cpu 20:22.45 total




# Model 1 commence run_mcmc
# Time taken:  659.4692330360413  seconds
# Model 1 over
# laying out figure
# plotting now
# Error in summary_plot: 'NoneType' object has no attribute 'create_artists'
# rsample=  [[-7.80875245e+00  1.11681751e+00 -1.17740899e-01 ...  3.77692705e+00
#    1.43633871e+02 -1.67548603e-01]
#  [-7.89222506e+00  1.16989269e+00 -1.21909327e-01 ...  3.75967177e+00
#    1.88613729e+02 -1.34278632e-01]
#  [-7.97352037e+00  1.22286297e+00 -1.24978031e-01 ...  3.76519553e+00
#    1.86907092e+02 -2.27098242e-01]
#  ...
#  [-7.57437371e+00  1.01049255e+00 -1.13060308e-01 ...  3.78015689e+00
#    1.29014302e+02  1.19430291e-03]
#  [-8.09994851e+00  1.27978791e+00 -1.27264416e-01 ...  3.80009093e+00
#    1.41355886e+02 -4.65616141e-01]
#  [-7.89249244e+00  1.19112397e+00 -1.25091268e-01 ...  3.76072559e+00
#    1.82607352e+02 -1.92741753e-01]]
# Error in draw: lf.log10phi() missing 1 required positional argument: 'z'
# Time right now:  2024-09-11 20:46:46

print("In lfg_multiple.py")

import sys
import numpy as np 
from composite import lf
from composite import lf_polyb
from summary_fromFile import summary_plot as sp
import drawlf
import bins
import traceback

##add time!!!!
import time
from datetime import datetime
start_time = time.time()
readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Time right now: ", readable_time)

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

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11),
             ('Selmaps_with_tiles/giallongo15_sel.dat', 0.047, 7),
             ('Selmaps_with_tiles/ukidss_sel_4.dat', 3370.0, 19),
             ('Selmaps_with_tiles/banados_sel_4.dat', 2500.0, 20)]

lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg1.bestfit(g, method=method)
print("\n****************************************************************************************\n")
print("Best fit parameters: ", b)
print("\n****************************************************************************************\n")

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

print("Model 1 commence run_mcmc")

lfg1.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 1 over")

try:
    sp(composite=lfg1, individuals=bins.lfs, sample=True)  # Try calling the function

except Exception as e:
    print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
    traceback.print_exc()

try:
    drawlf.draw(lfg1, show_individual_fit=True)  # Try calling the function

except Exception as e:
    print(f"Error in {drawlf.draw.__name__}: {e}")  # Catch and report the error, but continue to the next function
    traceback.print_exc()

#tried plotting.. but not working
# drawlf.draw(lfg1, show_individual_fit=True)
# sys.exit()
#------------------------------------------------------------

# Model 2 

start_time = time.time()
readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Time right now: ", readable_time)

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
             'Data_new/kashikawa15_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11)]

lfg2 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg2.bestfit(g, method=method)

lfg2.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg2.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 10.0, 200.0, 2.0])

assert(np.all(lfg2.prior_min_values < lfg2.prior_max_values))
assert(np.all(lfg2.bf.x < lfg2.prior_max_values))
assert(np.all(lfg2.prior_min_values < lfg2.bf.x))

print("Model 2 commence run_mcmc")

lfg2.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 2 over")

#------------------------------------------------------------

# Model 3
start_time = time.time()
readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Time right now: ", readable_time)

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
             'Data_new/kashikawa15_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11)]

lfg3 = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,2])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -1.30352181, -0.15925648])

method = 'Nelder-Mead'
b = lfg3.bestfit(g, method=method)

lfg3.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -5.0, -5.0])

lfg3.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 0.0, 5.0])

assert(np.all(lfg3.prior_min_values < lfg3.prior_max_values))
assert(np.all(lfg3.bf.x < lfg3.prior_max_values))
assert(np.all(lfg3.prior_min_values < lfg3.bf.x))

print("Model 3 commence run_mcmc")

lfg3.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 3 over")

#------------------------------------------------------------
###################################################################################################
## What this this doing here???
# import bins 
###################################################################################################
try:
    sp(composite=lfg1, individuals=bins.lfs, sample=True)  # Try calling the function

except Exception as e:
    print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function

try:
    drawlf.draw(lfg1, show_individual_fit=True)  # Try calling the function

except Exception as e:
    print(f"Error in {drawlf.draw.__name__}: {e}")  # Catch and report the error, but continue to the next function
