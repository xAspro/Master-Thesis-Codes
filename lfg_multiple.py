# Checked once. Didnt run. 
# Also check bins.py, importing it might cause this to run bins fully. that should be avoided in general.
# But that is for a later time.

# evolution6.pdf is correct alomst!
# Need to look at my error boundaries. that abs is making an issue i guess. but got the correct plot. not fully correct though. logical error.
print("In lfg_multiple.py")

import sys
import numpy as np 
from composite import lf
from composite import lf_polyb
from summary_fromFile import summary_plot as sp
import drawlf
# import bins
import traceback
import sounddevice as sd

import time
from datetime import datetime


"""
This program applies multiple models for the evolution of the quasar luminosity function.
"""

start_time = time.time()
readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Time right now: ", readable_time)


bins_lfs = np.load('bins_lfs.npy', allow_pickle=True)


###############################################################################################
###############################################################################################
###############################################################################################

pnum1 = [3,4,2,5]
g1 = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])
lfg1_prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg1_prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 20.0, 200.0, 2.0])

###############################################################################################

pnum2 = [3,4,2,5]
g2 = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])
lfg2_prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg2_prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 20.0, 200.0, 2.0])

###############################################################################################

pnum3 = [3,4,2,2]
g3 = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -1.30352181, -0.15925648])
lfg3_prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -5.0, -5.0])

lfg3_prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 0.0, 5.0])

###############################################################################################
###############################################################################################
###############################################################################################

###############################################################################################
###############################################################################################
###############################################################################################

# pnum1 = [5,5,5,5]
# g1 = np.array([-7.95061036, 1.15284665, -0.12037541, 0, 0,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026, 0,
#               -3.35945526, -0.26211017, 0, 0, 0,
#               -1.30352181, -0.15925648, 0, 0, 0])
# lfg1_prior_min_values = np.array([-25.0, -25.0, -25.0, -25.0, -25.0,
#                                  -50.0, -50.0, -50.0, -50.0, -50.0,
#                                  -15.0, -15.0, -15.0, -15.0, -15.0,
#                                  -10000.0, -10000.0, -10000.0, -10000.0, -10000.0,])

# lfg1_prior_max_values = np.array([25.0, 25.0, 25.0, 25.0, 25.0,
#                                  50.0, 5.0, 50.0, 50.0, 50.0,
#                                  15.0, 15.0, 15.0, 15.0, 15.0,
#                                  10000.0, 10000.0, 10000.0, 10000.0, 10000.0,])

# ###############################################################################################

# pnum2 = [5,5,5,5]
# g2 = np.array([-7.95061036, 1.15284665, -0.12037541, 0, 0,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026, 0,
#               -3.35945526, -0.26211017, 0, 0, 0,
#               -1.30352181, -0.15925648, 0, 0, 0])
# lfg2_prior_min_values = np.array([-25.0, -25.0, -25.0, -25.0, -25.0,
#                                  -50.0, -50.0, -50.0, -50.0, -50.0,
#                                  -15.0, -15.0, -15.0, -15.0, -15.0,
#                                  -10000.0, -10000.0, -10000.0, -10000.0, -10000.0,])

# lfg2_prior_max_values = np.array([25.0, 25.0, 25.0, 25.0, 25.0,
#                                  50.0, 5.0, 50.0, 50.0, 50.0,
#                                  15.0, 15.0, 15.0, 15.0, 15.0,
#                                  10000.0, 10000.0, 10000.0, 10000.0, 10000.0,])

# ###############################################################################################

# pnum3 = [5,5,5,5]
# g3 = np.array([-7.95061036, 1.15284665, -0.12037541, 0, 0,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026, 0,
#               -3.35945526, -0.26211017, 0, 0, 0,
#               -1.30352181, -0.15925648, 0, 0, 0])
# lfg3_prior_min_values = np.array([-15.0, 0.0, -5.0,
#                                  -30.0, -10.0, 0.0, -2.0,
#                                  -7.0, -5.0,
#                                  -5.0, -5.0])

# lfg3_prior_max_values = np.array([-5.0, 10.0, 5.0,
#                                  -10.0, -1.0, 2.0, 2.0,
#                                  -1.0, 5.0,
#                                  0.0, 5.0])

###############################################################################################
###############################################################################################
###############################################################################################


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
             'Data_new/banados_sample.dat',
             'Data_2019+/DESI_SV1_sample.dat',
             'Data_2019+/DESI_main_selection_sample.dat',
             'Data_2019+/CEERS_Kocevski_sample.dat',
             'Data_2019+/LRD_Data_CEERS_sample.txt',
             'Data_2019+/LRD_Data_JADES_sample.txt',
             'Data_2019+/LRD_Data_NGDEEP_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-COS_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-UDS_sample.txt',
             'Data_2019+/LRD_Data_UNCOVER_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16_gG_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16Q_gG_sample.txt',
            #  '../QLF - Database Details/SHELLQs_sample.txt',
             '../QLF - Database Details/milliquas-ref-2022/processed_data/SHELQ4_z_sample.txt',
             '../QLF - Database Details/milliquas-ref-2022/processed_data/SHELQ4_i_sample.txt',
             '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELLQ_i_sample.txt',
             '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELQS_z_sample.txt',
             '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELQS_i_sample.txt',
             '../QLF - Database Details/milliquas-ref-2019/processed_data/SHELQ3_i_sample.txt',
             ]

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
             ('Selmaps_with_tiles/banados_sel_4.dat', 2500.0, 20),
             ('smooth_maps/2019+/DESI_SV1_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/DESI_main_selection_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/CEERS_Kocevski_selfunc_3_with_tiles.dat', 0.009583, 26),
             ('smooth_maps/2019+/LRD_Data_CEERS_selfunc_with_tiles.dat', 0.024611, 151),
             ('smooth_maps/2019+/LRD_Data_JADES_selfunc_with_tiles.dat', 0.017250, 152),
             ('smooth_maps/2019+/LRD_Data_NGDEEP_selfunc_with_tiles.dat', 0.003167, 153),
             ('smooth_maps/2019+/LRD_Data_PRIMER-COS_selfunc_with_tiles.dat', 0.038583, 154),
             ('smooth_maps/2019+/LRD_Data_PRIMER-UDS_selfunc_with_tiles.dat', 0.067500, 155),
             ('smooth_maps/2019+/LRD_Data_UNCOVER_selfunc_with_tiles.dat', 0.012167, 156),
             ('smooth_maps/2019+/fake_sdss_dr16_selfunc_with_tiles.dat', 14555.0, 105),
             ('smooth_maps/2019+/fake_sdss_dr16Q_selfunc_with_tiles.dat', 14555.0, 106),
            #  ('smooth_maps/2019+/SHELLQs_selfunc_with_tiles.dat', 1200.0, 107)
             ('smooth_maps/2019+/SHELLQs_z_selfunc_with_tiles.dat', 1200.0, 108),
             ('smooth_maps/2019+/SHELLQs_i_selfunc_with_tiles.dat', 1200.0, 108),
             ('smooth_maps/2019+/SHELLQ_i_selfunc_with_tiles.dat', 1200.0, 109),
             ('smooth_maps/2019+/SHELQS_z_selfunc_with_tiles.dat', 1200.0, 110),
             ('smooth_maps/2019+/SHELQS_i_selfunc_with_tiles.dat', 1200.0, 110),
             ('smooth_maps/2019+/SHELQ3_i_selfunc_with_tiles.dat', 1200.0, 111),
            ]




lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=pnum1)

# lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[4,4,4,4])

# g1 = np.array([-7.95061036, 1.15284665, -0.12037541,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026,
#               -3.35945526, -0.26211017,
#               -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg1.bestfit(g1, method=method)
print("\n****************************************************************************************\n")
print("Best fit parameters: ", b)
print("\n****************************************************************************************\n")



lfg1.prior_min_values = lfg1_prior_min_values

lfg1.prior_max_values = lfg1_prior_max_values

# print("Prior min values:", lfg1.prior_min_values)
# print("Prior max values:", lfg1.prior_max_values)
# print("Best fit values:", lfg1.bf.x)
# print("Comparison result:", lfg1.prior_min_values < lfg1.prior_max_values)
# print("Comparison result:", lfg1.bf.x < lfg1.prior_max_values)
# print("Comparison result:", lfg1.prior_min_values < lfg1.bf.x)

# # assert(np.all(lfg1.prior_min_values < lfg1.prior_max_values))
# # assert(np.all(lfg1.bf.x < lfg1.prior_max_values))
# # assert(np.all(lfg1.prior_min_values < lfg1.bf.x))

# try:
#     assert(np.all(lfg1.prior_min_values < lfg1.prior_max_values))
#     assert(np.all(lfg1.bf.x < lfg1.prior_max_values))
#     assert(np.all(lfg1.prior_min_values < lfg1.bf.x))
# except AssertionError as e:
#     print(f"Assertion error: {e}")
#     traceback.print_exc()

print("Model 1 commence run_mcmc")

lfg1.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 1 over")

try:
    sp(composite=lfg1, individuals=bins_lfs, sample=True)  # calling the function

except Exception as e:
    print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
    traceback.print_exc()

duration = 3  # seconds
frequency = 440  # Hz, the frequency of the beep sound (440Hz is standard A note)

# Generate sound wave (440Hz sine wave)
sample_rate = 44100  # samples per second
t = np.linspace(0, duration, int(sample_rate * duration), False)
wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Play the generated sound wave
sd.play(wave, samplerate=sample_rate)
# sd.wait()  # Wait until the sound is finished playing


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
             'Data_new/kashikawa15_sample.dat',
             'Data_2019+/DESI_SV1_sample.dat',
             'Data_2019+/DESI_main_selection_sample.dat',
             'Data_2019+/CEERS_Kocevski_sample.dat',
             'Data_2019+/LRD_Data_CEERS_sample.txt',
             'Data_2019+/LRD_Data_JADES_sample.txt',
             'Data_2019+/LRD_Data_NGDEEP_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-COS_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-UDS_sample.txt',
             'Data_2019+/LRD_Data_UNCOVER_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16_gG_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16Q_gG_sample.txt',
             '../QLF - Database Details/SHELLQs_sample.txt',
             ]

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
             ('smooth_maps/2019+/DESI_SV1_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/DESI_main_selection_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/CEERS_Kocevski_selfunc_3_with_tiles.dat', 0.009583, 26),
             ('smooth_maps/2019+/LRD_Data_CEERS_selfunc_with_tiles.dat', 0.024611, 151),
             ('smooth_maps/2019+/LRD_Data_JADES_selfunc_with_tiles.dat', 0.017250, 152),
             ('smooth_maps/2019+/LRD_Data_NGDEEP_selfunc_with_tiles.dat', 0.003167, 153),
             ('smooth_maps/2019+/LRD_Data_PRIMER-COS_selfunc_with_tiles.dat', 0.038583, 154),
             ('smooth_maps/2019+/LRD_Data_PRIMER-UDS_selfunc_with_tiles.dat', 0.067500, 155),
             ('smooth_maps/2019+/LRD_Data_UNCOVER_selfunc_with_tiles.dat', 0.012167, 156),
             ('smooth_maps/2019+/fake_sdss_dr16_selfunc_with_tiles.dat', 14555.0, 105),
             ('smooth_maps/2019+/fake_sdss_dr16Q_selfunc_with_tiles.dat', 14555.0, 106),
             ('smooth_maps/2019+/SHELLQs_selfunc_with_tiles.dat', 1200.0, 107)
             ]

lfg2 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=pnum2)

# lfg2 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[4,4,4,4])

# g2 = np.array([-7.95061036, 1.15284665, -0.12037541,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026,
#               -3.35945526, -0.26211017,
#               -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg2.bestfit(g2, method=method)


# # lfg2.prior_max_values = np.array([-5.0, 10.0, 5.0,
# #                                  -10.0, -1.0, 2.0, 2.0,
# #                                  -1.0, 5.0,
# #                                  10.0, 10.0, 10.0, 200.0, 2.0])



lfg2.prior_min_values = lfg2_prior_min_values

lfg2.prior_max_values = lfg2_prior_max_values

# try:
#     assert(np.all(lfg2.prior_min_values < lfg2.prior_max_values))
#     assert(np.all(lfg2.bf.x < lfg2.prior_max_values))
#     assert(np.all(lfg2.prior_min_values < lfg2.bf.x))
# except AssertionError as e:
#     print(f"Assertion error: {e}")
#     traceback.print_exc()
# # assert(np.all(lfg2.prior_min_values < lfg2.prior_max_values))
# # assert(np.all(lfg2.bf.x < lfg2.prior_max_values))
# # assert(np.all(lfg2.prior_min_values < lfg2.bf.x))

print("Model 2 commence run_mcmc")

lfg2.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 2 over")


# try:
#     sp(composite=lfg2, individuals=bins.lfs, sample=True, output_file_name='evolution3.pdf')  # calling the function

# except Exception as e:
#     print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
#     traceback.print_exc()
# # Play the generated sound wave
# sd.play(wave, samplerate=sample_rate)

try:
    sp(composite=(lfg1,lfg2), individuals=bins_lfs, sample=True, output_file_name='evolution4_new.pdf')  # calling the function

except Exception as e:
    print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
    traceback.print_exc()

# Play the generated sound wave
sd.play(wave, samplerate=sample_rate)
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
             'Data_new/kashikawa15_sample.dat',
             'Data_2019+/DESI_SV1_sample.dat',
             'Data_2019+/DESI_main_selection_sample.dat',
             'Data_2019+/CEERS_Kocevski_sample.dat',
             'Data_2019+/LRD_Data_CEERS_sample.txt',
             'Data_2019+/LRD_Data_JADES_sample.txt',
             'Data_2019+/LRD_Data_NGDEEP_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-COS_sample.txt',
             'Data_2019+/LRD_Data_PRIMER-UDS_sample.txt',
             'Data_2019+/LRD_Data_UNCOVER_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16_gG_sample.txt',
             '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16Q_gG_sample.txt',
             '../QLF - Database Details/SHELLQs_sample.txt',
             ]

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
             ('smooth_maps/2019+/DESI_SV1_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/DESI_main_selection_selfunc_3_with_tiles.dat', 14000.0, 25),
             ('smooth_maps/2019+/CEERS_Kocevski_selfunc_3_with_tiles.dat', 0.009583, 26),
             ('smooth_maps/2019+/LRD_Data_CEERS_selfunc_with_tiles.dat', 0.024611, 151),
             ('smooth_maps/2019+/LRD_Data_JADES_selfunc_with_tiles.dat', 0.017250, 152),
             ('smooth_maps/2019+/LRD_Data_NGDEEP_selfunc_with_tiles.dat', 0.003167, 153),
             ('smooth_maps/2019+/LRD_Data_PRIMER-COS_selfunc_with_tiles.dat', 0.038583, 154),
             ('smooth_maps/2019+/LRD_Data_PRIMER-UDS_selfunc_with_tiles.dat', 0.067500, 155),
             ('smooth_maps/2019+/LRD_Data_UNCOVER_selfunc_with_tiles.dat', 0.012167, 156),
             ('smooth_maps/2019+/fake_sdss_dr16_selfunc_with_tiles.dat', 14555.0, 105),
             ('smooth_maps/2019+/fake_sdss_dr16Q_selfunc_with_tiles.dat', 14555.0, 106),
             ('smooth_maps/2019+/SHELLQs_selfunc_with_tiles.dat', 1200.0, 107)
             ]



# lfg3 = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,2])

# lfg3 = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[4,4,4,4])
lfg3 = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=pnum3)

# g3 = np.array([-7.95061036, 1.15284665, -0.12037541,
#               -18.64592897, -4.52638114, 0.47207865, -0.01890026,
#               -3.35945526, -0.26211017,
#               -1.30352181, -0.15925648])

method = 'Nelder-Mead'
b = lfg3.bestfit(g3, method=method)



lfg3.prior_min_values = lfg3_prior_min_values

lfg3.prior_max_values = lfg3_prior_max_values

# lfg3.prior_min_values = np.array([-15.0, 0.0, -5.0,
#                                  -30.0, -10.0, 0.0, -2.0,
#                                  -7.0, -5.0,
#                                  -10.0, -10.0, 0.0, -10.0, -2.0])

# lfg3.prior_max_values = np.array([-5.0, 10.0, 5.0,
#                                  -10.0, -1.0, 2.0, 2.0,
#                                  -1.0, 5.0,
#                                  10.0, 10.0, 20.0, 200.0, 2.0])

# print("Prior min values:", lfg3.prior_min_values)
# print("Prior max values:", lfg3.prior_max_values)
# print("Best fit values:", lfg3.bf.x)
# print("Comparison result:", lfg3.prior_min_values < lfg3.prior_max_values)
# print("Comparison result:", lfg3.bf.x < lfg3.prior_max_values)
# print("Comparison result:", lfg3.prior_min_values < lfg3.bf.x)

# # assert(np.all(lfg3.prior_min_values < lfg3.prior_max_values))
# # assert(np.all(lfg3.bf.x < lfg3.prior_max_values))
# # assert(np.all(lfg3.prior_min_values < lfg3.bf.x))

# try:
#     assert(np.all(lfg3.prior_min_values < lfg3.prior_max_values))
#     assert(np.all(lfg3.bf.x < lfg3.prior_max_values))
#     assert(np.all(lfg3.prior_min_values < lfg3.bf.x))
# except AssertionError as e:
#     print(f"Assertion error: {e}")
#     traceback.print_exc()

print("Model 3 commence run_mcmc")

lfg3.run_mcmc()

end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")
print("Model 3 over")

# try:
#     sp(composite=lfg3, individuals=bins.lfs, sample=True, output_file_name='evolution5.pdf')  # calling the function

# except Exception as e:
#     print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
#     traceback.print_exc()
# # Play the generated sound wave
# sd.play(wave, samplerate=sample_rate)

# try:
#     sp(composite=(lfg1,lfg2,lfg3), individuals=bins.lfs, sample=True, output_file_name='evolution6.pdf')  # calling the function

# except Exception as e:
#     print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
#     traceback.print_exc()

try:
    # Format the readable_time string to remove spaces and colons
    formatted_time = readable_time.replace(' ', '_').replace(':', '-')
    # Create the filename using the formatted time string
    filename = 'evolution6_new_' + formatted_time + '.pdf'

    sp(composite=lfg1, individuals=bins_lfs, sample=True, lfg_break=lfg2, lfg_polyb=lfg3, output_file_name=filename)  # calling the function

except Exception as e:
    print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function
    traceback.print_exc()

#------------------------------------------------------------
###################################################################################################
## What this this doing here???
# import bins 
###################################################################################################

# try:
#     sp(composite=lfg1, individuals=bins.lfs, sample=True)  # calling the function

# except Exception as e:
#     print(f"Error in {sp.__name__}: {e}")  # Catch and report the error, but continue to the next function

