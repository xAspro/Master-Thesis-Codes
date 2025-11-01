# The output pdfs of this code is saved in a folder with the current date and time.
# It can be turned off if required
import sys 
import numpy as np
import individual
import importlib
importlib.reload(individual)
from individual import lf
import mosaic
import drawlf
import time
import datetime
import os
import matplotlib.pyplot as plt



start_time = time.time()
curr_date_time = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")+"/"
print("curr_date_time:", curr_date_time)
os.makedirs(curr_date_time, exist_ok=True)



# # WRITE_PARAMS = sys.argv[1] if len(sys.argv) > 1 else False
# WRITE_PARAMS = True
# if WRITE_PARAMS: 
             

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
            #'Data_new/dr3z2p6_sample.dat', # This is DR3 not DR7 
            'Data_new/croom09sgp_sample.dat',
            'Data_new/croom09ngp_sample.dat',
            'Data_new/bossdr9color.dat',
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
            # 'Data_2019+/DESI_SV1_sample.dat',
            # 'Data_2019+/DESI_main_selection_sample.dat',
            # 'Data_2019+/CEERS_Kocevski_sample.dat',
            # 'Data_2019+/LRD_Data_CEERS_sample.txt',
            # 'Data_2019+/LRD_Data_JADES_sample.txt',
            # 'Data_2019+/LRD_Data_NGDEEP_sample.txt',
            # 'Data_2019+/LRD_Data_PRIMER-COS_sample.txt',
            # 'Data_2019+/LRD_Data_PRIMER-UDS_sample.txt',
            # 'Data_2019+/LRD_Data_UNCOVER_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16_gG_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2020/unprocessed_data/DR16Q_gG_sample.txt',
            # # '../QLF - Database Details/SHELLQs_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2022/processed_data/SHELQ4_z_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2022/processed_data/SHELQ4_i_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELLQ_i_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELQS_z_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2018/processed_data/SHELQS_i_sample.txt',
            # '../QLF - Database Details/milliquas-ref-2019/processed_data/SHELQ3_i_sample.txt',
            ]

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13, r'Richards et al. 2006'),
            #('Selmaps_with_tiles/dr3z2p6_selfunc.dat', 1622.0, 13, r'Richards et al. 2006'), # This is DR3 not DR7! 
            ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15, r'Croom et al. 2009'),
            ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15, r'Croom et al. 2009'),
            ('Selmaps_with_tiles/ross13_selfunc2.dat', 2236.0, 1, r'Ross et al. 2013'),
            ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13, r'Richards et al. 2006'),
            ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6, r'Glikman et al. 2011'),
            ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6, r'Glikman et al. 2011'),
            ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17, r'Yang et al. 2016'),
            ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8, r'McGreer et al. 2013'),
            ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8, r'McGreer et al. 2013'),
            ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18, r'Jiang et al. 2016'),
            ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18, r'Jiang et al. 2016'),
            ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18, r'Jiang et al. 2016'),
            ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10, r'Willott et al. 2010'),
            ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10, r'Willott et al. 2010'),
            ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11, r'Kashikawa et al. 2015'),
            ('Selmaps_with_tiles/giallongo15_sel.dat', 0.047, 7, r'Giallongo et al. 2015'),
            ('Selmaps_with_tiles/ukidss_sel_4.dat', 3370.0, 19, r'UKIDSS DXS'),
            ('Selmaps_with_tiles/banados_sel_4.dat', 2500.0, 20, r'Banados et al. 2016'),
            # ('smooth_maps/2019+/DESI_SV1_selfunc_3_with_tiles.dat', 14000.0, 25, r'DESI SV1'),
            # ('smooth_maps/2019+/DESI_main_selection_selfunc_3_with_tiles.dat', 14000.0, 25, r'DESI Main Selection'),
            # ('smooth_maps/2019+/CEERS_Kocevski_selfunc_3_with_tiles.dat', 0.009583, 26, r'CEERS Kocevski'),
            # ('smooth_maps/2019+/LRD_Data_CEERS_selfunc_with_tiles.dat', 0.024611, 151, r'LRD CEERS'),
            # ('smooth_maps/2019+/LRD_Data_JADES_selfunc_with_tiles.dat', 0.017250, 152, r'LRD JADES'),
            # ('smooth_maps/2019+/LRD_Data_NGDEEP_selfunc_with_tiles.dat', 0.003167, 153, r'LRD NGDEEP'),
            # ('smooth_maps/2019+/LRD_Data_PRIMER-COS_selfunc_with_tiles.dat', 0.038583, 154, r'LRD PRIMER-COS'),
            # ('smooth_maps/2019+/LRD_Data_PRIMER-UDS_selfunc_with_tiles.dat', 0.067500, 155, r'LRD PRIMER-UDS'),
            # ('smooth_maps/2019+/LRD_Data_UNCOVER_selfunc_with_tiles.dat', 0.012167, 156, r'LRD UNCOVER'),
            # ('smooth_maps/2019+/fake_sdss_dr16_selfunc_with_tiles.dat', 14555.0, 105, r'Milliquas DR16 gG'),
            # ('smooth_maps/2019+/fake_sdss_dr16Q_selfunc_with_tiles.dat', 14555.0, 106, r'Milliquas DR16Q gG'),
            # # ('smooth_maps/2019+/SHELLQs_selfunc_with_tiles.dat', 1200.0, 107, r'SHELLQs Matsuoka 2022'),
            # ('smooth_maps/2019+/SHELLQs_z_selfunc_with_tiles.dat', 1200.0, 108, r'SHELQ4 z Matsuoka 2022'),
            # ('smooth_maps/2019+/SHELLQs_i_selfunc_with_tiles.dat', 1200.0, 108, r'SHELQ4 i Matsuoka 2022'),
            # ('smooth_maps/2019+/SHELLQ_i_selfunc_with_tiles.dat', 1200.0, 109, r'SHELLQ i Matsuoka 2018'),
            # ('smooth_maps/2019+/SHELQS_z_selfunc_with_tiles.dat', 1200.0, 110, r'SHELQS z Matsuoka 2018'),
            # ('smooth_maps/2019+/SHELQS_i_selfunc_with_tiles.dat', 1200.0, 110, r'SHELQS i Matsuoka 2018'),
            # ('smooth_maps/2019+/SHELQ3_i_selfunc_with_tiles.dat', 1200.0, 111, r'SHELQ3 i Matsuoka 2019'),
            ]

method = 'Nelder-Mead'


zls = [(0.1, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2),
       (1.2, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.2), (2.2, 2.4),
       (2.4, 2.5), (2.5, 2.6), (2.6, 2.7), (2.7, 2.8), (2.8, 2.9),
       (2.9, 3.0), (3.0, 3.1), (3.1, 3.2), (3.2, 3.3), (3.3, 3.4),
       (3.4, 3.5), (3.7, 4.1), (4.1, 4.7), (4.7, 5.5), (5.5, 6.5),
       (6.5, 8.5)
    ]

# -----------------------------------------------------------------------------
# Mosaic Plot Toggle and Parameter Setup
# -----------------------------------------------------------------------------
# This section controls whether to run the mosaic plot visualization
# for our simulated function compared to the data.
#
# Workflow:
#   1️ First run with `mosaic_plot_run = False`:
#       - This executes the main analysis code and produces binned lf.
#       - Then `lfg_multiple` needs to be run and it will produce the best fit parameters.
#
#   2️ Then (manually for now) copy those fitted parameters into `params` below
#       and set `mosaic_plot_run = True` to generate the mosaic plot.
#
#   > TODO (Automation):
#       - Automate parameter transfer by saving the fitted parameters to a file 
#         and then loading them here directly.
#       - Once automated, the manual copy-paste of params will no longer be needed.
# -----------------------------------------------------------------------------


mosaic_plot_run = False  # Set to True after first run to produce the mosaic plot

if mosaic_plot_run:

    # params value from previous runs
    # params = [[-0.3261, 1.2184, -7.2610],
    #           [-0.0439, 0.3283, -1.6293, -23.3691],
    #           [-0.0552, 0.1424, -3.6888],
    #           [-0.1612, -1.4329]]
    
    params = [[-0.3280, 1.2183, -7.2532],
              [-0.0454, 0.3198, -1.6133, -23.3546],
              [-0.0549, 0.1392, -3.6832],
              [-0.1616, -1.4310]]

    # Calculate appropriate grid size for 26 plots
    n_plots = len(zls)  # 26
    n_plots = 25  # For testing purposes, use 25 plots
    n_cols = 6
    n_cols = 5  # For testing purposes, use 5 columns
    n_rows = int(np.ceil(n_plots / n_cols))  # This will be 5 rows

    # Create figure with shared axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                            sharex=True, sharey=True)
    axes = axes.flatten()

    for i, zl in enumerate(zls):
        if i < len(axes):
            lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)
            lfi.plot_bins_for_params_mosaic(params, fig=fig, ax=axes[i], 
                                          subplot_index=i, n_rows=n_rows, n_cols=n_cols, n_plots=n_plots)


    for j in range(len(zls), len(axes)):
        axes[j].set_visible(False)

    fig.text(0.5, 0.02, r'$M_{1450}$', ha='center', fontsize=16)
    fig.text(0.02, 0.5, r'$\log_{10}(\phi^*)$ [cMpc$^{-3}$ mag$^{-1}$]', va='center', rotation='vertical', fontsize=16)
    
    fig.suptitle('Quasar Luminosity Function across Redshift Bins', fontsize=28, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, bottom=0.04, top=0.94, hspace=0.0, wspace=0.0)
    plt.savefig("QLF_bins_mosaic.png", dpi=300, bbox_inches='tight')
    plt.savefig("QLF_bins_mosaic.pdf", bbox_inches='tight')
    plt.show()


    import sys
    sys.exit("Exiting early for testing purposes.")



# Main portion of the file!

# filename = 'bins_ultra_new_2.dat'
filename = 'bins_check.dat'

def main(cores):
    lfs = [] 
    cnt = 0

    # Notes down the parameters for it to be later used in summary files.
    WRITE_PARAMS2 = True
    if WRITE_PARAMS2: 
        with open(filename, 'w') as f:
            f.write('# zmean zmin  zmax  phi_star  phi_star_err    M_star  M_star_err        alpha  alpha_err        beta    beta_err\n')

    for i, zl in enumerate(zls):

        lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

        print('z =', zl)
        print('{:d} quasars in this bin.'.format(lfi.z.size))
        print('sids (samples): '+'  '.join(['{:2d}'.format(int(x)) for x in np.unique(lfi.sid)]))
        print('sids (maps): '+'  '.join(['{:2d}'.format(x.sid) for x in lfi.maps]))
        print(' ')

        cnt += lfi.z.size
        
        g = (np.log10(1.e-6), -25.0, -3.0, -1.5)      # Initial guess for log10(phi_star), M_star, alpha, beta
        b = lfi.bestfit(g, method=method)

        print("\n\n\n\n\n\n\nb:\n", b)
        print("\n\n\n\n\n\n")

        zmin, zmax = zl 
        
        if zmin < 0.3:
            lfi.prior_min_values = np.array([-14.0, -32.0, -7.0, -10.0])
        else:
            lfi.prior_min_values = np.array([-14.0, -32.0, -7.0, -4.0])

        if zmin > 5.4:
            # Special priors for z = 6 data.
            lfi.prior_max_values = np.array([-4.0, -20.0, -4.0, 0.0])

            # Change result of optimize.minimize so that emcee works.
            lfi.bf.x[2] = -5.0
        elif zmin < 0.3:
            lfi.prior_max_values = np.array([-1.0, -15.0, 0.0, 15.0])
        else:
            lfi.prior_max_values = np.array([-4.0, -20.0, 0.0, 0.0])

        assert(np.all(lfi.prior_min_values < lfi.prior_max_values))
        
        lfi.run_mcmc_with_bad_points(ncores=int(cores), dirname=curr_date_time)
        lfi.get_percentiles()
        drawlf.draw(lfi, dirname=curr_date_time, show_individual_fit=True, includes_bad_points=True)

        
        # FOR SUMMARY (Fig 4)
        if WRITE_PARAMS2: 
            with open(filename, 'a') as f:
                output = ([lfi.z.mean()] + list(zl) + lfi.phi_star
                        + lfi.M_star + lfi.alpha + lfi.beta)
                f.write(('{:.3f}  '*len(output)).format(*output))
                f.write('\n')
        
        lfs.append(lfi)

        # mosaic.draw(lfs)


        # lfi.clear_samples()
        lfi.sample_samples()

    end_time = time.time()



    print()
    print()
    elapsed_time = end_time - start_time
    print("Time taken in bins.py:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("\n\n\nlfs:", lfs)


    # np.save('bins_lfs_1.npy', lfs)
    np.save('bins_lfs_ultra_new_2.npy', lfs)

    print(type(lfs))
    print(dir(lfs))
    print("Total number of quasars in all bins:", cnt)


if __name__ == "__main__":
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cores)

