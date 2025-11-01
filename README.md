QLF
=====

Homogenized AGN catalogue and luminosity function analysis code from Kulkarni et al. 2018.

-----

This repository contains three things:
1. A homogeneous catalogue of 83,488 AGN between redshift 0 and 7.5, with rest-frame UV magnitudes, redshifts, and selection probabilities.
2. Code to derive luminosity functions from this data 
3. Code to model hydrogen and helium reionization 

The commit tagged `v3.0` reproduces the results from Kulkarni et al. 2018.

The `Data_new` subdirectory contains the AGN catalogue in ASCII text files.  Comments in the files explain their structure.  See Section 2 of the paper for details on the original references for these data and how we homogenise them.  Use `data.py` to visualise the full catalogue in the style of Figure 1 of the paper. Luminosity functions in bins of redshift and magnitude are implemented in `drawlf.py`.  These are shown as points in, e.g., Figure 3 of the paper. Detailed definitions are in Section 3.1 of the paper.Double-power-law luminosity function models are derived in the `lf` class, defined in `individual.py`.  See `lfi.py` or `bins.py` for examples on how to use this class. The latter is used by `mosaic.py` to produce Figure 3 of the paper. Details of this modelling is in Section 3.2 of the paper. Global models of luminosity function evolution are implemented in `composite.py`.  See `lfg.py` or `lfg_multiple.py` for examples of how to use these. Three such models are discussed in Section 3.3 of the paper. Hydrogen-ionizing emissivity is modelled in `gammapi.py`.  The hydrogen-photoionization rate is modelled in `rtg2.py`.  Code in `qhe.py` models Helium reionization. The methods behind these codes is discussed in Section 4 of the paper. 

-------

### FAQ

#### 1. Where do I start?

Begin by running lfi.py to get the luminosity function in a redshift bin.  

#### 2. How do I get the number density of AGN at a certain redshift? 

Pass luminosity function models (instances of `individual.lf` or `composite.lf`) to one of the functions in `rhoqso.py`.

#### 3. I just want to know the value of one of the double-power-law parameters at a redshift

Pass an instance of `composite.lf` to `paramz.parameter_atz()`.

-------

Girish Kulkarni (kulkarni@ast.cam.ac.uk)



-------

### Updates and Notes (2024–2025)

This section documents the updates and modifications made to the QLF repository as part of my MSc project at the Tata Institute of Fundamental Research (TIFR), Department of Astronomy and Astrophysics, under the supervision of Prof. Girish Kulkarni (DTP, TIFR).  
The primary goal was to modernize the codebase, incorporate a more flexible statistical framework, and prepare the repository for future continuation and analysis.

#### Summary of Updates

1. **Python 3 Compatibility**  
   - All scripts have been converted from Python 2 to Python 3 to ensure compatibility with modern environments.

2. **Documentation and Code Comments**  
   - Partial documentation has been added across several files to clarify functionality and data flow.  
   - Some sections still contain redundant or commented-out code blocks retained from intermediate testing stages. These can be cleaned or consolidated as development continues.
   - Full documentation can be added in future development phases to ensure consistency and maintainability.

3. **Mixture-Model Framework Integration**  
   - Implemented a mixture model following the approach of *Hogg et al., “Data Analysis Recipes: Fitting a Model to Data”* to handle unknown uncertainties and systematic effects in luminosity function fitting.  
   - The main implementation is distributed across `bins.py`, `individual.py`, `composite.py`, and `lfg_multiple.py`.  
   - The method downweights potential outliers, improving robustness to unknown systematics.

4. **rhoqso.py and Physical Model Validation**  
   - Preliminary mixture-model results have been added to `rhoqso.py`. The current implementation does not yet use weighted data in its integrals and therefore one should be careful while interpretting the data and curve together.  
   - The next step would be to generate a smooth, physically plausible luminosity function — one that remains finite and continuous across redshifts, even if it does not match the data.  
   - Since `rhoqso.py` integrates the fitted curve from the Bayesian mixture model rather than performing its own fitting, future updates should focus on consistency between the fitting and integration stages.

5. **Intermediate Data Handling**  
   - Many scripts automatically generate and read intermediate `.npy` and data `.dat` files to reduce repeated computation.  
   - This modular structure allows individual scripts (e.g., `lfg_multiple.py`) to reuse outputs from earlier stages like `bins.py`, significantly reducing run time and making debugging easier.  
   - Output figures and test results are stored in timestamped folders for progress tracking. Users may modify the save paths if they prefer simpler output handling.

6. **Automation and Code Structure**  
   - Some automation steps (e.g., linking intermediate outputs between scripts) are partially implemented and may require manual reading/writing of results between stages. 
   - Completing these automation steps would make the analysis more reproducible and reduce manual intervention in future runs. 
   - Later parts of the workflow were developed more procedurally rather than fully object-oriented. While this does not affect correctness, further modularization could improve readability and scalability.

#### Suggested Workflow

To reproduce or extend the analysis:
1. Start with `bins.py` to generate binned luminosity data.  
2. Proceed to `lfg_multiple.py` to fit the global luminosity function models.  
3. Optionally use `rhoqso.py` for preliminary number density comparisons (further refinement recommended).  
4. The core logic can be traced primarily through `bins.py` and `lfg_multiple.py`; other modules build on these outputs.

#### Related Work

For a conceptual overview and worked examples of the mixture-model approach (Hogg et al.), see my companion repository:  
[**David-Hogg-Data-Analysis-Recipes–Fitting-a-Model-to-Data**](https://github.com/xAspro/David-Hogg-Data-Analysis-Recipes--Fitting-a-model-to-data)  
This repository includes illustrative examples and test cases that formed the foundation for the mixture-model implementation here.

-------

**Sooryabalan Murugesan**  
Former MSc Student, Department of Astronomy and Astrophysics, TIFR, Mumbai  
Email: [sooryabalanm@gmail.com](mailto:sooryabalanm@gmail.com)
