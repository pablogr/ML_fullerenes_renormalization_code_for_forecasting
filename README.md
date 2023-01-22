# ML_fullerenes_renormalization_code_for_forecasting

========================================================================================================================
                 PROGRAM USED TO FORECAST THE RENORMALIZATION OF ELECTRONIC EIGENVALUES OF FULLERENES
      See 'ELECTRON-VIBRATIONAL RENORMALIZATION IN FULLERENES THROUGH AB INITION AND MACHINE LEARNING METHODS'
                                       BY PABLO GARCIA-RISUENO ET AL.
========================================================================================================================

This program forecasts the renormalizations of electronic eigenvalues of fullerenes using ab initio data calculated with
Quantum Espresso. If you want to have an estimate of the renormalization of electronic eigenvalues in a fullerene,
run a ground state calculation (preferably using Quantum Espresso with the provided pseudopotential files), then
extract the information of the regressors from its output file, write them to a file (called input_regressors_molecule_to_forecast.csv,
see the example file with that name) and run this program.

Code fully written by Pablo Garcia Risueno [garciaDOTrisuenoATgmailDOTcom], January 2023. Contact me if you have any doubt.
