# Dual-Surrogate Multi-Objective Optimization of WRF Physics Parameterizations

Code and derived data for the paper:

> Djari, I., et al. (2026). *Dual-Surrogate Multi-Objective Optimization of WRF Physics
> Parameterizations for Precipitation Forecasting in a Semi-Arid Region.*
> Submitted to Computers & Geosciences.

This repository contains the framework used to search the discrete space of
2,457,000 WRF physics configurations (microphysics, cumulus, planetary boundary
layer, surface layer, land surface, longwave and shortwave radiation) using two
machine-learning surrogates (one for precipitation RMSE, one for ETS) embedded
in the NSGA-II multi-objective genetic algorithm, together with the scripts that
automate the WRF ensemble simulations and the verification against GPM IMERG.

## Repository structure

```
.
├── README.md
├── LICENSE
├── surrogate/            FLAML surrogate training and the NSGA-II optimization loop
│                        
├── wrf_automation/       namelist generation, run WRF on HPC and batch scripts for the
│                         700-member LHS ensemble
├── dataprocessing/       categorical encodings and RMSE / ETS computation
│                         
└── Validation/
    └── imerg1.py   precipitation extraction (IMERG)
```

## Requirements

- Python 3.10.11 
- WRF-ARW v4.2.1 (compiled separately; see https://www2.mmm.ucar.edu/wrf/users/)
- Python packages: 
  (numpy, pandas, netCDF4, scikit-learn, flaml, xgboost, catboost, pymoo, earthaccess)



## Input data (public sources, not redistributed here)

- **GFS 0.25 degree forecast data** (initial and lateral boundary conditions):
  https://www.nco.ncep.noaa.gov/pmb/products/gfs/
  Successive daily 06 UTC cycles, forecast hours f000 to f024 at 1-hourly intervals.
- **GPM IMERG Final Run V07B daily precipitation** (verification reference):
  https://doi.org/10.5067/GPM/IMERGDF/DAY/07
  Downloaded via the `earthaccess` package; a free NASA Earthdata account is
  required (https://urs.earthdata.nasa.gov). Set your credentials as environment
  variables before running the download script.



No credentials are stored in this repository.

## Workflow

1. **Generate the training ensemble.** `wrf_automation/` samples 700 feasible
   configurations by Latin Hypercube Sampling, writes a WRF namelist for each,
   and submits the runs (single domain, 189 x 99 grid points at 1.3 km,
   35 vertical levels, 10-day window, first 24 h discarded as spin-up),
   and run wrf on hpc.
2. **Extract and verify precipitation.** `postprocessing/` extracts daily
   precipitation (RAINC + RAINNC + RAINSH) at the model grid point nearest to
   Batna (35.75 N, 6.17 E), `Validation/` downloads the collocated 0.1 degree IMERG pixel,
   and computes the RMSE and ETS (1 mm/day threshold) for each configuration.
3. **Train the surrogates.** `surrogate/` benchmarks 21 algorithm-encoding
   combinations with FLAML and selects, by external cross-validation RMSE, one
   surrogate per objective (CatBoost for RMSE, XGBoost for ETS in the paper).
4. **Run the optimization.** The two surrogates replace WRF evaluations inside
   NSGA-II (20,000 surrogate evaluations), producing the Pareto front of
   non-dominated configurations.
5. **Re-evaluate with WRF.** Selected Pareto configurations are re-run with
   full WRF over the independent validation windows and verified against IMERG.

Each stage has its own usage notes in the corresponding folder.



## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper above. A citable archive of this
repository is available on Zenodo: https://doi.org/10.5281/zenodo.21420487 
(replace with the DOI minted at release).

## Contact

Corresponding author: [Imene Djari], [imene.djari@univ-batna2.dz], University of Batna 2, Algeria.
