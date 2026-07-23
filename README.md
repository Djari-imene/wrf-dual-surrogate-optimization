# Dual-Surrogate Multi-Objective Optimization of WRF Physics Parameterizations

Code and derived data for the paper:

> Djari, I., Gassi, K. A. A., Seghir, R., Belhaouari, S. B., Kadache, N. (2026).
> *Dual-Surrogate Multi-Objective Optimization of WRF Physics Parameterizations
> for Precipitation Forecasting in a Semi-Arid Region.* Submitted to
> Computers & Geosciences.

This repository contains the framework used to search the discrete space of
2,457,000 WRF physics configurations (microphysics, cumulus, planetary boundary
layer, surface layer, land surface, longwave and shortwave radiation) using two
machine-learning surrogates (CatBoost with target encoding for precipitation
RMSE, XGBoost with one-hot encoding for ETS) embedded in the NSGA-II
multi-objective genetic algorithm, together with the scripts that automate the
WRF ensemble simulations and the verification against GPM IMERG at the Batna
reference site (35.75 N, 6.17 E).

## Repository structure

```
.
├── README.md
├── LICENSE
├── surrogate/                  Surrogate training and optimization
│   ├── flaml_benchmark.py        21 algorithm-encoding combinations, external 10-fold CV
│   ├── catboost_surrogate.py     S1: RMSE surrogate (CatBoost + target encoding)
│   ├── xgboost_surrogate.py      S2: ETS surrogate (XGBoost + one-hot encoding)
│   └── nsga2__final.py           NSGA-II loop driven by S1 and S2
├── wrf_automation/             WRF ensemble generation
│   ├── lhs.py                    Latin Hypercube sampling of 700 configurations
│   ├── modifier_nmlst.py         namelist.input generation per configuration
│   └── exe_par_hpc.py            batch execution on HPC (WPS once, then real/wrf per run)
├── dataprocessing/             Feature encodings and verification targets
│   ├── extract_daily_precip.py   daily series (RAINC+RAINNC+RAINSH) from wrfout at the
│   │                             Batna grid point                    
│   ├── dataset_targets_RMSE-ETS.py  RMSE and ETS per configuration 
│   ├── smoothing_target_encd.py  target-smoothing encoding (alpha = 10)
│   ├── one_h_encoding.py         one-hot encoding
│   └── encoder_binary.py         binary encoding
├── Validation/
└──   ├── get_imerg.py              IMERG daily precipitation at the collocated 0.1° cell
```

## Requirements

- Python 3.10+
- WRF-ARW v4.2.1 and WPS 4.2 (compiled separately; see
  <https://www2.mmm.ucar.edu/wrf/users/>)
- FLAML
- Python packages: numpy, pandas, netCDF4, h5py, joblib, scikit-learn, flaml,
  xgboost, catboost, lightgbm, matplotlib, earthaccess; shap (optional, for the
  feature-importance figure)

## Input data (public sources, not redistributed here)

- **GFS 0.25 degree operational data** (initial and lateral boundary
  conditions): <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>
  All simulations are initialized at 06 UTC from the GFS analysis (forecast
  hour f000) of a single cycle and forced at the lateral boundaries by the
  hourly forecasts of the same cycle, replicating an operational forecast
  configuration. Every ensemble member uses identical initial and boundary
  conditions, so score differences between configurations are attributable
  solely to the physics choices.
- **GPM IMERG Final Run V07B daily precipitation** (verification reference):
  <https://doi.org/10.5067/GPM/IMERGDF/DAY/07>
  Downloaded via the `earthaccess` package; a free NASA Earthdata account is
  required (<https://urs.earthdata.nasa.gov>). Set your credentials as
  environment variables (`EARTHDATA_USER`, `EARTHDATA_PASS`) before running the
  download script.

No credentials are stored in this repository.

## Workflow

1. **Generate the training ensemble.** `wrf_automation/lhs.py` samples 700
   configurations from the discrete space; `modifier_nmlst.py` writes a WRF
   namelist for each; `exe_par_hpc.py` runs WPS once and then executes
   real.exe and wrf.exe per configuration on the HPC cluster (single domain,
   189 x 99 grid points at 1.3 km, 35 vertical levels; each configuration is
   integrated continuously over the training window from one initialization).
2. **Extract and verify precipitation.**
   `dataprocessing/extract_daily_precip.py` builds the daily precipitation
   series (RAINC + RAINNC + RAINSH) at Batna;
   `Validation/get_imerg.py` extracts the collocated 0.1 degree IMERG cell;
   `dataprocessing/dataset_targets_RMSE-ETS.py` computes RMSE and ETS
   (1 mm/day threshold) for each configuration. Verification is site-specific
   by design (point-to-pixel at Batna); see Section 4.2 of the paper for the
   protocol and its justification.
3. **Train the surrogates.** `surrogate/flaml_benchmark.py` benchmarks 7
   algorithms x 3 categorical encodings x 2 objectives (42 combinations) with
   FLAML under external 10-fold cross-validation. One surrogate per objective
   is selected on lowest external-CV MAE: CatBoost with target encoding for
   RMSE (S1) and XGBoost with one-hot encoding for ETS (S2), trained by
   `catboost_surrogate.py` and `xgboost_surrogate.py`.
4. **Run the optimization.** `surrogate/nsga2__final.py` replaces WRF
   evaluations inside NSGA-II with S1 and S2 inference (population 100,
   200 generations, 20,000 evaluations), producing the Pareto front and
   selecting verification candidates by uniform spacing along the
   deduplicated front. All paper figures are generated by this script.
5. **Re-evaluate with WRF.** Selected Pareto configurations and the two
   baselines are re-run with full WRF over the three independent validation
   windows and verified against IMERG.

## Simulation periods

- Training window: May 2025 (see the paper, Section 3.2.1, for the exact days
  used to compute the targets).
- Validation windows: three 10-day spring windows in March, April, and May
  2025 (30 days in total), disjoint from the training window; the exact dates
  are listed in `Validation/get_imerg.py` and in the paper, Section 6.1.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper above. A citable archive of this
repository is available on Zenodo: <https://doi.org/10.5281/zenodo.21420487>

## Contact

Corresponding author: Imene Djari, <imene.djari@univ-batna2.dz>,
LaSTIC Laboratory, University of Batna 2, Algeria.
