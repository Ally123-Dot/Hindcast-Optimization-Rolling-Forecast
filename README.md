# Arctic Oil-Spill BO

Python framework for Bayesian parameter calibration and ensemble trajectory forecasting in Arctic ice-covered waters.

## Overview

This repository provides a framework for one-hour retrospective calibration followed by one-hour ensemble trajectory forecasting in high-ice-concentration waters. The model calibrates drift parameters against observed trajectories, then initializes the forecast from the observed position at the end of the calibration period.

The same framework can be applied to marginal ice zones and open water by replacing the sample list and configuring the corresponding input data and output paths.

## Repository Files

- `Base_Forcing_Loader_2024.py`: Loads environmental forcing data and maps the variables required by the forecasting framework.
- `Run_HyosPy_Batch_Group1_1h1h.py`: Reads samples and observations, performs Bayesian parameter calibration and ensemble trajectory forecasting, and saves trajectory results.

## Parameter Settings

The same parameter search ranges are used for all three environmental regimes:

| Parameter | Description | Search range |
| --- | --- | --- |
| `fw` | Wind drag coefficient | 0–0.06 |
| `Kh` | Horizontal diffusivity | 0–20 m²/s |
| `k_sic` | Sea ice concentration correction factor | 0.5–1.5 |

The configuration uses 20 initial Latin hypercube sampling (LHS) evaluations followed by 100 Bayesian optimization iterations. The Upper Confidence Bound (UCB) acquisition function uses κ = 2.576. Each simulation uses 10,000 particles, and the ensemble forecast contains 100 members.

## Requirements

Python and the following packages are required:

- NumPy
- pandas
- SciPy
- xarray
- scikit-learn
- OpenDrift
- bayesian-optimization
- A NetCDF backend compatible with the input files

Package versions must support the interfaces used in the code, including `bayes_opt.acquisition.UpperConfidenceBound`.

## Input Data

Prepare the following data before running the framework:

- A buoy observation CSV containing `buoyid`, `year`, `month`, `day`, `hour`, `lat`, and `lon`.
- A time-window sample list containing `BuoyID` and `Target_Time`. The script also supports `Start_Time`.
- Ocean, sea ice, and wind forcing data in NetCDF format.
- Perturbed forcing fields for 100 ensemble members, organized according to the directory structure expected by the script.

Observations and environmental forcing must cover both the calibration and forecast periods. Variable names, units, and time coordinates must match the loading configuration.

The required input datasets are not included in this repository.

## Running the Framework

1. Place both Python files in the same directory.
2. Check `BASE_DIR` in both files and verify the environmental data directories and variable mappings.
3. Configure the following paths in the forecasting script:
   - `LEVEL1_FILE`: Buoy observation file.
   - `INPUT_CSV`: Time-window sample list.
   - `ENSEMBLE_DIR`: Directory containing the ensemble forcing fields.
   - `OUT_DIR`: Directory for trajectory outputs.
4. Run the forecasting script from that directory:

```bash
python Run_HyosPy_Batch_Group1_1h1h.py
```

Trajectory results are saved as CSV files for individual time windows, with `Set`, `Hour`, `Lat`, and `Lon` fields.

The script skips windows whose output files already exist. To rerun a window, back up and move its existing output file before execution.

## Applying the Framework to Other Regimes

To use the framework in marginal ice zones or open water, replace `INPUT_CSV` with the sample list for the target regime and verify the corresponding observation, environmental forcing, and ensemble forcing paths. Use a separate `OUT_DIR` for each regime.

All three regimes use the same parameter search ranges, so `pbounds` does not require modification.

The Group 1 filename and high-ice-concentration messages identify the default configuration. They do not automatically detect the environmental regime or switch parameter ranges.

## Usage Status

The author has successfully run the code with local datasets and a configured software environment. Running it in another environment requires compatible dependencies and input data, together with verification of file paths, variable mappings, and temporal coverage.
