## Usage
This folder contains the Julia implementation of the **Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model with Topological Emissions (ds-HDP-HMM-TE)**.

This file contains three main Julia scripts:
## Files in this folder

1) TDA_Tools.jl which contains all of the functions used to process time series data into persistence images.
2) ds_HDP_HMM_AUX.jl which contains a number of auxiliary functions called by the main script.
3) ds_HDP_HMM_w_TDA_weak_limit.jl which is the primary script for running the algorithm.
1. **`TDA_Tools.jl`**
   - Functions for transforming time-series data into persistence diagrams and persistence images.
   - Includes utilities used during data preparation.

The first 60 lines of ds_HDP_HMM_w_TDA_weak_limit.jl are for user supplied data.  Most importantly, the user must supply a training data file (in the form of a Julia Serialized Object like that produced by the timeSeries_to_PIs function in the TDA_Tools.jl script) and a set of known states for the training data (as a comma separated file).  Optionally, test data of the same formate can be supplied.  The other parameters can be set as described in ds_HDP_HMM_w_TDA_weak_limit.jl.
2. **`ds_HDP_HMM_AUX.jl`**
   - Auxiliary/helper functions called by the main model script.

3. **`ds_HDP_HMM_w_TDA_weak_limit.jl`**
   - Main training/inference script.
   - The top section (roughly the first ~60 lines) is where you configure user inputs and run settings.

## Expected inputs

The main script expects:

- **Training features** as a Julia serialized object (for example, produced by `timeSeries_to_PIs` in `TDA_Tools.jl`).
- **Training labels/states** as a CSV file with known states.

Optional:

- **Test features** in the same serialized format.
- **Test labels/states** in CSV format.

## Typical workflow

1. Prepare or load time-series data.
2. Use functions in `TDA_Tools.jl` to convert data into persistence-image features.
3. Save feature outputs as Julia serialized objects.
4. Set the input file paths and model parameters in the top section of `ds_HDP_HMM_w_TDA_weak_limit.jl`.
5. Run the main script.

## Notes

- Parameter descriptions and defaults are documented inline in `ds_HDP_HMM_w_TDA_weak_limit.jl`.
- For sample input files and preparation details, see `../Sample_Data/README.md`.
