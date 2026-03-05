# hdp-hmm-te Sample Data
There are two types of sample time series data included here:  Two dimensional data of loops (with 1, 2, or 3 loops per frame),  and three dimensional time series of randomly selected from spherical normal random variables with different variances.
# Sample_Data

## Usage
The time series data (leaf_series_data and Var_Change_Points files) must first be converted to a Julia serialization object (.jls file) using the timeSeries_to_PIs function available in the TDA_Tools.jl script in the Source_Code folder.  The syntax is shown below.
This folder contains small example datasets for running and validating the HDP-HMM + Topological Data Analysis (TDA) workflow in this repository.

timeSeries_to_PIs(time_series, output_file, window_sz, slide; jump=1, max_dim=1, pi_sz=5, stack=false, converter=nothing)
## Included datasets

time_series is an array of points (as tuples)
### 1) `leaf_*` datasets (2D loop trajectories)
- `leaf_series_data_1.csv`
- `leaf_series_data_2.csv`
- `leaf_state_data_1.csv`
- `leaf_state_data_2.csv`

output_file is where the data will be stored (as a .jls file)
These are 2D time-series examples with loop-like structure (frames may contain 1, 2, or 3 loops).

window_sz is the number of data points that will be used to create the persistence image
### 2) `Var_Change_*` datasets (3D variance-change trajectories)
- `Var_Change_Points_Train.csv`
- `Var_Change_Points_Test.csv`
- `Var_Change_States_Train.csv`
- `Var_Change_States_Test.csv`

slide is how much the window will advance each iteration
These are 3D time-series examples generated from spherical normal variables with changing variance.

jump allows you to pick how many samples in the window to use for computation (jump = 10 would take every 10th sample)
### 3) `Var_Change_*2` datasets (alternate split/variant)
- `Var_Change_Points2_Train.csv`
- `Var_Change_Points2_Test.csv`
- `Var_Change_States2_Train.csv`
- `Var_Change_States2_Test.csv`

max_dim is the maximum dimension used in the persistene image (increasing this will greatly slow down the computation)
These are an additional variance-change train/test set with matching state-label files.

pi_sz is the size of the (typically) square matrix representing the persistence image.
        If this is an int, all dimensions will have the same base size.
        You can specify an array of sizes for each dimenion (array should be length max_dim + 1).
        
stack=true will stack all PIs into a single N X 1 vector
---

The method will return a converter object for producing persistence images which can be passed back to the method on the test data (so that both sets of images are produced with the same process).
## How to use these files

The state data files must be modified to the correct length depending on the choice of window size and slide rate.
The HDP-HMM scripts operate on persistence-image representations serialized as `.jls` files. To convert raw time series (`*_series_*` or `*_Points_*`) to persistence images, use `timeSeries_to_PIs` from `Source_Code/TDA_Tools.jl`.

```julia
timeSeries_to_PIs(time_series, output_file, window_sz, slide;
                  jump=1, max_dim=1, pi_sz=5, stack=false, converter=nothing)
```

### Parameter reference
- `time_series`: Array of points (tuples).
- `output_file`: Path to output serialized file (`.jls`).
- `window_sz`: Number of points per window.
- `slide`: Window shift between consecutive persistence-image calculations.
- `jump`: Subsampling stride inside each window (`jump=10` keeps every 10th sample).
- `max_dim`: Maximum homology dimension to include.
- `pi_sz`: Persistence image resolution.
  - If an integer, all dimensions use the same size.
  - You may also pass an array of sizes of length `max_dim + 1`.
- `stack=true`: Flattens and stacks persistence images into a single `N × 1` vector.
- `converter`: Optional converter returned from prior preprocessing (recommended for test data).

### Important workflow note
The function returns a `converter` object. Reuse the same converter for test data so train and test persistence images are generated consistently.

---

## Aligning state-label files

State-label files (`*_state_*` or `*_States_*`) must match the number of generated windows. If you change `window_sz`, `slide`, or `jump`, update label lengths accordingly.

A common window count approximation is:

```text
n_windows ≈ floor((n_points - window_sz) / slide) + 1
```

(assuming `n_points >= window_sz` and no additional filtering).

---

## Related docs
- Full project overview: `README.md` (repo root)
- Code and scripts: `Source_Code/README.md`
