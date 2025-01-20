# hdp-hmm-te Source Code

## Usage

This file contains three main Julia scripts:

1) TDA_Tools.jl which contains all of the functions used to process time series data into persistence images.
2) ds_HDP_HMM_AUX.jl which contains a number of auxiliary functions called by the main script.
3) ds_HDP_HMM_w_TDA_weak_limit.jl which is the primary script for running the algorithm.

The first 60 lines of ds_HDP_HMM_w_TDA_weak_limit.jl are for user supplied data.  Most importantly, the user must supply a training data file (in the form of a Julia Serialized Object like that produced by the timeSeries_to_PIs function in the TDA_Tools.jl script) and a set of known states for the training data (as a comma separated file).  Optionally, test data of the same formate can be supplied.  The other parameters can be set as described in ds_HDP_HMM_w_TDA_weak_limit.jl.
