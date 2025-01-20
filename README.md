# hdp-hmm-te
Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model with Topological Emissions
(developed in VS Code using Julia)

## Install
The primary code is found in the Source_Code folder.  Once Julia is installed, the following packages must also be available.

Clustering,
CSV,
DataFrames,
DelimitedFiles,
Distributions,
Hungarian,
LinearAlgebra,
PersistenceDiagrams,
PlotlyJS,
Printf,
ProgressBars,
Random,
Ripserer,
Serialization,
SpecialFunctions,
StatsBase,
Tables

Sample_Data contains data in CSV form.  The data must be first converted to vectorized persistence images and saved as a Julia Serialization object.  See the README in that folder for details.

## Usage
See READMEs in Source_Code for details.
