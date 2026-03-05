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

Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model with Topological Emissions (**ds-HDP-HMM-TE**).

This repository contains:
- Julia source code for training and evaluating the model
- utilities for topological data analysis (TDA)-based emissions
- sample datasets used by the provided scripts

## Repository layout

- `Source_Code/` — core Julia implementation and usage notes
  - `ds_HDP_HMM_w_TDA_weak_limit.jl` — main model implementation
  - `ds_HDP_HMM_AUX.jl` — helper routines used by the model
  - `TDA_Tools.jl` — TDA feature/persistence-diagram tooling
  - `README.md` — script-level usage details
- `Sample_Data/` — CSV sample datasets and data preparation notes
  - `README.md` — how to transform CSV data into serialized persistence-image inputs expected by the model

## Prerequisites

- Julia (version compatible with the scripts in `Source_Code/`)
- The following Julia packages:

```julia
Clustering
CSV
DataFrames
DelimitedFiles
Distributions
Hungarian
LinearAlgebra
PersistenceDiagrams
PlotlyJS
Printf
ProgressBars
Random
Ripserer
Serialization
SpecialFunctions
StatsBase
Tables
```

## Setup

1. Install Julia.
2. Ensure all required packages are available in your Julia environment.
3. Prepare the data:
   - Start from files in `Sample_Data/` (or your own equivalent data).
   - Convert the CSV inputs into vectorized persistence images.
   - Save those persistence-image outputs as Julia `Serialization` objects.
   - Follow `Sample_Data/README.md` for the expected preparation workflow.

## Running the code

1. Open the scripts in `Source_Code/`.
2. Follow the execution instructions in `Source_Code/README.md`.
3. Point script configuration to your prepared serialized input data.

## Data notes

The included sample data are intended as reference/test inputs for the repository workflow. File naming in `Sample_Data/` reflects train/test splits and state/change-point variants used by the example pipeline.

## Additional documentation

- For dataset preparation specifics: `Sample_Data/README.md`
- For script-level execution details and parameters: `Source_Code/README.md`

Sample_Data contains data in CSV form.  The data must be first converted to vectorized persistence images and saved as a Julia Serialization object.  See the README in that folder for details.
---

## Usage
See README in Source_Code for details.
If you use this project in research, consider documenting the exact Julia/package versions used in your environment to improve reproducibility.

