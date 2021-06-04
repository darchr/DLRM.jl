using DLRM
using Test
using Random

using Flux
using HDF5: HDF5
using NaturalSort: NaturalSort
using Zygote
using OneDNN
using StaticArrays

const DATASET_DIR = joinpath(@__DIR__, "dataset")

# # Embedding Table Lookup
# include("embedding/lookup.jl")
# include("embedding/map.jl")
# include("embedding/update.jl")

# Model Pipeline
include("model/interact.jl")
include("model/model.jl")

# Training
include("train/loss.jl")
include("train/backprop.jl")

# Data Loaders
include("data/criteo.jl")

# Larger integration tests
include("integration.jl")

# # include("dataset.jl")
