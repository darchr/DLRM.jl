using DLRM
using Test
using Random

# For checking gradients
using Flux
using Zygote
using OneDNN

const DATASET_DIR = joinpath(@__DIR__, "dataset")

# Embedding Table Lookup
include("embedding/lookup.jl")
include("embedding/map.jl")
include("embedding/update.jl")

# Model Pipeline
include("model/interact.jl")
include("model/model.jl")

# Training
include("train/loss.jl")
include("train/backprop.jl")

# Data Loaders
include("data/criteo.jl")

# include("dataset.jl")
