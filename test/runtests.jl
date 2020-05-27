using DLRM
using Test
using Random

# For checking gradients
using Flux
using Zygote

#include("interact.jl")
include("dataset.jl")
include("model.jl")
include("data.jl")
