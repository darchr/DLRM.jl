module DLRM

# stdlib
using Mmap

# External Dependencies
using Flux
using NNlib
using ProgressMeter

include("embedding.jl")
include("interact.jl")

# Data Utils
include("data.jl")
include("data_analysis.jl")

# The DLRM model implementation
include("model.jl")

end # module
