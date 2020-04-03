module DLRM

# stdlib
using Distributed   # for preprocessing dataset
using Mmap
using SparseArrays

# External Dependencies
import DataStructures
using Flux
import GZip
import MappedArrays
using NNlib
using ProgressMeter
import PrettyTables

include("embedding.jl")
include("interact.jl")

# Data Utils
include("data.jl")
include("data_analysis.jl")

# The DLRM model implementation
include("model.jl")

end # module
