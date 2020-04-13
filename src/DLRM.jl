module DLRM

# stdlib
using Distributed   # for preprocessing dataset
using Mmap
using SparseArrays

# "Internal" dependencies
using EmbeddingTables

# External Dependencies
import DataStructures
using Flux
import GZip
import MappedArrays
using NNlib
using ProgressMeter
import PrettyTables
import Zygote

# Extra Zygote Adjoints for improving the performance
# of broadcasting.
include("interact.jl")

# Data Utils
include("data.jl")
include("data_analysis.jl")

# The DLRM model implementation
include("model.jl")
include("train.jl")

end # module
