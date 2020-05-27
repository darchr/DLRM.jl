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
using NNlib
using ProgressMeter
import PrettyTables
import Zygote

# Extra Zygote Adjoints for improving the performance
# of broadcasting.
include("interact.jl")

# Data Utils
include("preprocess.jl")

# The DLRM model implementation
include("model.jl")
include("train/data.jl")
include("train/train.jl")

end # module
