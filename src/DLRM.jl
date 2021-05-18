module DLRM

export DLRMModel, dlrm

# stdlib
using Mmap: Mmap
using SparseArrays: SparseArrays
using Random: Random
using Serialization: Serialization
import Statistics: mean

# "Internal" dependencies
using OneDNN: OneDNN

# External Dependencies
using DataStructures: DataStructures
using Flux: Flux
using HDF5: HDF5
using NaturalSort: NaturalSort
using ProgressMeter: ProgressMeter
import UnPack: @unpack

include("utils/threading.jl")
include("embedding/embedding.jl")
using ._EmbeddingTables

include("model/model.jl")
using ._Model

include("train/train.jl")
using ._Train

include("data/criteo.jl")

#
# # Data Utils
# include("preprocess.jl")
# include("dataset.jl")
#
# # The DLRM model implementation
# include("model.jl")
# include("loss.jl")
# include("train/data.jl")
# include("train/train.jl")
#
end # module
