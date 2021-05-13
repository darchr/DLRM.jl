module DLRM

export DLRMModel, dlrm

# stdlib
import Mmap
import SparseArrays
import Random
import Serialization
import Statistics: mean

# "Internal" dependencies
using OneDNN: OneDNN

# External Dependencies
import DataStructures
import ProgressMeter
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
