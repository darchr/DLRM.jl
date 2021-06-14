module DLRM

export DLRMModel, dlrm

# stdlib
using LinearAlgebra: LinearAlgebra
using Mmap: Mmap
using SparseArrays: SparseArrays
using Random: Random
using Serialization: Serialization
import Statistics: mean

# "Internal" dependencies
using CachedArrays
using OneDNN: OneDNN

# External Dependencies
using DataStructures: DataStructures
using Flux: Flux
using HDF5: HDF5
using NaturalSort: NaturalSort
using ProgressMeter: ProgressMeter
import UnPack: @unpack

include("utils/utils.jl")
using ._Utils

include("embedding/embedding.jl")
using ._EmbeddingTables

include("model/model.jl")
using ._Model

include("train/train.jl")
using ._Train

include("data/criteo.jl")

#####
##### Keep these definitions here for now until we find a better home.
#####

struct ToCached{T,M}
    manager::M
end

tocached(m::CachedArrays.CacheManager) = tocached(Float32, m)
tocached(::Type{T}, m::M) where {T,M} = ToCached{T,M}(m)

OneDNN.ancestor(x::CachedArrays.CachedArray) = x
(f::ToCached{T})(x...) where {T} = CachedArrays.CachedArray{T}(undef, f.manager, x)
(f::ToCached)(::Type{T}, x...) where {T} = CachedArrays.CachedArray{T}(undef, f.manager, x)

#####
##### CachedArrays Compatility
#####

# Make the destination writable for initialization.
@hint function _Model.multithread_init(f, data::CachedArray)
    return __invoke__(f, __writable__(data))
end

@hint function Base.fill!(A::CachedArray, x)
    # Preserve the semantics of "fill!" by returning the filled object, but return
    # the original object rather than the potentially newly created writable one.
    __invoke__(__writable__(A), x)
    return A
end

const MaybeTranspose{T} = Union{T,LinearAlgebra.Transpose{<:Any,<:T}}
@hint function OneDNN.creatememory(x::MaybeTranspose{CachedArray}, desc)
    return __invoke__(__readable__(x), desc)
end

@hint function OneDNN.Dense(
    weights::MaybeTranspose{CachedArray}, bias::CachedArray, args...
)
    return __invoke__(__readable__(weights), __readable__(bias), args...)
end

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
