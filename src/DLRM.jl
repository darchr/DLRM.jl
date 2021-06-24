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
using ChainRulesCore: ChainRulesCore
using ConstructionBase: ConstructionBase
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

struct ToCached{T,M,S}
    manager::M
    status::S
end

function tocached(m, s = CachedArrays.NotBusy())
    return tocached(Float32, m, s)
end
function tocached(::Type{T}, m::M, s::S = CachedArrays.NotBusy()) where {T,M,S}
    return ToCached{T,M,S}(m, s)
end

OneDNN.ancestor(x::CachedArrays.CachedArray) = x
OneDNN.ancestor(x::CachedArrays.HeapArray) = x

function (f::ToCached{T,<:CachedArrays.CacheManager})(x...) where {T}
    return CachedArrays.CachedArray{T}(undef, f.manager, x; status = f.status)
end
function (f::ToCached{<:Any,<:CachedArrays.CacheManager})(::Type{T}, x...) where {T}
    return CachedArrays.CachedArray{T}(undef, f.manager, x; status = f.status)
end

function (f::ToCached{T,CachedArrays.HeapManager})(x...) where {T}
    return CachedArrays.HeapArray{T}(undef, f.manager, x)
end

function (f::ToCached{<:Any,CachedArrays.HeapManager})(::Type{T}, x...) where {U,T}
    return CachedArrays.HeapArray{T}(undef, f.manager, x)
end

#####
##### CachedArrays Compatility
#####

const UnwritableMemory = MemoryAround{UnwritableCachedArray}
const UnreadableMemory = MemoryAround{UnreadableCachedArray}

CachedArrays.@wrapper OneDNN.Memory array
function CachedArrays.constructorof(
    ::Type{_EmbeddingTables.SimpleEmbedding{_EmbeddingTables.Static{N},T,A}}
) where {N,T,A}
    return x -> _EmbeddingTables.SimpleEmbedding(x, Val(N))
end
CachedArrays.@wrapper _EmbeddingTables.SimpleEmbedding data

# Accessibility hooks
# Creating Model
#
# Make the destination writable for initialization.
@annotate function _Model.multithread_init(f, data::UnwritableCachedArray)
    return __recurse__(f, __writable__(data))
end

# Grab the bias that is being returned and convert it to NotBusy.
@annotate function Flux.create_bias(weights::CachedArray, bias::Bool, dims::Integer...)
    return __release__(__invoke__(weights, bias, dims...))
end

const MaybeTranspose{T} = Union{T,LinearAlgebra.Transpose{<:Any,<:T}}
@annotate function OneDNN.MemoryPtr(x::MaybeTranspose{<:UnreadableCachedArray}, desc)
    return __recurse__(__readable__(x), desc)
end

@annotate function _EmbeddingTables.lookup!(
    O,
    A::SimpleEmbedding{S,T,<:UnreadableCachedArray},
    I::AbstractVector{<:Integer},
) where {S,T,N}
    return __recurse__(O, __readable__(A), I)
end

@annotate function Flux.update!(
    x::_EmbeddingTables.SimpleEmbedding{<:Any,<:Any,<:UnwritableCachedArray},
    xbar::_EmbeddingTables.SparseEmbeddingUpdate{<:Any,<:Any,<:AbstractVector},
    numcols::Integer
)
    return __recurse__(__writable__(x), xbar, numcols)
end

# Since OneDNN kernels are long running, we can hook into the "access_pointer" API in order
# to circumvent the need to change the wrapped array type.
#
# However, we need to clean up the `__readable__` call to avoid creating an entire new array
# and instead just use a CachedArray callback to save on some allocations.
@annotate function OneDNN.access_pointer(x::UnreadableCachedArray, offset, ::OneDNN.Reading)
    return pointer(__readable__(x), offset)
end

@annotate function OneDNN.access_pointer(x::UnwritableCachedArray, offset, ::OneDNN.Writing)
    return pointer(__writable__(x), offset)
end

# Capture memories coming out of OneDNN kernels and convert them to "NotBusy".
@annotate function OneDNN.kernel_exit_hook(x::MemoryAround{CachedArray})
    return __release__(x)
end

@annotate function (dot::_Model.DotInteraction)(
    x::UnreadableCachedArray, ys::ReadableCachedArray; kw...
)
    return dot(__readable__(x), ys; kw...)
end

@annotate function ChainRulesCore.rrule(
    f::typeof(_Train.bce_loss), y::UnreadableCachedArray, x::CachedArray
)
    return __recurse__(f, __readable__(y), __readable__(x))
end

@annotate function _Model.process_batches_back(
    dot::_Model.DotInteraction, Δ::UnreadableCachedArray, x...
)
    return __recurse__(dot, __readable__(Δ), x...)
end

# Two update flavors.
@annotate function Flux.update!(o::Flux.Descent, x::UnwritableMemory, y::UnreadableMemory)
    return __recurse__(o, __writable__(x), __readable__(y))
end

@annotate function Flux.update!(o::Flux.Descent, x::UnwritableMemory, ix, y::UnreadableMemory, iy)
    return __recurse__(o, __writable__(x), ix, __readable__(y), iy)
end

end # module
