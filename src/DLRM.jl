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

function tocached(m::CachedArrays.CacheManager, s = CachedArrays.NotBusy())
    return tocached(Float32, m, s)
end
function tocached(::Type{T}, m::M, s::S = CachedArrays.NotBusy()) where {T,M,S}
    return ToCached{T,M,S}(m, s)
end

OneDNN.ancestor(x::CachedArrays.CachedArray) = x
function (f::ToCached{T})(x...) where {T}
    return CachedArrays.CachedArray{T}(undef, f.manager, x; status = f.status)
end
function (f::ToCached)(::Type{T}, x...) where {T}
    return CachedArrays.CachedArray{T}(undef, f.manager, x; status = f.status)
end

#####
##### CachedArrays Compatility
#####

# Accessibility hooks
# Creating Model
#
# Make the destination writable for initialization.
@annotate function _Model.multithread_init(f, data::CachedArray)
    return __invoke__(f, __writable__(data))
end

@annotate function Base.fill!(A::CachedArray, x)
    # Preserve the semantics of "fill!" by returning the filled object, but return
    # the original object rather than the potentially newly created writable one.
    __invoke__(__writable__(A), x)
    return A
end

const MaybeTranspose{T} = Union{T,LinearAlgebra.Transpose{<:Any,<:T}}
@annotate function OneDNN.creatememory(x::MaybeTranspose{CachedArray}, desc)
    return __invoke__(__readable__(x), desc)
end

@annotate function OneDNN.Dense(
    weights::MaybeTranspose{CachedArray}, bias::CachedArray, args...
)
    return __invoke__(__readable__(weights), __readable__(bias), args...)
end

# Embedding Tables
function CachedArrays.readable(x::SimpleEmbedding{T,A,N}) where {T,A<:CachedArray,N}
    a = CachedArrays.readable(x.data)
    return SimpleEmbedding(a, Val(N))
end

@annotate function _EmbeddingTables.lookup!(
    O,
    A::SimpleEmbedding{T,<:UnreadableCachedArray},
    I::AbstractVector{<:Integer},
    style::_EmbeddingTables.Static{N},
) where {T,N}
    return __recurse__(O, __readable__(A), I, style)
end

@annotate function OneDNN.access_pointer(x::UnreadableCachedArray, offset, ::OneDNN.Reading)
    return pointer(__readable__(x), offset)
end

@annotate function OneDNN.access_pointer(x::UnwritableCachedArray, offset, ::OneDNN.Writing)
    return pointer(__writable__(x), offset)
end

# Capture memories coming out of OneDNN kernels and convert them to "NotBusy".
function OneDNN.kernel_exit_hook(
    x::OneDNN.Memory{L,T,N,CachedArray{T,N,S,M}}
) where {L,T,N,S,M}
    return convert(OneDNN.Memory{L,T,N,CachedArray{T,N,CachedArrays.NotBusy,M}}, x)
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

@annotate function Flux.update!(
    o::Flux.Descent,
    x::OneDNN.Memory{<:Any,<:Any,<:Any,<:UnwritableCachedArray},
    y::OneDNN.Memory{<:Any,<:Any,<:Any,<:UnreadableCachedArray},
)
    return __recurse__(o, __writable__(x), __readable__(y))
end

end # module
