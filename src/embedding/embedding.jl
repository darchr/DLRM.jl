module _EmbeddingTables

# types
export AbstractEmbeddingTable, SimpleEmbedding, SplitEmbedding
export SparseEmbeddingUpdate, Static

# functions
export lookup, maplookup

# strategies
export DefaultStrategy, SimpleParallelStrategy, PreallocationStrategy

# local deps
using .._Utils

# deps
import ChainRulesCore
import Flux
import Polyester
import SIMD
import UnPack: @unpack
import Zygote

# Execution strategies describe how to perform `maplookup` across an ensemble of embedding
# tables.
# The `DefaulfExecutionStrategy` merely defaults to serializing `lookup` across each
# embedding table.
#
# This provides an entry point for developing strategies specialized for PMM
abstract type AbstractExecutionStrategy end

#####
##### Embedding Table API
#####

# Used to generate the lookup kernel.
#
# Static kernels have an unrolled kernel that generates significantly less code at the cost
# of requiring a fixed feature size.
abstract type AbstractLookupType end
struct Dynamic <: AbstractLookupType end
struct Static{N} <: AbstractLookupType end
Static(N) = Static{N}()

# For now, require nice alignment for static kernels.
const VECTOR_WIDTH_BYTES = 64
function require_cache_alignment(::Type{Static{N}}, ::Type{T}) where {N,T}
    rem = mod(sizeof(T) * N, VECTOR_WIDTH_BYTES)
    if !iszero(rem)
        msg = """
        Due to implementation limitations, the feature size for static lookup
        kernels must align to $VECTOR_WIDTH_BYTES bytes!

        For feature size $N, this is instead $(rem)!
        """
        throw(ArgumentError(msg))
    end
    return nothing
end

# Supertype for Embedding Tables
abstract type AbstractEmbeddingTable{S<:AbstractLookupType,T} <: AbstractArray{T,2} end
function require_cache_alignment(::AbstractEmbeddingTable{Static{N},T}) where {N,T}
    return require_cache_alignment(Static{N}, T)
end
require_cache_alignment(::AbstractEmbeddingTable{Dynamic}) = nothing

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

featuresize(A::AbstractMatrix) = size(A, 1)
featuresize(A::AbstractEmbeddingTable{Static{N}}) where {N} = N

function columnpointer(A::AbstractMatrix{T}, i::Integer) where {T}
    return pointer(A) + strides(A)[2] * sizeof(T) * (i - 1)
end
@inline columnview(A::AbstractMatrix, i) = view(A, 1:size(A, 1), i)

# Interface
include("simd.jl")
include("lookup.jl")
include("update.jl")

# Embedding Table struct implementations
include("simple.jl")
include("split.jl")

end
