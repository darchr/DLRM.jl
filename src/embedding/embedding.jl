module _EmbeddingTables

export SimpleEmbedding, lookup, maplookup, SparseEmbeddingUpdate, SimpleParallelStrategy

# For defining sparse adjoints.
import ChainRulesCore
using Flux
using Zygote

# To vectorize lookup operations
using SIMD
using MacroTools

# Execution strategies describe how to perform `maplookup` across an ensemble of embedding
# tables.
# The `DefaulfExecutionStrategy` merely defaults to serializing `lookup` across each
# embedding table.
#
# This provides an entry point for developing strategies specialized for PMM
abstract type AbstractExecutionStrategy end
struct DefaultExecutionStrategy <: AbstractExecutionStrategy end

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

# Super type for Embedding Tables
abstract type AbstractEmbeddingTable{T} <: AbstractArray{T,2} end

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

featuresize(A::AbstractMatrix) = size(A, 1)
lookuptype(::AbstractEmbeddingTable) = Dynamic()
columnpointer(A::AbstractMatrix, i::Integer) = pointer(A, featuresize(A) * (i-1) + 1)
@inline columnview(A::AbstractMatrix, i) = view(A, 1:size(A,1), i)

# Interface
include("simd.jl")
include("lookup.jl")
include("update.jl")

# Embedding Table struct implementations
include("simple.jl")
include("split.jl")

#####
##### Random Utility Functions
#####

# Check if `A` is aligned to a cache-line boundary.
# If not, get angry.
function cached_aligned_error(A)
    if !iszero(mod(convert(Int, pointer(A)), 64))
        error("Array must be aligned to a cache-line boundary (multiple of 64-bytes)!")
    end
end


end
