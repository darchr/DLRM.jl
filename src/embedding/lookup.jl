#####
##### Reference Implementations
#####

### Non-reducing
lookup(A::AbstractMatrix, I::AbstractVector{<:Integer}) = A[:, I]

### Reducing
function lookup(A::AbstractMatrix, II::AbstractMatrix{<:Integer})
    _A = [lookup(A, II[:, i]) for i = 1:size(II, 2)]
    _b = [[sum(_a[i, :]) for i = 1:size(_a, 1)] for _a in _A]
    return hcat(_b...)
end

#####
##### lookup
#####

const VecOrMat{T} = Union{Vector{T},Matrix{T}}
_trailing_size(x::AbstractVector) = length(x)
_trailing_size(x::AbstractMatrix) = size(x, 2)

function lookup(A::AbstractEmbeddingTable{S,T}, I::VecOrMat{<:Integer}) where {S,T}
    nrows = featuresize(A)
    O = similar(example(A), T, nrows, _trailing_size(I))
    # inner `lookup!` dispatches to either an optimized static or dynamic fallback
    # implementations.
    lookup!(O, A, I)
    return O
end

### Non-reducing

# Optimized static branch.
# Implementation is in "simd.jl"
@generated function lookup!(
    dst, src::AbstractEmbeddingTable{Static{N},T}, indices::AbstractVector{<:Integer}
) where {T,N}
    return emit_lookup(T, N)
end

# fallback dynamic implementation
function lookup!(
    O, A::AbstractEmbeddingTable{Dynamic,T}, I::AbstractVector{<:Integer}
) where {T}
    nrows = featuresize(A)
    for (col, i) in enumerate(I)
        @inbounds ptrA = columnpointer(A, i)
        @inbounds ptrO = columnpointer(O, col)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
end

### Reducing (sum)

# Optimized static branch.
# Implementation is in "simd.jl"
@generated function lookup!(
    dst, src::AbstractEmbeddingTable{Static{N},T}, indices::AbstractMatrix{<:Integer}
) where {T,N}
    return emit_lookup_reducing(T, N)
end

# fallback dynamic implementation
function lookup!(
    O, A::AbstractEmbeddingTable{Dynamic,T}, I::AbstractMatrix{<:Integer}
) where {T}
    nrows = featuresize(A)
    sz = size(i, 2)
    for j in enumerate(Base.OneTo(size(I, 1))), i in Base.OneTo(sz)
        col = @inbounds(I[i, j])
        @inbounds ptrA = columnpointer(A, j)
        @inbounds ptrO = columnpointer(O, col)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
end

#####
##### Embedding Updates
#####

# A sparse updater for embedding tables.
struct SparseEmbeddingUpdate{S<:AbstractLookupType,A<:AbstractMatrix,I<:AbstractArray}
    delta::A
    indices::I
end

function SparseEmbeddingUpdate{S}(delta::A, indices::I) where {S,A,I}
    return SparseEmbeddingUpdate{S,A,I}(delta, indices)
end

# Convert the compressed representations
function uncompress(
    x::SparseEmbeddingUpdate{<:Any,<:Any,<:AbstractVector},
    dstcols = maximum(x.indices);
    maxindices = length(x.indices),
)
    @unpack indices, delta = x
    dst = similar(delta, size(delta, 1), dstcols)
    dst .= zero(eltype(dst))
    count = 0
    for (column, update) in zip(indices, eachcol(delta))
        columnview(dst, column) .+= update
        count += 1
        count == maxindices && break
    end
    return dst
end

# Compress all updates for each column in place.
# The updates to the final embedding table can then all be performed at once.

_maybe_columnview(x::AbstractVector, i) = x[i]
_maybe_columnview(x::AbstractMatrix, i) = view(x, :, i)

function crunch(
    src::SparseEmbeddingUpdate{Static{N},A,<:AbstractVector},
    translation::Dict{Int,Int} = Dict{Int,Int}();
    mulby = one(eltype(A)),
) where {N,A}
    return _crunch!(src, src, translation, mulby)
end

function crunch(
    src::SparseEmbeddingUpdate{Static{N},A,<:AbstractMatrix},
    translation::Dict{Int,Int} = Dict{Int,Int}();
    mulby = one(eltype(A)),
) where {N,A}
    # Use our dictionary to count the number of unique indices.
    for i in src.indices
        translation[i] = 0
    end
    num_unique_indices = length(translation)

    # Now, create a new SparseEmbeddingUpdate that is large enough to hold our potentially
    # expanded result.
    src_delta = src.delta
    src_indices = src.indices
    dst_delta = similar(
        src_delta, eltype(src_delta), size(src_delta, 1), num_unique_indices
    )
    dst_indices = similar(src_indices, eltype(src_indices), num_unique_indices)
    dst = SparseEmbeddingUpdate{Static{N}}(dst_delta, dst_indices)
    return _crunch!(dst, src, translation, mulby)
end

# N.B.: This algorithm is written so `dst` and `src` can be the same object.
# This will only work if the number of unique indices in `src` is less than or equal
# the current `length(src.indices)`.
#
# For the multi-lookup case, we creatge a new `SparseEmbeddingUpdate` struct and use
# that.
function _crunch!(
    dst::SparseEmbeddingUpdate{Static{N},A,<:AbstractVector},
    src::SparseEmbeddingUpdate{Static{N},A,<:VecOrMat},
    translation::Dict{Int,Int},
    mulby,
) where {N,A}
    head = 1

    # Unpack Arguments
    dst_delta, dst_indices = dst.delta, dst.indices
    src_delta, src_indices = src.delta, src.indices

    empty!(translation)
    for i in Base.OneTo(size(src_delta, 2))
        for target_column in _maybe_columnview(src_indices, i)
            accumulation_column = get!(translation, target_column, head)
            if accumulation_column == head
                # Move this column to the head pointer and update the `indices` array
                # appropriately.
                # Since we're moving sequentially, we don't have to worry about destroying data.
                _dst = columnview(dst_delta, head)
                _src = columnview(src_delta, i)
                for j in Base.OneTo(N)
                    @inbounds(_dst[j] = mulby * _src[j])
                end

                dst_indices[head] = target_column
                head += 1
                continue
            end

            # We already have a column in `delta` for the target destination.
            # Add the next update in place.
            _dst = columnview(dst_delta, accumulation_column)
            _src = columnview(src_delta, i)
            for j in Base.OneTo(N)
                @inbounds(_dst[j] += mulby * _src[j])
            end
        end
    end
    return dst, head - 1
end

# -- pullback
function ChainRulesCore.rrule(::typeof(lookup), A::AbstractEmbeddingTable{S}, I) where {S}
    function lookup_pullback(Δ)
        return (
            ChainRulesCore.NoTangent(),
            SparseEmbeddingUpdate{S}(Δ, I),
            ChainRulesCore.NoTangent(),
        )
    end
    return lookup(A, I), lookup_pullback
end

#####
##### ColumnWrap
#####

# The `ColumnWrap` types lets us treat a 2D matrix as an array of arrays.
struct ColumnWrap{A}
    array::A
end

unwrap(x::ColumnWrap) = x.array
Base.eachindex(x::ColumnWrap) = Base.OneTo(size(x.array, 2))
Base.getindex(x::ColumnWrap, i::Integer) = view(x.array, :, i)

Base.length(x::ColumnWrap) = size(x.array, 2)
function Base.iterate(x::ColumnWrap, i = 1)
    return in(i, eachindex(x)) ? (@inbounds(x[i]), i + 1) : nothing
end

# Dispatch plumbing for-the-win.
colwrap(x::ColumnWrap) = x
colwrap(x::AbstractVector{<:AbstractVector}) = x
colwrap(x::AbstractMatrix) = ColumnWrap(x)

#####
##### maplookup
#####

function maplookup(x::Vector{A}, i...) where {A<:AbstractEmbeddingTable}
    return maplookup(DefaultStrategy(), x, i...)
end

function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::AbstractExecutionStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    I,
) where {S}
    function maplookup_pullback(Δs)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.@thunk(map(SparseEmbeddingUpdate{S}, Δs, colwrap(I))),
            ChainRulesCore.NoTangent(),
        )
    end
    result = maplookup(strategy, A, I)
    return result, maplookup_pullback
end

### Default Strategy
# Just all "lookup" on each table.
struct DefaultStrategy <: AbstractExecutionStrategy end
function maplookup(
    strategy::DefaultStrategy, x::Vector{A}, I
) where {A<:AbstractEmbeddingTable}
    return map(lookup, x, colwrap(I))
end

### Simple Parallel strategy.
# Thread lookups using Julia's normal multi-threading.
struct SimpleParallelStrategy <: AbstractExecutionStrategy end
function maplookup(::SimpleParallelStrategy, x::Vector{<:AbstractEmbeddingTable}, _I)
    out = Vector{typeof(example(x[1]))}(undef, length(x))
    I = colwrap(_I)
    Threads.@threads for i in eachindex(x, I)
        out[i] = lookup(x[i], I[i])
    end
    return out
end

### Preallocate destinations
# The idea of the preallocation strategy is to essentially merge the "Concat" step with
# the "EmbeddingLookup" step.
#
# This may involve preallocating some space in the first few rows of the destination
# array to make space for inserting the result of the bottom MLP.
struct PreallocationStrategy <: AbstractExecutionStrategy
    # Allow for extra rows to be placed at the beginning of the destination to allow
    # the results of dense computation to be inserted inplace.
    prependrows::Int
end
PreallocationStrategy() = PreallocationStrategy(0)

_batchsize(x::AbstractVector) = length(first(x))
_batchsize(x::AbstractMatrix) = size(x, 1)
_batchsize(x::ColumnWrap) = _batchsize(unwrap(x))

function maplookup(
    strategy::PreallocationStrategy, x::Vector{<:AbstractEmbeddingTable{T}}, _I
) where {T}
    # Preallocate destination.
    I = colwrap(_I)
    rows = featuresize.(x)
    offset = strategy.prependrows
    batchsize = _batchsize(_I)
    data = similar(example(x[1]), strategy.prependrows + sum(rows), batchsize)

    # For deciding where to index
    rows_sum = cumsum(rows)
    pushfirst!(rows_sum, 0)

    #Threads.@threads for i in eachindex(x, I)
    Polyester.@batch per = thread for i in eachindex(x)
        start = 1 + offset + rows_sum[i]
        stop = offset + rows_sum[i + 1]

        # Create destination view
        O = view(data, start:stop, Base.OneTo(batchsize))
        A = x[i]

        lookup!(O, A, I[i])
    end
    return data
end

function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::PreallocationStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    _I,
) where {S}
    I = colwrap(_I)
    data = maplookup(strategy, A, I)
    function maplookup_pullback(Δ)
        f = Slicer(strategy.prependrows + 1, 1, Δ)
        δs = map((y, x) -> SparseEmbeddingUpdate{S}(f(featuresize(y)), x), A, I)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            δs,
            ChainRulesCore.NoTangent(),
        )
    end
    return data, maplookup_pullback
end
