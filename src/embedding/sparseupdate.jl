#####
##### Embedding Updates
#####

# Compress all updates for each column in place.
# The updates to the final embedding table can then all be performed at once.
_maybe_columnview(x::AbstractVector, i) = x[i]
_maybe_columnview(x::AbstractMatrix, i) = columnview(x, i)

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
    x::SparseEmbeddingUpdate, dstcols = maximum(x.indices); maxindices = length(x.indices)
)
    @unpack indices, delta = x
    dst = zeros(eltype(delta), size(delta, 1), dstcols)
    count = 0
    for (column, update) in enumerate(eachcol(delta))
        for c in _maybe_columnview(indices, column)
            columnview(dst, c) .+= update
        end
        count += 1
        count == maxindices && break
    end
    return dst
end

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
    dst_indices = similar(src_delta, eltype(src_indices), num_unique_indices)
    dst = SparseEmbeddingUpdate{Static{N}}(dst_delta, dst_indices)
    return _crunch!(dst, src, translation, mulby)
end

# N.B.: This algorithm is written so `dst` and `src` can be the same object.
# This will only work if the number of unique indices in `src` is less than or equal
# the current `length(src.indices)`.
#
# For the multi-lookup case, we create a new `SparseEmbeddingUpdate` struct and use that.
function _crunch!(
    dst::SparseEmbeddingUpdate{Static{N},A,<:AbstractVector},
    src::SparseEmbeddingUpdate{Static{N},B,<:VecOrMat},
    translation::Dict{Int,Int},
    mulby,
) where {N,A,B}
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

# The job of "partition!" is to split up each sparse tables into multiple sparse
# tables using the power of views.
#
# For example, an update may look like this:
#
#  |*************************|
#  |*************************|   Delta
#  |*************************|
#
#  |*************************|   Indices
#
#  And we want to turn it into this
#
#  |*****|*****|*****|*****|*****|
#  |*****|*****|*****|*****|*****|   Delta
#  |*****|*****|*****|*****|*****|
#
#  |*****|*****|*****|*****|*****|   Indices
#
# Where each of the inner entries is ALSO a SparseEmbeddingUpdate,
#
# We use a stateful iterator to simplify downstream implementations that require using
# many such partitioners.
mutable struct UpdatePartitioner{S,A,I}
    update::SparseEmbeddingUpdate{S,A,I}
    base::Int
    batchsize::Int
end

function UpdatePartitioner(update::SparseEmbeddingUpdate, batchsize::Integer)
    @unpack delta, indices = update
    return UpdatePartitioner(update, 1, batchsize)
end

_maybeslice(x::AbstractVector, range) = view(x, range)
_maybeslice(x::AbstractMatrix, range) = view(x, :, range)

function Base.iterate(x::UpdatePartitioner{S,A,I}, _ = nothing) where {S,A,I}
    @unpack update, base, batchsize = x
    @unpack delta, indices = update
    last = size(delta, 2)

    # Are we done?
    base > last && return nothing

    # Still have room left. Lets create an update!
    upper = min(base + batchsize, last)
    subdelta = view(delta, :, base:upper)
    subindices = _maybeslice(indices, base:upper)
    x.base = upper + 1
    return SparseEmbeddingUpdate{S}(subdelta, subindices)
end

