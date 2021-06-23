# The split embedding table shards data into chunks.
# Each chunk is stored as a matrix.
# Parameter `F` is the feature size - used to generate high-performance lookups.
struct SplitEmbedding{S,T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{S,T}
    data::Vector{A}
    # All sub matrices (except for the last) should be the same size.
    # For the last, we provide some wiggle room to allow for non-perfectly sized
    # embedding tables.
    matrixsize::Tuple{Int,Int}

    # Inner constructor to ensure uniform sub-matrix sizing
    function SplitEmbedding(A::AbstractMatrix, cols_per_shard = 1)
        # Determine the number of shards we will have.
        nshards = ceil(Int, size(A,2) / cols_per_shard)

        data = map(1:nshards) do i
            start = cols_per_shard * (i - 1) + 1
            stop = min(i * cols_per_shard, size(A,2))

            B = similar(A, eltype(A), size(A, 1), stop - start + 1)
            @views B .= A[:, start:stop]
            return B
        end

        matrixsize = (size(A,1),cols_per_shard)
        return new{Static{size(A,1)},eltype(A), typeof(A)}(
            data,
            matrixsize,
        )
    end
end

#####
##### Array Interface
#####

# Helper Functions
function _divrem_index(i, x)
    a, b = divrem(i, x)
    # In the case where the remainder is zero, we actually need to step back one chunk.
    return iszero(b) ? (a, x) : (a+1, b)
end

chunkindex(A::SplitEmbedding, i::Int) = _divrem_index(i, prod(A.matrixsize))

# Interface
function Base.size(A::SplitEmbedding)
    nrows = A.matrixsize[1]
    ncols = A.matrixsize[2] * (length(A.data) - 1) + size(last(A.data), 2)
    return (nrows, ncols)
end

function Base.getindex(A::SplitEmbedding, i::Int)
    @boundscheck checkbounds(A, i)
    # Find which chunk the data is in, then lookup that chunk
    chunk, index = chunkindex(A, i)
    return A.data[chunk][index]
end

function Base.setindex!(A::SplitEmbedding, v, i::Int)
    @boundscheck checkbounds(A, i)
    # Find which chunk the data is in, then lookup that chunk
    chunk, index = chunkindex(A, i)
    return A.data[chunk][index] = v
end

#####
##### EmbeddingTables Interface
#####

# featuresize(::SplitEmbedding{T,A,F}) where {T,A,F} = F
# lookuptype(::SplitEmbedding{T,A,F}) where {T,A,F} = Static{F}()
example(A::SplitEmbedding) = first(A.data)

Base.@propagate_inbounds function columnpointer(A::SplitEmbedding, i::Integer)
    # Find the chunk and return the column from that chunk
    chunk, col = _divrem_index(i, A.matrixsize[2])
    return columnpointer(A.data[chunk], col)
end

# Return a column view of some underlying chunk
Base.@propagate_inbounds @inline function columnview(A::SplitEmbedding, i::Integer)
    chunk, col = _divrem_index(i, A.matrixsize[2])
    return columnview(A.data[chunk], col)
end

