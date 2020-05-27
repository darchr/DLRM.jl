#####
##### Extractor
#####

# Efficient extraction from the Kaggle Dataset
mutable struct Extractor{T}
    data::T
    batchsize::Int

    # Preloaded buffers for storing the data for the next batchsize
    labels::Vector{Float32}
    dense::Array{Float32, 2}
    sparse::Vector{Vector{UInt32}}
end

function Extractor(data, batchsize)
    return Extractor(
        data,
        batchsize,
        Array{Float32}(undef, batchsize),
        Array{Float32}(undef, num_continuous_features(DAC()), batchsize),
        [Array{UInt32}(undef, batchsize) for _ in 1:num_categorical_features(DAC())],
    )
end

function Base.iterate(E::Extractor, index = 1)
    batchsize = E.batchsize

    # Detect if we've run out of data.
    if index + batchsize - 1 > length(E.data)
        return nothing
    end

    return @inbounds(E[index]), index + batchsize
end

function Base.getindex(E::Extractor, index::Integer)
    batchsize = E.batchsize

    # Detect if we've run out of data.
    @boundscheck if index + batchsize - 1 > length(E.data)
        throw(BoundsError(E, index))
    end

    for (count,i) in enumerate(index:index + batchsize - 1)
        # Access this row of data.
        row = E.data[i]

        E.labels[count] = row.label
        E.dense[:,count] .= row.continuous

        for (indices, val) in zip(E.sparse, row.categorical)
            indices[count] = val
        end
    end
    return (dense = E.dense, sparse = E.sparse, labels = E.labels)
end

#####
##### DataGenerator
#####

# Generate Dummy Data for benchmarking larger models.
struct DataGenerator
    # Pre-loaded buffers
    labels::Vector{Float32}
    dense::Array{Float32,2}
    sparse::Vector{Vector{UInt32}}
    sparsesizes::Vector{Int32}
end

batchsize(x::DataGenerator) = length(x.labels)

"""
    DataGenerator(; batchsize, densesize, sparsesizes)

Construct a random `DataGenerator` for the DLRM network.

* `batchsize::Integer`: The batchsize for which to supply data.
* `densesize::Integer`: The number of dense features. For the Kaggle DLRM, this is 13.
* `sparsesizes::Vector{<:Integer}`: A vector of integers. The length of the vector denotes
    the number of embedding tables. The entry at index `i` indicates the number of entries
    in embedding table `i`.
"""
function DataGenerator(; batchsize, densesize, sparsesizes)
    labels = Vector{Float32}(undef, batchsize)
    dense = Array{Float32,2}(undef, densesize, batchsize)

    sparsesizes = Int64.(sparsesizes)
    sparse = [Vector{UInt32}(undef, batchsize) for _ in sparsesizes]
    return DataGenerator(
        labels,
        dense,
        sparse,
        sparsesizes,
    )
end

function Base.getindex(x::DataGenerator, args...)
    # Configure labels
    for i in eachindex(x.labels)
        @inbounds x.labels[i] = eltype(x.labels)(rand((0,1)))
    end

    # Configure dense
    for i in eachindex(x.dense)
        @inbounds x.dense[i] = randn(eltype(x.dense))
    end

    # Configure sparse
    for (vec, sz) in zip(x.sparse, x.sparsesizes)
        for i in eachindex(vec)
            @inbounds vec[i] = rand(1:sz)
        end
    end
    return (dense = x.dense, sparse = x.sparse, labels = x.labels)
end

