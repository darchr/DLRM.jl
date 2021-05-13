#####
##### Extractor
#####

# Efficient extraction from the Kaggle Dataset
mutable struct Extractor{T}
    data::T
    batchsize::Int

    # Preloaded buffers for storing the data for the next batchsize
    labels::Vector{Float32}
    dense::Array{Float32,2}
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

    for (count, i) in enumerate(index:(index + batchsize - 1))
        # Access this row of data.
        row = E.data[i]

        E.labels[count] = row.label
        E.dense[:, count] .= row.continuous

        for (indices, val) in zip(E.sparse, row.categorical)
            indices[count] = val
        end
    end
    return (dense = E.dense, sparse = E.sparse, labels = E.labels)
end
