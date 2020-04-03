#####
##### Utilities
#####

# Investigate properties of datasets.
function histogram(f, X)
    d = Dict{UInt32,Int64}()
    for x in X
        k = f(x)
        v = get!(d, k, zero(valtype(d)))
        d[k] = v + one(valtype(d))
    end
    return d
end

function cdf(d)
    A = Vector{valtype(d)}(undef, length(d))
    s = zero(valtype(d))
    for i in 1:length(d)
        s += get(d, convert(keytype(d), i), zero(valtype(d)))
        A[i] = s
    end
    return A
end

function increment!(d::AbstractDict{K,V}, k, v = one(V)) where {K,V}
    old = get(d, k, zero(V))
    d[k] = old + v
    return nothing
end

function reusedistance(X::AbstractArray)
    indexof = Dict{eltype(X),Int}()
    histogram = DataStructures.SortedDict{Int,Int}()
    for (i,x) in enumerate(X)
        # Get the last recorded depth of this item.
        index = get(indexof, x, i)

        # The different between the current index and the old index is the reuse distance.
        increment!(histogram, i - index)
        indexof[x] = i
    end
    return histogram
end

#####
##### Stats on DAC
#####

function reusedistance(dac::DAC, records::Vector{DACRecord})
    # Create a reuse-distance discionary for each categorical features.
    pmeter = Progress(num_categorical_features(dac), "Computing Reuse Distances")
    distances = map(1:num_categorical_features(dac)) do i
        A = MappedArrays.mappedarray(x -> x.categorical[i], records)
        next!(pmeter)
        return reusedistance(A)
    end
    return distances
end

# Generate `xy` pairs for plotting a CDF.
function xycdf(d::DataStructures.SortedDict)
    x = zeros(Int, 2 * length(d))
    y = zeros(Int, 2 * length(d))

    # Drop the first element since it will be appended onto
    # the end of the arrays
    i = 2
    y[1] = 0
    x[1] = 0
    for (k, v) in Iterators.drop(d, 1)
        x[i] = k
        x[i+1] = k

        s = y[i-1]
        y[i] = s
        y[i + 1] = s + v

        i += 2
    end

    # Append last values
    x[end] = x[end-1]
    y[end] = y[end-1] + d[0]
    return x, y
end

