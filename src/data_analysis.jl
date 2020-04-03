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

mutable struct HistogramTracker{T}
    # Items for just tracking the number of times each category is seen.
    histogram::Dict{T,Int}

    # Items for tracking reuse distance
    reusehistogram::DataStructures.SortedDict{Int,Int}
    history::Dict{T,Set{T}}
end

function HistogramTracker{T}() where {T}
    return HistogramTracker{T}(
        Dict{T,Int}(),
        DataStructures.SortedDict{Int,Int}(),
        Dict{T,Set{T}}()
    )
end


function update!(H::HistogramTracker, x::T) where {T}
    # Increment the basic histogram.
    increment!(H.histogram, x)

    return nothing
end

function statistics(records::Vector{DACRecord})
    # Construct a `HistogramTracker` for each categorical features.
    trackers = [HistogramTracker{UInt32}() for _ in 1:num_categorical_features(DAC())]
    @showprogress 1 "Calculating Reuse Distances" for record in records
        for (tracker, feature) in zip(trackers, record.categorical)
            update!(tracker, feature)
        end
    end
    return trackers
end

# Compute cache sizes for reuse distance
#
# Percents should be a *SORTED* iterable of numbers between 0 and 1.
function cachesize(histogram, percents)
    total = sum(values(histogram))
    rolling_sum = zero(valtype(histogram))

    sizes = Union{keytype(histogram),Missing}[]

    # Drop the first item since it is the sentinel value and counts
    # compulsory misses.
    index = 1
    for (k,v) in Iterators.drop(histogram, 1)
        rolling_sum += v
        if rolling_sum / total > percents[index]
            push!(sizes, k)
            index += 1
        end
        index > length(percents) && break
    end

    while length(sizes) < length(percents)
        push!(sizes, missing)
    end

    return sizes
end

function reusetable(
        stats::Vector{<:HistogramTracker};
        percents = [0.5, 0.75, 0.8, 0.9, 0.99, 0.999]
    )

    # Count the number of unique items per category
    nunique = length.(getproperty.(stats, :histogram))
    sizes = cachesize.(getproperty.(stats, :reusehistogram), Ref(percents))
    table = transpose(reduce(hcat, sizes))
    table = hcat(table, nunique)

    header = vcat(string.(percents), ["Num Unique"])

    missing_highlighter = PrettyTables.Highlighter(
        (data, i, j) -> ismissing(data[i,j]),
        PrettyTables.crayon"red bold",
    )
    # Print out the table
    PrettyTables.pretty_table(table, header; highlighters = (missing_highlighter,))

    #return table
end

#####
##### Old Stuff
#####

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

