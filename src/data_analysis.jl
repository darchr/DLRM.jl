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

function reusedistance(X::AbstractVector{T}) where {T}
    indexof = Dict{T,Int}()
    histogram = Dict{Int,Int}()
    for (i,x) in enumerate(X)
        # Get the last recorded depth of this item.
        index = get(indexof, x, i)

        # The different between the current index and the old index is the reuse distance.
        increment!(histogram, i - index)
        indexof[x] = i
    end
    return histogram
end
