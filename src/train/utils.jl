#####
##### Callback Utilities
#####

call(f, xs...) = f(xs...)
forcecall(f, xs...) = call(f, xs...)

runall(f) = f
runall(fs::AbstractVector) = (f = call) -> foreach(f, fs)

mutable struct Every{F}
    callback::F
    count::Int
    callat::Int
end

function Every(f::F, count) where {F}
    return Every(f, 0, count)
end

function (f::Every)()
    f.count += 1
    if f.count >= f.callat
        f.callback()
        f.count = 0
    end
    return nothing
end
forcecall(f::Every) = f.callback()

function test(model, data; record = Float32[], strategy)
    s = zero(Float32)
    correct = 0
    total = 0
    ProgressMeter.@showprogress 1 "Perfoming Test" for d in data
        labels, dense, sparse = d
        _predictions = CachedArrays.readable(model(dense, sparse; strategy))
        predictions = round.(convert.(Float32, _predictions))
        correct += count(predictions .== labels)
        total += length(predictions)
    end
    meanval = correct / total
    @show (correct, total, meanval)
    push!(record, meanval)
    return nothing
end
