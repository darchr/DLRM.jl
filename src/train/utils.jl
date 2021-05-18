#####
##### Callback Utilities
#####

call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

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

function test(loss, model, data; record = Float32[], kw...)
    s = zero(Float32)
    count = 0
    ProgressMeter.@showprogress 1 "Perfoming Test" for d in data
        v = loss(model, d...; kw...)
        s += v
        count += 1
    end
    meanval = s / count
    @show meanval
    push!(record, meanval)
    return nothing
end
