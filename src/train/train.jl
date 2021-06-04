module _Train

export bce_loss

# stdlib
using Statistics

# internal
using .._EmbeddingTables
using .._Utils

# deps
using ChainRulesCore: ChainRulesCore
using Flux: Flux
using OneDNN: OneDNN
using Polyester: Polyester
using ProgressMeter: ProgressMeter
import UnPack: @unpack
using Zygote: Zygote

include("utils.jl")

#####
##### Loss Functions
#####

bce_loss(ŷ, y) = Flux.Losses.binarycrossentropy(ŷ, y; agg = mean)

function ChainRulesCore.rrule(::typeof(bce_loss), ŷ, y)
    z = bce_loss(ŷ, y)

    function bce_pullback(Δ)
        ϵ = eps(eltype(ŷ))

        # Adjust for the mean
        Δ = Δ / length(y)

        a = @. Δ * ((one(Δ) - y) / (one(Δ) - ŷ + ϵ) - (y / (ŷ + ϵ)))
        b = @. Δ * (log(one(Δ) - ŷ + ϵ) - log(ŷ + ϵ))
        return (ChainRulesCore.NoTangent(), a, b)
    end

    return z, bce_pullback
end

struct LossWrapper{F,NT}
    f::F
    kw::NT
end

wrap_loss(loss_fn; kw...) = LossWrapper(loss_fn, (;kw...,))

function (wrapper::LossWrapper)(model, labels, args...)
    loss = wrapper.f(model(args...; wrapper.kw...), labels)
    return loss
end

#####
##### Parameter and Gradient Collection
#####

# You might ask: "Why not use the normal Flux approach of parameter and gradient
# accumulation?" Surely that would be more flexible.
#
# Well, you're probably right.
# The one thing that a more explicit representation gives us is more flexibility to
# control scheduling of updates, which may or may not end up being helpful.
#
# The other reason is that this approach lets us not have to use the Flux style
# "accum_param" semi-nightmare.

struct DLRMParams{W,B,E}
    weights::Vector{W}
    bias::Vector{B}
    embeddings::Vector{E}
end

function DLRMParams(model)
    W = typeof(first(model.bottom_mlp).weights)
    B = typeof(first(model.bottom_mlp).bias)
    E = eltype(model.embeddings)
    params = DLRMParams(W[], B[], E[])
    gather!(params, model)
    return params
end

function Base.empty!(params::DLRMParams)
    @unpack weights, bias, embeddings = params
    empty!(weights)
    empty!(bias)
    empty!(embeddings)
    return nothing
end

struct DLRMGrads{T,U}
    weights::Vector{T}
    bias::Vector{U}
    embeddings::Vector{SparseEmbeddingUpdate}
end

function DLRMGrads(model #=::DLRMModel=#)
    T = typeof(first(model.bottom_mlp).weights)
    U = typeof(first(model.bottom_mlp).bias)
    return DLRMGrads(T[], U[], SparseEmbeddingUpdate[])
end

function Base.empty!(grads::DLRMGrads)
    @unpack weights, bias, embeddings = grads
    empty!(weights)
    empty!(bias)
    empty!(embeddings)
    return nothing
end

function gather!(x, nt)
    @assert in(:bottom_mlp, propertynames(nt))
    @assert in(:embeddings, propertynames(nt))
    @assert in(:top_mlp, propertynames(nt))

    empty!(x)
    gather_mlp!(x, nt.bottom_mlp)
    gather_embeddings!(x, nt.embeddings)
    gather_mlp!(x, nt.top_mlp)
    return nothing
end

function gather_mlp!(x, chain)
    layers = chain.layers
    for dense in layers
        push!(x.weights, dense.weights)
        push!(x.bias, dense.bias)
    end
end

gather_embeddings!(x, vec) = append!(x.embeddings, vec)

#####
##### Train Loop
#####

const TIMES = UInt64[]

function train!(loss, model, data, opt; cb = () -> ())
    params = DLRMParams(model)
    grads = DLRMGrads(model)
    cb = runall(cb)

    count = 1
    ProgressMeter.@showprogress 1 for d in data
        _grads = Zygote.gradient(loss, model, d...)
        gather!(grads, _grads[1])
        custom_update!(opt, params, grads)
        count += 1
        count == 20 && break
        cb()
    end
end

function custom_update!(opt, params::DLRMParams, grads::DLRMGrads)
    # Weight Update
    param_weights = params.weights
    grads_weights = grads.weights
    #for i in eachindex(param_weights, grads_weights)
    static_thread(ThreadPool(Base.OneTo(length(param_weights))), eachindex(param_weights, grads_weights)) do i
        Flux.update!(opt, param_weights[i], grads_weights[i])
    end

    # Bias update
    param_bias = params.bias
    grads_bias = grads.bias
    for i in eachindex(param_bias, grads_bias)
    #static_thread(ThreadPool(Base.OneTo(length(param_bias))), eachindex(param_bias, grads_bias)) do i
        Flux.update!(opt, param_bias[i], grads_bias[i])
    end

    # Embedding Update
    param_embeddings = params.embeddings
    grads_embeddings = grads.embeddings
    start = time_ns()
    #Threads.@threads for i in eachindex(param_embeddings, grads_embeddings)
    Polyester.@batch per=thread for i in eachindex(param_embeddings, grads_embeddings)
        Flux.update!(opt, param_embeddings[i], grads_embeddings[i])
    end
    push!(TIMES, time_ns() - start)
end

end # module
