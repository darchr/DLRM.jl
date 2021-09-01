module _Train

export bce_loss, wrap_loss, train!

# stdlib
using Statistics

# internal
using EmbeddingTables
using .._Model
using .._Utils

# local deps
using CachedArrays: CachedArrays

# deps
using ChainRulesCore: ChainRulesCore
using Flux: Flux
using LoopVectorization: LoopVectorization
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

function ChainRulesCore.rrule(
    ::typeof(bce_loss), ŷ::AbstractVector{T}, y::AbstractVector{T}
) where {T}
    z = bce_loss(ŷ, y)
    ly = length(y)

    function bce_pullback(_Δ)
        # Adjust for the mean
        Δ = _Δ / ly
        ϵ = eps(T)

        # Because why not!
        a = LoopVectorization.@turbo @. Δ *
                                        ((one(Δ) - y) / (one(Δ) - ŷ + ϵ) - (y / (ŷ + ϵ)))
        b = LoopVectorization.@turbo @. Δ * (log(one(Δ) - ŷ + ϵ) - log(ŷ + ϵ))
        return (ChainRulesCore.NoTangent(), a, b)
    end

    return z, bce_pullback
end

struct LossWrapper{F,NT}
    f::F
    kw::NT
end

function __cb(loss::LossWrapper)
    kw = loss.kw
    if hasproperty(kw, :cb)
        return kw.cb
    end
    return donothing
end

wrap_loss(loss_fn; kw...) = LossWrapper(loss_fn, (; kw...))

function (wrapper::LossWrapper)(model, labels, args...)
    out = model(args...; wrapper.kw...)
    loss = _Model.callback(__cb(wrapper), :loss, wrapper.f, out, labels)
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
    # The weights and weight gradients end up with different layouts.
    # Here, we cache the indices from the grads to the weights to make the updating
    # process faster.
    weight_index_translations::Vector{Tuple{Vector{Int},Vector{Int}}}
    bias::Vector{B}

    # Embedding tables and pre-allocated dictionaries for doing the pre-compression
    # before embedding table update.
    embeddings::Vector{E}
    crunch_translations::Vector{Dict{Int,Int}}
end

function DLRMParams(model)
    W = typeof(first(model.bottom_mlp).weights)
    WIT = Tuple{Vector{Int},Vector{Int}}
    B = typeof(first(model.bottom_mlp).bias)
    E = eltype(model.embeddings)
    params = DLRMParams(W[], WIT[], B[], E[], Dict{Int,Int}[])
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

function DLRMGrads(model) #=::DLRMModel=#
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

function train!(loss, model, data, opt; cb = () -> (), maxiters = nothing, update_batchsize)
    # Run once to make sure data formats are initialized.
    _ = Zygote.gradient(loss, model, first(data)...)

    params = DLRMParams(model)
    grads = DLRMGrads(model)
    updater = _Model.BatchUpdater()

    cb = runall(cb)

    count = 0
    telemetry = __cb(loss)
    #ProgressMeter.@showprogress 1 for d in data
    println("Starting Training Loop")
    for d in data
        telemetry(:start)
        _grads = Zygote.gradient(loss, model, d...)
        telemetry(:grads_done)
        gather!(grads, _grads[1])
        custom_update!(opt, params, grads, updater, update_batchsize)
        telemetry(:update_done)

        count += 1
        count == maxiters && break
        cb()
    end
end

function custom_update!(
    opt,
    params::DLRMParams,
    grads::DLRMGrads,
    updater::_Model.BatchUpdater,
    update_batchsize::Integer,
)
    # Weight Update
    param_weights = params.weights
    grads_weights = grads.weights

    param_embeddings = params.embeddings
    grads_embeddings = grads.embeddings

    @assert length(param_weights) == length(grads_weights)
    @assert length(param_embeddings) == length(grads_embeddings)

    # Check if we need to populate the weight index translation tables
    if isempty(params.weight_index_translations)
        populate_translations!(params, grads)
    end

    crunch_translations = params.crunch_translations
    if isempty(crunch_translations)
        for _ in eachindex(param_embeddings)
            push!(crunch_translations, eltype(crunch_translations)())
        end
    end

    # Merge embedding table updates with weight updates.
    len = length(param_weights)
    index_translation = params.weight_index_translations

    # TODO: Hack Alert!!
    # Find the cache manager and disable movement.
    manager = CachedArrays.manager(parent(grads_weights[1]))
    manager.policy.movement_enabled = false

    Polyester.@batch per = core for i in Base.OneTo(len)
        Flux.update!(
            opt,
            param_weights[i],
            index_translation[i][1],
            grads_weights[i],
            index_translation[i][2],
        )
    end

    # Embedding Table Update
    feeder = EmbeddingTables.UpdatePartitioner.(grads_embeddings, update_batchsize)
    @time _Model.process!(updater, opt, param_embeddings, feeder, Threads.nthreads())
    manager.policy.movement_enabled = true

    # Bias update
    # Single thread since this is super quick anyways.
    param_bias = params.bias
    grads_bias = grads.bias
    for i in eachindex(param_bias, grads_bias)
        Flux.update!(opt, param_bias[i], grads_bias[i])
    end
end

# function train!(loss, model, data, opt; cb = () -> (), maxiters = nothing)
#     # Run once to make sure data formats are initialized.
#     _ = Zygote.gradient(loss, model, first(data)...)
#
#     params = DLRMParams(model)
#     grads = DLRMGrads(model)
#     #updater = BatchUpdater()
#
#     cb = runall(cb)
#
#     count = 0
#     telemetry = __cb(loss)
#     #ProgressMeter.@showprogress 1 for d in data
#     println("Starting Training Loop")
#     for d in data
#         telemetry(:start)
#         _grads = Zygote.gradient(loss, model, d...)
#         telemetry(:grads_done)
#         gather!(grads, _grads[1])
#         #custom_update!(opt, params, grads, updater, update_batchsize)
#         custom_update!(opt, params, grads)
#         telemetry(:update_done)
#
#         count += 1
#         count == maxiters && break
#         cb()
#     end
# end

# function custom_update!(opt, params::DLRMParams, grads::DLRMGrads)
#     # Weight Update
#     param_weights = params.weights
#     grads_weights = grads.weights
#
#     param_embeddings = params.embeddings
#     grads_embeddings = grads.embeddings
#
#     @assert length(param_weights) == length(grads_weights)
#     @assert length(param_embeddings) == length(grads_embeddings)
#
#     # Check if we need to populate the weight index translation tables
#     if isempty(params.weight_index_translations)
#         populate_translations!(params, grads)
#     end
#
#     crunch_translations = params.crunch_translations
#     if isempty(crunch_translations)
#         for _ in eachindex(param_embeddings)
#             push!(crunch_translations, eltype(crunch_translations)())
#         end
#     end
#
#     # Merge embedding table updates with weight updates.
#     m = length(param_weights)
#     len = length(param_weights) + length(param_embeddings)
#     index_translation = params.weight_index_translations
#
#     Polyester.@batch per=core for i in Base.OneTo(len)
#         if i <= m
#             Flux.update!(
#                 opt,
#                 param_weights[i],
#                 index_translation[i][1],
#                 grads_weights[i],
#                 index_translation[i][2],
#             )
#         else
#             j = i - m
#             Flux.update!(
#                 opt, param_embeddings[j], grads_embeddings[j], crunch_translations[j]
#             )
#         end
#     end
#
#     # Bias update
#     # Single thread since this is super quick anyways.
#     param_bias = params.bias
#     grads_bias = grads.bias
#     for i in eachindex(param_bias, grads_bias)
#         Flux.update!(opt, param_bias[i], grads_bias[i])
#     end
# end

function populate_translations!(params, grads)
    empty!(params.weight_index_translations)
    for (param, grad) in zip(params.weights, grads.weights)
        @assert isa(param, OneDNN.Memory)
        @assert isa(grad, OneDNN.Memory)
        param_indices = OneDNN.generate_linear_indices(param)
        grad_indices = OneDNN.generate_linear_indices(grad)
        push!(params.weight_index_translations, (param_indices, grad_indices))
    end
    return nothing
end

end # module
