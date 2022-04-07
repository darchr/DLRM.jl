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

#@inline maxlog(x) = max(log(x), -100)
function bce_loss(x, y)
    # Binary cross entropy loss:
    # L = -y * ln(x) + (y-1) * ln(1-x)
    s = zero(eltype(y))
    LoopVectorization.@turbo for i in eachindex(y, x)
        s += -y[i] * max(log(x[i]), -100) + (y[i] - 1) * max(log(1 - x[i]), -100)
    end
    return s / length(x)
end

#bce_loss(x::AbstractVector{OneDNN.BFloat16}, y; kw...) = bce_loss(convert.(Float32, x), y)

function ChainRulesCore.rrule(
    ::typeof(bce_loss), x::AbstractVector{T}, y::AbstractVector{T}
) where {T}
    z = bce_loss(x, y)
    ly = length(y)

    function bce_pullback(_Δ)
        # Adjust for the mean
        Δ = _Δ / ly
        ϵ = eps(T)

        # Because why not!
        dx = similar(x)
        dy = similar(y)

        LoopVectorization.@turbo for i in eachindex(dx, dy)
            c = one(Δ) - x[i] + ϵ
            d = x[i] + ϵ

            dx[i] = Δ * ((one(Δ) - y[i]) / c - (y[i] / d))
            dy[i] = Δ * (log(c) - log(d))
        end
        return (ChainRulesCore.NoTangent(), dx, dy)
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
    bias::Vector{B}

    # Embedding tables and pre-allocated dictionaries for doing the pre-compression
    # before embedding table update.
    embeddings::Vector{E}
    indexers::Vector{EmbeddingTables.Indexer}
end

mantissa_trick(::DLRMParams{<:Any,Nothing}) = false

function DLRMParams(model)
    weight_example = first(model.bottom_mlp).weights
    bias_example = first(model.bottom_mlp).bias

    W = typeof(weight_example)
    B = typeof(bias_example)

    B = typeof(first(model.bottom_mlp).bias)
    E = eltype(model.embeddings)
    params = DLRMParams(W[], B[], E[], EmbeddingTables.Indexer[])
    gather!(params, model)
    return params
end

function Base.empty!(params::DLRMParams)
    (; weights, bias, embeddings) = params
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
    (; weights, bias, embeddings) = grads
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
        if hasproperty(dense, :weights) && hasproperty(dense, :bias)
            push!(x.weights, dense.weights)
            push!(x.bias, dense.bias)
        end
    end
end

gather_embeddings!(x, vec) = append!(x.embeddings, vec)

#####
##### Train Loop
#####

function train!(
    loss,
    model,
    data,
    opt;
    cb = () -> (),
    maxiters = nothing,
    embedding_threads = 12,
)
    # Run once to make sure data formats are initialized.
    _ = Zygote.gradient(loss, model, first(data)...)

    params = DLRMParams(model)
    grads = DLRMGrads(model)
    cb = runall(cb)

    count = 0
    telemetry = __cb(loss)
    println("Starting Training Loop")

    iterlen = (maxiters === nothing) ? length(data) : min(length(data), maxiters)
    meter = ProgressMeter.Progress(iterlen, 1)

    losses = Float32[]
    iteration_times = UInt64[]

    for d in data
        telemetry(:start)
        start = time_ns()

        # Run forward and backward pass.
        l, pullback = Zygote._pullback(loss, model, d...)
        _grads = pullback(Zygote.sensitivity(l))

        telemetry(:grads_done)

        # Weight Update
        gather!(grads, _grads[2])
        custom_update!(opt, params, grads, telemetry; embedding_threads)

        # Callbacks
        push!(iteration_times, time_ns() - start)
        push!(losses, l)
        telemetry(:update_done)
        ProgressMeter.next!(meter)
        count += 1
        count == maxiters && break
        cb()
    end
    cb()
    return (; iteration_times, losses)
end

function custom_update!(
    opt, params::DLRMParams, grads::DLRMGrads, telemetry = donothing; embedding_threads = 12,
)
    # Weight Update
    param_weights = params.weights
    grads_weights = grads.weights

    param_embeddings = params.embeddings
    grads_embeddings = grads.embeddings
    @assert length(param_weights) == length(grads_weights)
    @assert length(param_embeddings) == length(grads_embeddings)

    # Weight update - thread across all dense layers
    len = length(param_weights)
    index = Threads.Atomic{Int}(1)
    Polyester.@batch (per = thread) for _ in Base.OneTo(Threads.nthreads())
        while true
            local_index = Threads.atomic_add!(index, 1)
            local_index > len && break
            Flux.update!(opt, param_weights[local_index], grads_weights[local_index])
        end
    end

    # Bias update
    # Single thread since this is super quick anyways.
    param_bias = params.bias
    grads_bias = grads.bias
    for i in eachindex(param_bias, grads_bias)
        Flux.update!(opt, param_bias[i], grads_bias[i])
    end
    telemetry(:weight_update_done)

    # Allocate dictionaries once, to avoid allocating them every time we need to do a
    # gradient update.
    (; indexers) = params
    if isempty(indexers)
        for _ in eachindex(param_embeddings)
            push!(indexers, EmbeddingTables.Indexer())
        end
    end

    EmbeddingTables.update!(
        opt,
        param_embeddings,
        grads_embeddings,
        indexers;
        num_splits = 8,
        nthreads = embedding_threads,
    )

    return telemetry(:embedding_update_done)
end

# function populate_translations!(params, grads)
#     ptr_map = PtrMap{Float32}()
#     for (param, grad) in zip(params.weights, grads.weights)
#         @assert isa(param, OneDNN.Memory)
#         @assert isa(grad, OneDNN.Memory)
#         param_indices = OneDNN.generate_linear_indices(param)
#         grad_indices = OneDNN.generate_linear_indices(grad)
#
#         grad_parent = parent(grad)
#         param_parent = parent(param)
#         for i in eachindex(param_indices, grad_indices)
#             src_ptr = pointer(grad_parent, grad_indices[i])
#             dst_ptr = pointer(param_parent, param_indices[i])
#             push!(ptr_map, src_ptr => dst_ptr)
#         end
#     end
#     sort!(ptr_map; by = last)
#     simdcompress!(params.weight_index_translations, ptr_map)
#     return nothing
# end

end # module
