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
struct DLRMParams{W,M,B,C,E}
    weights::Vector{W}
    mantissas::Vector{M}

    # The weights and weight gradients end up with different layouts.
    # Here, we cache the indices from the grads to the weights to make the updating
    # process faster.
    #weight_index_translations::Vector{Tuple{Vector{Int},Vector{Int}}}
    weight_index_translations::Vector{Tuple{Ptr{Float32},Ptr{Float32}}}
    bias::Vector{B}
    bias_mantissas::Vector{C}

    # Embedding tables and pre-allocated dictionaries for doing the pre-compression
    # before embedding table update.
    embeddings::Vector{E}
    indexers::Vector{EmbeddingTables.Indexer}
end

mantissa_trick(::DLRMParams) = true
mantissa_trick(::DLRMParams{<:Any,Nothing}) = false

function DLRMParams(model; mantissa_trick = false)
    weight_example = first(model.bottom_mlp).weights
    bias_example = first(model.bottom_mlp).bias

    W = typeof(weight_example)
    B = typeof(bias_example)
    if mantissa_trick
        if eltype(weight_example) != OneDNN.BFloat16
            msg = """
            You requested the "mantissa trick" be performed for an eltype that is not
            BFloat16. Since this doesn't really make sense, here's an error for you :)
            """
            throw(ArgumentError(msg))
        end
        println("Performing the Mantissa Trick")
        _w = typeof(parent(weight_example))
        _b = typeof(parent(bias_example))
        M = Base.promote_op(similar, _w, Type{UInt16}, Int)
        C = Base.promote_op(similar, _b, Type{UInt16}, Int)
    else
        M = Nothing
        C = Nothing
    end

    #WIT = Tuple{Vector{Int},Vector{Int}}
    WIT = Tuple{Ptr{Float32},Ptr{Float32}}
    B = typeof(first(model.bottom_mlp).bias)
    E = eltype(model.embeddings)
    params = DLRMParams(W[], M[], WIT[], B[], C[], E[], EmbeddingTables.Indexer[])
    gather!(params, model)

    # Construct the mantissas
    if mantissa_trick
        mantissas = params.mantissas
        for weight in params.weights
            mantissa = similar(parent(weight), UInt16, sizeof(weight))
            mantissa .= zero(eltype(mantissa))
            push!(mantissas, mantissa)
        end

        bias_mantissas = params.bias_mantissas
        for bias in params.bias
            mantissa = similar(parent(bias), UInt16, sizeof(bias))
            mantissa .= zero(eltype(mantissa))
            push!(bias_mantissas, mantissa)
        end
    end
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

function _tof32(::Type{OneDNN.Memory{T,N,A}}) where {T<:AbstractFloat,N,A}
    return OneDNN.Memory{Float32,N,_tof32(A)}
end

_tof32(::Type{Array{T,N}}) where {T<:AbstractFloat,N} = Array{Float32,N}
function _tof32(::Type{CachedArrays.CachedArray{T,N,S,M}}) where {T<:AbstractFloat,N,S,M}
    return CachedArrays.CachedArray{Float32,N,S,M}
end

function DLRMGrads(model) #=::DLRMModel=#
    T = _tof32(typeof(first(model.bottom_mlp).weights))
    U = _tof32(typeof(first(model.bottom_mlp).bias))
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
    mantissa_trick = false,
    embedding_threads = 12,
)
    # Run once to make sure data formats are initialized.
    _ = Zygote.gradient(loss, model, first(data)...)

    params = DLRMParams(model; mantissa_trick = mantissa_trick)
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
        println()

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
    opt, params::DLRMParams, grads::DLRMGrads, telemetry = donothing; embedding_threads = 12
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

    # Merge embedding table updates with weight updates.
    #m = length(param_weights)
    #index_translation = params.weight_index_translations
    translations = params.weight_index_translations
    eta = opt.eta
    Polyester.@batch (per = core) for i in eachindex(translations)
        src, dst = translations[i]
        update = unsafe_load(dst) - eta * unsafe_load(src)
        unsafe_store!(dst, update)
    end
    # Polyester.@batch (per = core) for i in Base.ONeTo(m)
    #     # # Decide if we're going to do the mantissa trick or not.
    #     # # Remember, the mantissa trick glues the lower 16 bits of the mantissa to bf16
    #     # # weights, allowing for higher precision intermediate values during training.
    #     # if mantissa_trick(params)
    #     #     weights = OneDNN.Mirrored(param_weights[i], param_mantissas[i])
    #     # else
    #     #     weights = param_weights[i]
    #     # end
    #     weights = param_weights[i]
    #     Flux.update!(
    #         opt, weights, index_translation[i][1], grads_weights[i], index_translation[i][2]
    #     )
    # end

    # Bias update
    # Single thread since this is super quick anyways.
    param_bias = params.bias
    param_bias_mantissas = params.bias_mantissas
    grads_bias = grads.bias
    # if mantissa_trick(params)
    #     for i in eachindex(param_bias, grads_bias)
    #         _bias = OneDNN.Mirrored(param_bias[i], param_bias_mantissas[i])
    #         Flux.update!(opt, _bias, grads_bias[i])
    #     end
    # else
    #     for i in eachindex(param_bias, grads_bias)
    #         Flux.update!(opt, param_bias[i], grads_bias[i])
    #     end
    # end

    for i in eachindex(param_bias, grads_bias)
        Flux.update!(opt, param_bias[i], grads_bias[i])
    end

    telemetry(:weight_update_done)

    # Allocate dictionaries once, to avoid allocating them every time we need to do a
    # gradient update.
    indexers = params.indexers
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

function populate_translations!(params, grads)
    translations = params.weight_index_translations
    empty!(translations)
    for (param, grad) in zip(params.weights, grads.weights)
        @assert isa(param, OneDNN.Memory)
        @assert isa(grad, OneDNN.Memory)
        param_indices = OneDNN.generate_linear_indices(param)
        grad_indices = OneDNN.generate_linear_indices(grad)

        grad_parent = parent(grad)
        param_parent = parent(param)
        for i in eachindex(param_indices, grad_indices)
            src_ptr = pointer(grad_parent, grad_indices[i])
            dst_ptr = pointer(param_parent, param_indices[i])
            push!(translations, (src_ptr, dst_ptr))
        end
        #push!(params.weight_index_translations, (param_indices, grad_indices))
    end
    sort!(translations; by = last)
    return nothing
end

end # module
