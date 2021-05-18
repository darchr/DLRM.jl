module _Train

export bce_loss

# stdlib
using Statistics

# internal
using .._EmbeddingTables

# deps
using ChainRulesCore: ChainRulesCore
using Flux: Flux
using OneDNN: OneDNN
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
        Δ = Δ ./ length(y)

        a = @. Δ * ((one(Δ) - y) / (one(Δ) - ŷ + ϵ) - (y / (ŷ + ϵ)))
        b = @. Δ * (log(one(Δ) - ŷ + ϵ) - log(ŷ + ϵ))
        return (ChainRulesCore.NO_FIELDS, a, b)
    end

    return z, bce_pullback
end

struct LossWrapper{F}
    f::F
end

wrap_loss(loss_fn) = LossWrapper(loss_fn)

function (wrapper::LossWrapper)(model, labels, args...; kw...)
    loss = wrapper.f(model(args...; kw...), labels)
    # Zygote.ignore() do
    #     @show loss
    # end
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

function train!(loss, model, data, opt; cb = () -> ())
    params = DLRMParams(model)
    grads = DLRMGrads(model)
    cb = runall(cb)

    ProgressMeter.@showprogress 1 for d in data
        _grads = Zygote.gradient(loss, model, d...)
        gather!(grads, _grads[1])
        custom_update!(opt, params, grads)
        cb()
    end
end

function custom_update!(opt, params::DLRMParams, grads::DLRMGrads)
    # Weight Update
    param_weights = params.weights
    grads_weights = grads.weights
    for i in eachindex(param_weights, grads_weights)
        Flux.update!(opt, param_weights[i], grads_weights[i])
    end

    # Bias update
    param_bias = params.bias
    grads_bias = grads.bias
    for i in eachindex(param_bias, grads_bias)
        Flux.update!(opt, param_bias[i], grads_bias[i])
    end

    # Embedding Update
    param_embeddings = params.embeddings
    grads_embeddings = grads.embeddings
    Threads.@threads for i in eachindex(param_embeddings, grads_embeddings)
        Flux.update!(opt, param_embeddings[i], grads_embeddings[i])
    end
end


# include("backprop")

# mutable struct TestRun{F,D}
#     f::F
#     dataset::D
#     count::Int
#     every::Int
# end
#
# function (T::TestRun)()
#     T.count += 1
#     !iszero(mod(T.count, T.every)) && return nothing
#
#     # Run the test set.
#     total = 0
#     correct = 0
#     println("Testing")
#     @time for (dense, sparse, labels) in T.dataset
#         result = round.(Int, T.f(dense, sparse))
#         labels = clamp.(labels, 0, 1)
#
#         # Update the total
#         total += length(result)
#
#         #println(result .- labels)
#         # Update the number correct.
#         correct += count(x -> x[1] == x[2], zip(result, labels))
#     end
#     println("Iteration: $(T.count)")
#     println("Accuracy: $(correct / total)")
#     println()
#
#     if div(T.count, T.every) == 10
#         throw(Flux.Optimise.StopException())
#     end
# end
#
# # Routines for training DLRM.
# function top(;
#     debug = false,
#     # Percent of the dataset to reserve for testing.
#     testfraction = 0.125,
#     train_batchsize = 128,
#     test_batchsize = 16384,
#     #test_batchsize = 128,
#     test_frequency = 10000,
#     learning_rate = 0.1,
# )
#
#     #####
#     ##### Import dataset
#     #####
#
#     dataset = load(DAC(), joinpath(homedir(), "data", "dac", "train.bin"))
#
#     # Split into training and testing regions.
#     num_test_samples = ceil(Int, testfraction * length(dataset))
#
#     trainset = @views dataset[1:(end - num_test_samples - 1)]
#     testset = @views dataset[(end - num_test_samples):end]
#
#     train_loader = Extractor(trainset, train_batchsize)
#     test_loader = Extractor(testset, test_batchsize)
#
#     model = kaggle_dlrm()
#
#     loss =
#         (dense, sparse, labels) -> begin
#             # Clamp for stability
#             forward = model(dense, sparse)
#             ls = sum(Flux.binarycrossentropy.(forward, vec(labels))) / length(forward)
#             isnan(ls) && throw(error("NaN Loss"))
#             return ls
#         end
#
#     # The inner training loop
#     callbacks = [TestRun(model, test_loader, 0, test_frequency)]
#
#     # Warmup
#     opt = Flux.Descent(0.01)
#
#     count = 1
#     params = Flux.params(model)
#     for (dense, sparse, labels) in train_loader
#         grads = gradient(params) do
#             loss(dense, sparse, labels)
#         end
#         Flux.Optimise.update!(opt, params, grads)
#         count += 1
#         count == 1000 && break
#     end
#     opt = Flux.Descent(learning_rate)
#
#     if !debug
#         Flux.train!(loss, Flux.params(model), train_loader, opt; cb = callbacks)
#     end
#
#     return model, loss, train_loader, opt
# end
#

end # module
