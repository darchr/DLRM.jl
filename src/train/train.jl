module _Train

export bce_loss

# stdlib
using Statistics

# deps
import ChainRulesCore
import Flux
import Zygote

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

function wrap_loss(model, loss_fn)
    function dlrm_loss(labels, dense, sparse)
        return loss_fn(model(dense, sparse), labels)
    end
    return dlrm_loss
end

#####
##### Train Loop
#####

function train!(loss, params, data, opt)
    for d in data
        grads = Zygote.gradient(params) do
            loss(d...)
        end
        custom_update!(opt, params, grads)
    end
end

function custom_update!(opt, params::Zygote.Params, grads::Zygote.Grads)
    for param in params
        grad = grads[param]
        grad == nothing && continue
        Flux.update!(opt, param, grad)
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
