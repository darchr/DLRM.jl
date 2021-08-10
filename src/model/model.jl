module _Model

export DLRMModel, dlrm

# internal deps
using EmbeddingTables
using .._Utils

# local deps
using CachedArrays: CachedArrays

# deps
import ChainRulesCore
import DataStructures
import Flux
import LoopVectorization
import ManualMemory
import NNlib
import OneDNN
import Polyester
import ProgressMeter
import UnPack: @unpack
import Zygote

# DocStringExtensions
using DocStringExtensions

const POST_INTERACTION_PAD_TO_MUL = 16
cdiv(x, y) = 1 + div(x - 1, y)
up_to_mul_of(x, y) = y * cdiv(x, y)

include("interact.jl")
include("embedding_update.jl")

# Strategy:
# We can take advantage of Julia's dynamic typing to build the intermediate parts of the
# network using an untyped array.
#
# Then, we dump everying into a `Flux.Chain` to become concretely typed.
function create_mlp(sizes, sigmoid_index; weight_init)
    layers = Any[]
    for i = 1:(length(sizes) - 1)
        in = sizes[i]
        out = sizes[i + 1]

        # Choose between sigmoid or relu activation function.
        if i == sigmoid_index - 1
            activation = Flux.sigmoid
        else
            activation = Flux.relu
        end
        layer = OneDNN.Dense(Flux.Dense(in, out, activation; init = weight_init))
        push!(layers, layer)
    end
    return Flux.Chain(layers...)
end

# Initialization
function multithread_init(f, A)
    Threads.@threads for i in eachindex(A)
        A[i] = f(A)
    end
end

struct OneInit end
(f::OneInit)(A) = one(eltype(A))

struct ZeroInit end
(f::ZeroInit)(A) = zero(eltype(A))

struct GlorotNormal end
(f::GlorotNormal)(A) = randn() * sqrt(2.0f0 / sum(Flux.nfan(size(A)...)))

# Create embeddings.
# Allow for a constructor to be passed.
function create_embeddings(ncols::Integer, rowcounts::AbstractVector{<:Integer}; kw...)
    return create_embeddings(SimpleEmbedding, ncols, rowcounts; kw...)
end

function create_embeddings(finish, ncols, rowcounts, initialize)
    progress_meter = ProgressMeter.Progress(
        length(rowcounts), 1, "Building Embedding Tables ..."
    )
    embeddings = map(1:length(rowcounts)) do i
        nrows = rowcounts[i]
        # Construct and initialize the underlying data.
        # Then construct the embedding table
        data = initialize(ncols, nrows)
        table = finish(data)

        ProgressMeter.next!(progress_meter)
        return table
    end
    ProgressMeter.finish!(progress_meter)
    return embeddings
end

#####
##### DLRM Model
#####

struct DLRMModel{B,E,I,T}
    bottom_mlp::B
    embeddings::Vector{E}
    interaction::I
    top_mlp::T
end

Flux.@functor DLRMModel (bottom_mlp, embeddings, top_mlp)

# Setup a scheme for recording callbacks.
back(x) = Symbol("$(x)_back")

function callback(cb::C, sym::Symbol, f::F, x...) where {C,F}
    y = f(x...)
    cb(sym)
    return y
end

Zygote.@adjoint function callback(cb::C, sym::Symbol, f::F, x...) where {C,F}
    y, pullback = Zygote._pullback(__context__, f, x...)
    cb(sym)
    function callback_pullback(Δ...)
        # Perform pullback computation the call the callback
        dx = pullback(Δ...)
        cb(back(sym))
        return (nothing, nothing, dx...)
    end
    return y, callback_pullback
end

# Model Constructor
function (D::DLRMModel)(
    dense,
    sparse;
    strategy = DefaultStrategy(),
    # Arbitrary callback.
    # Symbol will be passed describing action.
    cb = donothing,
)
    # Wrap everything in a `callback` - default will compile away.
    y = callback(cb, :lookup, maplookup, strategy, D.embeddings, sparse)
    x = callback(cb, :bottom_mlp, D.bottom_mlp, dense)
    z = callback(cb, :interaction, D.interaction, x, y)
    out = callback(cb, :top_mlp, D.top_mlp, z)
    return vec(OneDNN.materialize(out))
end

# # inplace version of `Flux.glorot_normal`
# glorot_normal!(x) = randn!(x) .* sqrt(2.0f0 / sum(Flux.nfan(size(x)...)))
# zeros!(x) = (x .= zero(eltype(x)))

# Test entry point for now.
function dlrm(
    bottom_mlp_sizes,
    top_mlp_sizes,
    sparse_feature_size,
    embedding_sizes;
    # Default to the `dot` interaction method.
    interaction = dot_interaction,
    embedding_constructor = SimpleEmbedding,

    # Modify internal arrays
    constructor = (x...) -> Array{Float32}(undef, x...),
    weight_init_kernel = GlorotNormal(),
)
    # Use the passed kernels to construct the actual functions that will initialize the
    # weights of the model
    weight_init = function (x...)
        data = constructor(x...)
        multithread_init(weight_init_kernel, data)
        return data
    end

    # Create the bottom MLP
    bottom_mlp = create_mlp(bottom_mlp_sizes, 0; weight_init = weight_init)
    embeddings = create_embeddings(
        embedding_constructor, sparse_feature_size, embedding_sizes, weight_init
    )

    # Compute the size of the first layer for the top mlp.
    num_features = length(embedding_sizes)
    bottom_out_size = last(bottom_mlp_sizes)

    # Do some math with sizes.
    # This is gross ...
    @assert iszero(mod(sparse_feature_size * num_features, bottom_out_size))
    pre_triangle_size = div(sparse_feature_size * num_features, bottom_out_size) + 1

    # top_layer_input_size = up_to_mul_of(
    #     div(pre_triangle_size^2 - pre_triangle_size, 2) + bottom_out_size,
    #     POST_INTERACTION_PAD_TO_MUL,
    # )
    top_layer_input_size = div(pre_triangle_size^2 - pre_triangle_size, 2) + bottom_out_size
    top_mlp_sizes = vcat([top_layer_input_size], top_mlp_sizes)
    top_mlp = create_mlp(top_mlp_sizes, lastindex(top_mlp_sizes); weight_init = weight_init)

    return DLRMModel(bottom_mlp, embeddings, interaction, top_mlp)
end

end # module
