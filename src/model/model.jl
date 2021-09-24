module _Model

export DLRMModel, dlrm

# stdlib
import Random

# internal deps
using EmbeddingTables
using .._Utils

# local deps
using CachedArrays: CachedArrays

# deps
import ChainRulesCore
import DataStructures
import Distributions
import Flux
import LoopVectorization
import ManualMemory
import NNlib
import OneDNN
import Polyester
import ProgressMeter
import UnPack: @unpack
import Zygote

# Documentation
using DocStringExtensions

const POST_INTERACTION_PAD_TO_MUL = 1
cdiv(x, y) = 1 + div(x - 1, y)
up_to_mul_of(x, y) = y * cdiv(x, y)

include("interact.jl")
include("embedding_update.jl")

# Initialization
function multithread_init(f, A)
    Threads.@threads for i in eachindex(A)
        A[i] = f(A)
    end
end

function singlethread_init(f, A)
    for i in eachindex(A)
        A[i] = f(A)
    end
end

struct OneInit end
(f::OneInit)(A) = one(eltype(A))

struct ZeroInit end
(f::ZeroInit)(A) = zero(eltype(A))

struct GlorotNormal end
(f::GlorotNormal)(A) = rand(Distributions.Normal(zero(Float32), sqrt(2.0f0 / sum(size(A)))))

struct ScaledUniform end
function (f::ScaledUniform)(A)
    sz = inv(sqrt(Float32(size(A, 2))))
    return rand(Distributions.Uniform(-sz, sz))
end

# Strategy:
# We can take advantage of Julia's dynamic typing to build the intermediate parts of the
# network using an untyped array.
#
# Then, we dump everying into a `Flux.Chain` to become concretely typed.
function create_mlp(sizes, sigmoid_index; init)
    layers = Any[]
    for i = Base.OneTo(length(sizes) - 1)
        in = sizes[i]
        out = sizes[i + 1]
        issigmoid = (i == sigmoid_index - 1)

        # Choose between sigmoid or relu activation function.
        if issigmoid
            activation = identity
        else
            activation = Flux.relu
        end
        layer = OneDNN.Dense(Flux.Dense(in, out, activation; init = init))
        push!(layers, layer)
        if issigmoid
            push!(layers, x -> OneDNN.toeltype(Float32, x))
            push!(layers, x -> OneDNN.eltwise(Flux.sigmoid, x))
        end
    end
    return Flux.Chain(layers...)
end

# Create embeddings.
function create_embeddings(finish, ncols, rowcounts; init)
    progress_meter = ProgressMeter.Progress(
        length(rowcounts), 1, "Building Embedding Tables ..."
    )
    embeddings = map(Base.OneTo(length(rowcounts))) do i
        nrows = rowcounts[i]
        # Construct and initialize the underlying data.
        # Then construct the embedding table
        table = finish(init(ncols, nrows))
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

    # Options for initializaztion
    weight_init_kernel = GlorotNormal(),
    weight_eltype::Type{W} = Float32,

    embedding_constructor = SimpleEmbedding,
    embedding_init_kernel = ScaledUniform(),
    embedding_eltype::Type{E} = Float32,

    # Where does memory come from?
    allocator = default_allocator,
    embedding_allocator = allocator,
) where {W,E}
    Random.seed!(51234)

    # Create closures for constructing the weights, biases, and embedding tables.
    function init_weight(dims...)
        data = allocator(W, dims...)
        multithread_init(weight_init_kernel, data)
        return data
    end

    function init_embedding(dims...)
        data = embedding_allocator(E, dims...)
        multithread_init(embedding_init_kernel, data)
        return data
    end

    # Create the bottom MLP
    bottom_mlp = create_mlp(bottom_mlp_sizes, 0; init = init_weight)
    embeddings = create_embeddings(
        embedding_constructor, sparse_feature_size, embedding_sizes; init = init_embedding,
    )

    # Compute the size of the first layer for the top mlp.
    num_features = length(embedding_sizes)
    bottom_out_size = last(bottom_mlp_sizes)

    # Do some math with sizes.
    # This is gross ...
    @assert iszero(mod(sparse_feature_size * num_features, bottom_out_size))
    pre_triangle_size = div(sparse_feature_size * num_features, bottom_out_size) + 1

    top_layer_input_size = up_to_mul_of(
        div(pre_triangle_size^2 - pre_triangle_size, 2) + bottom_out_size,
        POST_INTERACTION_PAD_TO_MUL,
    )

    # top_layer_input_size = div(pre_triangle_size^2 - pre_triangle_size, 2) + bottom_out_size
    top_mlp_sizes = vcat([top_layer_input_size], top_mlp_sizes)
    top_mlp = create_mlp(top_mlp_sizes, lastindex(top_mlp_sizes); init = init_weight)

    return DLRMModel(bottom_mlp, embeddings, interaction, top_mlp)
end

end # module
