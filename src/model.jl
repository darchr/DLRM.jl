# This
function create_mlp(sizes, sigmoid_index)
    layers = []
    for i in 1:(length(sizes)-1)
        in = sizes[i]
        out = sizes[i+1]

        # Choose between sigmoid or relu activation function.
        if i == sigmoid_index-1
            activation = Flux.sigmoid
        else
            activation = Flux.relu
        end
        layer = Dense(
            in,
            out,
            activation;
            initW = Flux.glorot_normal,
            initb = Flux.glorot_normal,
        )
        push!(layers, layer)
    end
    return Chain(layers...)
end

function create_embeddings(ncols, rowcounts)
    embeddings = map(rowcounts) do nrows
        data = Flux.glorot_normal(ncols, nrows)
        return SimpleEmbedding(data)
    end
    return embeddings
end

struct DLRMModel{B,E,I,T}
    bottom_mlp::B
    embeddings::Vector{E}
    interaction::I
    top_mlp::T
end

Flux.@functor DLRMModel (bottom_mlp, embeddings, top_mlp)

function (D::DLRMModel)(dense, sparse::Vector{<:Vector{<:Integer}})
    x = D.bottom_mlp(dense)
    y = map(maplookup, D.embeddings, sparse)
    z = D.interaction(x, y)
    out = D.top_mlp(z)
    return vec(out)
end

# Test entry point for now.
function dlrm(
        bottom_mlp_sizes,
        top_mlp_sizes,
        sparse_feature_size,
        embedding_sizes;
        # Default to the `dot` interaction method.
        interaction = dot_interaction,
    )

    # Create the bottom MLP
    bottom_mlp = create_mlp(bottom_mlp_sizes, 0)
    embeddings = create_embeddings(sparse_feature_size, embedding_sizes)

    # Compute the size of the first layer for the top mlp.
    num_features = length(embedding_sizes)
    bottom_out_size = last(bottom_mlp_sizes)

    # Do some math with sizes.
    @assert iszero(mod(sparse_feature_size * num_features, bottom_out_size))
    pre_triangle_size = div(sparse_feature_size * num_features, bottom_out_size) + 1

    top_layer_input_size = div(pre_triangle_size^2 - pre_triangle_size, 2) + bottom_out_size
    top_mlp_sizes = vcat([top_layer_input_size], top_mlp_sizes)
    @show top_mlp_sizes
    top_mlp = create_mlp(top_mlp_sizes, lastindex(top_mlp_sizes))

    return DLRMModel(
        bottom_mlp,
        embeddings,
        interaction,
        top_mlp,
    )
end
