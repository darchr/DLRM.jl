# This
function create_mlp(sizes, sigmoid_index)
    layers = []
    for i in 1:(length(sizes)-1)
        in = sizes[i]
        out = sizes[i+1]

        # Choose between sigmoid or relu activation function.
        if i == sigmoid_index
            activation = Flux.σ
        else
            activation = Flux.sigmoid
        end
        push!(layers, Dense(in, out, activation))
    end
    return Chain(layers...)
end

function create_embeddings(ncols, rowcounts)
    embeddings = map(rowcounts) do nrows
        data = rand(Float32, ncols, nrows)
        return Embedding(data)
    end
    return embeddings
end

struct DLRMModel{B,E <: AbstractEmbedding,I,T}
    bottom_mlp::B
    embeddings::Vector{E}
    interaction::I
    top_mlp::T
end

function (D::DLRMModel)(dense, sparse)
    x = D.bottom_mlp(dense)
    y = map(D.embeddings, sparse) do E, I
        return E(+, I)
    end
    z = D.interaction(x, y)
    return D.top_mlp(z)
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
    num_features = length(embedding_sizes) + 1
    bottom_out_size = last(bottom_mlp_sizes)
    top_layer_input_size = div((num_features * (num_features - 1)), 2) + bottom_out_size
    @show num_features

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
