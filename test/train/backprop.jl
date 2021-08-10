#####
##### Test forward and backward model against a reference model.
#####

struct DLRMReference{B,E,T}
    bottom_mlp::B
    embeddings::E
    top_mlp::T
end

function reference_lookup(embeddings::Vector{<:AbstractMatrix}, ids)
    return map(reference_lookup, embeddings, ids)
end

function reference_lookup(embedding::AbstractMatrix, ids::AbstractVector)
    return embedding[:, ids]
end

function (model::DLRMReference)(dense, sparse)
    x = model.bottom_mlp(dense)
    y = reference_lookup(model.embeddings, sparse)
    z = DLRM._Model.dot_interaction_reference(x, y)
    out = model.top_mlp(z)
    return vec(out)
end

function loss(model, dense, sparse, labels)
    forward = model(dense, sparse)
    return DLRM.bce_loss(forward, labels)
end

@testset "Testing full model backprop" begin
    Random.seed!(1234)
    num_dense = 13
    num_categorical = 26
    max_ind = 1000
    feature_size = 64
    batchsize = 128

    # Generate input data
    dense = rand(Float32, num_dense, batchsize)
    categorical = [rand(1:max_ind, batchsize) for _ = 1:num_categorical]
    labels = rand((zero(Float32), one(Float32)), batchsize)

    # Build reference model
    bottom_mlp = Flux.Chain(
        Flux.Dense(num_dense, 256, Flux.relu; init = Flux.glorot_normal),
        Flux.Dense(256, 128, Flux.relu; init = Flux.glorot_normal),
        Flux.Dense(128, feature_size, Flux.relu; init = Flux.glorot_normal),
    )

    # Increase the weight of the embeddings for better numerical stability of testing.
    embeddings = [10 .* rand(Float32, feature_size, max_ind) for _ = 1:num_categorical]

    top_mlp = Flux.Chain(
        Flux.Dense(415, 256, Flux.relu; init = Flux.glorot_normal),
        Flux.Dense(256, 128, Flux.relu; init = Flux.glorot_normal),
        Flux.Dense(128, 1, Flux.sigmoid; init = Flux.glorot_normal),
    )

    model_reference = DLRMReference(bottom_mlp, embeddings, top_mlp)

    forward_reference = model_reference(dense, categorical)
    loss_reference = loss(model_reference, dense, categorical, labels)
    grads_reference = Zygote.gradient(loss, model_reference, dense, categorical, labels)

    #####
    ##### Create Optimized Model
    #####

    opt_bottom_mlp = Flux.Chain(OneDNN.Dense.(bottom_mlp)...)
    opt_embeddings = DLRM.SimpleEmbedding.(embeddings)
    opt_top_mlp = Flux.Chain(OneDNN.Dense.(top_mlp)...)

    model_opt = DLRM.DLRMModel(
        opt_bottom_mlp, opt_embeddings, DLRM._Model.dot_interaction, opt_top_mlp
    )

    forward_opt = model_opt(dense, categorical)
    loss_opt = loss(model_opt, dense, categorical, labels)
    grads_opt = Zygote.gradient(loss, model_opt, dense, categorical, labels)

    @test isapprox(forward_reference, forward_opt)
    @test isapprox(loss_reference, loss_opt)

    #####
    ##### Gradient Checks
    #####

    @test length(grads_reference) == 4
    @test length(grads_opt) == 4

    # Dense Argument
    @test isapprox(grads_reference[2], OneDNN.materialize(grads_opt[2]))

    # Start descending through the grads tuples
    grads_model_ref = grads_reference[1]
    grads_model_opt = grads_opt[1]

    # Bottom MLP
    # Use a "let" block to keep variable names from escaping.
    let
        bottom_mlp_grads_ref = grads_model_ref.bottom_mlp.layers
        bottom_mlp_grads_opt = grads_model_opt.bottom_mlp.layers

        @test length(bottom_mlp_grads_ref) == length(bottom_mlp_grads_opt)
        for i in eachindex(bottom_mlp_grads_ref, bottom_mlp_grads_opt)
            println("Testing Bottom MLP layer: $i")
            ref = bottom_mlp_grads_ref[i]
            @test isa(ref, NamedTuple)
            @test issubset((:weight, :bias, :σ), propertynames(ref))

            opt = bottom_mlp_grads_opt[i]
            @test isa(opt, NamedTuple)
            @test issubset((:weights, :bias), propertynames(opt))
            @test isa(opt.weights, OneDNN.Memory)
            @test isa(opt.bias, OneDNN.Memory)

            @test isapprox(ref.weight, transpose(OneDNN.materialize(opt.weights)))
            @test isapprox(ref.bias, OneDNN.materialize(opt.bias))
        end
    end

    # Top MLP
    let
        top_mlp_grads_ref = grads_model_ref.top_mlp.layers
        top_mlp_grads_opt = grads_model_opt.top_mlp.layers

        @test length(top_mlp_grads_ref) == length(top_mlp_grads_opt)
        for i in eachindex(top_mlp_grads_ref, top_mlp_grads_opt)
            println("Testing Top MLP layer: $i")
            ref = top_mlp_grads_ref[i]
            @test isa(ref, NamedTuple)
            @test issubset((:weight, :bias, :σ), propertynames(ref))

            opt = top_mlp_grads_opt[i]
            @test isa(opt, NamedTuple)
            @test issubset((:weights, :bias), propertynames(opt))
            @test isa(opt.weights, OneDNN.Memory)
            @test isa(opt.bias, OneDNN.Memory)

            @test isapprox(ref.weight, transpose(OneDNN.materialize(opt.weights)))
            @test isapprox(ref.bias, OneDNN.materialize(opt.bias))
        end
    end

    # Embeddings
    let
        embedding_grads_ref = grads_model_ref.embeddings
        embedding_grads_opt = grads_model_opt.embeddings
        @test length(embedding_grads_ref) == length(embedding_grads_opt)
        for i in eachindex(embedding_grads_ref, embedding_grads_opt)
            ref = embedding_grads_ref[i]
            opt = embedding_grads_opt[i]
            @test isa(opt, EmbeddingTables.SparseEmbeddingUpdate)
            @test isapprox(ref, EmbeddingTables.uncompress(opt, max_ind))
        end
    end
end
