@testset "Testing Against Large PyTorch Reference" begin
    reference_path = "/home/mark/projects/intermediate_values.hdf5"
    io = HDF5.h5open(reference_path)
    batchsize = 128

    model = DLRM.load_hdf5(io)
    labels, dense, sparse = DLRM.load_inputs(io)

    y = DLRM.maplookup(DLRM.SimpleParallelStrategy(), model.embeddings, sparse)
    x = model.bottom_mlp(dense)
    z = model.interaction(x, y)
    out = model.top_mlp(z)
    loss = DLRM._Train.bce_loss(vec(OneDNN.materialize(out)), labels)

    # Unload reference and compare
    x_ref = read(io["mlp_bottom"])
    @test isapprox(x_ref, OneDNN.materialize(x))

    z_ref = read(io["output_interaction"])
    @test isapprox(z_ref, OneDNN.materialize(z))

    out_ref = read(io["mlp_top"])
    @test isapprox(out_ref, OneDNN.materialize(out))

    loss_ref = read(io["loss"])
    @test isapprox(loss_ref, loss)

    # More agressive
    loss_fn = DLRM._Train.wrap_loss(DLRM._Train.bce_loss)
    @test isapprox(loss_ref, loss_fn(model, labels, dense, sparse))

    #####
    ##### Gradients!
    #####

    learning_rate = 10.0
    opt = Flux.Descent(learning_rate)

    params = DLRM._Train.DLRMParams(model)
    grads = DLRM._Train.DLRMGrads(model)
    _grads = Zygote.gradient(loss_fn, model, labels, dense, sparse)
    DLRM._Train.gather!(grads, _grads[1])
    DLRM._Train.custom_update!(opt, params, grads)

    # Top MLP
    start = length(model.bottom_mlp) + 1
    let
        update_names = sort(
            filter(startswith("update_top"), keys(io)); lt = NaturalSort.natural
        )
        update_prefixes = unique(first.(splitext.(update_names)))

        original_names = sort(
            filter(startswith("top_l"), keys(io)); lt = NaturalSort.natural
        )
        original_prefixes = unique(first.(splitext.(original_names)))

        iter = eachindex(original_prefixes, update_prefixes, grads.weights[start:end])

        for i in iter
            # Weights
            original = read(io["$(original_prefixes[i]).weight"])
            update = read(io["$(update_prefixes[i]).weight"])
            pytorch_grad = original .- update

            jl_grad = learning_rate .* OneDNN.materialize(grads.weights[start + i - 1])
            @test !isapprox(original, update)
            @test isapprox(jl_grad, pytorch_grad)
            @test isapprox(update, OneDNN.materialize(model.top_mlp[i].weights))

            # Bias
            original = read(io["$(original_prefixes[i]).bias"])
            update = read(io["$(update_prefixes[i]).bias"])
            pytorch_grad = original .- update

            jl_grad = learning_rate .* OneDNN.materialize(grads.bias[start + i - 1])
            @test !isapprox(original, update)
            @test isapprox(jl_grad, pytorch_grad)
            @test isapprox(update, OneDNN.materialize(model.top_mlp[i].bias))
        end
    end

    # Bottom MLP
    let
        update_names = sort(
            filter(startswith("update_bot"), keys(io)); lt = NaturalSort.natural
        )
        update_prefixes = unique(first.(splitext.(update_names)))

        original_names = sort(
            filter(startswith("bot_l"), keys(io)); lt = NaturalSort.natural
        )
        original_prefixes = unique(first.(splitext.(original_names)))

        iter = eachindex(original_prefixes, update_prefixes, grads.weights[1:(start - 1)])

        for i in iter
            # Weights
            original = read(io["$(original_prefixes[i]).weight"])
            update = read(io["$(update_prefixes[i]).weight"])
            pytorch_grad = original .- update

            jl_grad = learning_rate .* OneDNN.materialize(grads.weights[i])
            @test !isapprox(original, update)
            @test isapprox(jl_grad, pytorch_grad)
            @test isapprox(update, OneDNN.materialize(model.bottom_mlp[i].weights))

            # Bias
            original = read(io["$(original_prefixes[i]).bias"])
            update = read(io["$(update_prefixes[i]).bias"])
            pytorch_grad = original .- update

            jl_grad = learning_rate .* OneDNN.materialize(grads.bias[i])
            @test !isapprox(original, update)
            @test isapprox(jl_grad, pytorch_grad)
            @test isapprox(update, OneDNN.materialize(model.bottom_mlp[i].bias))
        end
    end

    # Embedding Tables
    let
        update_names = sort(
            filter(startswith("update_emb"), keys(io)); lt = NaturalSort.natural
        )

        original_names = sort(
            filter(startswith("emb_"), keys(io)); lt = NaturalSort.natural
        )

        for i in eachindex(original_names, update_names, model.embeddings)
            original = read(io[original_names[i]])
            reference = read(io[update_names[i]])
            @test isapprox(reference, model.embeddings[i])
            @test !isapprox(original, reference)
        end
    end
end
