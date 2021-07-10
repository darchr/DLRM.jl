function validate(path, strategy; learning_rate = 10.0)
    io = HDF5.h5open(path)

    model = DLRM.load_hdf5(io)
    labels, dense, sparse = DLRM.load_inputs(io)
    loss_fn = DLRM._Train.wrap_loss(DLRM._Train.bce_loss; strategy = strategy)

    #####
    ##### Inference
    #####

    loss_ref = read(io["loss"])
    jl_loss = loss_fn(model, labels, dense, sparse)
    if !isapprox(loss_ref, jl_loss)
        msg = """
        Loss mismatch between Julia and Pytorch Inference.
        Pytorch: $loss_ref
        Julia: $jl_loss
        """
        error(msg)
    end

    learning_rate = 10.0
    opt = Flux.Descent(learning_rate)

    # Run once to update data formats
    _ = Zygote.gradient(loss_fn, model, labels, dense, sparse)

    params = DLRM._Train.DLRMParams(model)
    grads = DLRM._Train.DLRMGrads(model)
    _grads = Zygote.gradient(loss_fn, model, labels, dense, sparse)
    DLRM._Train.gather!(grads, _grads[1])
    DLRM._Train.custom_update!(opt, params, grads)

    # Top MLP
    start = length(model.bottom_mlp) + 1
    validate_mlp(model, io, grads, "top", start, learning_rate)

    # Bottom MLP
    start = 1
    validate_mlp(model, io, grads, "bottom", start, learning_rate)
    validate_embeddings(model, io, grads)
    return true
end

function __updatename(key)
    updatenames = Dict(
        "top" => "update_top",
        "bottom" => "update_bot",
    )
    return updatenames[key]
end

function __originalname(key)
    originalnames = Dict(
        "top" => "top_l",
        "bottom" => "bot_l",
    )
    return originalnames[key]
end


function get(model::DLRMModel, k::AbstractString)
    if k == "top"
        return model.top_mlp
    elseif k == "bottom"
        return model.bottom_mlp
    end
    error("Unknown Key: $k")
end

prefix(x) = first(splitext(x))

function validate_mlp(model, io, grads, key::AbstractString, start, learning_rate)
    update_names = sort(
        filter(startswith(__updatename(key)), keys(io)); lt = NaturalSort.natural
    )
    update_prefixes = unique(prefix.(update_names))

    original_names = sort(
        filter(startswith(__originalname(key)), keys(io)); lt = NaturalSort.natural
    )
    original_prefixes = unique(prefix.(original_names))

    # Use "eachindex" to ensure all lengths are the same
    len = length(original_prefixes)
    iter = eachindex(original_prefixes, update_prefixes, grads.weights[start:(start + len - 1)])

    for i in iter
        println("Checking $(titlecase(key)) MLP: ", i)
        # Weights
        original = read(io["$(original_prefixes[i]).weight"])
        update = read(io["$(update_prefixes[i]).weight"])
        pytorch_grad = original .- update

        jl_grad = learning_rate .* OneDNN.materialize(grads.weights[start + i - 1])
        if isapprox(original, update)
            error("Pytorch original and updated weights match!")
        end
        if !isapprox(jl_grad, pytorch_grad)
            error("Julia and Pytorch gradients are not approximately equal!")
        end
        if !isapprox(update, OneDNN.materialize(get(model, key)[i].weights))
            error("Julia and Pytorch updated weights are not the same!")
        end

        # Bias
        original = read(io["$(original_prefixes[i]).bias"])
        update = read(io["$(update_prefixes[i]).bias"])
        pytorch_grad = original .- update

        jl_grad = learning_rate .* OneDNN.materialize(grads.bias[start + i - 1])
        if isapprox(original, update)
            error("Pytorch original and updated weights match!")
        end
        if !isapprox(jl_grad, pytorch_grad)
            error("Julia and Pytorch gradients are not approximately equal!")
        end
        if !isapprox(update, OneDNN.materialize(get(model, key)[i].bias))
            error("Julia and Pytorch updated weights are not the same!")
        end
    end
end

function validate_embeddings(model, io, grads)
    update_names = sort(
        filter(startswith("update_emb"), keys(io)); lt = NaturalSort.natural
    )

    original_names = sort(
        filter(startswith("emb_"), keys(io)); lt = NaturalSort.natural
    )

    for i in eachindex(original_names, update_names, model.embeddings)
        println("Checking Embedding Tables: ", i)
        original = read(io[original_names[i]])
        reference = read(io[update_names[i]])
        if !isapprox(reference, model.embeddings[i])
            error("Julia and Pytorch updated embeddings don't match!")
        end

        if isapprox(original, reference)
            error("Pytorch original and updated embeddings match!")
        end
    end
end
