# We have a problem where lookups mapped across a vector of Embedding Tables doesn't
# correctly register the gradients.
#
# This test checks both that this case still fails and tests my workaround.
function create_ensemble(dims)
    return [DLRM._EmbeddingTables.SimpleEmbedding(rand(Float32, dim)) for dim in dims]
end

_normalize(x::AbstractVector) = reduce(vcat, x)
_normalize(x) = x

loss_wrap(f) = (y, x...) -> Flux.mse(_normalize(f(x...)), y)

@testset "Testing Map" begin
    EmbeddingTables = DLRM._EmbeddingTables

    # Create three lookup tables
    dims = [(5, 5), (5, 10), (5, 15)]

    # Create input indices and the "expected" result.
    batchsize = 5
    I = [rand(1:last(d), batchsize) for d in dims]
    y = rand(Float32, sum(first.(dims)), batchsize)

    # Create the function over normal arrays to avoid hitting our workaround.
    tables = create_ensemble(dims)
    grads = Zygote.gradient(loss_wrap((x...) -> map(DLRM.lookup, x...)), y, tables, I)

    # Gradients for the lookup table are at index 2
    for (i, grad) in enumerate(grads[2])
        @test isa(grad, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
    end

    #####
    ##### Now, verify that `SimpleEmbedding` works.
    #####

    grads = Zygote.gradient(loss_wrap(DLRM.maplookup), y, tables, I)
    for (i, grad) in enumerate(grads[2])
        @test isa(grad, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
    end

    #####
    ##### Try "PreallocationStrategy"
    #####

    f = (x...) -> DLRM.maplookup(DLRM.PreallocationStrategy(), x...)
    grads2 = Zygote.gradient(loss_wrap(f), y, tables, I)

    for (i, grad) in enumerate(grads2[2])
        @test isa(grad, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
        @test grad.delta == grads[2][i].delta
        @test grad.indices == grads[2][i].indices
    end

    # Retry with prepadding
    # Use a view of the output to keep dimensions happy with the "y" matrix used as the
    # dummy target.
    f = (x...) -> @views(DLRM.maplookup(DLRM.PreallocationStrategy(20), x...)[21:end, :])
    grads2 = Zygote.gradient(loss_wrap(f), y, tables, I)

    for (i, grad) in enumerate(grads2[2])
        @test isa(grad, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
        @test grad.delta == grads[2][i].delta
        @test grad.indices == grads[2][i].indices
    end
end
