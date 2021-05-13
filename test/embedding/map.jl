# We have a problem where lookups mapped across a vector of Embedding Tables doesn't
# correctly register the gradients.
#
# This test checks both that this case still fails and tests my workaround.
create_ensemble(dims; kw...) = create_ensemble((x...) -> map(DLRM._EmbeddingTables.lookup, x...), dims; kw...)
function create_ensemble(f, dims; kw...)
    params = map(dims) do dim
        return DLRM._EmbeddingTables.SimpleEmbedding(rand(Float32, dim))
    end

    return (I, y) -> Flux.mse(vcat(f(params, I)...), y), Flux.Params(params)
end

@testset "Testing Map" begin
    EmbeddingTables = DLRM._EmbeddingTables

    # Create three lookup tables
    dims = [
        (5, 5),
        (5, 10),
        (5, 15),
    ]

    # Create input indices and the "expected" result.
    batchsize = 5
    I = [rand(1:last(d), batchsize) for d in dims]
    y = rand(Float32, sum(first.(dims)), batchsize)

    # Create the function over normal arrays to avoid hitting our workaround.
    f, params = create_ensemble(dims)

    grads = Zygote.gradient(params) do
        f(I, y)
    end

    for (i, p) in enumerate(params)
        u = grads.grads[p]
        @test isa(u, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test u.indices == I[i]
    end

    #####
    ##### Now, verify that `SimpleEmbedding` works.
    #####

    f, params = create_ensemble(EmbeddingTables.maplookup, dims)
    grads = Zygote.gradient(params) do
        f(I, y)
    end
    for (i, p) in enumerate(params)
        u = grads.grads[p]
        @test isa(u, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test u.indices == I[i]
    end

    #####
    ##### Test that the ExecutionStrategy default works.
    #####

    tables = map(dims) do dim
        return DLRM._EmbeddingTables.SimpleEmbedding(rand(Float32, dim))
    end
    f = (I,y) -> Flux.mse(vcat(EmbeddingTables.maplookup(tables, I)...), y)
    params2 = Flux.Params(tables)

    grads2 = Zygote.gradient(params2) do
        f(I, y)
    end
    for (i, p) in enumerate(params2)
        u = grads2.grads[p]
        @test isa(u, DLRM._EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test u.indices == I[i]
    end
end
