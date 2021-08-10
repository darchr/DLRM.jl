#####
##### Playground for testing Embedding Table Update Strategies.
#####

function make_tables(
    sizes::AbstractVector, featuresize::Int, manager::CachedArrays.CacheManager
)
    constructor = tocached(
        Float32, manager, CachedArrays.ReadWrite(); priority = CachedArrays.ForceRemote
    )
    function init(args...)
        data = constructor(args...)
        _Model.multithread_init(_Model.ZeroInit(), data)
        return data
    end

    f = x -> SplitEmbedding(x, 1024)
    g = SplitEmbedding{Static{featuresize}}

    return _Model.create_embeddings(g, featuresize, sizes, init)
    #return _Model.create_embeddings(f, featuresize, sizes, init)
end

function make_updates(
    sizes::AbstractVector,
    featuresize::Int,
    manager::CachedArrays.CacheManager;
    indices_per_result = 1,
    batchsize = 16,
)
    constructor = tocached(Float32, manager, CachedArrays.ReadWrite())
    function init(args...)
        data = constructor(args...)
        Random.randn!(data)
        return data
    end

    arrays = [init(featuresize, batchsize) for _ in sizes]
    if indices_per_result == 1
        indices = [rand(1:sz, batchsize) for sz in sizes]
    else
        indices = [rand(1:sz, indices_per_result, batchsize) for sz in sizes]
    end
    return (_Model.SparseEmbeddingUpdate{Static{featuresize}}).(arrays, indices)
end
