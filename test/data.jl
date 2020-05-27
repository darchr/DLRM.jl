# Test putting in the random data generator.
@testset "Testing Random Data Generator" begin
    model = DLRM.kaggle_dlrm()
    loss = (dense, sparse, labels) -> begin
        # Clamp for stability
        forward = model(dense, sparse)
        ls = sum(Flux.binarycrossentropy.(forward, vec(labels))) / length(forward)
        isnan(ls) && throw(error("NaN Loss"))
        return ls
    end

    batchsize = 128

    generator = DLRM.DataGenerator(;
        batchsize = batchsize,
        densesize = DLRM.num_continuous_features(DLRM.DAC()),
        sparsesizes = DLRM.KAGGLE_EMBEDDING_SIZES
    )

    # Just run the model with data from the generator.
    # Make sure it doesn't error out.
    val = loss(generator[]...)
    @test isa(val, Float32)
end
