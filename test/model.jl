# Test the whole model.
function makefunction()
    dlrm  = DLRM.dlrm(
       [512, 512, 64],
       [1024, 1024, 1024, 1],
       64,
       [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
    )

    return f = (d,s,e) -> Flux.crossentropy(dlrm(d,s), e), dlrm
end

@testset "Testing Whole Pipeline" begin
    # Base this on the benchmark code that the `dlrm` repo provides.
    batchsize = 128

    forward, dlrm = makefunction()
    dense = rand(Float32, 512, batchsize)
    sparse = [[rand(1:1000000) for _ in 1:batchsize] for _ in 1:8]

    labels = Float32.((0,1))
    expected = rand(labels, batchsize)

    # Try to take a gradient
    grads = gradient(params(dlrm)) do
        forward(dense, sparse, labels)
    end
end
