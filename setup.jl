using DLRM, Flux, Zygote, Profile

function makefunction()
    dlrm = DLRM.dlrm(
        [512, 512, 64],
        [1024, 1024, 1024, 1],
        64,
        [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000],
    )

    #return f = (d,s,e) -> (sum(dlrm(d,s) .+ e)), dlrm
    return f = (d, s, e) -> Flux.mse(dlrm(d, s), e), dlrm
end

function _profile(f, iters)
    @profile for _ in 1:iters
        f()
    end
end

function makeforward(batchsize = 2048)
    forward, dlrm = makefunction()
    dense = rand(Float32, 512, batchsize)
    sparse = [[rand(1:1000000) for _ in 1:batchsize] for _ in 1:8]

    labels = Float32.((0, 1))
    expected = rand(labels, batchsize)

    return () -> forward(dense, sparse, expected)
end

function makebackward(batchsize = 2048)
    forward, dlrm = makefunction()
    dense = rand(Float32, 512, batchsize)
    sparse = [[rand(1:1000000) for _ in 1:batchsize] for _ in 1:8]

    labels = Float32.((0, 1))
    expected = rand(labels, batchsize)
    params = Flux.params(dlrm)
    return () -> gradient(params) do
        forward(dense, sparse, expected)
    end
end

function profilef(f, iters)
    Profile.clear()
    _profile(f, 3)
    Profile.clear()
    return _profile(f, iters)
end

#####
##### Debugging stuff
#####

function test(batchsize)
    mlp = DLRM.create_mlp([1024, 1024, 1024, 1], -1)
    x = rand(Float32, 1024, batchsize)
    expected = rand(Float32, batchsize)

    return (x, y) -> Flux.mse(vec(mlp(x)), y), Flux.params(mlp), x, expected
end
