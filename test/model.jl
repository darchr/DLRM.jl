# Test the whole model.
function makefunction()
    dlrm  = DLRM.dlrm(
       [512, 512, 64],
       [1024, 1024, 1024, 1],
       64,
       [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
    )

    return f = (d,s,e) -> Flux.mse(dlrm(d,s), e), dlrm
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
        forward(dense, sparse, expected)
    end
end

# We have data from the original py-torceh implementation.
#
# Here, we test against those values
@testset "Testing Against Pytorch" begin
    #####
    ##### Bottom MLP
    #####

    dense_input = [
        0.03685 0.2673  0.34    0.86827 0.50025;
        0.33139 0.61687 0.08192 0.09998 0.17455;
        0.26606 0.43873 0.05581 0.8632  0.01412;
        0.03511 0.24039 0.3384  0.86125 0.16903;
    ]' |> collect

    # Expected Outputs
    output_of_bottom_mlp = [
        0.      0.      0.48116 0.2543
        0.10157 0.      1.21686 0.12463
        0.10026 0.      0.86805 0.29431
        0.      0.      0.46845 0.18102
    ]' |> collect

    bottom_mlp = DLRM.create_mlp([5, 4], -1)

    # Manually configure the weights
    dense1 = bottom_mlp[1]
    dense1.W .= [
         0.74967  0.56771  0.12462  0.25165  0.24943;
        -0.48992  0.27794 -1.07279 -0.14344  0.05737;
         0.03564  1.2152  -0.50384 -0.19891 -0.05389;
         0.36123 -0.1329  -0.32115  0.25878  0.2262 ;
    ]
    dense1.b .= [-0.57597, -0.69637, 0.52599, 0.04786]

    @test isapprox(bottom_mlp(dense_input), output_of_bottom_mlp)

    #####
    ##### Embedding Tables
    #####

    # Adjust from Python's zero-based indexing
    sparse_inputs = [
        [3, 1, 4, 2] .+ 1,
        [1, 1, 2, 4] .+ 1,
        [2, 3, 1, 4] .+ 1,
    ]

    output_of_embedding_tables = [
        [-2.44501e-01  2.17411e-01 -3.56797e-01  1.62552e-01;
         -1.94420e-01 -2.32488e-01 -2.44200e-01  3.98621e-01;
          3.02468e-01 -2.25959e-01  7.72339e-06 -2.31782e-01;
          3.92202e-01  2.09187e-01  5.49969e-02 -3.05629e-01]',
        [-0.08135 -0.43825  0.26487 -0.36162;
         -0.08135 -0.43825  0.26487 -0.36162;
         -0.22964 -0.23195 -0.07373  0.34383;
          0.09969 -0.02979  0.42599 -0.23596]',
        [-0.19309  0.33247  0.13907 -0.24246;
         -0.4414  -0.06371 -0.15425 -0.08834;
          0.24136  0.12974  0.14533 -0.28756;
          0.25169  0.34613  0.21345  0.44208]',
    ]

    embeddings = DLRM.create_embeddings(4, [5,5,5])

    embeddings[1].data .= [
         1.38703e-01 -3.49615e-02  2.91594e-01 -2.36776e-01;
        -1.94420e-01 -2.32488e-01 -2.44200e-01  3.98621e-01;
         3.92202e-01  2.09187e-01  5.49969e-02 -3.05629e-01;
        -2.44501e-01  2.17411e-01 -3.56797e-01  1.62552e-01;
         3.02468e-01 -2.25959e-01  7.72339e-06 -2.31782e-01;
    ]'

    embeddings[2].data .= [
        -0.22749  0.15044 -0.25153 -0.03229;
        -0.08135 -0.43825  0.26487 -0.36162;
        -0.22964 -0.23195 -0.07373  0.34383;
         0.13122  0.10117 -0.2659   0.05052;
         0.09969 -0.02979  0.42599 -0.23596;
    ]'

    embeddings[3].data .= [
         0.24587 -0.03799  0.22436  0.06881;
         0.24136  0.12974  0.14533 -0.28756;
        -0.19309  0.33247  0.13907 -0.24246;
        -0.4414  -0.06371 -0.15425 -0.08834;
         0.25169  0.34613  0.21345  0.44208;
    ]'

    Ys = map((e,i) -> e(i), embeddings, sparse_inputs)

    for i in 1:3
        @test isapprox(Ys[i], output_of_embedding_tables[i])
    end

    #####
    ##### Dot Interaction
    #####

    # NOTE: We do the triangular slicing in a different order.
    top_mlp = DLRM.create_mlp([10, 5, 1], [3])

    top_mlp[1].W .= [
        -0.17529 -0.77823  0.21247  0.23766  0.18327 -0.45722 -0.03328  0.22742 -0.19802 -0.28992;
        -0.11675 -0.58319  0.627    0.08652  0.20797 -0.10146 -0.31713 -0.22506 0.41106  0.01268 ;
         0.02179 -0.41254  0.28599 -0.76669 -0.02132  0.5079   0.75233  0.16296 0.04205  0.59095 ;
         0.01036  0.06319 -0.32023  0.45371  0.05588 -0.12152  0.23366  0.12804 0.30971  0.01763 ;
         0.07816  0.24648 -0.52729  0.10986  0.1919   0.32612  0.14791  0.10404 0.05035 -0.2003  ;
    ]
    top_mlp[1].b .= [-0.20719, -0.54316, 0.73679, 0.28494, 0.4195]

    top_mlp[2].W .= [0.15213, -0.12579, -0.69831, -0.9994,  -0.20551]'
    top_mlp[2].b .= [1.80363]

    dlrm = DLRM.DLRMModel(bottom_mlp, embeddings, DLRM.dot_interaction, top_mlp)
    expected =[0.72185, 0.30605, 0.92103, 0.78934]

    @show dlrm(dense_input, sparse_inputs)
end
