# Test the whole model.
function makefunction()
    #dot = DLRM._Model.DotInteraction(Matrix{Float32}(undef, 1, 1))

    dlrm = DLRM.dlrm(
        [512, 512, 64],
        [1024, 1024, 1024, 1],
        64,
        [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000];
        interaction = DLRM._Model.dot_interaction,
    )

    return f = (d, s, e) -> Flux.mse(dlrm(d, s), e), dlrm
end

@testset "Testing Whole Pipeline" begin
    # Base this on the benchmark code that the `dlrm` repo provides.
    batchsize = 128

    forward, dlrm = makefunction()
    dense = rand(Float32, 512, batchsize)
    sparse = [[rand(1:100000) for _ in 1:batchsize] for _ in 1:8]

    labels = Float32.((0, 1))
    expected = rand(labels, batchsize)

    # Try to take a gradient
    grads = gradient(params(dlrm)) do
        forward(dense, sparse, expected)
    end
end

#####
##### Test manually running the model versus the full run
#####

# Since we manually run the model when comparing against PyTorch, this will detect changes
# we make to the code so we don't accidentally invalidate tests.
@testset "Manually running Model" begin
    bottom_mlp = Chain(Flux.Dense(10, 20, Flux.relu), Flux.Dense(20, 30, Flux.relu))

    embedding_tables = [
        DLRM.SimpleEmbedding(Flux.glorot_normal(30, 100)),
        DLRM.SimpleEmbedding(Flux.glorot_normal(30, 100)),
        DLRM.SimpleEmbedding(Flux.glorot_normal(30, 100)),
    ]

    interaction = DLRM._Model.dot_interaction_reference

    top_mlp = Chain(
        Flux.Dense(36, 20, Flux.relu),
        Flux.Dense(20, 30, Flux.relu),
        Flux.Dense(30, 1, Flux.sigmoid),
    )

    model = DLRM.DLRMModel(bottom_mlp, embedding_tables, interaction, top_mlp)

    batchsize = 20
    dense_inputs = randn(Float32, 10, batchsize)
    sparse_inputs = [rand(1:100, batchsize) for _ in 1:length(embedding_tables)]

    # Run the model automatically.
    reference = model(dense_inputs, sparse_inputs)
    @test !any(isnan, reference)

    # run manually
    X = model.bottom_mlp(dense_inputs)
    Ys = DLRM.maplookup(model.embeddings, sparse_inputs)
    Z = model.interaction(X, Ys)
    out = vec(model.top_mlp(Z))

    # should be equal
    # If this test fails, then our comparison with PyTorch will have to be revisited.
    @test out == reference
end

# We have data from the original py-torceh implementation.
#
# Here, we test against those values
@testset "Testing Against Pytorch" begin
    # Reference points
    #
    # mlp top arch 2 layers, with input to output dimensions:
    # [10  5  1]
    # # of interactions
    # 10
    # mlp bot arch 1 layers, with input to output dimensions:
    # [5 4]
    # # of features (sparse and dense)
    # 4
    # dense feature size
    # 5
    # sparse feature size
    # 4
    # # of embeddings (= # of sparse features) 3, with dimensions 4x:
    # [5 5 5]

    # Some things to keep track of:
    #
    # PyTorch is row major while Julia is column major.
    # However, OneDNN introduces a wierd mix of row-major + column-major headaches.
    #
    # All of the `py` prefixed variables are entered as they are in PyTorch.

    ### Inputs and Expected Outputs
    py_dense_input =
        Float32.(
            [
                0.03685 0.2673 0.34 0.86827 0.50025
                0.33139 0.61687 0.08192 0.09998 0.17455
                0.26606 0.43873 0.05581 0.8632 0.01412
                0.03511 0.24039 0.3384 0.86125 0.16903
            ],
        )

    # add `1` because PyTorch in index-0 while julia is index-1
    py_sparse_input = [[3, 1, 4, 2] .+ 1, [1, 1, 2, 4] .+ 1, [2, 3, 1, 4] .+ 1]

    py_expected_result = Float32.([0.72185, 0.30605, 0.92103, 0.78934])

    ### Initial Weights ###

    # bottom mlp
    py_dense1_weights =
        Float32.(
            [
                0.74967 0.56771 0.12462 0.25165 0.24943
                -0.48992 0.27794 -1.07279 -0.14344 0.05737
                0.03564 1.2152 -0.50384 -0.19891 -0.05389
                0.36123 -0.1329 -0.32115 0.25878 0.2262
            ],
        )
    py_dense1_bias = Float32.([-0.57597, -0.69637, 0.52599, 0.04786])

    # embedding tables
    py_embedding1_weights =
        Float32.(
            [
                1.38703e-01 -3.49615e-02 2.91594e-01 -2.36776e-01
                -1.94420e-01 -2.32488e-01 -2.44200e-01 3.98621e-01
                3.92202e-01 2.09187e-01 5.49969e-02 -3.05629e-01
                -2.44501e-01 2.17411e-01 -3.56797e-01 1.62552e-01
                3.02468e-01 -2.25959e-01 7.72339e-06 -2.31782e-01
            ],
        )

    py_embedding2_weights =
        Float32.(
            [
                -0.22749 0.15044 -0.25153 -0.03229
                -0.08135 -0.43825 0.26487 -0.36162
                -0.22964 -0.23195 -0.07373 0.34383
                0.13122 0.10117 -0.2659 0.05052
                0.09969 -0.02979 0.42599 -0.23596
            ],
        )

    py_embedding3_weights =
        Float32.(
            [
                0.24587 -0.03799 0.22436 0.06881
                0.24136 0.12974 0.14533 -0.28756
                -0.19309 0.33247 0.13907 -0.24246
                -0.4414 -0.06371 -0.15425 -0.08834
                0.25169 0.34613 0.21345 0.44208
            ],
        )

    # top mlp
    py_dense2_weights =
        Float32.(
            [
                -0.17529 -0.77823 0.21247 0.23766 0.18327 -0.45722 -0.03328 0.22742 -0.19802 -0.28992
                -0.11675 -0.58319 0.627 0.08652 0.20797 -0.10146 -0.31713 -0.22506 0.41106 0.01268
                0.02179 -0.41254 0.28599 -0.76669 -0.02132 0.5079 0.75233 0.16296 0.04205 0.59095
                0.01036 0.06319 -0.32023 0.45371 0.05588 -0.12152 0.23366 0.12804 0.30971 0.01763
                0.07816 0.24648 -0.52729 0.10986 0.1919 0.32612 0.14791 0.10404 0.05035 -0.2003
            ],
        )
    py_dense2_bias = Float32.([-0.20719, -0.54316, 0.73679, 0.28494, 0.4195])

    py_dense3_weights = Float32.([0.15213, -0.12579, -0.69831, -0.9994, -0.20551]')
    py_dense3_bias = Float32.([1.80363])

    ### expected intermediate results ###
    py_bottom_mlp_output =
        Float32.(
            [
                0.0 0.0 0.48116 0.2543
                0.10157 0.0 1.21686 0.12463
                0.10026 0.0 0.86805 0.29431
                0.0 0.0 0.46845 0.18102
            ],
        )

    py_embedding_outputs = [
        Float32.(
            [
                -2.44501e-01 2.17411e-01 -3.56797e-01 1.62552e-01
                -1.94420e-01 -2.32488e-01 -2.44200e-01 3.98621e-01
                3.02468e-01 -2.25959e-01 7.72339e-06 -2.31782e-01
                3.92202e-01 2.09187e-01 5.49969e-02 -3.05629e-01
            ],
        ),
        Float32.(
            [
                -0.08135 -0.43825 0.26487 -0.36162
                -0.08135 -0.43825 0.26487 -0.36162
                -0.22964 -0.23195 -0.07373 0.34383
                0.09969 -0.02979 0.42599 -0.23596
            ],
        ),
        Float32.(
            [
                -0.19309 0.33247 0.13907 -0.24246
                -0.4414 -0.06371 -0.15425 -0.08834
                0.24136 0.12974 0.14533 -0.28756
                0.25169 0.34613 0.21345 0.44208
            ],
        ),
    ]

    # note: [:, 1:4] is `py_bottom_mlp_output`, [:, 5:end] is the flattened BMM result.
    py_interaction_output =
        Float32.(
            [
                0.0 0.0 0.48116 0.2543 -0.13034 0.03548 -0.22868 0.00526 0.03046 -0.00548
                0.10157 0.0 1.21686 0.12463 -0.26723 0.26898 -0.09113 -0.24355 0.10308 0.05492
                0.10026 0.0 0.86805 0.29431 -0.03788 0.01416 -0.09674 0.06572 0.11034 -0.1951
                0.0 0.0 0.46845 0.18102 -0.02956 0.15684 0.12841 0.18002 0.04775 0.00139
            ],
        )

    py_top_mlp_output = Float32.([0.77095, 0.73668, 0.7734, 0.69538])

    #####
    ##### Testing
    #####

    # We construct two Julia DLRM models.
    # One uses standard Flux layers and the other uses our OneDNN layers.
    # Both should yield approximately equal results and be equal to the PyTorch
    # implementation.

    ### reference implementation
    bottom_mlp = Flux.Dense(py_dense1_weights, py_dense1_bias, Flux.relu)
    embedding_tables = [
        DLRM.SimpleEmbedding(collect(py_embedding1_weights')),
        DLRM.SimpleEmbedding(collect(py_embedding2_weights')),
        DLRM.SimpleEmbedding(collect(py_embedding3_weights')),
    ]
    interaction = DLRM._Model.dot_interaction_reference
    top_mlp = Flux.Chain(
        Flux.Dense(py_dense2_weights, py_dense2_bias, Flux.relu),
        Flux.Dense(py_dense3_weights, py_dense3_bias, Flux.σ),
    )

    reference_dlrm = DLRM.DLRMModel(bottom_mlp, embedding_tables, interaction, top_mlp)

    # bottom mlp
    reference_bottom_mlp_output = reference_dlrm.bottom_mlp(transpose(py_dense_input))
    @test isapprox(reference_bottom_mlp_output, transpose(py_bottom_mlp_output))

    # embedding table
    reference_embedding_outputs = DLRM.maplookup(
        reference_dlrm.embeddings, py_sparse_input
    )

    for (py_ref, jl_ref) in zip(py_embedding_outputs, reference_embedding_outputs)
        @test isapprox(py_ref', jl_ref)
    end

    # interaction
    reference_interaction = reference_dlrm.interaction(
        reference_bottom_mlp_output, reference_embedding_outputs
    )

    @test isapprox(reference_interaction, py_interaction_output')

    # finally, the last dense layer
    reference_top_mlp_output = reference_dlrm.top_mlp(reference_interaction)
    @test isapprox(reference_top_mlp_output, py_top_mlp_output')
end

