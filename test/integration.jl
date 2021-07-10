@testset "Testing Against Large PyTorch Reference" begin
    #reference_path = "/home/mark/projects/intermediate_values.hdf5"
    reference_dir = joinpath(@__DIR__, "..", "ref")
    for modelpath in readdir(reference_dir; join = true)
        println("Testing ", modelpath)
        io = HDF5.h5open(modelpath)
        model = DLRM.load_hdf5(io)
        labels, dense, sparse = DLRM.load_inputs(io)

        strategy = DLRM.PreallocationStrategy(size(last(model.bottom_mlp).weights, 2))
        y = DLRM.maplookup(strategy, model.embeddings, sparse)
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
        strategies = [
            # DLRM.DefaultStrategy(),
            # DLRM.SimpleParallelStrategy(),
            DLRM.PreallocationStrategy(size(last(model.bottom_mlp).weights, 2))
        ]

        for strategy in strategies
            @show strategy
            @test DLRM.validate(modelpath, strategy)
        end
    end
end
