#####
##### Test Sparse Updates
#####

@testset "Testing Sparse Update" begin
    EmbeddingTables = DLRM._EmbeddingTables

    # We have an optimized version of the reducing lookup when the feature-size of the
    # embedding table is known at compile-time.
    #
    # Here, we test the update routine by comparing with the generic dynamic fallback.
    #
    # STEPS
    # 1. Create a base array to create two embedding tables: a statically sized one and a
    #    "dynamically" sized one.
    #
    # 2. Manually construct a SparseEmbeddingUpdate so we know what the update is going to
    #    be.
    #
    # 3. Run both of the updates and see if we get the same result.
    # @testset "Static-Reduction Update" begin
    #     featuresizes = [16,32,48,64,128,256]
    #     for featuresize in featuresizes
    #         ncols = 100
    #         nlookups = 40
    #         batchsize = 320

    #         base = zeros(Float32, featuresize, ncols)

    #         # Construct dynamic and static tables
    #         dynamic = EmbeddingTables.SimpleEmbedding(copy(base))
    #         static = EmbeddingTables.SimpleEmbedding(copy(base), Val(featuresize))
    #         split_static = EmbeddingTables.SplitEmbedding(copy(base), 10)

    #         # Manually construct an update
    #         indices = rand(1:ncols, nlookups, batchsize)
    #         delta = randn(Float32, featuresize, batchsize)
    #         update = EmbeddingTables.SparseEmbeddingUpdate(delta, indices)

    #         # Apply the update to the dynamic and static tables
    #         Flux.Optimise.update!(dynamic, update)
    #         Flux.Optimise.update!(static, update)
    #         Flux.Optimise.update!(split_static, update)

    #         @test isapprox(dynamic, static)
    #         @test isapprox(static, split_static)
    #     end
    # end
end

#####
##### Update Pipeline
#####

function update_routine(baseline, new, iters; lookups_per_output = 1)
    EmbeddingTables = DLRM._EmbeddingTables
    @test size(baseline) == size(new)
    @test length(baseline) == length(new)

    nrows, ncols = size(new)

    # Create a loss function and loss input
    loss(A, I, x) = Flux.mse(EmbeddingTables.lookup(A, I), x)

    opt = Flux.Descent(0.1)
    for iter in 1:iters
        # Create lookup indices and an example loss.
        #
        # Keep the number of indices to lookup small, but large enough to get the cache-line
        # alignment.
        nindices = 10
        if lookups_per_output == 1
            I = rand(1:ncols, nindices)
        else
            I = [rand(1:ncols) for _ in 1:lookups_per_output, _ in 1:nindices]
        end

        loss_input = randn(Float32, nrows, nindices)

        # Gradient Computations
        grads_baseline = Zygote.gradient(Params((baseline,)))  do
            loss(baseline, I, loss_input)
        end

        grads_new = Zygote.gradient(Params((new,)))  do
            loss(new, I, loss_input)
        end

        # Test that produces updates are approximately equal
        uncompressed = EmbeddingTables.uncompress(grads_new[new], size(baseline, 2))
        @test isapprox(grads_baseline[baseline], uncompressed)

        Flux.Optimise.update!(opt, baseline, grads_baseline.grads[baseline])
        Flux.Optimise.update!(opt, new, grads_new.grads[new])

        # Update should affect both the `baseline` and the `new` table the same.
        equal = isapprox(baseline, new)
        @test equal
        if !equal
            # Find all the mismatching columns.
            mismatch_cols = findall(eachcol(baseline) .!= eachcol(new))
            @show mismatch_cols
            # printstyled(stdout, "Baseline\n"; color = :cyan)
            # display(baseline)
            # printstyled(stdout, "New\n"; color = :cyan)
            # display(new)
            # printstyled(stdout, "Difference\n"; color = :cyan)
            # display(!isapprox.(new, baseline))
            println()
        end
    end
end

@testset "Testing Crunch" begin
    EmbeddingTables = DLRM._EmbeddingTables

    delta = rand(Float32, 16, 5)
    delta_old = copy(delta)

    indices = [4,1,4,2,1]
    # Idiot check
    @test length(indices) == size(delta, 2)

    update = EmbeddingTables.SparseEmbeddingUpdate{EmbeddingTables.Static{size(delta,1)}}(
        delta,
        indices,
    )
    newlength = EmbeddingTables.crunch!(update)
    @test newlength == length(unique(indices))
    @test view(update.indices, 1:newlength)  == unique(indices)
    @test view(delta, :, 1) == delta_old[:,1] + delta_old[:,3]
    @test view(delta, :, 2) == delta_old[:,2] + delta_old[:,5]
    @test view(delta, :, 3) == delta_old[:,4]
end

@testset "Testing Update" begin
    EmbeddingTables = DLRM._EmbeddingTables

    nrows = [64, 80, 128]
    ncols = 100
    numtests = 10

    @testset "Simple" begin
        for rows in nrows
            # # Dynamic
            # base = randn(Float32, rows, ncols)
            # A = copy(base)
            # B = EmbeddingTables.SimpleEmbedding(copy(base))
            # update_routine(A, B, numtests)

            # Static
            base = randn(Float32, rows, ncols)
            A = copy(base)
            B = EmbeddingTables.SimpleEmbedding(copy(base), Val(rows))
            update_routine(A, B, numtests)
        end
    end

    @testset "Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]
        for rows in nrows
            base = randn(Float32, rows, ncols)
            for cols_per_chunk in chunk_sizes
                A = copy(base)
                B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
                update_routine(A, B, numtests)
            end
        end
    end

    # @testset "Reducing Simple" begin
    #     for rows in nrows
    #         base = randn(Float32, rows, ncols)
    #         A = copy(base)
    #         B = EmbeddingTables.SimpleEmbedding(copy(base), Val(size(base,1)))
    #         update_routine(A, B, numtests; lookups_per_output = 5)
    #     end
    # end

    # @testset "Reducing Split" begin
    #     chunk_sizes = [10, 20, 30, 40, 50]
    #     for rows in nrows
    #         base = randn(Float32, rows, ncols)
    #         for cols_per_chunk in chunk_sizes
    #             A = copy(base)
    #             B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
    #             update_routine(A, B, numtests; lookups_per_output = 5)
    #         end
    #     end
    # end
end
