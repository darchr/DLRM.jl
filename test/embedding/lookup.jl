function test_routine(baseline, new; lookups_per_output = 1, ntests = 10)
    EmbeddingTables = DLRM._EmbeddingTables

    @test size(baseline) == size(new)
    @test length(baseline) == length(new)

    nrows, ncols = size(new)

    # Generate indices for looking up.
    #
    # Determine whether the lookup struct is an Array or Matrix based on the argument
    # `lookups_per_output`.
    for _ in 1:ntests
        if lookups_per_output == 1
            I = shuffle(1:ncols)
        else
            I = reduce(hcat, [shuffle(1:ncols) for _ in 1:lookups_per_output])
        end

        lookup_baseline = EmbeddingTables.lookup(baseline, I)
        lookup_new = EmbeddingTables.lookup(new, I)

        equal = (lookup_baseline == lookup_new)
        @test equal
        if !equal
            # Find all the mismatching columns.
            mismatch_cols = findall(eachcol(lookup_baseline) .!= eachcol(lookup_new))
            @show mismatch_cols
        end
    end

    # Now, allow repeates in the lookup
    for _ in 1:ntests
        if lookups_per_output == 1
            I = [rand(1:ncols) for _ in 1:20]
        else
            I = [rand(1:ncols) for _ in 1:lookups_per_output, _ in 1:20]
        end

        lookup_baseline = EmbeddingTables.lookup(baseline, I)
        lookup_new = EmbeddingTables.lookup(new, I)
        equal = lookup_baseline == lookup_new
        @test equal
        if !equal
            @show size(baseline)
            @show size(I)
        end
    end
end

@testset "Testing Lookup" begin
    EmbeddingTables = DLRM._EmbeddingTables

    # Run across a range of rows to test the unrolling kernel
    # Throw in the 1504 sized kernel as an oddball
    nrows = [32,64,128,256,512,1024,1504]
    ncols = 1000

    @testset "Testing Standard Simple" begin
        for rows in nrows
            base = rand(Float32, rows, ncols)

            # Dynamic Sized
            A = copy(base)
            B = EmbeddingTables.SimpleEmbedding(copy(base))
            test_routine(A, B)

            # Static Sized
            A = copy(base)
            B = EmbeddingTables.SimpleEmbedding(copy(base), Val(size(base, 1)))
            test_routine(A, B)
        end
    end

    @testset "Testing Standard Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]
        for rows in nrows
            base = rand(Float32, rows, ncols)

            for cols_per_chunk in chunk_sizes
                A = copy(base)
                B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
                test_routine(A, B)
            end
        end
    end

    # @testset "Testing Reducing Simple" begin
    #     for rows in nrows
    #         base = rand(Float32, rows, ncols)

    #         # For now, only the static case is fully implemented
    #         A = copy(base)
    #         B = EmbeddingTables.SimpleEmbedding(copy(base), Val(size(base,1)))
    #         test_routine(A, B; lookups_per_output = 40)
    #     end
    # end

    # @testset "Testing Reducing Split" begin
    #     chunk_sizes = [10, 20, 30, 40, 50]
    #     for rows in nrows
    #         base = rand(Float32, rows, ncols)

    #         for cols_per_chunk in chunk_sizes
    #             A = copy(base)
    #             B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
    #             test_routine(A, B; lookups_per_output = 40)
    #         end
    #     end
    # end
end
