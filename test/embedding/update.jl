function non_reducing_update(
    table::DLRM.AbstractEmbeddingTable, baseline::Matrix; numtests = 10
)
    @test size(table) == size(baseline)
    @test length(table) == length(table)
    nrows, ncols = size(table)

    opt = Flux.Descent(10.0)
    for _ = 1:numtests
        # Generate random lookup indices which may include repeats.
        indices = rand(1:ncols, ncols)

        out_ref, back_ref = Zygote._pullback(DLRM.lookup, baseline, indices)
        out, back = Zygote._pullback(DLRM.lookup, table, indices)

        @test out == out_ref

        # Seed for the sensitivity.
        diff_out = randn(Float32, size(out))

        # The results here have different types, so we can't compare them directly.
        # Instead, we need to use the `uncompress` function to turn the `diff_table`
        # into a full array.
        diff_baseline = back_ref(diff_out)
        @test length(diff_baseline) == 3
        @test diff_baseline[1] === nothing
        @test diff_baseline[3] === nothing
        diff_baseline = diff_baseline[2]

        diff_table = back(diff_out)
        @test length(diff_table) == 3
        @test diff_table[1] === nothing
        @test diff_table[3] === nothing
        diff_table = diff_table[2]

        @test isa(diff_table, DLRM.SparseEmbeddingUpdate)
        uncompressed = DLRM._EmbeddingTables.uncompress(diff_table, size(diff_baseline, 2))
        @test isapprox(diff_baseline, uncompressed)

        # Try crunching and decompressing again, the result should still be the same.
        diff_table, maxindices = DLRM._EmbeddingTables.crunch(diff_table)
        uncompressed = DLRM._EmbeddingTables.uncompress(
            diff_table, size(diff_baseline, 2); maxindices = maxindices
        )
        @test isapprox(diff_baseline, uncompressed)

        # N.B: Zygote is dropping gradients when there are repeated indices.

        # Next - make sure the Flux update pipeline works as expected with the compressed
        # update.
        #
        #
        # diff_table = back(diff_out)[2]
        # zeros_baseline = similar(baseline)
        # zeros_baseline .= zero(eltype(zeros_baseline))
        # zeros_table = zeros(table)

        # Flux.Optimise.update!(opt, zeros_baseline, diff_baseline)
        # Flux.Optimise.update!(opt, zeros_table, diff_table)
        # @test isapprox(zeros_baseline, zeros_table)
    end
end

#####
##### Tests
#####

@testset "Testing Crunch" begin
    delta = rand(Float32, 16, 5)
    delta_old = copy(delta)

    indices = [4, 1, 4, 2, 1]
    # Idiot check
    @test length(indices) == size(delta, 2)

    update = DLRM.SparseEmbeddingUpdate{DLRM.Static{size(delta, 1)}}(delta, indices)
    update, newlength = DLRM._EmbeddingTables.crunch(update)

    @test newlength == length(unique(indices))
    @test view(update.indices, 1:newlength) == unique(indices)
    @test view(delta, :, 1) == delta_old[:, 1] + delta_old[:, 3]
    @test view(delta, :, 2) == delta_old[:, 2] + delta_old[:, 5]
    @test view(delta, :, 3) == delta_old[:, 4]
end

@testset "Testing Update" begin
    EmbeddingTables = DLRM._EmbeddingTables

    nrows = [64, 80, 128]
    ncols = 100
    numtests = 10

    @testset "Simple" begin
        for rows in nrows
            # Static
            base = randn(Float32, rows, ncols)
            A = DLRM.SimpleEmbedding{DLRM.Static{rows}}(copy(base))
            B = copy(base)
            non_reducing_update(A, B; numtests = numtests)
        end
    end
end
