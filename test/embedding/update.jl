non_reducing_update(x...; kw...) = _update_inner(i -> (i,), x...; kw...)
reducing_update(x...; kw...) = _update_inner(i -> (i, i), x...; kw...)

function _update_inner(
    f::F, table::DLRM.AbstractEmbeddingTable, baseline::Matrix; numtests = 10
) where {F}
    @test size(table) == size(baseline)
    @test length(table) == length(table)
    nrows, ncols = size(table)

    opt = Flux.Descent(10.0)
    for _ = 1:numtests
        # Generate random lookup indices which may include repeats.
        indices_base = rand(1:ncols, f(ncols)...)
        indices = copy(indices_base)

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

        # Need to create a new closure because "chunch" effectively destroys the old
        # version of "indices" that gets captured by the original pullback.
        out_ref, back_ref = Zygote._pullback(DLRM.lookup, baseline, copy(indices_base))
        out, back = Zygote._pullback(DLRM.lookup, table, copy(indices_base))

        diff_baseline = back_ref(diff_out)[2]
        diff_table = back(diff_out)[2]

        zeros_baseline = zeros(eltype(baseline), size(baseline))
        zeros_table = zeros(table)

        @test isa(zeros_table, DLRM.AbstractEmbeddingTable)
        Flux.Optimise.update!(opt, zeros_baseline, diff_baseline)
        Flux.Optimise.update!(opt, zeros_table, diff_table)
        @test isapprox(zeros_baseline, zeros_table)
    end
end

#####
##### Tests
#####

findcols(v, x::AbstractVector) = findall(isequal(v), x)
function findcols(v, x::AbstractMatrix)
    r = Int[]
    for (i, col) in enumerate(eachcol(x))
        for _ in Base.OneTo(count(isequal(v), col))
            push!(r, i)
        end
    end
    return r
end

@testset "Testing Crunch" begin
    ### Single Lookup Case
    delta = rand(Float32, 16, 5)
    delta_old = copy(delta)

    old_indices = [4, 1, 4, 2, 1]
    indices = copy(old_indices)
    # Idiot check
    @test length(indices) == size(delta, 2)

    update_old = DLRM.SparseEmbeddingUpdate{DLRM.Static{size(delta, 1)}}(delta, indices)
    update, newlength = DLRM._EmbeddingTables.crunch(update_old)
    # Single lookup update should return the same underlying object.
    # Ensure this invariant.
    @test update === update_old

    @test newlength == length(unique(indices))
    @test view(update.indices, 1:newlength) == unique(indices)
    @test view(delta, :, 1) == delta_old[:, 1] + delta_old[:, 3]
    @test view(delta, :, 2) == delta_old[:, 2] + delta_old[:, 5]
    @test view(delta, :, 3) == delta_old[:, 4]

    for i in 1:newlength
        expected = sum(getindex.(Ref(delta_old), :, findcols(update.indices[i], old_indices)))
        @test view(delta, :, i) == expected
    end

    ### Multiple Lookup Case
    delta = rand(Float32, 16, 5)
    delta_old = copy(delta)

    old_indices = [
        4 1 4 2 1;
        5 1 3 3 2;
    ]
    indices = copy(old_indices)

    update_old = DLRM.SparseEmbeddingUpdate{DLRM.Static{size(delta, 1)}}(delta, indices)
    update, newlength = DLRM._EmbeddingTables.crunch(update_old)

    # By necessity, a different object should be returned.
    @test update !== update_old
    @test newlength == length(unique(indices))
    @test view(update.indices, 1:newlength) == unique(indices)
    delta = update.delta

    for i in 1:newlength
        expected = sum(getindex.(Ref(delta_old), :, findcols(update.indices[i], old_indices)))
        @test view(delta, :, i) == expected
    end
end

@testset "Testing Update" begin
    EmbeddingTables = DLRM._EmbeddingTables

    nrows = [64, 80, 128]
    ncols = 100
    numtests = 10

    @testset "Simple Nonreducing" begin
        for rows in nrows
            # Static
            base = randn(Float32, rows, ncols)
            A = DLRM.SimpleEmbedding{DLRM.Static{rows}}(copy(base))
            B = copy(base)
            non_reducing_update(A, B; numtests = numtests)
        end
    end

    @testset "Simple Reducing" begin
        for rows in nrows
            # Static
            base = randn(Float32, rows, ncols)
            A = DLRM.SimpleEmbedding{DLRM.Static{rows}}(copy(base))
            B = copy(base)
            reducing_update(A, B; numtests = numtests)
        end
    end
end
