#####
##### Setup
#####

function non_reducing_lookup(
    table::DLRM.AbstractEmbeddingTable, baseline::Array; numtests = 10
)
    # Is "size" working properly?
    @test size(table) == size(baseline)
    @test length(table) == length(baseline)
    nrows, ncols = size(table)

    ### Test 1: No repeats per lookup.
    for _ in Base.OneTo(numtests)
        indices = shuffle(1:ncols)

        lookup_ref = DLRM.lookup(baseline, indices)
        lookup_test = DLRM.lookup(table, indices)
        @test lookup_ref == lookup_test
    end

    ### Test 2: Allow repeats
    for _ in Base.OneTo(numtests)
        indices = [rand(1:ncols) for _ = 1:ncols]

        lookup_ref = DLRM.lookup(baseline, indices)
        lookup_test = DLRM.lookup(table, indices)
        @test lookup_ref == lookup_test
    end
    return nothing
end

function reducing_lookup(
    table::DLRM.AbstractEmbeddingTable, baseline::Array; numtests = 10
)
    # Is "size" working properly?
    @test size(table) == size(baseline)
    @test length(table) == length(baseline)
    nrows, ncols = size(table)

    ### Test 1: No repeats per lookup.
    lookups_per_output = 12
    for _ in Base.OneTo(numtests)
        indices = reduce(vcat, [shuffle(2:ncols)' for _ in 1:lookups_per_output])
        lookup_ref = DLRM.lookup(baseline, indices)
        lookup_test = DLRM.lookup(table, indices)
        @test lookup_ref == lookup_test
    end

    ### Test 2: Allow repeats
    for _ in Base.OneTo(numtests)
        indices = [rand(1:ncols) for _ in 1:lookups_per_output, _ in 1:ncols]
        lookup_ref = DLRM.lookup(baseline, indices)
        lookup_test = DLRM.lookup(table, indices)
        @test lookup_ref == lookup_test
    end
    return nothing
end

#####
##### Tests
#####

@testset "Testing Lookup" begin
    # Run across a range of rows to test the unrolling kernel
    # Throw in the 1504 sized kernel as an oddball
    nrows = [32, 64, 128, 256, 512, 1024, 1504]
    ncols = 1000

    @testset "Testing Standard Simple" begin
        for rows in nrows
            base = rand(Float32, rows, ncols)

            # Dynamic
            table = DLRM.SimpleEmbedding(copy(base))
            baseline = copy(base)
            non_reducing_lookup(table, baseline)

            # Static Sized
            table = DLRM.SimpleEmbedding{DLRM.Static{rows}}(copy(base))
            baseline = copy(base)
            non_reducing_lookup(table, baseline)
        end
    end

    @testset "Testing Reducing Simple" begin
        for rows in nrows
            base = rand(Float32, rows, ncols)

            # Dynamic
            table = DLRM.SimpleEmbedding(copy(base))
            baseline = copy(base)
            reducing_lookup(table, baseline)

            # Static Sized
            table = DLRM.SimpleEmbedding{DLRM.Static{rows}}(copy(base))
            baseline = copy(base)
            reducing_lookup(table, baseline)
        end
    end

    @testset "Testing Standard Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]
        for rows in nrows
            base = rand(Float32, rows, ncols)

            for cols_per_chunk in chunk_sizes
                table = DLRM.SplitEmbedding(copy(base), cols_per_chunk)
                baseline = copy(base)
                non_reducing_lookup(table, baseline)
            end
        end
    end

    @testset "Testing Reducing Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]
        for rows in nrows
            base = rand(Float32, rows, ncols)

            for cols_per_chunk in chunk_sizes
                table = DLRM.SplitEmbedding(copy(base), cols_per_chunk)
                baseline = copy(base)
                reducing_lookup(table, baseline)
            end
        end
    end
end
