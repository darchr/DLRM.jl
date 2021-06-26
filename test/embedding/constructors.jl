@testset "Testing Embeddingtable Constructors" begin
    # Make sure we get error for invalid constructors.

    # Feature size does not evenly distribute across cache lines.
    bad_size = rand(Float32, 65, 10)
    good_size = rand(Float32, 64, 10)

    # Make sure we have success
    x = DLRM.SimpleEmbedding{DLRM.Static{64}}(good_size)
    @test size(x) == size(good_size)

    # Paramter mismatch
    @test_throws ArgumentError DLRM.SimpleEmbedding{DLRM.Static{32}}(good_size)
    # Wrong type
    @test_throws ArgumentError DLRM.SimpleEmbedding{DLRM.Static{64.0}}(good_size)
    # bad alignment
    @test_throws ArgumentError DLRM.SimpleEmbedding{DLRM.Static{65}}(bad_size)

    # Creating dynamic tables should pretty much always work.
    x = DLRM.SimpleEmbedding(bad_size)
    @test size(x) == size(bad_size)
    @test isa(x, DLRM.SimpleEmbedding{DLRM._EmbeddingTables.Dynamic})
end
