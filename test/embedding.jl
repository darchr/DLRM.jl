# This is the default way of doing embedding in Julia that we're trying to optimize.
# TODO: Apply this syntax to the Embedding Tables?
reference_embedding(A, I) = A[:, I]

@testset "Testing Embedding Utils" begin
    # Our embedding table
    A = rand(Float32, 16, 1000)
    E = DLRM.Embedding(A)

    # Test across a wide range of lookups.
    indmax = size(A, 2)
    length_max = 32
    ntests = 1000
    for _ in 1:ntests
        I = rand(1:indmax, rand(1:length_max))
        x = E(I)
        y = reference_embedding(A, I)
        @test x == y
    end
end
