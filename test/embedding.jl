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

    # Now, we have to test that our back-prop hooks work correctly.
    A = rand(Float32, 16, 1000)  
    B = copy(A)

    # Just have some random target here.
    batchsize = 200 
    I = rand(1:size(A, 2), batchsize)
    target = rand(Float32, 16, batchsize)

    # Kind of complicated, but this lets us pass in either our `reference_embedding`
    # as `g` or `DLRM.embedding_lookup`.
    f = (x,i,g) -> Flux.crossentropy(g(x,i), target)

    ### Perform the update on A using the reference embedding.
    grad1 = Zygote.gradient(f, A, I, reference_embedding) 

    # Should have no gradients for either the categorical inputs or the function we
    # passed in.
    @test grad1[2] == nothing
    @test grad1[3] == nothing

    # Perform the update operation.
    Flux.update!(A, grad1[1])

    # Now, feed in our custom lookup using B
    grad2 = Zygote.gradient(f, B, I, DLRM.embedding_lookup) 
    @test grad2[2] == nothing
    @test grad2[3] == nothing
    Flux.update!(B, grad2[1])

    @test isapprox(A, B)
end
