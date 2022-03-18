@testset "Testing Utils" begin
    a1 = randn(Float32, 16 * 1024)
    a2 = randn(Float32, 16 * 2048)
    a3 = [1.0f0]

    b1 = copy(a1)
    b2 = copy(a2)
    b3 = copy(a3)

    v = Vector{Pair{Ptr{Float32},Ptr{Float32}}}()
    for i in eachindex(a1, b1)
        push!(v, pointer(a1, i) => pointer(b1, i))
    end
    for i in eachindex(a2, b2)
        push!(v, pointer(a2, i) => pointer(b2, i))
    end
    push!(v, pointer(a3) => pointer(b3))

    # Test success case.
    # Since we constructed "b1" and "b2" as copies of "a1" and "a2", applying the SGD
    # optimizer with an "eta" value of -1 should result in "b1" and "b2" being doubled
    # in value.
    #@test DLRM._Utils.cancompress(v)
    u = DLRM._Utils.simdcompress(v)
    eta = -1.0f0
    DLRM._Utils.sgd!(u, eta)
    @test b1 == 2 .* a1
    @test b2 == 2 .* a2
    @test b3 == 2 .* a3
end
