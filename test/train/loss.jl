@testset "Testing BCE Loss Backprop" begin
    # Make sure our backprop is correct.
    ŷ = rand(Float32, 20)
    y = rand(Float32, 20)

    ref, back_ref = Zygote._pullback(Flux.Losses.binarycrossentropy, ŷ, y)
    z, back_z = Zygote._pullback(DLRM.bce_loss, ŷ, y)

    @test isapprox(ref, z)

    grads_ref = back_ref(one(Float32))
    grads_z = back_z(one(Float32))

    @test length(grads_ref) == length(grads_z) == 3
    @test grads_ref[1] === grads_z[1] === nothing
    @test isapprox(grads_ref[2], grads_z[2])
    @test isapprox(grads_ref[3], grads_z[3])
end
