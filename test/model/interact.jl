#####
##### Test against reference implementation
#####

@testset "Testing Interaction Reference" begin
    #####
    ##### Test piece by piece
    #####

    x = randn(Float32, 10, 10)
    ys = [randn(Float32, 10, 10) for _ in 1:5]

    X = OneDNN.Memory(x)
    Ys = OneDNN.Memory.(ys)

    # Concatenation
    d, batchsize = size(x)

    combined_ref, combined_back_ref = Zygote._pullback(vcat, x, ys...)
    combined_t, combined_back_t = Zygote._pullback(DLRM._Model.fast_vcat, X, Ys)

    @test isapprox(combined_ref, OneDNN.materialize(combined_t))

    Δcombined = randn(Float32, size(combined_ref))
    grads_combined_ref = combined_back_ref(Δcombined)
    grads_combined_t = combined_back_t(OneDNN.Memory(Δcombined))

    @test isapprox(grads_combined_ref[2], grads_combined_t[2])
    for i in 1:length(grads_combined_t[3])
        @test isapprox(grads_combined_ref[2 + i], grads_combined_t[3][i])
    end

    # Self Batched Mul
    T = reshape(randn(Float32, size(combined_ref)), d, :, batchsize)

    bmm_ref, bmm_back_ref = Zygote._pullback(DLRM._Model.self_batched_mul_reference, T)
    bmm_t, bmm_back_t = Zygote._pullback(DLRM._Model.self_batched_mul, T)
    @test isapprox(bmm_ref, OneDNN.materialize(bmm_t))

    Δbmm = randn(Float32, size(bmm_ref))
    grads_bmm_ref = bmm_back_ref(Δbmm)
    grads_bmm_t = bmm_back_t(Δbmm)
    @test isapprox(grads_bmm_ref[2], grads_bmm_t[2])

    # Triangular Slice
    y_ref, back_ref = Zygote._pullback(DLRM._Model.triangular_slice_reference, bmm_ref)
    y_opt, back_opt = Zygote._pullback(DLRM._Model.triangular_slice, bmm_ref)
    @test isapprox(y_ref, y_opt)

    dy = randn(Float32, size(y_opt))
    dx_ref = back_ref(dy)
    dx_opt = back_opt(dy)
    @test isapprox(dx_ref[2], dx_opt[2])

    #####
    ##### Test whole pipeline
    #####

    # x = randn(Float32, 256, 128)
    # ys = [randn(Float32, 256, 128) for _ in 1:20]

    x = randn(Float32, 256, 2^15)
    ys = [randn(Float32, 256, 2^15) for _ in 1:20]

    # x = randn(Float32, 10, 5)
    # ys = [randn(Float32, 10, 5) for _ in 1:4]

    X = OneDNN.Memory(x)
    Ys = OneDNN.Memory.(ys)

    ref, back_ref = Zygote._pullback(DLRM._Model.dot_interaction_reference, x, ys)
    t, back_t = Zygote._pullback(DLRM._Model.dot_interaction, X, Ys)

    # Make sure the forward pass at least is the same.
    @test isapprox(ref, OneDNN.materialize(t))

    # See if the backprop is the same.
    Δ = randn(Float32, size(ref))
    grads_ref = back_ref(Δ)
    grads_t = back_t(OneDNN.Memory(Δ))

    # First result should be `nothing`
    @test grads_ref[1] === grads_t[1] === nothing

    # second entry is the gradient for `x`
    @test isapprox(grads_ref[2], grads_t[2])

    # third entry is the gradient for each of the `ys`
    # all should be approximate
    @test length(grads_ref[3]) == length(grads_t[3])
    for (r, t) in zip(grads_ref[3], grads_t[3])
        @test isapprox(r, t)
    end
end
