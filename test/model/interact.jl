#####
##### Test against reference implementation
#####

@testset "Testing Interaction" begin

    @testset "Testing Triangular Slicing" begin
        # Test docstring examples
        x = [
            1 4 7
            2 5 8
            3 6 9
        ]
        y = zeros(Int, 3)
        DLRM._Model.triangular_slice_kernel!(y, x)
        @test y == [4, 7, 8]

        z = similar(x)
        DLRM._Model.triangular_slice_back_kernel!(z, y)
        @test z == [
            0 4 7
            0 0 8
            0 0 0
        ]

        z = similar(x)
        DLRM._Model.triangular_slice_back_fuse_add_transpose_kernel!(z, y)
        @test z == [
            0 4 7
            4 0 8
            7 8 0
        ]

        # Try some more examples.
        ntests = 20
        for i in range(5; step = 1, length = ntests)
            x = rand(Float32, i, i)
            y = similar(x, Float32, div(i * (i - 1), 2))
            DLRM._Model.triangular_slice_kernel!(y, x)
            y_ref, back = Zygote._pullback(DLRM._Model.triangular_slice_reference, x)
            @test y == y_ref

            # Test some back prop stuff.
            # Gradient returned from "back" is a two-tuple where we're interested in the
            # second element.
            dx_ref = back(y)
            @test length(dx_ref) == 2
            @test dx_ref[1] === nothing
            dx_ref = dx_ref[2]

            dx = similar(dx_ref)
            DLRM._Model.triangular_slice_back_kernel!(dx, y)
            @test dx == dx_ref

            dx = similar(dx_ref)
            DLRM._Model.triangular_slice_back_fuse_add_transpose_kernel!(dx, y)
            @test dx == dx_ref + transpose(dx_ref)
        end

        # Try the batched methods.
        ntests = 20
        for i in range(5; step = 1, length = ntests)
            x = rand(Float32, i, i, i)
            y, back = Zygote._pullback(DLRM._Model.triangular_slice, x)
            y_ref, back_ref = Zygote._pullback(DLRM._Model.triangular_slice_reference, x)
            @test y == y_ref

            # Test some back prop stuff.
            # Gradient returned from "back" is a two-tuple where we're interested in the
            # second element.
            dx_ref = back_ref(y)
            @test length(dx_ref) == 2
            @test dx_ref[1] === nothing
            dx_ref = dx_ref[2]

            dx = back(y)
            @test length(dx) == 2
            @test dx[1] === nothing
            dx = dx[2]

            @test dx == dx_ref
        end
    end

    @testset "Testing fast_vcat" begin
        # Full copy path.
        x = OneDNN.Memory(rand(Float32, 128, 128))
        ys = [rand(Float32, 128, 128) for _ = 1:20]

        z = DLRM._Model.fast_vcat(x, ys)
        @inferred DLRM._Model.fast_vcat(x, ys)
        mycat(x, ys) = vcat(OneDNN.materialize(x), reduce(vcat, ys))
        @test OneDNN.materialize(z) == mycat(x, ys)

        # Check pullbacks
        z, back = Zygote._pullback(DLRM._Model.fast_vcat, x, ys)
        z_ref, back_ref = Zygote._pullback(mycat, x, ys)

        @test OneDNN.materialize(z) == z_ref

        dx = back(z)
        dx_ref = back_ref(z_ref)

        @test dx[1] === dx_ref[1] === nothing
        @test dx[2] == dx_ref[2]
        @test all(dx[3] .== dx_ref[3])

        # Optimized copy path.
        xx = zeros(eltype(x), size(x))
        yys = vcat(xx, reduce(vcat, ys))
        z = DLRM._Model.fast_vcat(parent(x), yys)
        @test z == mycat(x, ys)

        z, back = Zygote._pullback(DLRM._Model.fast_vcat, parent(x), yys)
        @test z == mycat(x, ys)
        dx = back(z)
        @test dx[1] === nothing
        @test dx[2] == dx_ref[2]

        # To compare the pullbacks, we need to take an appropriate view of the large matrix
        # returned by `fast_vcat_pullback` and concat the reference sensitivities together.
        #
        # It's quite a dance ...
        @test view(dx[3], (size(x, 1) + 1):size(dx[3], 1), :) == reduce(vcat, dx_ref[3])
    end

    @testset "Testing Utilities" begin
        # gemmavx!
        Random.seed!(1234)
        for _ = 1:100
            x = rand(Float32, rand(10:100), rand(10:100))
            xt = transpose(x)
            y = similar(x, size(x, 1), size(x, 1))

            DLRM._Model.gemmavx!(y, x, xt)
            @test y == x * xt
        end

        # sumavx
        # Pass in some exotic types to try to trip up LoopVectorization
        x = rand(Float32, 128, 128)
        vx = view(x, 1:12, :)
        vtx = view(transpose(x), 1:12, :)
        y = DLRM._Model.sumavx(vx, vtx)
        @test y == vx + vtx

        # process slice.
        # TODO: Better utilities for generating arrays of the correct size ...
        _len = div(100 * (100 - 1), 2)
        dst = Vector{Float32}(undef, 10 + _len)
        concat = rand(Float32, 10)
        src = rand(Float32, 128, 100)
        DLRM._Model.process_slice!(dst, src, concat)

        ref = vcat(concat, DLRM._Model.triangular_slice_reference(transpose(src) * src))
        @test isapprox(ref, dst)
    end

    @testset "Testing DotInteraction" begin
        # Create a `DotInteraction` struct.
        # The size of the passed array doesn't matter since the contents will be
        # sized appropriately on the first call.
        dot = DLRM._Model.DotInteraction(Matrix{Float32}(undef, 1, 1))

        x = randn(Float32, 128, 1024)
        ys = [randn(Float32, 128, 1024) for _ = 1:10]

        # Construct the mimic array for the PreallocationStrategy output.
        ys_preallocated = vcat(zeros(Float32, size(x)), reduce(vcat, ys))
        @test size(ys_preallocated) == (size(x, 1) + sum(size.(ys, 1)), size(x, 2))
        z = dot(x, ys_preallocated)
        @test !isa(z, Tuple)

        # Test unwrapping of OneDNN.memory
        @test z == dot(OneDNN.Memory(x), ys_preallocated)
        @test isapprox(z, DLRM._Model.dot_interaction_reference(x, ys))

        # Test pullbacks
        z, back = Zygote._pullback(dot, x, ys_preallocated)
        z_ref, back_ref = Zygote._pullback(DLRM._Model.dot_interaction_reference, x, ys)
        @test isapprox(z, z_ref)

        dx = back(z)
        dx_ref = back_ref(z)
        @test dx[1] === dx_ref[1] === nothing
        @test isapprox(dx[2], dx_ref[2])
        @test isapprox(
            view(dx[3], (size(dx[2], 1) + 1):size(dx[3], 1), :), reduce(vcat, dx_ref[3])
        )
    end

    @testset "Testing Interaction Implementation 2" begin
        #####
        ##### Test piece by piece
        #####

        x = randn(Float32, 10, 10)
        ys = [randn(Float32, 10, 10) for _ = 1:5]

        X = OneDNN.Memory(x)

        # Concatenation
        d, batchsize = size(x)

        combined_ref, combined_back_ref = Zygote._pullback(vcat, x, ys...)
        combined_t, combined_back_t = Zygote._pullback(DLRM._Model.fast_vcat, X, ys)

        @test isapprox(combined_ref, OneDNN.materialize(combined_t))

        Δcombined = randn(Float32, size(combined_ref))
        grads_combined_ref = combined_back_ref(Δcombined)
        grads_combined_t = combined_back_t(OneDNN.Memory(Δcombined))

        @test isapprox(grads_combined_ref[2], grads_combined_t[2])
        for i = 1:length(grads_combined_t[3])
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

        batchsize = 512
        x = randn(Float32, 256, batchsize)
        ys = [randn(Float32, 256, batchsize) for _ = 1:20]

        X = OneDNN.Memory(x)

        ref, back_ref = Zygote._pullback(DLRM._Model.dot_interaction_reference, x, ys)
        t, back_t = Zygote._pullback(DLRM._Model.dot_interaction, X, ys)

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
end

