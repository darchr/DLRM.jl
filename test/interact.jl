function symm(A)
    B = copy(A)

    # Zero the diagonal
    for j in axes(B,1), i in 1:j
        if i == j
            B[i,i,:] .= zero(eltype(B))
        else
            B[j,i,:] .= B[i,j,:]
        end
    end
    return B
end

# The easy way of doing things.
#
# I went to all this work - hopefully it's actually faster.
function default_triangular_slice(A)
    sz = size(A,2)

    indices = ((i,j) for j in 1:sz-1 for i in j+1:sz)
    _A = map(x-> A[x[1], x[2],:], indices)
    return reduce(hcat, _A)
end

@testset "Testing Triangular Slicing" begin
    # Our final target is the matrix
    A = [
        0 1 2;
        1 0 3;
        2 3 0;
    ]

    # Turn into a 3D array
    A = reshape(A, (1, 3, 3) )
    A = vcat(A, A)

    # When we slice, we should get 
    # [
    #  1 2 3;
    #  1 2 3;
    # ]
    
    B = [
        1 2 3;
        1 2 3;
    ]

    C = DLRM.triangular_slice(A)
    @test C == B

    # This was constructed to be the case.
    # It doesn't hold in general.
    @test DLRM.triangular_slice_adjoint(C, size(A)) == A

    # Test for different batchsizes and such
    batchsizes = (1, 10, 100)  
    widths = (10,20,40)

    for (batchsize, w) in Iterators.product(batchsizes, widths)
        A = rand(Float32, batchsize, w, w)
        @time DLRM.triangular_slice(A)
        @time default_triangular_slice(A)

        @test DLRM.triangular_slice(A) == default_triangular_slice(A)
    end

    # Time to try some derivatives.
    batchsize = 128
    sz = 48
    target = rand(Float32, batchsize, div(sz * (sz-1), 2))
    X = rand(Float32, batchsize, sz, sz)

    f = (g,t,x) -> Flux.crossentropy(g(x), t)

    for _ in 1:10
        @time grad1 = Zygote.gradient(f, DLRM.triangular_slice, target, X)
    end
end
