# Do this weird dot-product interaction thingie.
function dot_interaction(X, Ys)
    d, batchsize = size(X)

    # Merge the two into a single array so we can use the fast "vcat" in Base.
    combined = reduce(vcat, vcat([X], Ys))

    # TODO: Make sure the order of this is correct.
    T = reshape(combined, :, d, batchsize)
    Z = NNlib.batched_mul(T, NNlib.batched_transpose(T))

    # Slice out triangular indices of Z into a single 2D slice
    Zflat = triangular_slice(Z)
    return vcat(X, Zflat)
end

# TODO: Adjoint
function triangular_slice(X::AbstractArray{T,3}) where {T}
    # Compute the output size - don't do self interaction be default

    # Number of rows is the batchsize
    batchsize = size(X, 3)

    # Number of columns computed by counting the number of entries in the lower
    # triangle
    sz1 = size(X, 1)
    ncols = div((sz1 * sz1) - sz1, 2)

    O = similar(X, eltype(X), ncols, batchsize)
    row = 1
    for i in 2:sz1, j in 1:i-1
        # Again, descend to pointer madness because views aren't good enough.
        ptrX = pointer(X, batchsize * (sz1 * (j - 1) + i - 1) + 1)
        ptrO = pointer(O, batchsize * (row - 1) + 1)
        unsafe_copyto!(ptrO, ptrX, batchsize)
        row += 1
    end
    return O
end

