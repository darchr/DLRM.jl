# Do this weird dot-product interaction thingie.
function dot_interaction(X, Ys)
    @show size(X)
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

    # First dimension is the batch size
    batchsize = size(X, 1)

    # Number of columns computed by counting the number of entries in the lower
    # triangle
    sz1 = size(X, 2)
    ncols = div((sz1 * sz1) - sz1, 2)

    O = similar(X, eltype(X), batchsize, ncols)
    row = 1
    @inbounds for j in 1:(sz1-1), i in (j+1):sz1
        # Again, descend to pointer madness because views aren't good enough.
        Xindex = batchsize * (sz1 * (j - 1) + i - 1) + 1
        Oindex = batchsize * (row - 1) + 1
        #unsafe_copyto!(O, Oindex, X, Xindex, batchsize)
        copyto!(O, Oindex, X, Xindex, batchsize)
        row += 1
    end
    return O
end

# Capture the size of the original matrix
# We'll return a symmetric matrix of the result.
function triangular_slice_adjoint(Δ, sz::NTuple{3, Int})
    A = similar(Δ, eltype(Δ), sz)
    batchsize = sz[1]
    nrows = sz[2]
    @inbounds for j in axes(A,3), i in axes(A,2)
        # Fast case - fill diagonal with zeros
        if i == j
            for k in 1:batchsize
                A[k,i,j] = zero(eltype(Δ))
            end
            continue
        end

        # Index computation galore!
        Δindex = begin
            a, b = (i > j) ? (i,j) : (j,i)

            # This is some indexing magic.
            # See the explanation below for why this shit works.
            start = ( (b-1) * (2 * nrows - b) ) >> 1
            batchsize * (start + (a - b - 1)) + 1
        end
        Aindex = batchsize * (nrows * (j - 1) + (i - 1)) + 1
        #unsafe_copyto!(A, Aindex, Δ, Δindex, batchsize)
        copyto!(A, Aindex, Δ, Δindex, batchsize)
    end
    return A
end

# Explanation of that indexing magic.
# Here's a 2D slice of a 3D array,
# pretend the first dimension extrudes behind the screen.
#
# (This extruded dimension is the batchsize)
#
# The area selected by the triangular slicing is shown in the outlined box
# The sliced array is shown on the right, with dividers where we switch columns
# on the original array.
#
#        Original Array (OA)             Sliced Array (SA)
#
#   (1,1)  (1,2)  (1,3)  (1,4)               (2,1)
#  |-----|                                   (3,1)
#  |(2,1)| (2,2)  (2,3)  (2,4)               (4,1)
#  |     |------|                            -----
#  |(3,1)  (3,2)| (3,3)  (3,4)               (3,2)
#  |            |------|                     (4,2)
#  |(4,1)  (4,2)  (4,3)| (4,4)               -----
#  |-------------------|                     (4,3)
#
# The problem we have to answer is: given an (a,b) coordinate in the original matrix,
# with (a > b), how do we index into the sliced array?
#
# 1. Compute a start point.
#
# Looking at `b`  we compute a starting point for the indexing.
# Let N = 4 (the number of rows)
#
# When b=1, we stat indexing SA at 1, when b=2, we start at 4, and when b=3, we start at 6
#
# In table form:
#
# b  start
#
# 1  1      0 + 1
# 2  4      (N-1) + 1
# 3  6      (N-1) + (N-2) + 1
#
# We can generalize this to
#
# start = (b-1)*N - (1 + 2 + ... b-1) + 1
#       = (b-1)*N - (b-1)*b / 2 + 1
#       = (b-1)(2N + b) / 2 + 1
#
# Then, we correct for the `a` offset by noting that the indexing jump from our start point
# is (start + (b - a - 1))
#
# This gives a final index of
#
# (b-1)(2N + b) / 2 + (b - a)
#
# The last step is to adjust for the batchsize in the first dimension, which yields
# 
# batchsize * ( (b-1)(2N+b)/2 + (b-a) - 1 ) + 1

Zygote.@adjoint function triangular_slice(x) 
    return (
        triangular_slice(x), 
        Δ -> (triangular_slice_adjoint(Δ, size(x)),),
    )
end

