# Do this weird dot-product interaction thingie.
_nrows(x) = size(x, 1)
function dot_interaction(X, Ys)
    d, batchsize = size(X)

    # Merge the two into a single array so we can use the fast "vcat" in Base.
    combined = vcat(X, Ys...)
    T = reshape(combined, :, d, batchsize)

    # TODO: NNlib eventually will have a more efficient version of this.
    # For now, this is causing significant slowdown ...
    TT = permutedims(T, (2,1,3))
    Z = NNlib.batched_mul(T, TT)

    # Slice out triangular indices of Z into a single 2D slice
    Zflat = triangular_slice(Z)
    return vcat(X, Zflat)
end

function triangular_slice(X::AbstractArray{T,3}) where {T}
    # Compute the output size - don't do self interaction be default

    # First dimension is the batch size
    batchsize = size(X, 3)

    # Number of columns computed by counting the number of entries in the lower
    # triangle
    sz = size(X, 2)
    ncols = div((sz * sz) - sz, 2)

    O = similar(X, eltype(X), ncols, batchsize)
    for batch in 0:(batchsize-1)
        Xbase = batch * sz * sz
        Obase = batch * ncols
        row = 1
        @inbounds for i in 1:(sz-1)
            Xindex = sz * (i-1) + i + 1 + Xbase
            Oindex = row + Obase
            elements_copied = sz - i
            #unsafe_copyto!(O, Oindex, X, Xindex, batchsize)
            copyto!(O, Oindex, X, Xindex, elements_copied)
            row += elements_copied
        end
    end
    return O
end

# Capture the size of the original matrix
# We'll return a symmetric matrix of the result.
function triangular_slice_adjoint(Δ, sz::NTuple{3, Int})
    A = similar(Δ, eltype(Δ), sz)
    batchsize = sz[3]
    nrows = sz[2]
    for batch in 1:batchsize
        for j in axes(A,2), i in axes(A,1)
            # Fast case - fill diagonal with zeros
            if i >= j
                A[i,j,batch] = zero(eltype(Δ))
                continue
            end

            # Index computation galore!
            Δindex = begin
                a, b = (i > j) ? (i,j) : (j,i)

                # This is some indexing magic.
                # See the explanation below for why this shit works.
                start = ( (b-1) * (2 * nrows - b) ) >> 1
                start + (a - b)
            end
            A[i,j,batch] = Δ[Δindex,batch]
        end
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

