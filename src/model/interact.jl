# The basic interaction layer
function dot_interaction(X, Ys)
    d, batchsize = size(X)

    combined = fast_vcat(X, Ys)
    T = reshape(OneDNN.materialize(combined), d, :, batchsize)
    Z = self_batched_mul(T)

    # Slice out triangular indices of Z into a single 2D slice
    Zflat = triangular_slice(Z)
    return OneDNN.concat([X, OneDNN.Memory(Zflat)], 1)
end

#####
##### OneDNN offloads
#####

### Concatenation
# Note: type inference here is a little picky ...
function fast_vcat(X::T, Ys) where {T}
    # Help out inference?
    A = [OneDNN.Memory(i)::T for i in Ys]
    pushfirst!(A, X)
    return OneDNN.concat(A, 1)
end

function ChainRulesCore.rrule(
    ::typeof(fast_vcat), x::AbstractMatrix, ys::Vector{<:AbstractMatrix}
)
    xsize = size(x, 1)
    ysizes = size.(ys, 1)
    out = fast_vcat(x, ys)

    function fast_vcat_pullback(Δ)
        Δmaterialized = OneDNN.materialize(Δ; allowreorder = false)

        # view into 'X'
        δX = view(Δmaterialized, 1:xsize, :)

        # use OneDNN.Slicer to help with type inference.
        f = Slicer(xsize + 1, 1, Δmaterialized)
        δYs = map(f, ysizes)

        return (ChainRulesCore.NO_FIELDS, δX, δYs)
    end

    return out, fast_vcat_pullback
end

# Deal with the `PreallocationStrategy`.
function fast_vcat(X, ys::AbstractMatrix)
    x = OneDNN.materialize(X)

    # Sanity check that the proper amount of space was reserved.
    vy = view(ys, 1:size(x, 1), :)
    #Threads.@threads for i in eachindex(x, vy)
     for i in eachindex(x, vy)
        @inbounds(vy[i] = x[i])
    end
    return ys
end

function ChainRulesCore.rrule(
    ::typeof(fast_vcat), x::AbstractMatrix, ys::AbstractMatrix
)
    xsize = size(x, 1)
    ysize = size(ys, 1)

    out = fast_vcat(x, ys)
    function fast_vcat_pullback(Δ)
        Δmaterialized = OneDNN.materialize(Δ; allowreorder = false)
        Δx = view(Δmaterialized, 1:xsize, :)
        return (ChainRulesCore.NO_FIELDS, Δx, Δ)
    end
    return out, fast_vcat_pullback
end

### Self Interaction

# Note on OneDNN matrix multiplication:
#
# We need to reverse the arguments passed to `matmul` because Julia is column major while
# OneDNN is row major.
function batched_transpose(x::AbstractArray{T,3}) where {T}
    return PermutedDimsArray{T,3,(2, 1, 3),(2, 1, 3),typeof(x)}(x)
end

batched_transpose(x::OneDNN.Memory{OneDNN.Opaque,<:Any,3}) = permutedims(x, (2, 1, 3))

function self_batched_mul(x::OneDNN.Memory, xt::OneDNN.Memory)
    y = OneDNN.matmul(x, xt)
    return OneDNN.materialize(y; allowreorder = false)
end

function self_batched_mul(_x)
    x = OneDNN.Memory(_x)
    xt = batched_transpose(x)
    return self_batched_mul(xt, x)
end

function ChainRulesCore.rrule(::typeof(self_batched_mul), x)
    X = OneDNN.Memory(x)
    Xt = batched_transpose(X)
    y = self_batched_mul(Xt, X)

    back = function self_batched_mul_adjoint(_Δ)
        Δ = OneDNN.Memory(_Δ)
        out = OneDNN.matmul(X, batched_transpose(Δ))

        # Accumulate the results of the second multiplication into `partial` using post-ops
        postops = OneDNN.PostOps()
        OneDNN.appendsum!(postops)
        attributes = OneDNN.Attributes()
        append!(attributes, postops)
        OneDNN.matmul!(out, X, Δ; attributes = attributes)
        return (ChainRulesCore.NO_FIELDS, OneDNN.materialize(out))
    end

    return y, back
end

#####
##### Take the lower triangular slice
#####

function triangular_slice(X::AbstractArray{T,3}) where {T}
    # Compute the output size - don't do self interaction be default
    batchsize = size(X, 3)

    # Number of columns computed by counting the number of entries in the lower
    # triangle
    sz = size(X, 2)
    ncols = div((sz * sz) - sz, 2)

    O = similar(X, eltype(X), ncols, batchsize)
    Threads.@threads for batch in 0:(batchsize - 1)
        Xbase = batch * sz * sz
        Obase = batch * ncols
        row = 1
        @inbounds for i = 1:(sz - 1)
            Xindex = sz * i + Xbase + 1
            Oindex = row + Obase
            elements_copied = i
            copyto!(O, Oindex, X, Xindex, elements_copied)
            row += elements_copied
        end
    end
    return O
end

# Capture the size of the original matrix
# We'll return a symmetric matrix of the result.
function triangular_slice_adjoint(Δ, sz::NTuple{3,Int})
    A = similar(Δ, eltype(Δ), sz)
    batchsize = sz[3]
    nrows = sz[2]
    Threads.@threads for batch = 1:batchsize
        for j in axes(A, 2), i in axes(A, 1)
            # Fast case - fill diagonal with zeros
            if i >= j
                A[i, j, batch] = zero(eltype(Δ))
                continue
            end

            # Index computation galore!
            Δindex = begin
                a, b = i, j

                # This is some indexing magic.
                # See the explanation below for why this works.
                start = a + ((b - 2) * (b - 1)) >> 1
            end
            A[i, j, batch] = Δ[Δindex, batch]
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
#
#
# UPDATE
# The access pattern for the forward pass has been changed from the lower triangle to the
# upper triangle in order to match the pytorch behavior while still maintaining access
# pattern locality
#
#
#        Original Array (OA)             Sliced Array (SA)
#
#   (1,1) |(1,2)  (1,3)  (1,4)               (1,2)
#         +-----+                            (1,3)
#   (2,1)  (2,2)| (2,3)  (2,4)               (2,3)
#               +------+                     -----
#   (3,1)  (3,2)  (3,3)| (3,4)               (1,4)
#                      +-------              (2,4)
#   (4,1)  (4,2)  (4,3)  (4,4)               -----
#                                            (3,4)
#
#
# The problem is now: Given a coordinate `(a,b)` with `a < b`, how do we index into
# the sliced array?
# In this case, the starting point is determined by the triangular numbers.
#
# In table form:
#
# | b   |   start       |
# |-----|---------------|
# | 2   |  1            |
# | 3   |  1 + 1        |
# | 4   |  1 + 1 + 2    |
#
# In general
#
# start = 1 + (1 + 2 + 3 + ... + (b-2))
#       = 1 + ((b - 2) / 2) * (b - 1)
#
# The given `a`, we adjust the start by `a-1`.
# The result is
#
# start = a + ((b - 2) / 2) * (b - 1)
#

function ChainRulesCore.rrule(::typeof(triangular_slice), x)
    sz = size(x)
    function triangular_slice_pullback(Δ)
        return (
            ChainRulesCore.NO_FIELDS, triangular_slice_adjoint(OneDNN.materialize(Δ), sz)
        )
    end
    return triangular_slice(x), triangular_slice_pullback
end

############################################################################################

#####
##### Reference Implementation
#####

function dot_interaction_reference(X, Ys)
    # Get the size of `X`
    d, batchsize = size(X)

    # Concate all arrays together
    combined = vcat(X, Ys...)
    T = reshape(combined, d, :, batchsize)
    Z = self_batched_mul_reference(T)

    # Triangular slice
    Zflat = triangular_slice_reference(Z)

    # Concat
    return vcat(X, Zflat)
end

self_batched_mul_reference(x) = NNlib.batched_mul(NNlib.batched_transpose(x), x)

function triangular_slice_reference(x)
    xflat = map(2:size(x, 2)) do i
        view(x, 1:(i - 1), i, :)
    end
    return vcat(xflat...)
end
