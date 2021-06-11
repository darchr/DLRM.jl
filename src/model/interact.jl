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

        return (ChainRulesCore.NoTangent(), δX, δYs)
    end

    return out, fast_vcat_pullback
end

# Deal with the `PreallocationStrategy`.
function fast_vcat(X, ys::AbstractMatrix)
    x = OneDNN.materialize(X)

    # Sanity check that the proper amount of space was reserved.
    vy = view(ys, 1:size(x, 1), :)
    #Threads.@threads for i in eachindex(x, vy)
    @inbounds for i in eachindex(x, vy)
        vy[i] = x[i]
    end
    return ys
end

function ChainRulesCore.rrule(::typeof(fast_vcat), x::AbstractMatrix, ys::AbstractMatrix)
    xsize = size(x, 1)
    out = fast_vcat(x, ys)
    function fast_vcat_pullback(Δ)
        Δmaterialized = OneDNN.materialize(Δ; allowreorder = false)
        Δx = view(Δmaterialized, 1:xsize, :)
        return (ChainRulesCore.NoTangent(), Δx, Δ)
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

    back = function self_batched_mul_back(_Δ)
        Δ = OneDNN.Memory(_Δ)
        Δsum = Δ + batched_transpose(Δ)

        # N.B.: Remember that the order of multiplication is reversed because of the whole
        # column major vs row majer kerfuffle.
        out = OneDNN.matmul(X, Δsum)
        return (ChainRulesCore.NoTangent(), OneDNN.materialize(out))
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
    #Threads.@threads for batch in 0:(batchsize - 1)
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vX = view(X, :, :, batch)
        vO = view(O, :, batch)
        triangular_slice_kernel!(vO, vX)
    end
    return O
end

function triangular_slice_kernel!(O::AbstractVector, X::AbstractMatrix)
    Oindex = 0
    sz = size(X, 2)
    for i in Base.OneTo(sz - 1)
        Xindex = sz * i
        # Give the inner loop a small kick with LoopVectorization.
        LoopVectorization.@turbo for j in Base.OneTo(i)
            O[Oindex + j] = X[Xindex + j]
        end
        Oindex += i
    end
    return nothing
end

# Capture the size of the original matrix
# We'll return a symmetric matrix of the result.
function triangular_slice_back(Δ, sz::NTuple{3,Int})
    A = similar(Δ, eltype(Δ), sz)
    batchsize = sz[3]
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vA = view(A, :, :, batch)
        vΔ = view(Δ, :, batch)
        triangular_slice_back_kernel!(vA, vΔ)
    end
    return A
end

function triangular_slice_back_kernel!(A::AbstractMatrix{T}, O::AbstractVector) where {T}

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        if i >= j
            v = zero(T)
        else
            # Indexing magic.
            # Perform this computation inline to allow for future optimizations
            # that may implement loop reordering.
            m = i + (((j - 2) * (j - 1)) >> 1)
            v = O[m]
        end
        A[i, j] = v
    end
    return nothing
end

function triangular_slice_back_fuse_add_transpose_kernel!(
    A::AbstractMatrix{T}, O::AbstractVector
) where {T}

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        if i == j
            v = zero(T)
        elseif i > j
            m = j + (((i - 2) * (i - 1)) >> 1)
            v = O[m]
        else
            # Indexing magic.
            # Perform this computation inline to allow for future optimizations
            # that may implement loop reordering.
            m = i + (((j - 2) * (j - 1)) >> 1)
            v = O[m]
        end
        A[i, j] = v
    end
    return nothing
end

function ChainRulesCore.rrule(::typeof(triangular_slice), x)
    # Hoist out of closure to avoid capturing `x`.
    sz = size(x)
    function triangular_slice_pullback(Δ)
        return (
            ChainRulesCore.NoTangent(), triangular_slice_back(OneDNN.materialize(Δ), sz)
        )
    end
    return triangular_slice(x), triangular_slice_pullback
end

############################################################################################

# Alternative implementation - perform the whole thing one batch at a time
function gemmavx!(C, A, B)
    LoopVectorization.@turbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

function process_slice!(
    dst::AbstractVector,
    src::AbstractMatrix,
    concat::AbstractVector,
    scratch = similar(src, size(src, 2), size(src, 2)),
)
    # First, perform the concatenation step
    LoopVectorization.@turbo for i in Base.OneTo(length(concat))
        @inbounds dst[i] = concat[i]
    end

    # Then, perform the self matrix multiplication and triangular slice operation.
    gemmavx!(scratch, transpose(src), src)
    offset = length(concat) + 1
    triangular_slice_kernel!(view(dst, offset:length(dst)), scratch)
    return nothing
end

struct LazyCat{T}
    layers::Vector{T}
end

#function Base.view(

#####
##### DotInteraction
#####

# Contains pre-allocated scratchpads to help facilitate faster computation.
struct DotInteraction{T<:AbstractMatrix}
    scratchpads::Vector{T}
end

function DotInteraction(x::T) where {T<:AbstractMatrix}
    return DotInteraction{T}([similar(x, eltype(x), (1, 1)) for _ = 1:Threads.nthreads()])
end

Base.getindex(x::DotInteraction) = x.scratchpads[Threads.threadid()]

function init!(x::DotInteraction{T}, sz) where {T}
    scratchpads = x.scratchpads
    example = first(scratchpads)
    if size(example) != (sz, sz)
        for i in eachindex(scratchpads)
            scratchpads[i] = similar(example, eltype(example), (sz, sz))
        end
    end
    return nothing
end

function (dot::DotInteraction)(X, Ys::AbstractMatrix; return_t = false)
    d, batchsize = size(X)
    x = OneDNN.materialize(X)

    combined = fast_vcat(X, Ys)
    T = reshape(OneDNN.materialize(combined), d, :, batchsize)
    out = process_batches(dot, T, x)

    # Rely on constant propagation to optimize out any instability caused by
    # the contidional return type.
    if return_t
        return out, T
    else
        return out
    end
end

# Define a custom pullback to vectorize the accumulation of "X".
# Otherwrise, the accumulation is single threaded.
function dot_back(dot::DotInteraction, Δ, T, xlen)
    batchsize = size(T, 3)

    dx1, dt = process_batches_back(dot, Δ, T, xlen)
    dt_reshaped = reshape(dt, :, batchsize)

    # Reverse the concatenation process.
    dx2 = view(dt_reshaped, 1:xlen, :)

    # Finally, we need to add together both the contributions to the "x" input.
    dx = dosum(dx1, dx2)
    return dx, dt_reshaped
end

# Helper function.
function dosum(a, b)
    c = similar(a)
    Polyester.@batch for j in axes(c, 2), i in axes(c, 1)
        c[i, j] = a[i, j] + b[i, j]
    end
    return c
end

# Only support the preallocation strategy for now.
function ChainRulesCore.rrule(dot::DotInteraction, X, Y::AbstractMatrix)
    forward, t = dot(X, Y; return_t = true)
    xlen = size(X, 1)

    function dot_pullback(Δ)
        dx, dy = dot_back(dot, Δ, t, xlen)
        return (ChainRulesCore.NoTangent(), dx, dy)
    end
    return forward, dot_pullback
end

function process_batches(dot::DotInteraction, T, x)
    sz = size(T, 2)
    offset, batchsize = size(x)

    dst = similar(x, eltype(x), (div(sz^2 - sz, 2) + offset, batchsize))
    init!(dot, sz)

    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vT = view(T, :, :, batch)
        vdst = view(dst, :, batch)
        vx = view(x, :, batch)
        process_slice!(vdst, vT, vx, dot[])
    end
    return dst
end

function process_batches_back(dot::DotInteraction, Δ::AbstractMatrix{T}, t, xlen) where {T}
    # Take two view, one view is the concatenated version of "x" and the other will be
    # fed back to reverse the slicing process.
    batchsize = size(Δ, 2)
    dx = view(Δ, Base.OneTo(xlen), :)
    remainder = view(Δ, (xlen + 1):size(Δ, 1), :)

    dt = similar(Δ, T, size(t))
    #Polyester.@batch per=core for batch in Base.OneTo(batchsize)
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        # Now, reverse the triangular slice
        scratch = dot[]

        vΔ = view(remainder, :, batch)
        triangular_slice_back_fuse_add_transpose_kernel!(scratch, vΔ)

        # Reverse the matrix multiplication.
        vdt = view(dt, :, :, batch)
        vt = view(t, :, :, batch)
        gemmavx!(vdt, vt, scratch)
    end
    return dx, dt
end

# Unwrap arguments
function process_batches_back(dot::DotInteraction, Δ::OneDNN.Memory, x...)
    return process_batches_back(dot, OneDNN.materialize(Δ), x...)
end

function ChainRulesCore.rrule(::typeof(process_batches), dot::DotInteraction, T, x)
    forward = process_batches(dot, T, x)
    xlen = size(x, 1)
    function process_batches_pullback(Δ)
        dx, dt = process_batches_back(dot, Δ, T, xlen)
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), dt, dx)
    end
    return forward, process_batches_pullback
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
