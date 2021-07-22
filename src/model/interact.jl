# We begin with a reference implementation, the build up two more optimized methods.

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

    sz = size(Zflat, 1)
    padded_size = up_to_mul_of(sz, POST_INTERACTION_PAD_TO_MUL)
    padding_amount = padded_size - sz
    pad = zeros(eltype(Zflat), padding_amount, size(Zflat, 2))

    # Concat
    return vcat(X, Zflat, pad)
end

self_batched_mul_reference(x) = NNlib.batched_mul(NNlib.batched_transpose(x), x)

_colons(::Val{N}) where {N} = ntuple(_ -> :, Val(N))
function triangular_slice_reference(x::AbstractArray{T,N}) where {T,N}
    xflat = map(2:size(x, 2)) do i
        view(x, 1:(i - 1), i, _colons(Val(N - 2))...)
    end
    return vcat(xflat...)
end


#####
##### Triangular slicing methods.
#####

"""
$(TYPEDSIGNATURES)
Take the upper triangular slice of `x` and store it into `y`.

# Example
```
julia> x = [
    1 4 7;
    2 5 8;
    3 6 9;
];

julia> y = zeros(Int, 3)
3-element Vector{Int64}:
 0
 0
 0

julia> DLRM._Model.triangular_slice_kernel!(y, x)

julia> y
3-element Vector{Int64}:
 4
 7
 8
```
"""
function triangular_slice_kernel!(y::AbstractVector, x::AbstractMatrix)
    yindex = 0
    sz = size(x, 2)
    for i in Base.OneTo(sz - 1)
        xindex = sz * i
        # Give the inner loop a small kick with LoopVectorization.
        LoopVectorization.@turbo for j in Base.OneTo(i)
            y[yindex + j] = x[xindex + j]
        end
        yindex += i
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Place the entries in `y` on the upper triangle of `x` and zero all other entries in `x`.
This is the reverse operation of [`triangular_slice_kernel`](@ref).
Matrix `x` does not need to be initialized.

# Example
```
julia> x = ones(Int, 3, 3)
3×3 Matrix{Int64}:
 1  1  1
 1  1  1
 1  1  1

julia> y = [1,2,3];

julia> DLRM._Model.triangular_slice_back_kernel!(x, y)

julia> x
3×3 Matrix{Int64}:
 0  1  2
 0  0  3
 0  0  0
```
"""
function triangular_slice_back_kernel!(x::AbstractMatrix{T}, y::AbstractVector) where {T}
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        if i >= j
            v = zero(T)
        else
            # Indexing magic.
            # Perform this computation inline to allow for future optimizations
            # that may implement loop reordering.
            m = i + (((j - 2) * (j - 1)) >> 1)
            v = y[m]
        end
        x[i, j] = v
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Place the entries in `y` on the upper and lower triangles of `x` such that `x` is
symmetric with zeros on the diagonal.

This is a fusion of the operation.
```
julia> DLRM._Model.triangular_slice_back_kernel!(x, y)

julia> x = x + transpose(x)
```

# Example
```
julia> x = ones(Int, 3, 3)
3×3 Matrix{Int64}:
 1  1  1
 1  1  1
 1  1  1

julia> y = [1,2,3];

julia> DLRM._Model.triangular_slice_back_fuse_add_transpose_kernel!(x, y)

julia> x
3×3 Matrix{Int64}:
 0  1  2
 1  0  3
 2  3  0
```
"""
function triangular_slice_back_fuse_add_transpose_kernel!(
    x::AbstractMatrix{T}, y::AbstractVector
) where {T}
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        if i == j
            v = zero(T)
        elseif i > j
            m = j + (((i - 2) * (i - 1)) >> 1)
            v = y[m]
        else
            # Indexing magic.
            # Perform this computation inline to allow for future optimizations
            # that may implement loop reordering.
            m = i + (((j - 2) * (j - 1)) >> 1)
            v = y[m]
        end
        x[i, j] = v
    end
    return nothing
end

# Batched triangular slicing.
function triangular_slice(X::AbstractArray{T,3}) where {T}
    # Compute the output size - don't do self interaction be default
    batchsize = size(X, 3)

    # Number of columns computed by counting the number of entries in the lower
    # triangle
    sz = size(X, 2)
    ncols = div((sz * sz) - sz, 2)

    O = similar(X, eltype(X), ncols, batchsize)
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vX = view(X, :, :, batch)
        vO = view(O, :, batch)
        triangular_slice_kernel!(vO, vX)
    end
    return O
end

# Capture the size of the original matrix
# We'll return a symmetric matrix of the result.
function triangular_slice_back(Δ::AbstractMatrix, sz::NTuple{3,Int})
    A = similar(Δ, eltype(Δ), sz)
    batchsize = sz[3]
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vA = view(A, :, :, batch)
        vΔ = view(Δ, :, batch)
        triangular_slice_back_kernel!(vA, vΔ)
    end
    return A
end

function ChainRulesCore.rrule(::typeof(triangular_slice), x)
    # Hoist size out of closure to avoid capturing `x` unnecessarily.
    sz = size(x)
    function triangular_slice_pullback(Δ)
        return (
            ChainRulesCore.NoTangent(), triangular_slice_back(OneDNN.materialize(Δ), sz)
        )
    end
    return triangular_slice(x), triangular_slice_pullback
end

#####
##### Optimized Concatenation
#####

# """
# $(TYPEDSIGNATURES)
#
# Quickly concatenate `x` and all elements of `ys` along axis 1.
# """
# function fast_vcat(x::MemoryAround{A}, ys::AbstractVector{A}) where {A}
#     z = OneDNN.Memory.(ys)
#     #z = [OneDNN.Memory(i)::typeof(x) for i in ys]
#     pushfirst!(z, x)
#     return OneDNN.concat(z, 1)
# end

#####
##### LazyVcat
#####

struct LazyVcat{T}
    args::Vector{T}
end

Base.reshape(x::LazyVcat, dims) = LazyVcat(map(x -> reshape(x, dims), x.args))
height(x::AbstractArray) = size(x, 2)
height(x::LazyVcat) = sum(height, x.args)
featuresize(x::LazyVcat) = size(first(x.args), 1)

getslice(x::AbstractArray{T,N}, i) where {T,N} = view(x, _colons(Val(N - 1))..., i)
function getslice(x::LazyVcat, i)
    for array in x.args
        h = height(array)
        if h <= i
            result = getslice(array, i)
            return result
        end
        i -= h
    end
    return error("Out of Bounds")
end

function fast_vcat(x::MemoryAround{A}, ys::AbstractVector{A}, docopy = false) where {A}
    return fast_vcat(OneDNN.materialize(x), ys, docopy)
end

function fast_vcat(x::A, ys::AbstractVector{A}, docopy = false) where {A}
    _ys = docopy ? copy(ys) : ys
    pushfirst!(_ys, OneDNN.materialize(x))
    return LazyVcat(_ys)
end

_batchsize(x::LazyVcat) = size(first(x.args), 3)

# Implementation note - the forward pass combines many different arrays into a single
# large array.
#
# Thus, the pullback will recieve one large array.
# Views into this array are returned on the backward to avoid unnecessary copies.
# If a copy is indeed needed, the view can always be `collect`ed.
function ChainRulesCore.rrule(
    ::typeof(fast_vcat), x::A, ys::AbstractVector{A}, docopy = false
) where {A}
    xsize = size(x, 1)
    ysizes = size.(ys, 1)
    out = fast_vcat(x, ys, docopy)

    function fast_vcat_pullback(Δ)
        Δmaterialized = OneDNN.materialize(Δ; allowreorder = false)
        δx = view(Δmaterialized, 1:xsize, :)

        # Use a `Slicer` to air type inference.
        # Begin taking slices after the chunk used as a pullback for `x`.
        f = Slicer(xsize + 1, 1, Δmaterialized)
        δys = map(f, ysizes)

        return (ChainRulesCore.NoTangent(), δx, δys, ChainRulesCore.NoTangent())
    end

    return out, fast_vcat_pullback
end

"""
$(TYPEDSIGNATURES)

Copy `x` into the top of `ys` - assumes `ys` was allocated with this intent in mind.
This is used as the followup to the [`PreallocationStrategy`](@ref) to avoid unecessary
copying.
"""
function fast_vcat(x::AbstractMatrix, ys::AbstractMatrix)
    # IF there's not enough space in `ys`, then this view creation will fail.
    vy = view(ys, 1:size(x, 1), :)

    # Manually unroll to two loops to help Polyester do the right thing.
    #Polyester.@batch per=core for j in axes(vy, 2), i in axes(vy, 1)
    for j in axes(vy, 2), i in axes(vy, 1)
        vy[i, j] = x[i, j]
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

#####
##### Implementation 1
#####

# The strategy for this implementation involves performing the matrix multiplication
# and triangular slice one each entry in the batch as a single fused operation.
#
# This benefits from cache-locality for both the matrix multiplication (small-ish matrix
# sizes) and for the triangular slice (pretty small matrix sizes.)
#
# To make this more efficient, we use LoopVectorization to generate the `gemm` kernel
# (via the `gemmavx!` method). We can't use the default Julia implementation because
# it goes through BLAS and will try to multithread an individual BLAS, but we want
# threading at a different level.
#
# Then, we use a custom slicing kernel to pull out the appropriate triangular slice.
#
# It's worth noting that currently, this implementation only works for the "preallocation"
# strategy which implements the (almost) zero copy post-lookup concatenation.
#
# Copying of the dense result "x" is performed lazily as the first step in
# `process_slice!".

# Thank you LoopVectorization for being awesome.
function gemmavx!(C, A, B)
    LoopVectorization.@turbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

# sumavx
function sumavx(a, b)
    c = similar(a)
    # Manually unroll "eachindex/CartesianIndices" to help Polyester work correctly.
    Polyester.@batch for j in axes(c, 2), i in axes(c, 1)
        c[i, j] = a[i, j] + b[i, j]
    end
    return c
end

function process_slice!(
    dst::AbstractVector{T},
    src::AbstractMatrix,
    concat::AbstractVector,
    padding_amount::Int = 0,
    scratch = similar(src, size(src, 2), size(src, 2)),
) where {T}
    # First, perform the concatenation step
    # TODO: Is there a good way to abstract this?
    LoopVectorization.@turbo for i in Base.OneTo(length(concat))
        @inbounds dst[i] = concat[i]
    end

    # Then, perform the self matrix multiplication and triangular slice operation.
    gemmavx!(scratch, transpose(src), src)
    offset = length(concat) + 1
    dst_length = length(dst)
    triangular_slice_kernel!(view(dst, offset:(dst_length - padding_amount)), scratch)

    # Zero out any extra padding we may have applied.
    if !iszero(padding_amount)
        v = view(dst, (dst_length - padding_amount + 1):dst_length)
        for i in eachindex(v)
            @inbounds v[i] = zero(T)
        end
    end

    return nothing
end

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

function init!(x::DotInteraction{T}, sz1, sz2 = sz1) where {T}
    scratchpads = x.scratchpads
    example = first(scratchpads)
    if size(example) != (sz1, sz2)
        for i in eachindex(scratchpads)
            scratchpads[i] = similar(example, eltype(example), (sz1, sz2))
        end
    end
    return nothing
end

function (dot::DotInteraction)(x::OneDNN.Memory, ys; kw...)
    return dot(OneDNN.materialize(x), ys; kw...)
end

function (dot::DotInteraction)(x::AbstractMatrix, ys; training = false)
    d, batchsize = size(x)
    combined = fast_vcat(x, ys)
    T = reshape(combined, (d, length(ys), :))

    # The "reshape" function should create a new array sharing memory with the old array.
    # This should work for both normal arrays and CachedArrays.
    @assert !isa(T, Base.ReshapedArray)
    out, padding = process_batches(dot, T, x)

    # Rely on constant propagation to optimize out any instability caused by
    # the contidional return type.
    #CachedArrays.softevict!(x)
    #foreach(CachedArrays.softevict!, ys)

    if training
        return out, T, padding
    else
        return out
    end
end

# Define a custom pullback to vectorize the accumulation of "X".
# Otherwrise, the accumulation is single threaded.
function dot_back(dot::DotInteraction, Δ, T, xlen, ysizes, padding)
    batchsize = _batchsize(T)

    dx1, dt = process_batches_back(dot, Δ, T, xlen, padding)
    dt_reshaped = reshape(dt, :, batchsize)

    # Reverse the concatenation process.
    dx2 = view(dt_reshaped, 1:xlen, :)

    # Finally, we need to add together both the contributions to the "x" input.
    dx = sumavx(dx1, dx2)
    f = Slicer(1, 1, dt_reshaped)
    return dx, map(f, ysizes)
end

function ChainRulesCore.rrule(dot::DotInteraction, X, Y)
    forward, t, padding = dot(X, Y; training = true)
    xlen = size(X, 1)
    ysizes = size.(Y, 1)

    function dot_pullback(Δ)
        dx, dy = dot_back(dot, Δ, t, xlen, ysizes, padding)
        return (ChainRulesCore.NoTangent(), dx, dy)
    end
    return forward, dot_pullback
end

function process_batches(dot::DotInteraction, T, x)
    sz = height(T)
    offset, batchsize = size(x)

    # Round so that the size along the smallest dimension is a multiple of 16.
    unpadded_size = div(sz^2 - sz, 2) + offset
    padded_size = up_to_mul_of(unpadded_size, POST_INTERACTION_PAD_TO_MUL)
    padding_amount = padded_size - unpadded_size

    dst = similar(x, eltype(x), (padded_size, batchsize))
    init!(dot, sz)

    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        vT = getslice(T, batch)
        vdst = getslice(dst, batch)
        vx = getslice(x, batch)
        process_slice!(vdst, vT, vx, padding_amount, dot[])
    end
    return dst, padding_amount
end

function process_batches_back(
    dot::DotInteraction, Δ::AbstractMatrix{T}, t, xlen, padding
) where {T}
    # Take two view, one view is the concatenated version of "x" and the other will be
    # fed back to reverse the slicing process.
    batchsize = size(Δ, 2)
    dx = view(Δ, Base.OneTo(xlen), :)
    remainder = view(Δ, (xlen + 1):(size(Δ, 1) - padding), :)

    dt = similar(Δ, T, (featuresize(t), height(t), _batchsize(t)))
    Polyester.@batch per = core for batch in Base.OneTo(batchsize)
        # Now, reverse the triangular slice
        scratch = dot[]

        vΔ = getslice(remainder, batch)
        triangular_slice_back_fuse_add_transpose_kernel!(scratch, vΔ)

        # Reverse the matrix multiplication.
        vdt = getslice(dt, batch)
        vt = getslice(t, batch)
        gemmavx!(vdt, vt, scratch)
    end
    return dx, dt
end

# Unwrap OneDNN arguments
function process_batches_back(dot::DotInteraction, Δ::OneDNN.Memory, x...)
    return process_batches_back(dot, OneDNN.materialize(Δ), x...)
end

#####
##### Implementation 2
#####

# Coarser interaction layer.
# Use OneDNN to implement the batched matrix multiplication then use a custom kernel
# for the triangular slicing.
function dot_interaction(X, Ys)
    d, batchsize = size(X)

    combined = fast_vcat(X, Ys)
    T = reshape(OneDNN.materialize(combined), d, :, batchsize)
    Z = self_batched_mul(T)

    # Slice out triangular indices of Z into a single 2D slice
    Zflat = triangular_slice(Z)
    return OneDNN.concat([X, OneDNN.Memory(Zflat)], 1)
end

### Self Interaction

# Note on OneDNN matrix multiplication:
#
# We need to reverse the arguments passed to `matmul` because Julia is column major while
# OneDNN is row major.
function batched_transpose(x::AbstractArray{T,3}) where {T}
    return PermutedDimsArray{T,3,(2, 1, 3),(2, 1, 3),typeof(x)}(x)
end

batched_transpose(x::OneDNN.Memory{<:Any,3}) = permutedims(x, (2, 1, 3))

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

