#####
##### Reference Implementations
#####

const VI = AbstractVector{<:Integer}
const MI = AbstractMatrix{<:Integer}

lookup(A::AbstractMatrix, I::VI) = A[:, I]

function lookup(A::AbstractMatrix, II::MI)
    _A = [lookup(A, II[:, i]) for i = 1:size(II, 2)]
    _b = [[sum(_a[i, :]) for i = 1:size(_a, 1)] for _a in _A]
    return hcat(_b...)
end

#####
##### Embedding Updates
#####

# A sparse updater for embedding tables.
struct SparseEmbeddingUpdate{S<:AbstractLookupType,A<:AbstractMatrix,I<:AbstractArray}
    delta::A
    indices::I

    function SparseEmbeddingUpdate{S}(delta::A, indices::I) where {S,A,I}
        return new{S,A,I}(delta, indices)
    end
end

function uncompress(
    x::SparseEmbeddingUpdate{<:Any,<:Any,<:AbstractVector}, ncols = maximum(x.indices)
)
    O = similar(x.delta, size(x.delta, 1), ncols)
    O .= zero(eltype(O))
    for (column, update) in zip(x.indices, eachcol(x.delta))
        vO = columnview(O, column)
        vO .+= update
    end
    return O
end

function uncompress(
    x::SparseEmbeddingUpdate{<:Any,<:Any,<:AbstractMatrix}, ncols = maximum(x.indices)
)
    O = similar(x.delta, size(x.delta, 1), ncols)
    O .= zero(eltype(O))
    for (column, update) in zip(eachcol(x.indices), eachcol(x.delta))
        for c in column
            vO = columnview(O, c)
            vO .+= update
        end
    end
    return O
end

# Compress all updates for each column in place.
# The updates to the final embedding table can then all be performed at once.
function crunch!(
    x::SparseEmbeddingUpdate{Static{N},A,<:AbstractVector},
    translation::Dict{Int,Int} = Dict{Int,Int}(),
) where {N,A}
    head = 1
    @unpack delta, indices = x
    empty!(translation)

    for i in Base.OneTo(size(delta, 2))
        target_column = indices[i]
        accumulation_column = get!(translation, target_column, head)
        if accumulation_column == head
            # Move this column to the head pointer and update the `indices` array
            # appropriately.
            # Since we're moving sequentially, we don't have to worry about destroying
            # data.
            _dst = columnview(delta, head)
            _src = columnview(delta, i)
            for i in Base.OneTo(N)
                @inbounds(_dst[i] = _src[i])
            end

            indices[head] = target_column
            head += 1
            continue
        end

        # We already have a column in `delta` for the target destination.
        # Add the next update in place.
        _dst = columnview(delta, accumulation_column)
        _src = columnview(delta, i)
        for i in Base.OneTo(N)
            @inbounds(_dst[i] += _src[i])
        end
    end
    return head - 1
end

#####
##### Custom Implementation
#####

#-- No reduction function
function lookup(A::AbstractEmbeddingTable{S,T}, I::VI) where {S,T}
    nrows = featuresize(A)
    O = similar(example(A), T, nrows, length(I))

    # dispatch based on Static or Dynamic lookup type
    lookup!(O, A, I)
    return O
end

# fallback dynamic implementation
function lookup!(O, A::AbstractEmbeddingTable{Dynamic,T}, I::VI) where {T}
    nrows = featuresize(A)
    for (col, i) in enumerate(I)
        @inbounds ptrA = columnpointer(A, i)
        @inbounds ptrO = columnpointer(O, col)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
end

#-- Reduction function
function lookup(A::AbstractEmbeddingTable{T}, II::MI) where {F,T}
    nrows = size(A, 1)
    O = similar(example(A), T, nrows, size(II, 2))

    # simd generated function
    lookup!(O, A, II)
    return O
end

#-- Pullbacks
function ChainRulesCore.rrule(::typeof(lookup), A::AbstractEmbeddingTable{S}, I) where {S}
    function lookup_pullback(Δ)
        return (
            ChainRulesCore.NoTangent(),
            SparseEmbeddingUpdate{S}(Δ, I),
            ChainRulesCore.NoTangent(),
        )
    end
    return lookup(A, I), lookup_pullback
end

#####
##### maplookup
#####

const VecAET = Vector{<:AbstractEmbeddingTable}

# Dispatch plumbing for-the-win.
_colwrap(I::AbstractVector{<:AbstractVector}) = I
_colwrap(I::AbstractMatrix) = collect(eachcol(I))
maplookup(x::VecAET, i...) = maplookup(DefaultStrategy(), x, i...)
function maplookup(strategy::DefaultStrategy, x::VecAET, I)
    return map(lookup, x, _colwrap(I))
end

# Generate the same pullback regardless of execution strategy.
function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::AbstractExecutionStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    I,
) where {S}
    function maplookup_pullback(Δs)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.@thunk(map(SparseEmbeddingUpdate{S}, Δs, _colwrap(I))),
            ChainRulesCore.NoTangent(),
        )
    end
    result = maplookup(strategy, A, I)
    return result, maplookup_pullback
end

# Note - this is probably a hack that violates some part of Zygote's API - but it's
# kind of the only way I've been able to get this to work reliably.
#
# TODO: Ditch?
function Zygote.accum_param(
    cx::Zygote.Context, v::Vector{<:AbstractEmbeddingTable}, I::AbstractVector
)
    for i in eachindex(v, I)
        Zygote.accum_param(cx, v[i], I[i])
    end
    return I
end

#####
##### Simple Parallel strategy.
#####

struct SimpleParallelStrategy <: AbstractExecutionStrategy end
function maplookup(::SimpleParallelStrategy, x::Vector{<:AbstractEmbeddingTable}, _I)
    out = Vector{typeof(example(x[1]))}(undef, length(x))
    I = _colwrap(_I)
    Threads.@threads for i in eachindex(x, I)
        #Polyester.@batch for i in eachindex(x, I)
        out[i] = lookup(x[i], I[i])
    end
    return out
end

#####
##### Preallocate destinations
#####

# The idea of the preallocation strategy is to essentially merge the "Concat" step with
# the "EmbeddingLookup" step.
#
# This may involve preallocating some space in the first few rows of the destination
# array to make space for inserting the result of the bottom MLP.
struct PreallocationStrategy <: AbstractExecutionStrategy
    # Allow for extra rows to be placed at the beginning of the destination to allow
    # the results of dense computation to be inserted inplace.
    prependrows::Int
end
PreallocationStrategy() = PreallocationStrategy(0)

_batchsize(x::AbstractVector) = length(first(x))
_batchsize(x::AbstractMatrix) = size(x, 1)

function maplookup(
    strategy::PreallocationStrategy, x::Vector{<:AbstractEmbeddingTable{T}}, _I
) where {T}
    # Preallocate destination.
    I = _colwrap(_I)
    rows = featuresize.(x)
    offset = strategy.prependrows
    batchsize = _batchsize(_I)
    data = similar(example(x[1]), strategy.prependrows + sum(rows), _batchsize(_I))

    # For deciding where to index
    rows_sum = cumsum(rows)
    pushfirst!(rows_sum, 0)

    #Threads.@threads for i in eachindex(x, I)
    Polyester.@batch per = thread for i in eachindex(x)
        start = 1 + offset + rows_sum[i]
        stop = offset + rows_sum[i + 1]

        # Create destination view
        O = view(data, start:stop, Base.OneTo(batchsize))
        A = x[i]

        lookup!(O, A, I[i])
    end
    return data
end

function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::PreallocationStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    _I,
) where {S}
    I = _colwrap(_I)
    data = maplookup(strategy, A, I)
    function maplookup_pullback(Δ)
        f = Slicer(strategy.prependrows + 1, 1, Δ)
        δs = map((y, x) -> SparseEmbeddingUpdate{S}(f(featuresize(y)), x), A, I)

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            δs,
            ChainRulesCore.NoTangent(),
        )
    end
    return data, maplookup_pullback
end
