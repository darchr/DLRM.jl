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

# A sparse updater for embedding tables.
struct SparseEmbeddingUpdate{A<:AbstractMatrix,I<:AbstractArray}
    delta::A
    indices::I
end

function uncompress(
    x::SparseEmbeddingUpdate{<:Any,<:AbstractVector}, ncols = maximum(x.indices)
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
    x::SparseEmbeddingUpdate{<:Any,<:AbstractMatrix}, ncols = maximum(x.indices)
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

#####
##### Custom Implementation
#####

#-- No reduction function
function lookup(A::AbstractEmbeddingTable{T}, I::VI) where {T}
    nrows = featuresize(A)
    O = similar(example(A), T, nrows, length(I))

    # dispatch based on Static or Dynamic lookup type
    lookup!(O, A, I, lookuptype(A))
    return O
end

# fallback dynamic implementation
function lookup!(O, A::AbstractEmbeddingTable{T}, I::VI, ::Dynamic) where {T}
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
    lookup!(O, A, II, lookuptype(A))
    return O
end

#-- Pullbacks
function ChainRulesCore.rrule(::typeof(lookup), A::AbstractEmbeddingTable, I)
    function lookup_pullback(Δ)
        return (
            ChainRulesCore.NO_FIELDS,
            SparseEmbeddingUpdate(Δ, I),
            ChainRulesCore.DoesNotExist(),
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
_colwrap(I::AbstractMatrix) = eachcol(I)
maplookup(x::VecAET, i...) = maplookup(DefaultExecutionStrategy(), x, i...)
function maplookup(strategy::DefaultExecutionStrategy, x::VecAET, I)
    return map(lookup, x, _colwrap(I))
end

# Generate the same pullback regardless of execution strategy.
function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::AbstractExecutionStrategy,
    A::Vector{<:AbstractEmbeddingTable},
    I,
)
    function maplookup_pullback(Δs)
        return (
            ChainRulesCore.NO_FIELDS,
            ChainRulesCore.NO_FIELDS,
            ChainRulesCore.@thunk(map(SparseEmbeddingUpdate, Δs, _colwrap(I))),
            ChainRulesCore.DoesNotExist(),
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
    I = collect(_colwrap(_I))
    Threads.@threads for i in eachindex(x, I)
        out[i] = lookup(x[i], I[i])
    end
    return out
end

#####
##### Preallocate destinations
#####

struct PreallocationStrategy <: AbstractExecutionStrategy
    # Allow for extra rows to be placed at the beginning of the destination to allow
    # the results of dense computation to be inserted inplace.
    prependrows::Int
end
PreallocationStrategy() = PreallocationStrategy(0)

struct ConcatLookup{T}
    data::T
    offset::Int
    rows::Vector{Int}
end

_batchsize(x::AbstractVector) = length(first(x))
_batchsize(x::AbstractMatrix) = size(x, 1)

function maplookup(
    strategy::PreallocationStrategy, x::Vector{<:AbstractEmbeddingTable{T}}, _I
) where {T}
    # Preallocate destination.
    I = collect(_colwrap(_I))
    rows = featuresize.(x)
    offset = strategy.prependrows
    data = similar(example(x[1]), strategy.prependrows + sum(rows), _batchsize(_I))

    # For deciding where to index
    rows_sum = cumsum(rows)
    pushfirst!(rows_sum, 0)

    Threads.@threads for i in eachindex(x, I)
        start = 1 + offset + rows_sum[i]
        stop = offset + rows_sum[i + 1]
        # Create destination view
        O = view(data, start:stop, :)
        A = x[i]
        lookup!(O, A, I[i], lookuptype(A))
        @show start:stop
    end
    return ConcatLookup(data, offset, rows)
end

# Need to slightly modify the "rrule" since it's likely that the resulting sensitivity
# will be a single matrix rather than a vector of matrices.
function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::PreallocationStrategy,
    A::Vector{<:AbstractEmbeddingTable},
    I,
)
    lookup = maplookup(strategy, A, I)
    function maplookup_pullback(Δ)
        f = Slicer(lookup.offset + 1, 1, Δ)
        δs = map(f, lookup.rows)

        return (
            ChainRulesCore.NO_FIELDS,
            ChainRulesCore.NO_FIELDS,
            δs,
            ChainRulesCore.DoesNotExist(),
        )
    end
    return lookup, maplookup_pullback
end
