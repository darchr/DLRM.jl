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

#-- Adjoints
# Zygote.@adjoint function lookup(A::AbstractEmbeddingTable, I)
#     return lookup(A, I), Δ -> (SparseEmbeddingUpdate(Δ, I), nothing)
# end

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
_rowwrap(I::AbstractVector{<:AbstractVector}) = I
_rowwrap(I::AbstractMatrix) = eachrow(I)
maplookup(x::VecAET, i...) = maplookup(DefaultExecutionStrategy(), x, i...)
function maplookup(strategy::DefaultExecutionStrategy, x::VecAET, I)
    return map(lookup, x, _rowwrap(I))
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
            ChainRulesCore.@thunk(map(SparseEmbeddingUpdate, Δs, _rowwrap(I))),
            ChainRulesCore.DoesNotExist(),
        )
    end
    result = maplookup(strategy, A, I)
    return result, maplookup_pullback
end

# Note - this is probably a hack that violates some part of Zygote's API - but it's
# kind of the only way I've been able to get this to work reliably.
function Zygote.accum_param(
    cx::Zygote.Context, v::Vector{<:AbstractEmbeddingTable}, I::AbstractVector
)
    for i in eachindex(v, I)
        Zygote.accum_param(cx, v[i], I[i])
    end
    return I
end

