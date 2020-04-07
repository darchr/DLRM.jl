abstract type AbstractEmbedding end

struct Embedding{A <: AbstractMatrix} <: AbstractEmbedding
    # The underlying data matrix
    data::A
end

# Standard Embedding Lookup.
(E::Embedding)(I) = embedding_lookup(E, I)

function embedding_lookup(E::Embedding, I::AbstractVector{<:Integer})
    A = E.data
    nrows = size(A, 1)
    O = similar(A, eltype(A), nrows, length(I))
    O .= zero(eltype(O))

    @inbounds for (col, i) in enumerate(I)
        ptrA = pointer(A, nrows * (i - 1) + 1)
        ptrO = pointer(O, nrows * (col - 1) + 1)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
    return O
end

# Basically the embedding bag approach.
function (E::Embedding)(f, I::AbstractVector{T}) where {T <: Union{AbstractVector,Tuple}}
    A = E.data
    nrows = size(A, 1)
    O = similar(A, eltype(A), nrows, length(I))
    O .= zero(eltype(O))

    # The compiler is unable to optimize out approaches using views.
    # Manually unroll everything and apply "f".
    @inbounds for (col, indices) in enumerate(I)
        for i in indices
            for row in 1:nrows
                O[row, col] = f(O[row, col], A[row, i])
            end
        end
    end

    return O
end

struct EmbeddingUpdate{T <: AbstractVector}
    # Map from columns to the row updates.
    cols::DataStructures.SortedDict{Int,T}
end

function Flux.update!(x::AbstractArray, x̄::EmbeddingUpdate)
    # Sanity check on sizes
    @assert size(x, 1) == length(first(values(x̄.cols)))

    # Perform the update
    for (col, update) in x̄.cols
        @views x[:, col] .-= update
    end
    return nothing
end

# Zygote Adjoints for these layers.
#
# NOTE: This modifies the embedding object directly, and thus stops gradients
# from propogating further.
#
# Heres, Δ is the adjoint sensitivity.
function lazy_embedding_adjoint(
        I::AbstractVector{<:Integer},
        Δ::V
    ) where {T,N,V <: AbstractArray{T,N}}

    # Construct an embedding update from the indices and deltas.
    cols = DataStructures.SortedDict{Int, V{T,1}}()
    nrows = size(Δ, 1)
    for (i,v) in zip(I, eachcol(Δ))
        c = get!(cols, i, zeros(T, nrows))
        c .+= v
    end
    return cols
    # A = E.data
    # nrows = size(A, 1)
    # # Don't apply `@inbounds` until sure this works correctly.
    # for (col, indices) in enumerate(I)
    #     for row in 1:nrows
    #         A.data[row, i] += Δ[row, col]
    #     end
    # end
    # return (A,)
end

Zygote.@adjoint function (E::Embedding)(I)
    return E(I), Δ -> (embedding_update(I,Δ),)
end

#####
##### Higher Level Embedding Layers
#####

