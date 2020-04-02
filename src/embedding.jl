abstract type AbstractEmbedding end

struct Embedding{A <: AbstractMatrix} <: AbstractEmbedding
    # The underlying data matrix
    data::A
end

# Standard Embedding Lookup.
function (E::Embedding)(I::AbstractVector{<:Integer})
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

