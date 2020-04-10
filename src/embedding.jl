abstract type AbstractEmbedding end

struct Embedding{A <: AbstractMatrix} <: AbstractEmbedding
    # The underlying data matrix
    data::A
end

function embedding_ensemble(E::Vector{Embedding{A}}, I::AbstractMatrix) where {A}
    return map(E, eachrow(I)) do e,i
        embedding_lookup(e.data, i)
    end
end

# Mark the parameters of the embedding as requiring gradients
Flux.@functor Embedding

embedding_lookup(A::AbstractMatrix, I) = A[:, I]
#function embedding_lookup(A::AbstractMatrix, I)
#    nrows = size(A, 1)
#    O = similar(A, eltype(A), nrows, length(I))
#    @inbounds for (col, i) in enumerate(I)
#        ptrA = pointer(A, nrows * (i - 1) + 1)
#        ptrO = pointer(O, nrows * (col - 1) + 1)
#        unsafe_copyto!(ptrO, ptrA, nrows)
#    end
#    return O
#end


# TODO: Make more general so we can pump this through the various optimizers correctly.
struct LazyEmbeddingUpdate{T <: AbstractVector{<:Integer}, A <: AbstractArray}
    indices::T
    Δ::A
end

# TODO
# function Flux.update!(x::AbstractArray, x̄::EmbeddingUpdate)
#     # Sanity check on sizes
#     @assert size(x, 1) == length(first(values(x̄.cols)))
#
#     # Perform the update
#     @inbounds for (col, update) in x̄.cols
#         for row in axes(x, 1)
#             x[row, col] -= update[row]
#         end
#     end
#     return nothing
# end

Zygote.@adjoint function embedding_ensemble(E, I)
    return (
        embedding_ensemble(E, I),
        Δ -> (LazyEmbeddingUpdate.(eachrow(I),Δ), nothing),
    )
end

