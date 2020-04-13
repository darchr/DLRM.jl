# abstract type AbstractEmbedding end
#
# struct Embedding{A <: AbstractMatrix} <: AbstractEmbedding
#     # The underlying data matrix
#     data::A
# end
#
# function embedding_ensemble(E::Vector, I::AbstractMatrix) where {A}
#     return map(E, collect(eachrow(I))) do e,i
#         embedding_lookup(e.data, i)
#     end
# end
#
# # Mark the parameters of the embedding as requiring gradients
# Flux.@functor Embedding
#
# embedding_lookup(A::AbstractMatrix, I) = A[:, I]
# #function embedding_lookup(A::AbstractMatrix, I)
# #    nrows = size(A, 1)
# #    O = similar(A, eltype(A), nrows, length(I))
# #    @inbounds for (col, i) in enumerate(I)
# #        ptrA = pointer(A, nrows * (i - 1) + 1)
# #        ptrO = pointer(O, nrows * (col - 1) + 1)
# #        unsafe_copyto!(ptrO, ptrA, nrows)
# #    end
# #    return O
# #end
#
# # TODO: Make more general so we can pump this through the various optimizers correctly.
# struct LazyEmbeddingUpdate{T <: AbstractVector{<:Integer}, A <: AbstractArray}
#     indices::T
#     Δ::A
# end
#
# function lazy_embedding_update(indices, Δ)
#     return LazyEmbeddingUpdate(indices, Δ)
# end
#
# function Flux.Optimise.update!(opt, x, x̄::LazyEmbeddingUpdate)
#     return Flux.update!(x, Flux.Optimise.apply!(opt, x, x̄))
# end
#
# function Flux.Optimise.apply!(opt::Flux.Descent, x, Δ::LazyEmbeddingUpdate)
#     Δ.Δ .*= opt.eta
#     return Δ
# end
#
# function Flux.update!(x::AbstractArray, x̄::LazyEmbeddingUpdate)
#     # Perform the update
#     for (col, update) in zip(x̄.indices, eachcol(x̄.Δ))
#         for row in axes(x, 1)
#             x[row, col] -= update[row]
#         end
#     end
#     return nothing
# end
#
# Zygote.@adjoint function embedding_lookup(A, I)
#     return (
#         embedding_lookup(A, I),
#         Δ -> (lazy_embedding_update(I, Δ), nothing),
#     )
#     #     Δ -> begin
#     #         println(typeof(Δ))
#     #         embedding_updates = lazy_embedding_update.(eachrow(I), Δ)
#     #         @show typeof(embedding_updates)
#     #         return (embedding_updates, nothing)
#     #     end
#     # )
# end
#
