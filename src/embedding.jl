abstract type AbstractEmbedding end

struct Embedding{A <: AbstractMatrix} <: AbstractEmbedding
    # The underlying data matrix
    data::A
end

# Mark the parameters of the embedding as requiring gradients
Flux.@functor Embedding

# Standard Embedding Lookup.
(E::Embedding)(I) = embedding_lookup(E.data, I)

function embedding_lookup(A::AbstractMatrix, I::AbstractVector{<:Integer})
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

# TODO: Make more general so we can pump this through the various optimizers correctly.
struct EmbeddingUpdate{T <: AbstractVector}
    # Map from columns to the row updates.
    cols::DataStructures.SortedDict{Int,T}
end

function Flux.update!(x::AbstractArray, x̄::EmbeddingUpdate)
    # Sanity check on sizes
    @assert size(x, 1) == length(first(values(x̄.cols)))

    # Perform the update
    @inbounds for (col, update) in x̄.cols
        for row in axes(x, 1)
            x[row, col] -= update[row]
        end
    end
    return nothing
end

# This is kind of a hack to allow CachedArrays to potentiall hook into this translation
# mechanism.
change_eltype(::AbstractVector, ::Type{T}) where {T} = Vector{T}

# Zygote Adjoints for these layers.
#
# NOTE: This modifies the embedding object directly, and thus stops gradients
# from propogating further.
#
# Heres, Δ is the adjoint sensitivity.
function lazy_embedding_adjoint(I::AbstractVector{<:Integer}, Δ)
    # Construct an embedding update from the indices and deltas.
    cols = DataStructures.SortedDict{Int, change_eltype(I,eltype(Δ))}()
    nrows = size(Δ, 1)
    for (i,v) in zip(I, eachcol(Δ))
        c = get!(cols, i, zeros(eltype(Δ), nrows))
        c .+= v
    end
    return EmbeddingUpdate(cols)
end

Zygote.@adjoint function embedding_lookup(A, I)
    return embedding_lookup(A, I), Δ -> (lazy_embedding_adjoint(I,Δ), nothing)
end

#####
##### Higher Level Embedding Layers
#####

