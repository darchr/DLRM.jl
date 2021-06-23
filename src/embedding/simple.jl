# Very simple, basic implementation of an embedding lookup.
struct SimpleEmbedding{S,T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{S,T}
    data::A

    # -- Inner constructors
    # Have two flavors - one for dynamic sizes, one for static sizes
    SimpleEmbedding(A::AbstractMatrix{T}) where {T} = new{Dynamic,T,typeof(A)}(A)
    SimpleEmbedding(A::AbstractMatrix{T}, ::Val{N}) where {T,N} = new{Static{N},T,typeof(A)}(A)
end

#####
##### Array Interface
#####

# Implement Array Interface
Base.size(A::SimpleEmbedding) = size(A.data)
Base.getindex(A::SimpleEmbedding, i::Int) = A.data[i]
Base.setindex!(A::SimpleEmbedding, v, i::Int) = (A.data[i] = v)

#####
##### EmbeddingTable Interface
#####

columnpointer(A::SimpleEmbedding, i::Integer) = columnpointer(A.data, i)
example(A::SimpleEmbedding) = A.data

# # Dispatch between static and dynamic modes
# featuresize(x::SimpleEmbedding{T,A,Nothing}) where {T,A} = size(x, 1)
# featuresize(::SimpleEmbedding{T,A,N}) where {T,A,N} = N
#
# lookuptype(::SimpleEmbedding{T,A,Nothing}) where {T,A} = Dynamic()
# lookuptype(::SimpleEmbedding{T,A,N}) where {T,A,N} = Static{N}()

