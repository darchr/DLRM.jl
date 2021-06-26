# Very simple, basic implementation of an embedding lookup.
struct SimpleEmbedding{S,T,A<:AbstractMatrix{T}} <: AbstractEmbeddingTable{S,T}
    data::A

    # -- Inner constructors
    # Have two flavors - one for dynamic sizes, one for static sizes
    SimpleEmbedding(A::AbstractMatrix{T}) where {T} = new{Dynamic,T,typeof(A)}(A)
    function SimpleEmbedding{Static{N}}(A::AbstractMatrix{T}) where {N,T}
        if !isa(N, Int)
            msg = """
            Expected the type parameter for `Static{N}` to be an Int.
            Instead, it's a $(typeof(N))!
            """
            throw(ArgumentError(msg))
        end

        if N != size(A, 1)
            msg = """
            Parameter `N` should match the number of rows in the passed Matrix.
            Instead, `N = $N` while `size(A,1) = $(size(A, 1))`.
            """
            throw(ArgumentError(msg))
        end
        table = new{Static{N},T,typeof(A)}(A)
        require_cache_alignment(table)
        return table
    end
end

function Base.zeros(x::SimpleEmbedding{S,T}) where {S,T}
    newdata = similar(x.data)
    newdata .= zero(T)
    return SimpleEmbedding{S}(newdata)
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

Base.pointer(A::SimpleEmbedding) = pointer(A.data)
columnpointer(A::SimpleEmbedding, i::Integer) = columnpointer(A.data, i)
example(A::SimpleEmbedding) = A.data

