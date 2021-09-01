module _Utils

# Imports
import OneDNN

# utility functions
export zero!, donothing
zero!(x) = x .= zero(eltype(x))
donothing(x...) = nothing

# utility type aliases.
export MemoryAround
const MemoryAround{A} = OneDNN.Memory{<:Any,<:Any,<:A}

export ThreadPool, dynamic_thread, static_thread
include("threading.jl")

#####
##### ObjectCache
#####

export ObjectCache, Cache, return!
struct ObjectCache{T,Args}
    cache::Vector{T}
    args::Args
    lock::ReentrantLock
end

const Cache{T} = ObjectCache{T,Tuple{}}

ObjectCache{T}() where {T} = ObjectCache{T,Tuple{}}(Vector{T}(), (), ReentrantLock())

function Base.getindex(cache::ObjectCache{T}) where {T}
    if !isempty(cache.cache)
        return Base.@lock cache.lock begin
            isempty(cache.cache) ? T(cache.args...) : pop!(cache.cache)
        end
    end
    return T(cache.args...)
end

function return!(cache::ObjectCache{T}, x::T) where {T}
    Base.@lock cache.lock push!(cache.cache, x)
end

end
