module _Utils

# Imports
import Flux
import OneDNN

# see: https://github.com/FluxML/ZygoteRules.jl/pull/21
import ZygoteRules: ZygoteRules, _pullback, AContext, literal_getproperty, literal_getfield

function pullback_for_default_literal_getproperty(cx::AContext, x, ::Val{f}) where {f}
    return _pullback(cx, literal_getfield, x, Val{f}())
end

function ZygoteRules._pullback(cx::AContext, ::typeof(literal_getproperty), x::Flux.Chain, ::Val{f}) where {f}
    return pullback_for_default_literal_getproperty(cx, x, Val{f}())
end


# utility functions
export zero!, donothing, default_allocator
zero!(x) = x .= zero(eltype(x))
donothing(x...) = nothing
default_allocator(::Type{T}, dims...) where {T} = Array{T}(undef, dims...)

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
