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

end
