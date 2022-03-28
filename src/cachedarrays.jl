CachedArrays.tostring(::Type{<:DLRMModel}) = "DLRMModel"
CachedArrays.tostring(::Type{<:Flux.Chain}) = "Chain"
CachedArrays.tostring(::Type{<:Zygote.Pullback}) = "Pullback"

# Hacks to get BF16 working with our customized LLVM stuff
function Base.getindex(A::ReadableCachedArray{OneDNN.BFloat16}, i::Int)
    @boundscheck checkbounds(A, i)
    v = CachedArrays.LoadStore.unsafe_custom_load(Ptr{UInt16}(pointer(A)), i)
    return OneDNN.BFloat16(v)
end

function Base.setindex!(A::WritableCachedArray{OneDNN.BFloat16}, v, i::Int)
    @boundscheck checkbounds(A, i)
    return CachedArrays.LoadStore.unsafe_custom_store!(
        Ptr{UInt16}(pointer(A)),
        convert(OneDNN.BFloat16, v).val,
        i,
    )
end

#####
##### CachedArrays Compatility
#####

const UnwritableMemory = MemoryAround{UnwritableCachedArray}
const UnreadableMemory = MemoryAround{UnreadableCachedArray}
CachedArrays.@wrapper OneDNN.Memory array

# Accessibility hooks
# Creating Model
#
# Make the destination writable for initialization.
@annotate function _Model.multithread_init(f, data::UnwritableCachedArray)
    return __recurse__(f, writable(data))
end

@annotate function _Model.singlethread_init(f, data::UnwritableCachedArray)
    return __recurse__(f, writable(data))
end

# Grab the bias that is being returned and convert it to NotBusy.
@annotate function Flux.create_bias(weights::CachedArray, bias::Bool, dims::Integer...)
    return release(__invoke__(weights, bias, dims...))
end

const MaybeTranspose{T} = Union{T,LinearAlgebra.Transpose{<:Any,<:T}}
@annotate function OneDNN._MemoryPtr(x::MaybeTranspose{<:UnreadableCachedArray}, desc)
    return __recurse__(readable(x), desc)
end

# Since OneDNN kernels are long running, we can hook into the "access_pointer" API in order
# to circumvent the need to change the wrapped array type.
#
# However, we need to clean up the `__readable__` call to avoid creating an entire new array
# and instead just use a CachedArray callback to save on some allocations.
@annotate function OneDNN.access_pointer(x::UnreadableCachedArray, offset, ::OneDNN.Reading)
    return pointer(readable(x), offset)
end

@annotate function OneDNN.access_pointer(x::UnwritableCachedArray, offset, ::OneDNN.Writing)
    return pointer(writable(x), offset)
end

# Capture memories coming out of OneDNN kernels and convert them to "NotBusy".
@annotate function OneDNN.kernel_exit_hook(x::MemoryAround{CachedArray})
    return release(x)
end

@annotate function (dot::_Model.DotInteraction)(
    x::UnreadableCachedArray, ys::ReadableCachedArray; kw...
)
    return dot(readable(x), ys; kw...)
end

@annotate function (dot::_Model.DotInteraction)(
    x::UnreadableCachedArray, ys::Vector{<:UnreadableCachedArray}; kw...
)
    return dot(readable(x), map(readable, ys); kw...)
end

@annotate function ChainRulesCore.rrule(
    f::typeof(_Train.bce_loss), y::UnreadableCachedArray, x::CachedArray
)
    return __recurse__(f, readable(y), readable(x))
end

@annotate function _Model.process_batches_back(
    dot::_Model.DotInteraction, Δ::UnreadableCachedArray, x...
)
    return __recurse__(dot, readable(Δ), x...)
end

# Two update flavors.
@annotate function Flux.update!(o::Flux.Descent, x::UnwritableMemory, y::UnreadableMemory)
    return __recurse__(o, writable(x), readable(y))
end

@annotate function Flux.update!(
    o::Flux.Descent, x::UnwritableMemory, ix, y::UnreadableMemory, iy
)
    return __recurse__(o, writable(x), ix, readable(y), iy)
end

