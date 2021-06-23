# Set to 64 for AVX-512
# Set to 32 for AVX2
const VECTOR_WIDTH_BYTES = 64

# Lookup - nonreducing
@generated function lookup!(
    dst, src::AbstractEmbeddingTable{Static{N}, T}, indices::AbstractVector{<:Integer}
) where {T,N}
    return emit_lookup_2(Float32, N)
#    return emit_lookup(T, div(VECTOR_WIDTH_BYTES, sizeof(T)), N)
end

# # Lookup - reduction
# @generated function lookup!(
#     O, A::AbstractEmbeddingTable{T}, II::AbstractMatrix{<:Integer}, ::Static{N}
# ) where {T,N}
#     return emit_reducing_lookup(T, div(VECTOR_WIDTH_BYTES, sizeof(T)), N)
# end

@generated function __update!(x::AbstractEmbeddingTable{Static{N},T}, xbar) where {T,N}
    return emit_update(T, div(VECTOR_WIDTH_BYTES, sizeof(T)), N)
end

#####
##### emit_lookup
#####

# For the normal lookup, we can use streaming stores to improve our performance
function emit_lookup(::Type{T}, vecwidth::Integer, numelements) where {T}
    if !iszero(mod(numelements, vecwidth))
        error("Static Lookup Size must be a multiple of the vector width: $vecwidth")
    end

    maxunroll = 32
    f = x -> lookup_loop(SIMD.Vec{vecwidth,T}, x)
    inner = unroll(f, sizeof(T) * numelements, maxunroll)

    return quote
        # NEED to make sure the destination array is 64-byte aligned.
        cached_aligned_error(O)

        for (col, i) in enumerate(I)
            @inbounds ptrA = columnpointer(A, i)
            @inbounds ptrO = columnpointer(O, col)
            $(inner...)
        end

        # Since we used non-temporal stores, we need to put a `sfence` here.
        sfence()
    end
end

function sfence()
    str = raw"""
        tail call void asm sideeffect "sfence", "~{memory},~{dirflag},~{fpsr},~{flags}"()
        ret void
        """
    return Base.llvmcall(str, Nothing, Tuple{})
end

function lookup_loop(::Type{T}, unroll) where {T<:SIMD.Vec}
    syms = [Symbol("i_$j") for j = 0:unroll]
    load_exprs = map(0:(unroll - 1)) do j
        x = syms[j + 1]
        offset = sizeof(T) * j
        :($x = vload($T, ptrA + macro_offset + $offset))
    end

    # Use non-remporal stores
    store_exprs = map(0:(unroll - 1)) do j
        x = syms[j + 1]
        offset = sizeof(T) * j
        :(vstore($x, ptrO + macro_offset + $offset, nothing, Val{true}(), Val{true}()))
    end

    return quote
        $(load_exprs...)
        $(store_exprs...)
    end
end

#####
##### emit_reducing_lookup
#####

# Generate a static lookup function.
# Here, `N` for the parameter `Static` defines the feature size of the lookup.
function emit_reducing_lookup(::Type{T}, vecwidth::Integer, numelements) where {T}
    # For now, assert the `numelements` is a multiple of the vector width.
    # Later, we'll deal with loop peeling.
    if !iszero(mod(numelements, vecwidth))
        error("Static Lookup Size must be a multiple of the vector width: $vecwidth")
    end

    maxunroll = 32
    f = x -> reducing_lookup_loop(SIMD.Vec{vecwidth,T}, x)
    inner = unroll(f, sizeof(T) * numelements, maxunroll)

    return quote
        for col = 1:size(II, 2)
            # Get the column pointer
            ptrO = columnpointer(O, col)

            # Dump the inner generated code
            $(inner...)
        end
    end
end

function reducing_lookup_loop(::Type{T}, unroll) where {T<:SIMD.Vec}
    syms = [Symbol("i_$j") for j = 0:unroll]
    zero_exprs = map(0:(unroll - 1)) do j
        x = syms[j + 1]
        offset = sizeof(T) * j
        :($x = zero($T))
    end

    # First, unroll the innermost level
    inner_exprs = map(0:(unroll - 1)) do j
        x = syms[j + 1]
        offset = sizeof(T) * j
        return :($x += vload($T, ptrA + macro_offset + $offset))
    end

    store_exprs = map(0:(unroll - 1)) do j
        x = syms[j + 1]
        offset = sizeof(T) * j
        return :(vstore($x, ptrO + macro_offset + $offset))
    end

    # Then, emit the loop
    return quote
        $(zero_exprs...)
        for row = 1:size(II, 1)
            @inbounds slice = II[row, col]
            @inbounds ptrA = columnpointer(A, slice)
            $(inner_exprs...)
        end
        $(store_exprs...)
    end
end

#####
##### emit_update
#####

function emit_update(::Type{T}, vecwidth::Integer, numelements) where {T}
    if !iszero(mod(numelements, vecwidth))
        error("Static Update Size must be a multiple of the vector width: $vecwidth")
    end

    # Keep at 16 registers because we need the other 16 registers to store intermediate
    # values.
    maxunroll = 16
    f = x -> update_loop(SIMD.Vec{vecwidth,T}, x)
    inner = unroll(f, sizeof(T) * numelements, maxunroll)

    return quote
        for col = 1:size(xbar.indices, 2)
            # Load the set of values to update
            src_ptr = columnpointer(xbar.delta, col)
            $(inner...)
        end
    end
end

function update_loop(::Type{T}, unroll) where {T<:SIMD.Vec}
    syms_preload = [Symbol("i_$j") for j = 0:unroll]
    syms_load = [Symbol("j_$j") for j = 0:unroll]

    # Step 1: Preload the update values into registers
    # TODO: Apply learning parameter here?
    preload_exprs = map(0:(unroll - 1)) do j
        x = syms_preload[j + 1]
        unroll_offset = sizeof(T) * j
        :($x = vload($T, src_ptr + macro_offset + $unroll_offset))
    end

    # Step 2: Fetch the value we are going to update
    load_exprs = map(0:(unroll - 1)) do j
        x = syms_load[j + 1]
        unroll_offset = sizeof(T) * j
        :($x = vload($T, dst_ptr + macro_offset + $unroll_offset))
    end

    # Step 3: Add the values together
    sub_exprs = map(0:(unroll - 1)) do j
        x = syms_preload[j + 1]
        y = syms_load[j + 1]
        :($y -= $x)
    end

    # Step 4: Store the results back
    store_exprs = map(0:(unroll - 1)) do j
        x = syms_load[j + 1]
        unroll_offset = sizeof(T) * j
        :(vstore($x, dst_ptr + macro_offset + $unroll_offset))
    end

    return quote
        $(preload_exprs...)
        for row = 1:size(xbar.indices, 1)
            column_index = @inbounds(xbar.indices[row, col])
            dst_ptr = columnpointer(x, column_index)
            $(load_exprs...)
            $(sub_exprs...)
            $(store_exprs...)
        end
    end
end

#####
##### Unrolling Logic
#####

# Top level unroller
function unroll(f, len::Union{Integer,Symbol}, unroll::Integer)
    exprs = Expr[]
    push!(exprs, :(macro_offset = 0))

    # Emit the reducing while loop for the max unroll
    bytes = unroll * VECTOR_WIDTH_BYTES
    if mustemit(len, bytes)
        loopbody = quote
            $(f(unroll))
            $(adjust(len, bytes))
            macro_offset += $bytes
        end
        push!(exprs, emitloop(len, bytes, loopbody))
    end

    # Potentially adjust the length.
    len = sub(len, bytes * tripcount(len, bytes))

    # Now generate `if` statements to peel the rest of the loop
    unroll = unroll >> 1
    while !iszero(unroll)
        bytes = unroll * VECTOR_WIDTH_BYTES
        if mustemit(len, bytes)
            body = quote
                $(f(unroll))
                $(adjust(len, bytes))
                macro_offset += $bytes
            end
            push!(exprs, emitif(len, bytes, body))
            len = sub(len, bytes)
        end
        unroll = unroll >> 1
    end

    return exprs
end

# Dynamic Case
sub(x::Symbol, v) = x
mustemit(::Symbol, ::Integer) = true
tripcount(::Symbol, ::Integer) = 0
adjust(x::Symbol, bytes::Integer) = :($x -= $bytes)

function emitloop(x::Symbol, y::Integer, ex)
    return quote
        while ($x >= $y)
            $ex
        end
    end
end

function emitif(x::Symbol, y::Integer, ex)
    return quote
        if $x >= $y
            $ex
        end
    end
end

# Static Case
sub(x::Integer, v) = x - v
mustemit(x::Integer, y::Integer) = (x >= y)
tripcount(x::Integer, y::Integer) = div(x, y)
adjust(::Integer, ::Integer) = nothing

function emitloop(x::Integer, y::Integer, ex)
    trip_count = div(x, y)
    return quote
        for _ = 1:($(trip_count))
            $ex
        end
    end
end

function emitif(x::Integer, y::Integer, ex)
    @assert x >= y
    return ex
end

#####
##### version 2
#####

function emit_lookup_2(::Type{T}, numelements::Integer) where {T}
    return quote
        cached_aligned_error(dst)
        f = identity
        for (dst_col, src_col) in enumerate(indices)
            src_ptr = columnpointer(src, src_col)
            dst_ptr = columnpointer(dst, dst_col)
            $(generate_moveto(T, numelements, true))
        end
        sfence()
    end
end

function generate_moveto(::Type{T}, numelements::Integer, store_nontemporal::Bool) where {T}
    # For now, only support optimized movement if the feature size is a "nice" multiple
    # of the AVX vector size.
    bytes_to_move = sizeof(T) * numelements
    @assert iszero(mod(bytes_to_move, VECTOR_WIDTH_BYTES))

    # How many instructions do we need to emit?
    num_instructions = div(bytes_to_move, VECTOR_WIDTH_BYTES)
    vecsize = div(VECTOR_WIDTH_BYTES, sizeof(T))
    vectype = SIMD.Vec{vecsize,T}

    # Rely on LLVM to perform the constant propagation and loop unrolling if it thinks
    # it will be useful.
    var = gensym("x")
    return quote
        for i in Base.OneTo($num_instructions)
            # Apply an arbitrary function "f".
            # Usually, this will be the identity, but might as well add it in so we
            # can support reductions in the future.
            j = i - 1
            $var = f(SIMD.vload($vectype, src_ptr + sizeof($vectype) * j))
            SIMD.vstore(
                $var,
                dst_ptr + sizeof($vectype) * j,
                nothing,
                Val($store_nontemporal),
                Val($store_nontemporal),
            )
        end
    end
end

