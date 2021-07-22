# Set to 64 for AVX-512
# Set to 32 for AVX2
# TODO: At some point, make this automatic?

function sfence()
    str = raw"""
        tail call void asm sideeffect "sfence", "~{memory},~{dirflag},~{fpsr},~{flags}"()
        ret void
        """
    return Base.llvmcall(str, Nothing, Tuple{})
end

function cache_aligned_error(A)
    if !iszero(mod(convert(Int, pointer(A)), 64))
        error("Array must be aligned to a cache-line boundary (multiple of 64-bytes)!")
    end
end

#####
##### Non-reducing Lookup and Update
#####

function emit_lookup(::Type{T}, numelements::Integer) where {T}
    return quote
        cache_aligned_error(dst)
        for (dst_col, src_col) in enumerate(indices)
            src_ptr = columnpointer(src, src_col)
            dst_ptr = columnpointer(dst, dst_col)
            $(generate_moveto(T, numelements, true))
        end
        # Since we're using non-temporal stores - ensure all stores are visible before
        # exiting the lookup function.
        sfence()
    end
end

function emit_lookup_reducing(::Type{T}, numelements::Integer) where {T}
    svector = SVector{numelements,T}
    return quote
        cache_aligned_error(dst)
        sz = size(indices, 1)
        for dst_col in Base.OneTo(size(indices, 2))
            # Move the first element to the destination, then sum.
            src_col = @inbounds(indices[1, dst_col])

            accum = unsafe_load(Ptr{$svector}(columnpointer(src, src_col)))
            for offset in 2:sz
                src_col = @inbounds(indices[offset, dst_col])
                accum += unsafe_load(Ptr{$svector}(columnpointer(src, src_col)))
            end
            unsafe_store!(Ptr{$svector}(columnpointer(dst, dst_col)), accum)
        end
    end
end

function emit_update(::Type{T}, numelements::Integer) where {T}
    function emit_sub(var, vectype)
        tmp = gensym()
        return quote
            $tmp = SIMD.vload($vectype, dst_ptr + sizeof($vectype) * j)
            $tmp - $var
        end
    end
    return quote
        src = xbar.delta
        dst = x
        for (src_col, dst_col) in enumerate(view(xbar.indices, Base.OneTo(numcols)))
            src_ptr = columnpointer(src, src_col)
            dst_ptr = columnpointer(dst, dst_col)
            $(generate_moveto(T, numelements, false; injector = emit_sub))
        end
    end
end

function generate_moveto(
    ::Type{T},
    numelements::Integer,
    store_nontemporal::Bool;
    injector = (var, _) -> :($var),
) where {T}
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
    var1 = gensym("x")
    var2 = gensym("y")
    return quote
        for i in Base.OneTo($num_instructions)
            j = i - 1
            $var1 = SIMD.vload($vectype, src_ptr + sizeof($vectype) * j)
            $var2 = $(injector(var1, vectype))
            SIMD.vstore(
                $var2,
                dst_ptr + sizeof($vectype) * j,
                nothing,
                Val($store_nontemporal),
                Val($store_nontemporal),
            )
        end
    end
end

