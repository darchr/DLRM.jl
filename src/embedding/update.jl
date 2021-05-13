#####
##### Updates
#####

### Non-reduction case.
function Flux.Optimise.update!(
        x::AbstractEmbeddingTable,
        xbar::SparseEmbeddingUpdate{A,I}
    ) where {A,I <: AbstractVector}

    for (col, update) in zip(xbar.indices, eachcol(xbar.delta))
        # Update on a column-by-column basis
        v = columnview(x, col)
        v .-= update
    end
end

### Reducing Case
# Use `Flux.Optimise.update!` to dispatch to either the dynamic implementation or the
# unrolled, optimized static implementation.
#
# These implementations live under `__update!`.
# The static case lives in `simd.jl`.
function Flux.Optimise.update!(
        x::AbstractEmbeddingTable,
        xbar::SparseEmbeddingUpdate{A,I}
    ) where {A,I <: AbstractMatrix}

    return __update!(x, xbar, lookuptype(x))
end

# Dynamic update case
function __update!(x::AbstractEmbeddingTable, xbar, ::Dynamic)
    for col in 1:size(xbar.indices, 2)
        for row in 1:size(xbar.indices, 1)
            @inbounds slice = xbar.indices[row, col]

            # Why doesn't LLVM vectorize this?
            # Probably because it doesn't know that `x` and `xbar` don't alias?
            @inbounds for offset in 1:featuresize(x)
                x[offset, slice] -= xbar.delta[offset, col]
            end
        end
    end
end

#####
##### Optimizers
#####

# For now, just hijack a higher level of the Flux update chain.
# TODO: lazy wrapper for the learning rate to apply in `__update!`.
function Flux.Optimise.update!(opt, x, xbar::SparseEmbeddingUpdate)
    return Flux.update!(x, Flux.Optimise.apply!(opt, x, xbar))
end

function Flux.Optimise.apply!(opt::Flux.Descent, x, xbar::SparseEmbeddingUpdate)
    xbar.delta .*= opt.eta
    return xbar
end

