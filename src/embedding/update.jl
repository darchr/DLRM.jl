#####
##### Updates
#####

### Non-reduction case.
function Flux.Optimise.update!(
    x::AbstractEmbeddingTable,
    xbar::SparseEmbeddingUpdate{Dynamic,A,I},
    numcols::Integer
) where {A,I <: AbstractVector}
    for (src_col, dst_col) in enumerate(view(xbar.indices, Base.OneTo(numcols)))
        # Update on a column-by-column basis
        update = columnview(xbar.delta, src_col)
        v = columnview(x, dst_col)
        v .-= update
    end
    return nothing
end

@generated function Flux.Optimise.update!(
    x::AbstractEmbeddingTable{Static{N},T} ,
    xbar::SparseEmbeddingUpdate{Static{N},A,I},
    numcols::Integer
) where {N,T,A,I <: AbstractVector}
    return emit_update(T, N)
end

#####
##### Optimizers
#####

function Flux.Optimise.update!(opt, x, xbar::SparseEmbeddingUpdate, args...)
    return Flux.update!(x, Flux.Optimise.apply!(opt, x, xbar, args...)...)
end

# Prepare the SparseEmbeddingUpdate by first crunching it so we have less to write
# to the actual embedding table.
# During the crunch, we might as well multiply the gradiants by the learning rate
# since we're accessing
function Flux.Optimise.apply!(
        opt::Flux.Descent,
        x,
        xbar::SparseEmbeddingUpdate,
        translation = Dict{Int,Int}()
    )

    eta = convert(eltype(xbar.delta), opt.eta)
    return crunch(xbar, translation; mulby = eta)
end

