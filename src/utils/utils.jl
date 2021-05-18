module _Utils

export ThreadPool, dynamic_thread, static_thread
include("threading.jl")

#####
##### Slicing
#####

export Slicer

# TODO: See if the #15276 style problems still exist in Julia 1.6.
#
# map` seems to be having Julia issue #15276 is problems when keeping track of where
# we are indexing to create views.
#
# As such, we have to build this `Slicer` struct below in order to give inference
# some help.
mutable struct Slicer{T,N,A<:AbstractArray{T,N}}
    current_index::Int
    concat_dim::Int
    captured_array::A
end

function (S::Slicer{T,N})(sz) where {T,N}
    current_index = S.current_index
    range = current_index:(current_index + sz - 1)
    inds = ntuple(
        i -> i == S.concat_dim ? range : Base.OneTo(size(S.captured_array, i)), Val(N)
    )
    S.current_index += sz
    return view(S.captured_array, inds...)
end

end
