# Zygote.jl has problems with adjoints for broadcasting.
# Here, we define some custom adjoints to help that issue.
#
# The definitions we use here are things that are specifically used in the DLRM code.
# zeroone(x) = x > zero(x) ? one(x) : zero(x)
#
# Zygote.@adjoint function Base.broadcasted(::typeof(NNlib.relu), x::Zygote.Numeric)
#     broadcast(relu, x), Δ -> (nothing, zeroone.(x))
# end
