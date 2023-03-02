# ------- Unpack/Pack ForwardDiff Dual ------
# This section from Mohamed Tarek, @mohamed82008, see https://github.com/JuliaDiff/ForwardDiff.jl/issues/579
fd_value(x) = ForwardDiff.value.(x)
fd_partials(x) = reduce(vcat, transpose.(ForwardDiff.partials.(x)))

"""
unpack ForwardDiff Dual return value and derivative.
"""
function unpack_dual(x)
    xv = fd_value(x)
    dx = fd_partials(x)
    return xv, dx
end

"""
Create a ForwardDiff Dual with value yv, derivatives dy, and Dual type T
"""
pack_dual(yv::AbstractFloat, dy, T) = ForwardDiff.Dual{T}(yv, ForwardDiff.Partials(Tuple(dy)))
pack_dual(yv::AbstractVector, dy, T) = ForwardDiff.Dual{T}.(yv, ForwardDiff.Partials.(Tuple.(eachrow(dy))))

# -----------------------------------------

"""
Linear solve A x = b  (where A is computed in drdy and b is computed in jvp).
"""
linear_solve(A, b) = A\b
linear_solve(A::Number, b) = b / A  # scalar division for 1D case

# # ------- In-Place Unpack/Pack ForwardDiff Dual ------
# fd_value!(val, x::AbstractArray) = map!(ForwardDiff.value, val, x)

# # base case 1: No partials (not a Dual number)
# fd_partials!(partials::AbstractArray, x) where {M} = (partials .= 0)

# # base case 2: Dual number
# function fd_partials!(partials::AbstractVector, x::ForwardDiff.Dual{<:Any, <:Any, N}) where {N}
#     # error case
#     @assert length(partials)==N "Invalid partials array: expected length $N, got $(length(partials))"

#     partials .= x.partials

#     return partials
# end

# # array case
# """
#     fd_partials!(partials::AbstractArray, x::AbstractArray{<:ForwardDiff.Dual})

# Store the partials of `x` in `partials`, such that `partials[i, j, ..., :]`
# are the partials of `x[i, j, ...]`

# ```@example
# import ForwardDiff: Dual

# x = [Dual(1.0, 1.5, 2.0), Dual(3, 3.5, 4.0), Dual(5,5.5,6.0), Dual(7,7.5,8.0)]

# partials = zeros(size(x)..., eltype(x).parameters[3])

# ImplicitAD.fd_partials!(partials, x)
# ```

# ```@example
# import ForwardDiff: Dual

# x = [ Dual(1, 1.25, 1.5, 1.75, 2.0) Dual(2, 2.25, 2.5, 2.75, 3.0) Dual(3, 3.25, 3.5, 3.75, 4.0) Dual(4, 4.25, 4.5, 4.75, 5.0)
#       Dual(5, 5.25, 5.5, 5.75, 6.0) Dual(6, 6.25, 6.5, 6.75, 7.0) Dual(7, 7.25, 7.5, 7.75, 8.0) Dual(8, 8.25, 8.5, 8.75, 9.0) ]

# partials = zeros(size(x)..., eltype(x).parameters[3])

# ImplicitAD.fd_partials!(partials, x)
# ```
# """
# function fd_partials!(partials::AbstractArray{<:Any, M},
#                         x::AbstractArray{<:ForwardDiff.Dual{<:Any, <:Any, N}, L}) where {M, N, L}

#     # error cases
#     @assert M == L+1 "partials array expected to have $(L+1) dimensions; got M=$M"
#     for l in 1:L
#         @assert size(partials, l)==size(x, l) "partials array expected to have size (tuple(size(x)..., N)); got $(size(partials))"
#     end
#     @assert size(partials, M)==N "partials array expected to have size (tuple(size(x)..., N)); got $(size(partials))"

#     ind = CartesianIndices(size(partials)[1:end-1])

#     for (i, dual) in enumerate(x)
#         partials[ind[i], :] .= dual.partials
#     end

#     return partials
# end

# function pack_dual!(y, yv::AbstractFloat, dy, T)
#     y .= ForwardDiff.Dual{T}(yv, ForwardDiff.Partials(Tuple(dy)))
# end

# function pack_dual!(y, yp, yt, yv::AbstractVector, dy, T)
#     yt .= Tuple.(eachrow(dy))
#     yp .= ForwardDiff.Partials.(yt)
#     y .= ForwardDiff.Dual{T}.(yv, yp)
# end

# # -----------------------------------------

# function linear_solve(y, A, b)
#     y .= A\b
#     return nothing
# end