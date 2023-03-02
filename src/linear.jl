"""
    apply_factorization(A, factfun)

Apply a matrix factorization to the primal portion of a dual number.
Avoids user from needing to add ForwardDiff as a dependency.
`Afactorization = factfun(A)`
"""
function apply_factorization(A::AbstractArray{<:ForwardDiff.Dual{T}}, factfun) where {T}
    Av = fd_value(A)
    return factfun(Av)
end

# just apply factorization normally for case with no Duals
apply_factorization(A, factfun) = factfun(A)

# default factorization
apply_factorization(A) = apply_factorization(A, factorize)

# struct LinearCache{TVF1, TVF2, TMF1, TMF2, TT, TP, TMD}
#     yv::TVF1
#     bv::TVF2
#     rhs::TMF1
#     ydot::TMF2
#     yt::TT
#     yp::TP
#     yd::TMD
# end

# function create_linear_cache(T, ny, nd)

#     yv = zeros(ny)

#     if nd == 0
#         bv = 0.0
#         rhs = 0.0
#         ydot = 0.0
#         yt = 0.0
#         yp = 0.0
#         yd = 0.0
#     else
#         bv = zeros(ny)
#         rhs = zeros(ny, nd)
#         ydot = zeros(ny, nd)
#         yt = Vector{NTuple{nd, Float64}}(undef, ny)
#         yp = Vector{ForwardDiff.Partials{nd, Float64}}(undef, ny)
#         yd = Vector{ForwardDiff.Dual{T}}(undef, ny)
#     end

#     return LinearCache(bv, yv, rhs, ydot, yt, yp, yd)
# end


# create_linear_cache(A, b) = create_linear_cache(Float64, length(b), 0)
# create_linear_cache(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}) where {T} = create_linear_cache(T, length(b), eltype(A).parameters[3])  # length(A[1].partials)
# create_linear_cache(A::AbstractArray{<:ForwardDiff.Dual{T}}, b) where {T} = create_linear_cache(T, length(b), eltype(A).parameters[3])
# create_linear_cache(A, b::AbstractArray{<:ForwardDiff.Dual{T}}) where {T} = create_linear_cache(T, length(b), eltype(b).parameters[3])


"""
    implicit_linear(A, b; lsolve=linear_solve, Af=factorize)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
This version is for linear equations Ay = b

# Arguments
- `A::matrix`, `b::vector`: components of linear system ``A y = b``
- `lsolve::function`: lsolve(A, b). Function to solve the linear system, default is backslash operator.
- `Af::factorization`: An optional factorization of A, useful to override default factorize, or if multiple linear solves will be performed with same A matrix.
"""
implicit_linear(A, b; lsolve=linear_solve, Af=nothing) = _implicit_linear(A, b, lsolve, Af)


# If no AD, just solve normally.
_implicit_linear(A, b, lsolve, Af) = isnothing(Af) ? lsolve(A, b) : lsolve(Af, b)
# function implicit_linear(A, b, lsolve, Af, cache)
#     if isnothing(cache)
#         if isnothing(Af)
#             return lsolve(A, b)
#         else
#             return lsolve(Af, b)
#         end
#     else
#         if isnothing(Af)
#             lsolve(cache.yv, A, b)
#         else
#             lsolve(cache.yv, Af, b)
#         end
#         return cache.yv
#     end
# end

# catch three cases where one or both contain duals
_implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
_implicit_linear(A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
_implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
# implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af, cache) where {T} = isnothing(cache) ? linear_dual(A, b, lsolve, Af, T) : linear_dual(A, b, lsolve, Af, T, cache)
# implicit_linear(A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af, cache) where {T} = isnothing(cache) ? linear_dual(A, b, lsolve, Af, T) : linear_dual(A, b, lsolve, Af, T, cache)
# implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, Af, cache) where {T} = isnothing(cache) ? linear_dual(A, b, lsolve, Af, T) : linear_dual(A, b, lsolve, Af, T, cache)


# implicit_linear!(ydot, A, b; lsolve=linear_solve, Af=nothing, cache=nothing) = implicit_linear!(ydot, A, b, lsolve, Af)


# # If no AD, just solve normally.
# implicit_linear!(ydot, A, b, lsolve, Af) = isnothing(Af) ? lsolve(A, b) : lsolve(Af, b)

# # catch three cases where one or both contain duals
# implicit_linear!(ydot, A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual!(ydot, A, b, lsolve, Af, T)
# implicit_linear!(ydot, A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual!(ydot, A, b, lsolve, Af, T)
# implicit_linear!(ydot, A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, Af) where {T} = linear_dual!(ydot, A, b, lsolve, Af, T)

# Both A and b contain duals
function linear_dual(A, b, lsolve, Af, T)

    # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
    bv = fd_value(b)

    # save factorization since we will perform two linear solves
    Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

    # evaluate linear solver
    yv = lsolve(Afact, bv)

    # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
    rhs = fd_partials(b - A*yv)

    # solve for new derivatives
    ydot = lsolve(Afact, rhs)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end

# function linear_dual(A, b, lsolve, Af, T, cache)

#     # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
#     fd_value!(cache.bv, b)

#     # save factorization since we will perform two linear solves
#     Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

#     # evaluate linear solver
#     lsolve(cache.yv, Afact, cache.bv)

#     # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
#     # rhs = fd_partials(b - A*yv)
#     fd_partials!(cache.rhs, b - A*cache.yv)
#     # map!( (Arow, bel) -> ForwardDiff.partials(bel - Arow*yv), cache.rhs, eachrow(A), b)

#     # solve for new derivatives
#     lsolve(cache.ydot, Afact, cache.rhs)

#     # repack in ForwardDiff Dual
#     return pack_dual!(cache.yd, cache.yp, cache.yt, cache.yv, cache.ydot, T)
# end

# function linear_dual!(yd, A, b, lsolve, Af, T)

#     # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
#     bv = fd_value(b)

#     # save factorization since we will perform two linear solves
#     Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

#     # evaluate linear solver
#     yv = lsolve(Afact, bv)

#     # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
#     rhs = fd_partials(b - A*yv)

#     # solve for new derivatives
#     ydot = lsolve(Afact, rhs)

#     # repack in ForwardDiff Dual
#     println(size(rhs))
#     println(b)
#     println(length(yd))
#     yd = reinterpret(ForwardDiff.Dual{T, Float64, nd}, view(yd, 1:length(yd)))
#     pack_dual!(yd, yv, ydot, T)
#     return yd
# end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_implicit_linear), A, b, lsolve, Af)

    # save factorization
    Afact = isnothing(Af) ? factorize(ReverseDiff.value(A)) : Af

    # evaluate solver
    y = lsolve(Afact, b)

    function implicit_pullback(ybar)
        u = lsolve(Afact', ybar)
        return NoTangent(), -u*y', u, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b, lsolve, Af)
ReverseDiff.@grad_from_chainrules _implicit_linear(A, b::Union{TrackedArray, AbstractArray{<:TrackedReal}}, lsolve, Af)
ReverseDiff.@grad_from_chainrules _implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b::Union{TrackedArray, AbstractVector{<:TrackedReal}}, lsolve, Af)


# function implicit_linear_inplace(A, b, y, Af)
#     Afact = isnothing(Af) ? A : Af
#     ldiv!(y, Afact, b)
# end

# function implicit_linear_inplace(A, b, y::AbstractVector{<:ForwardDiff.Dual{T}}, Af) where {T}

#     # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
#     bv = fd_value(b)
#     yv, ydot = unpack_dual(y)

#     # save factorization since we will perform two linear solves
#     Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

#     # evaluate linear solver
#     ldiv!(yv, Afact, bv)

#     # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
#     rhs = fd_partials(b - A*yv)

#     # solve for new derivatives
#     ldiv!(ydot, Afact, rhs)

#     # reassign y to this value
#     y .= pack_dual(yv, ydot, T)
# end

