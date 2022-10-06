module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore
using LinearAlgebra: factorize

# main function
export implicit, implicit_linear


# ---------------------------------------------------------------------------

# ------- Unpack/Pack ForwardDiff Dual ------

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

# ---------- core methods -----------------

"""
Compute Jacobian vector product b = -B*xdot where B_ij = ∂r_i/∂x_j
This takes in the dual directly for x = (xv, xdot) since it is already formed that way.
"""
function jvp(residual, y, xd, p)
    
    # evaluate residual function
    rd = residual(y, xd, p)  # constant y
    
    # extract partials
    b = -fd_partials(rd) 

    return b

end

"""
Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c
"""
function vjp(residual, y, x, p, v)
    return ReverseDiff.gradient(xtilde -> v' * residual(y, xtilde, p), x)  # instead of using underlying functions in ReverseDiff just differentiate -v^T * r with respect to x to get -v^T * dr/dx
end


"""
compute A_ij = ∂r_i/∂y_j
"""
function drdy_forward(residual, y::AbstractVector, x, p)
    A = ForwardDiff.jacobian(ytilde -> residual(ytilde, x, p), y)
    return A
end

function drdy_forward(residual, y::Number, x, p)  # 1D case
    A = ForwardDiff.derivative(ytilde -> residual(ytilde, x, p), y)
    return A
end


"""
Linear solve A x = b  (where A is computed in drdy and b is computed in jvp).
"""
linear_solve(A, b) = A\b

linear_solve(A::Number, b) = b / A  # scalar division for 1D case

# -----------------------------------------



# ---------- Overloads for implicit ---------

"""
    implicit(solve, residual, x; drdy=drdy_forward, lsolve=linear_solve)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x, p). Solve implicit function for the residual defined below.
- `residual::function`: Either r = residual(y, x, p) or in-place residual!(r, y, x, p). Set residual r, given state y, variables x and fixed parameters p.
- `x::vector{float}`: evaluation point.
- `p::tuple`: fixed parameters
- `drdy::function`: drdy(residual, y, x, p).  Provide (or compute yourself): ∂r_i/∂y_j.  Default is forward mode AD.
- `lsolve::function`: lsolve(A, b).  Linear solve A x = b  (where A is computed in drdy and b is computed in jvp, or it solves A^T x = c where c is computed in vjp).  Default is backslash operator.
"""
function implicit(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

    # ---- check for in-place version and wrap as needed -------
    new_residual = residual
    if applicable(residual, 1.0, 1.0, 1.0, 1.0)  # in-place
        
        function residual_wrap(yw, xw, pw)  # wrap residual function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(eltype(xw), eltype(yw))
            rw = zeros(T, length(yw))  # match type of input variables
            residual(rw, yw, xw, pw)
            return rw
        end
        new_residual = residual_wrap
    end
    
    return implicit(solve, new_residual, x, p, drdy, lsolve)
end



# If no AD, just solve normally.
implicit(solve, residual, x, p, drdy, lsolve) = solve(x, p)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function implicit(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}
    
    # evalute solver
    xv = fd_value(x)
    yv = solve(xv, p)
    
    # solve for Jacobian-vector product
    b = jvp(residual, yv, x, p)

    # comptue partial derivatives
    A = drdy(residual, yv, xv, p)
    
    # linear solve
    ydot = lsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit), solve, residual, x, p, drdy, lsolve)

    # evalute solver
    y = solve(x, p)

    # comptue partial derivatives
    A = drdy(residual, y, x, p)

    function pullback(ybar)
        u = lsolve(A', ybar)
        xbar = vjp(residual, y, x, p, -u)
        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit(solve, residual, x::TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules implicit(solve, residual, x::AbstractVector{<:TrackedReal}, p, drdy, lsolve)


# ------ 1D version ------------

# """
#     implicit_1d(solve, residual, x)

# TODO
# """
# implicit_1d(solve, residual, x) = solve(x)

# function implicit_1d(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}) where {T}
    
#     # evalute solver
#     xv = fd_value(x)
#     yv = solve(xv)

#     # -drdx * xdot
#     rdot = jvp(residual, x, yv)
    
#     # partial derivatives
#     drdy = ForwardDiff.derivative(y -> residual(xv, y), yv)

#     # new derivatives
#     ydot = rdot / drdy

#     # repack in ForwardDiff Dual
#     return pack_dual(yv, ydot, T)
# end


# ------ linear case ------------

"""
    implicit_linear(A, b; lsolve=linear_solve, fact=factorize)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
This version is for linear equations Ay = b

# Arguments
- `A::matrix`, `b::vector`: components of linear system ``A y = b``
- `lsolve::function`: lsolve(A, b). Function to solve the linear system, default is backslash operator.
- `fact::function`: fact(A).  Factorize matrix A and save it so we can reuse it for solving system and solving derivatives.  Default is factorize.
"""
implicit_linear(A, b; lsolve=linear_solve, fact=factorize) = implicit_linear(A, b, lsolve, fact)


# If no AD, just solve normally.
implicit_linear(A, b, lsolve, fact) = lsolve(fact(A), b)

# catch three cases where one or both contain duals
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, fact) where {T} = linear_dual(A, b, lsolve, fact, T)
implicit_linear(A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, fact) where {T} = linear_dual(A, b, lsolve, fact, T)
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, fact) where {T} = linear_dual(A, b, lsolve, fact, T)


# Both A and b contain duals
function linear_dual(A, b, lsolve, fact, T)
    
    # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
    Av = fd_value(A)
    bv = fd_value(b)

    # save factorization since we will perform two linear solves
    Af = fact(Av)
    
    # evalute linear solver
    yv = lsolve(Af, bv)
    
    # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
    rhs = fd_partials(b - A*yv)
    
    # solve for new derivatives
    dy = lsolve(Af, rhs)

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_linear), A, b, lsolve, fact)

    # save factorization
    Af = fact(ReverseDiff.value(A))

    # evalute solver
    y = lsolve(Af, b)
    
    function implicit_pullback(ybar)
        u = lsolve(Af', ybar)
        return NoTangent(), -u*y', u, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b, lsolve, fact)
ReverseDiff.@grad_from_chainrules implicit_linear(A, b::Union{TrackedArray, AbstractArray{<:TrackedReal}}, lsolve, fact)
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b::Union{TrackedArray, AbstractVector{<:TrackedReal}}, lsolve, fact)


end
