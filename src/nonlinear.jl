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
    implicit(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x, p). Solve implicit function returning state variable y, for input variables x, and fixed parameters p.
- `residual::function`: Either r = residual(y, x, p) or in-place residual!(r, y, x, p). Set residual r (scalar or vector), given state y (scalar or vector), variables x (vector) and fixed parameters p (tuple).
- `x::vector{float}`: evaluation point.
- `p::tuple`: fixed parameters. default is empty tuple.
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

    return _implicit(solve, new_residual, x, p, drdy, lsolve)
end



# If no AD, just solve normally.
_implicit(solve, residual, x, p, drdy, lsolve) = solve(x, p)



# Overloaded for ForwardDiff inputs, providing exact derivatives using
# Jacobian vector product.
function _implicit(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}

    # evaluate solver
    xv = fd_value(x)
    yv = solve(xv, p)

    # solve for Jacobian-vector product
    b = jvp(residual, yv, x, p)

    # compute partial derivatives
    A = drdy(residual, yv, xv, p)

    # linear solve
    ydot = lsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_implicit), solve, residual, x, p, drdy, lsolve)

    # evaluate solver (create local copy of the output to guard against `y` getting overwritten)
    y = copy(solve(x, p))

    function pullback(ybar)
        A = drdy(residual, y, x, p)
        u = lsolve(A', ybar)
        xbar = vjp(residual, y, x, p, -u)
        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _implicit(solve, residual, x::TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules _implicit(solve, residual, x::AbstractVector{<:TrackedReal}, p, drdy, lsolve)