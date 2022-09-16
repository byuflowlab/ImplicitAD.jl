module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore

# main function
export implicit_function, ImplicitOptions


# ---------------------------------------------------------------------------

# ------- Unpack/Pack ForwardDiff Dual ------
"""
unpack ForwardDiff Dual return value and derivative.
"""
function unpack_dual(x) 
    xv = ForwardDiff.value.(x)
    dx = reduce(vcat, transpose.(ForwardDiff.partials.(x)))
    return xv, dx
end

"""
Create a ForwardDiff Dual with value yv, derivatives dy, and Dual type T
"""
pack_dual(yv::AbstractFloat, dy, T) = ForwardDiff.Dual{T}(yv, ForwardDiff.Partials(Tuple(dy)))
pack_dual(yv::AbstractVector, dy, T) = ForwardDiff.Dual{T}.(yv, ForwardDiff.Partials.(Tuple.(eachrow(dy))))

# -----------------------------------------

# ---------- core methods -----------------

struct FJVPTag end

"""
Compute Jacobian vector product b = B*v where B_ij = ∂r_i/∂x_j
"""
function jvp_forward(residual, x, y, v)
    
    # create new seed using v
    xd = pack_dual(x, v, FJVPTag)
    
    # evaluate residual function
    rd = residual(xd, y)  # constant y
    
    # extract partials
    _, b = unpack_dual(rd) 

    return b

end

"""
internal convenience method for when user can provide drdx
"""
function jvp_provide(residual, x, y, v, drdx)
    
    B = drdx(residual, x, y)
    return B*v

end

"""
Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c
"""
function vjp_reverse(residual, x, y, v)
    return ReverseDiff.gradient(xtilde -> v' * residual(xtilde, y), x)  # instead of using underlying functions in ReverseDiff just differentiate -v^T * r with respect to x to get -v^T * dr/dx
end

"""
internal convenience method for when user can provide drdx
"""
function vjp_provide(residual, x, y, v, drdx)
    
    B = drdx(residual, x, y)
    return (v'*B)'
    
end


"""
compute A_ij = ∂r_i/∂y_j
"""
function drdy_forward(residual, x, y)
    A = ForwardDiff.jacobian(ytilde -> residual(x, ytilde), y)
    return A
end


"""
Linear solve A x = b  (where A is computed in drdy and b is computed in jvp)
"""
linear_solve(A, b) = A\b

"""
transpose linear solve A^T x = c  (where A is computed in drdy and c is computed in vjp)
"""
transpose_linear_solve(A, b) = (b' / A)'

# -----------------------------------------



# ---------- Overloads for implicit_function ---------

"""
    implicit_function(solve, residual, x; jvp=jvp_forward, vjp=vjp_reverse, drdx=nothing, drdy=drdy_forward, lsolve=linear_solve, tlsolve=transpose_linear_solve)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x). Solve implicit function (where residual(x, y) = 0) given input x and return states y(x).
- `residual::function`: r = residual(x, y). Return residual given input x and state y.
- `x::vector{float}`: evaluation point.
- `jvp::function`: jvp(residual, x, y, v).  Compute Jacobian vector product b = B*v where B_ij = ∂r_i/∂x_j.  Default is forward mode AD where jacobian is not explicitly constructed.
- `vjp::function`: vjp(residual, x, y, v).  Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c  Default is reverse mode AD where Jacobian is not explicitly constructed.
- `drdx::function`: drdx(residual, x, y).  Provide (or compute yourself): ∂r_i/∂x_j.  Not a required function. Default is nothing.  If provided, then jvp and jvp are ignored and explicitly multiplied against this provided Jacobian.
- `drdy::function`: drdy(residual, x, y).  Provide (or compute yourself): ∂r_i/∂y_j.  Default is forward mode AD.
- `lsolve::function`: lsolve(A, b).  Linear solve A x = b  (where A is computed in drdy and b is computed in jvp).  Default is backslash operator.
- `tlsolve::function`: tlsolve(A, b).  Transpose linear solve A^T x = c  (where A is computed in drdy and c is computed in vjp).  Default is forwardslash operator.
"""
function implicit_function(solve, residual, x; jvp=jvp_forward, vjp=vjp_reverse, drdx=nothing, drdy=drdy_forward, lsolve=linear_solve, 
    tlsolve=transpose_linear_solve)
    
    # if dr/dx is provided, then just directly multiply
    if !isnothing(drdx)
        newjvp(residual, x, y, v) = jvp_provide(residual, x, y, v, drdx)
        jvp = newjvp
        newvjp(residual, x, y, v) = vjp_provide(residual, x, y, v, drdx)
        vjp = newvjp
    end

    return implicit_function(solve, residual, x, jvp, vjp, drdy, lsolve, tlsolve)
end



# If no AD, just solve normally.
implicit_function(solve, residual, x, jvp, vjp, drdy, lsolve, tlsolve) = solve(x)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function implicit_function(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, jvp, vjp, drdy, lsolve, tlsolve) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solver
    yv = solve(xv)
    
    # solve for Jacobian-vector product
    b = jvp(residual, xv, yv, -dx)

    # comptue partial derivatives
    A = drdy(residual, xv, yv)
    
    # linear solve
    dy = lsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_function), solve, residual, x, jvp, vjp, drdy, lsolve, tlsolve)

    # evalute solver
    y = solve(x)

    # comptue partial derivatives
    A = drdy(residual, x, y)

    function implicit_pullback(dy)
        u = tlsolve(A, dy)
        dx = vjp(residual, x, y, -u)
        return NoTangent(), NoTangent(), NoTangent(), dx, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit_function(solve, residual, x::TrackedArray, jvp, vjp, drdy, lsolve, tlsolve)
ReverseDiff.@grad_from_chainrules implicit_function(solve, residual, x::AbstractVector{<:TrackedReal}, jvp, vjp, drdy, lsolve, tlsolve)

end
