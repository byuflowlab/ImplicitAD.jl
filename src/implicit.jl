using ForwardDiff
using ReverseDiff
using ChainRulesCore


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


# # --------- customization options for efficiency ---------
struct ImpFuncOptions{T1, T2, T3, T4, T5}
    drdy::T1  # function to compute A = dr/dy
    linsolve::T2  # function to solve A x = b for some provided b
    tlinsolve::T3  # function to solve A^T x = b for some provided b
    jvp::T4  # function to compute jacobian vector product B v for some provided v where B = dr/dx
    vjp::T5  # function to compute vector jacobian product v' * B for some provided v where B = dr/dx
end

ImpFuncOptions() = ImpFuncOptions(forward_partial_y, default_linear_solve, default_transpose_linear_solve, forwardJVP, reverseVJP)

# ------ compute dr/dy ------------
"""
    drdy(residual, x, y)

residual(x, y) is a function where x is the input and y is the output 
from solving the implicit function where r = residual(x, y) = 0.
User can provide own implementations of drdy.  It must return
A_{ij} = dr_i/dy_j where r are the residuals and y are the states.
Can optionally provide a factorization for A.
"""
function drdy(residual, x, y) end  # just for documentation

"""
Use forward AD to compute partials.
"""
function forward_partial_y(residual, x, y)
    # evaluate partial derivatives with ForwardDiff
    ry(ytilde) = residual(x, ytilde)
    A = ForwardDiff.jacobian(ry, y)  # TODO: cached version
    return A
end

"""
Use reverse AD to compute partials.
"""
function reverse_partial_y(residual, x, y)
    # evaluate partial derivatives with ForwardDiff
    ry(ytilde) = residual(x, ytilde)
    A = ReverseDiff.jacobian(ry, y)  # TODO: cached version
    return A
end

# --------- linear solve ----------
"""
    linear_solve(A, b)

Solve the linear system Ax = b for x where A is the partial derivatives
from drdy (which presumably you have used a custom implementation for if you are replacing this method).
b is some vector that will be input.
"""
function linear_solve(A, b) end  # just for documentation

"""
    transpose_linear_solve(A, b)

Solve the linear system A^T x = b for x where A is the partial derivatives
from drdy.  Note that a factorization for A will be provided not A^T so you will want to use a matrix right division.
"""
function transpose_linear_solve(A, b) end  # just for documentation


default_linear_solve(A, b) = A \ b
default_transpose_linear_solve(A, b) = (b' / A)'


# ------ compute dr/dx * v (JVP)  or v^T dr/dx  (VJP) ---------

"""
    jvp(residual, x, y, v)

Return Jacobian vector product: B * v
where B_ij = dr_i/dx_j.  
"""
function jvp(residual, x, y, v) end  # just for documentation

"""
    vjp(residual, x, y, v)

Return vector Jacobian product: v^T * B
where B_ij = dr_i/dx_j.  
"""
function vjp(residual, x, y, v) end  # just for documentation


struct FJVPTag end

"""
seed forward mode AD with dx to compute Jacobian vector product without ever forming the Jacobian.
"""
function forwardJVP(residual, x, y, v)
    # create new seed using v
    xd = pack_dual(x, v, FJVPTag)
    
    # evaluate residual function
    rd = residual(xd, y)  # constant y
    
    # extract partials
    _, dr = unpack_dual(rd) 
    
    return dr
end

"""
Form Jacobian with forward AD then multiply
"""
function forward_post_multiply(residual, x, y, v)
    rx(xtilde) = residual(xtilde, y)
    B = ForwardDiff.jacobian(rx, x)
    return B * v
end

"""
seed reverse mode AD with dr to compute vector-Jacobian-product without ever forming the Jacobian.
"""
function reverseVJP(residual, x, y, v)
    wrapper(xtilde) = v'*residual(xtilde, y)  # instead of using underlying functions in ReverseDiff just differentiate v^T * r with respect to x to get v^T * dr/dx
    dx = ReverseDiff.gradient(wrapper, x)
    return dx
end

"""
Form Jacobian with forward AD then pre-multiply
"""
function forward_pre_multiply(residual, x, y, v)
    rx(xtilde) = residual(xtilde, y)
    B = ForwardDiff.jacobian(rx, x)
    return v' * B
end


# ---------- Overloaded functions for solve_implicit ---------

"""
    solve_implicit(solve, residual, x, partials=forward_ad_partials, linearsolve=default_linear_solve)

default to forward AD for partials, and standard linear solve
"""
solve_implicit(solve, residual, x, options=ImpFuncOptions()) = solve_implicit(solve, residual, x, options)



# If no AD, just solve normally.
solve_implicit(solve, residual, x, options) = solve(x)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function solve_implicit(solve, residual, x::Vector{<:ForwardDiff.Dual{T}}, options) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solvers
    yv = solve(xv)
    
    # solve for Jacobian-vector product
    b = options.jvp(residual, xv, yv, -dx)
    A = options.drdy(residual, xv, yv)
    dy = options.linsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(solve_implicit), solve, residual, x, options)
    println("rule defined")

    # evalute solvers
    y = solve(x)

    A = options.drdy(residual, x, y)

    function implicit_pullback(dy)
        u = options.tlinsolve(A, dy)
        dx = options.vjp(residual, x, y, -u)
        return NoTangent(), NoTangent(), NoTangent(), dx, NoTangent()
    end

    return y, implicit_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules solve_implicit(solve, residual, x::TrackedArray, options)


# ----- example --------

function lin_example(x)
    A = [x[1] x[2] 3.0;
        3.0 4.0 x[3];
        x[4] 5.0 10.0]
    b = [7.0, 9.0, 10.0]
    return A, b
end

function lin_solver(x)
    println("calling solver")
    A, b = lin_example(x)
    y = A\b
    return y
end


function lin_residual(x, y)
    println("calling residual")
    A, b = lin_example(x)
    r = A*y - b
    return r
end


function lin_overload(x)
    y1 = 2.0*x
    y2 = solve_implicit(lin_solver, lin_residual, y1)
    z = 2.0*y2
    return z
end

function lin_ad_through(x)
    y1 = 2.0*x
    y2 = lin_solver(y1)
    z = 2.0*y2
    return z
end


x = [1.0, 3.0, 6.0, 11.0]
println("compute J1")
J1 = ForwardDiff.jacobian(lin_overload, x)
println("compute J2")
J2 = ForwardDiff.jacobian(lin_ad_through, x)
println("compute J3")
J3 = ReverseDiff.jacobian(lin_overload, x)
println("compute J4")
J4 = ReverseDiff.jacobian(lin_ad_through, x)

println(maximum(abs.(J1 - J2)))
println(maximum(abs.(J3 - J2)))
println(maximum(abs.(J4 - J2)))