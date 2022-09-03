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


# --------- partial derivatives ---------
"""
    forward_ad_partials(residual, x, y)

Default is to use ForwardDiff for partials
"""
function forward_ad_partials(residual, x, y)
    ry(ytilde) = residual(x, ytilde)
    rx(xtilde) = residual(xtilde, y)
    A = ForwardDiff.jacobian(ry, y)  # TODO: cached version
    B = ForwardDiff.jacobian(rx, x)
    return A, B
end

# --------- linear solve ----------
"""
    default_linear_solve(A, b)

standard linear solver
"""
default_linear_solve(A, b) = A\b



# ---------- Overloaded functions for implicit ---------

"""
    solve_implicit(solve, residual, x, partials=forward_ad_partials, linearsolve=default_linear_solve)

default to forward AD for partials, and standard linear solve
"""
solve_implicit(solve, residual, x, partials=forward_ad_partials, linearsolve=default_linear_solve) = solve_implicit(solve, residual, x, partials, linearsolve)



# If no AD, just solve normally.
solve_implicit(solve, residual, x, partials, linearsolve) = solve(x)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function solve_implicit(solve, residual, x::Vector{<:ForwardDiff.Dual{T}}, partials, linearsolve) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solvers
    yv = solve(xv)
    
    # compute residual partial derivative matrices
    A, B = partials(residual, xv, yv)

    # solve for Jacobian-vector product
    b = -B * dx  
    dy = linearsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(solve_implicit), solve, residual, x, partials, linearsolve)
    println("rule defined")

    # evalute solvers
    y = solve(x)

    # compute residual partial derivative matrices
    A, B = partials(residual, x, y)

    function implicit_pullback(dy)
        u = linearsolve(transpose(A), dy)
        dx = -transpose(B) * u
        return NoTangent(), NoTangent(), NoTangent(), dx, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules solve_implicit(solve, residual, x::TrackedArray, partials, linearsolve)


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