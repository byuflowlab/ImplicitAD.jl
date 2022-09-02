using ForwardDiff
using ReverseDiff
using ChainRulesCore

# ---------- Structs ---------

"""
    ImplicitFunction(solver, residual)

Provide residual function: `r = residual(x, y)`
and corresponding solve function: `y = solve(x)`
"""
struct ImplicitFunction{S, R}
    solve::S
    residual::R
end


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



# ---------- Overloaded functions for implicit ---------

"""
    implicit(x)

Functor, if not dual numbers, just solve normally.
"""
# (implicit::ImplicitFunction)(x) = implicit.solve(x)
solve_implicit(implicit, x) = implicit.solve(x)


"""
    implicit(x)

Functor overload for ForwardDiff inputs, providing exact derivatives using 
Jacobian vector product.
"""
# function (implicit::ImplicitFunction)(x::Vector{<:ForwardDiff.Dual{T}}) where {T}
function solve_implicit(implicit, x::Vector{<:ForwardDiff.Dual{T}}) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solvers
    yv = implicit(xv)
    
    # compute residual partial derivative matrices
    ry(ytilde) = implicit.residual(xv, ytilde)
    rx(xtilde) = implicit.residual(xtilde, yv)

    A = ForwardDiff.jacobian(ry, yv)  # computing partials with ForwardDiff (could use other methods)
    B = ForwardDiff.jacobian(rx, xv)

    # solve for Jacobian-vector product
    b = -B * dx  
    dy = A \ b

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# function ChainRulesCore.rrule(implicit::typeof(ImplicitFunction), x)
function ChainRulesCore.rrule(::typeof(solve_implicit), implicit, x)
    println("rule defined")

    # evalute solvers
    y = implicit(x)

    # compute residual partial derivative matrices
    ry(ytilde) = implicit.residual(x, ytilde)
    rx(xtilde) = implicit.residual(xtilde, y)

    A = ForwardDiff.jacobian(ry, y)  # computing partials with ForwardDiff (could use other methods)
    B = ForwardDiff.jacobian(rx, x)

    function implicit_pullback(dy)
        println("calling pullback")
        u = transpose(A) \ dy
        dx = -transpose(B) * u
        return NoTangent(), NoTangent(), dx
    end

    return y, implicit_pullback
end

ReverseDiff.@grad_from_chainrules solve_implicit(implicit, x::TrackedArray)





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
    lin_imp = ImplicitFunction(lin_solver, lin_residual)
    y2 = solve_implicit(lin_imp, y1)
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