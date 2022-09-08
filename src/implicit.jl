using ForwardDiff
using ReverseDiff
using ChainRulesCore


struct FJVPTag end

"""
Compute Jacobian vector product b = B*v where B_ij = ∂r_i/∂x_j
"""
function jvp(residual, x, y, v, provideB)
    println("jvp")
    
    if isnothing(provideB)
        # create new seed using v
        xd = pack_dual(x, v, FJVPTag)
        
        # evaluate residual function
        rd = residual(xd, y)  # constant y
        
        # extract partials
        _, b = unpack_dual(rd) 
        
        return b

    else
        B = provideB(x, y)
        return B*v

    end
end

"""
Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c
"""
function vjp(residual, x, y, v, provideB)
    println("vjp")
    
    if isnothing(provideB)
        return ReverseDiff.gradient(xtilde -> v' * residual(xtilde, y), x)  # instead of using underlying functions in ReverseDiff just differentiate -v^T * r with respect to x to get -v^T * dr/dx

    else
        B = provideB(x, y)  # function really provides ∂r/∂x. not residual
        return (v'*B)'
    
    end
end


"""
compute A_ij = ∂r_i/∂y_j
"""
function computeA(residual, x, y, provideA)
    println("computeA")
    
    # compute A = ∂r/∂y
    if isnothing(provideA)
        A = ForwardDiff.jacobian(ytilde -> residual(x, ytilde), y)

    else
        A = provideA(x, y)  # function is really ∂r/∂y. not residual
    
    end

    return A
end



# ---------- Overloaded functions for solve_implicit ---------

"""
    solve_implicit(solve, residual, x, partials=forward_ad_partials, linearsolve=default_linear_solve)

default to forward AD for partials, and standard linear solve
"""
solve_implicit(solve, residual, x; provideA=nothing, provideB=nothing) = solve_implicit(solve, residual, x, provideA, provideB)


# If no AD, just solve normally.
solve_implicit(solve, residual, x, provideA, provideB) = solve(x)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function solve_implicit(solve, residual, x::Vector{<:ForwardDiff.Dual{T}}, provideA, provideB) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solvers
    yv = solve(xv)
    
    # solve for Jacobian-vector product
    b = jvp(residual, xv, yv, -dx, provideB)

    # comptue partial derivatives
    A = computeA(residual, xv, yv, provideA)
    
    # linear solve
    dy = A \ b

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(solve_implicit), solve, residual, x, provideA, provideB)
    println("rule defined")

    # evalute solvers
    y = solve(x)

    # comptue partial derivatives
    A = computeA(residual, x, y, provideA)

    function implicit_pullback(dy)
        println("pullback")
        u = (dy' / A)'
        dx = vjp(residual, x, y, -u, provideB)
        return NoTangent(), NoTangent(), NoTangent(), dx, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules solve_implicit(solve, residual, x::TrackedArray, provideA, provideB)


# ----- example --------

using NLsolve
using FiniteDiff

function residual(x, y)
    T = promote_type(eltype(x), eltype(y))
    r = zeros(T, 2)
    r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
    r[2] = sin(y[2]*exp(y[1])-1)*x[4]
    return r
end

function solve(x)
    rwrap(y) = residual(x[1:4], y)
    res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
    return res.zero
end

function program(x)
    z = 2.0*x
    w = z + x.^2
    y = solve(w)
    return y[1] .+ w*y[2]
end

function modprogram(x)
    z = 2.0*x
    w = z + x.^2
    y = solve_implicit(solve, residual, w)
    return y[1] .+ w*y[2]
end

x = [1.0; 2.0; 3.0; 4.0; 5.0]
program(x)
modprogram(x)

J1 = FiniteDiff.finite_difference_jacobian(program, x)
J2 = ForwardDiff.jacobian(modprogram, x)
J3 = ReverseDiff.jacobian(modprogram, x)

println(maximum(abs.(J1 - J2)))
println(maximum(abs.(J2 - J3)))

