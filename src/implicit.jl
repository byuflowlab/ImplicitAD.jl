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

# -----------------------------------------

# ---------- core methods -----------------

struct FJVPTag end

"""
Compute Jacobian vector product b = B*v where B_ij = ∂r_i/∂x_j
"""
function jvp(residual, x, y, v, drdx)
    
    if isnothing(drdx)
        # create new seed using v
        xd = pack_dual(x, v, FJVPTag)
        
        # evaluate residual function
        rd = residual(xd, y)  # constant y
        
        # extract partials
        _, b = unpack_dual(rd) 

        return b

    else
        B = drdx(x, y)
        return B*v
    end
end

"""
Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c
"""
function vjp(residual, x, y, v, drdx)
    
    if isnothing(drdx)
        return ReverseDiff.gradient(xtilde -> v' * residual(xtilde, y), x)  # instead of using underlying functions in ReverseDiff just differentiate -v^T * r with respect to x to get -v^T * dr/dx

    else
        B = drdx(x, y)
        return (v'*B)'
    
    end
end


"""
compute A_ij = ∂r_i/∂y_j
"""
function computeA(residual, x, y, drdy)
    
    # compute A = ∂r/∂y
    if isnothing(drdy)
        A = ForwardDiff.jacobian(ytilde -> residual(x, ytilde), y)

    else
        A = drdy(x, y)
    
    end

    return A
end

# linear solve (creating function so users can overload and replace residual with struct containing other data)
lsolve(residual, A, b) = A\b
# transpose linear solve (A^T x = b) using factorization in A
tlsolve(residual, A, b) = (b' / A)'
# -----------------------------------------


# ---------- Overloads for implicit_function ---------

"""
    implicit_function(solve, residual, x, drdy=nothing, drdx=nothing)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x). Solve implicit function (where residual(x, y) = 0) given input x and return states y(x).
- `residual::function`: r = residual(x, y). Return residual given input x and state y.
- `x::vector{float}`: evaluation point.
- `drdy::function`: ∂r/∂y = drdy(x, y). Defaults to nothing. Set to a function if you can provide the partials directly.
- `drdx::function`: ∂r/∂x = drdx(x, y). Defaults to nothing. Set to a function if you can provide the partials directly.
"""
implicit_function(solve, residual, x; drdy=nothing, drdx=nothing) = implicit_function(solve, residual, x, drdy, drdx)


# If no AD, just solve normally.
implicit_function(solve, residual, x, drdy, drdx) = solve(x)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function implicit_function(solve, residual, x::Vector{<:ForwardDiff.Dual{T}}, drdy, drdx) where {T}

    # unpack dual numbers
    xv, dx = unpack_dual(x) 
    
    # evalute solver
    yv = solve(xv)
    
    # solve for Jacobian-vector product
    b = jvp(residual, xv, yv, -dx, drdx)

    # comptue partial derivatives
    A = computeA(residual, xv, yv, drdy)
    
    # linear solve
    dy = lsolve(residual, A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, dy, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_function), solve, residual, x, drdy, drdx)

    # evalute solver
    y = solve(x)

    # comptue partial derivatives
    A = computeA(residual, x, y, drdy)

    function implicit_pullback(dy)
        u = tlsolve(residual, A, dy)
        dx = vjp(residual, x, y, -u, drdx)
        return NoTangent(), NoTangent(), NoTangent(), dx, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit_function(solve, residual, x::TrackedArray, drdy, drdx)


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

function drdy(x, y)
    A = zeros(2, 2)
    A[1, 1] = y[2]^3-x[2]
    A[1, 2] = 3*y[2]^2*(y[1]+x[1])
    u = exp(y[1])*cos(y[2]*exp(y[1])-1)*x[4]
    A[2, 1] = y[2]*u
    A[2, 2] = u
    return A
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
    y = implicit_function(solve, residual, w)
    return y[1] .+ w*y[2]
end

function modprogram2(x)
    z = 2.0*x
    w = z + x.^2
    y = implicit_function(solve, residual, w, drdy=drdy)
    return y[1] .+ w*y[2]
end

x = [1.0; 2.0; 3.0; 4.0; 5.0]
program(x)
modprogram(x)

J1 = FiniteDiff.finite_difference_jacobian(program, x)
J2 = ForwardDiff.jacobian(modprogram, x)
J3 = ReverseDiff.jacobian(modprogram, x)
J4 = ForwardDiff.jacobian(modprogram2, x)
J5 = ReverseDiff.jacobian(modprogram2, x)

println(maximum(abs.(J1 - J2)))
println(maximum(abs.(J2 - J3)))
println(maximum(abs.(J2 - J4)))
println(maximum(abs.(J2 - J5)))

using LinearAlgebra: Symmetric, factorize
using SparseArrays: sparse
function test3(x)
    A = [x[1]*x[2] x[3]+x[4];
        x[3]+x[4] 0.0]
    b = [2.0, 3.0]
    Af = factorize(Symmetric(sparse(A)))
    w = Af\b
    return 2*w
end

function solvelin(x)
    A = [x[1]*x[2] x[3]+x[4];
        x[3]+x[4] 0.0]
    b = [2.0, 3.0]
    Af = factorize(Symmetric(sparse(A)))
    return Af \ b
end

function Alin(x, y)
    return [x[1]*x[2] x[3]+x[4];
        x[3]+x[4] 0.0]
end

# r = A * y - b
# drdx = dAdx * y
function Blin(x, y)
    B = [x[2]*y[1]  x[1]*y[1]  y[2]  y[2];
         0.0  0.0  y[1]  y[1]]
    return B
end

function test4(x)
    w = implicit_function(solvelin, nothing, x, drdy=Alin, drdx=Blin)
    return 2*w
end

ForwardDiff.jacobian(test4, [1.0, 2.0, 3.0, 4.0])
FiniteDiff.finite_difference_jacobian(test4, [1.0, 2.0, 3.0, 4.0])