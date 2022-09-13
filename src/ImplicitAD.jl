module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore

# main function
export implicit_function

# may overload
export jvp, vjp, computeA, lsolve, tlsolve



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

end
