module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore
using LinearAlgebra: factorize, ldiv!

# main function
export implicit, implicit_linear, apply_factorization, provide_rule


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
    
    return implicit(solve, new_residual, x, p, drdy, lsolve)
end



# If no AD, just solve normally.
implicit(solve, residual, x, p, drdy, lsolve) = solve(x, p)



# Overloaded for ForwardDiff inputs, providing exact derivatives using 
# Jacobian vector product.
function implicit(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}
    
    # evaluate solver
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

    # evaluate solver
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



# ------ linear case ------------

"""
Apply a matrix factorization to the primal portion of a dual number.
Avoids user from needing to add ForwardDiff as a dependency.
"""
function apply_factorization(A::AbstractArray{<:ForwardDiff.Dual{T}}, factfun) where {T}
    Av = fd_value(A)
    return factfun(Av)
end

# just apply factorization normally for case with no Duals
apply_factorization(A, factfun) = factfun(A)

# default factorization
apply_factorization(A) = apply_factorization(A, factorize)



"""
    implicit_linear(A, b; lsolve=linear_solve, fact=factorize)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
This version is for linear equations Ay = b

# Arguments
- `A::matrix`, `b::vector`: components of linear system ``A y = b``
- `lsolve::function`: lsolve(A, b). Function to solve the linear system, default is backslash operator.
- `Af::factorization`: An optional factorization of A, useful to override default factorize, or if multiple linear solves will be performed with same A matrix.
"""
implicit_linear(A, b; lsolve=linear_solve, Af=nothing) = implicit_linear(A, b, lsolve, Af)


# If no AD, just solve normally.
implicit_linear(A, b, lsolve, Af) = isnothing(Af) ? lsolve(A, b) : lsolve(Af, b)

# catch three cases where one or both contain duals
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
implicit_linear(A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)

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


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_linear), A, b, lsolve, Af)

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
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b, lsolve, Af)
ReverseDiff.@grad_from_chainrules implicit_linear(A, b::Union{TrackedArray, AbstractArray{<:TrackedReal}}, lsolve, Af)
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b::Union{TrackedArray, AbstractVector{<:TrackedReal}}, lsolve, Af)


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


# -------------- non AD-able operations ------------------------

"""
    provide_rule(func, x, p, mode="ffd"; jacobian=nothing, jvp=nothing, vjp=nothing) = provide_rule(func, x, p, mode, jacobian, jvp, vjp)

Provide partials rather than rely on AD.  For cases where AD is not available 
or to provide your own rule, or to use mixed mode, etc.

# Arguments
- `func::function`, `x::vector{float}`, `p::tuple`:  function of form: y = func(x, p), where x are variables and p are fixed parameters.
- `mode::string`: 
    - "ffd": forward finite difference
    - "cfd": central finite difference
    - "cs": complex step
    - "jacobian": provide your own jacobian (see jacobian function)
    - "vector product": provide jacobian vector product and vector jacobian product (see jvp and vjp)
- `jacobian::function`: only used if mode="jacobian". J = jacobian(x, p) provide J_ij = dy_i / dx_j
- `jvp::function`: only used if mode="vp" and in forward mode. ydot = jvp(x, p, v) provide Jacobian vector product J*v
- `vjp::function`: only used if mode="vp" and in revese mode. xbar = vjp(x, p, v) provide vector Jacobian product v'*J
"""
provide_rule(func, x, p, mode="ffd"; jacobian=nothing, jvp=nothing, vjp=nothing) = provide_rule(func, x, p, mode, jacobian, jvp, vjp)

provide_rule(func, x, p, mode, jacobian, jvp, vjp) = func(x, p)

function provide_rule(func, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, mode, jacobian, jvp, vjp) where {T}
    
    # unpack dual
    xv, xdot = unpack_dual(x)
    nx, nd = size(xdot)

    # call function with primal values
    yv = func(xv, p)
    ny = length(yv)

    # initialize
    ydot = Matrix{Float64}(undef, ny, nd)
    
    if mode == "ffd"  # forward finite diff

        h = sqrt(eps(Float64))  # theoretical optimal (absolute) step size
    
        # check whether we should do a series of JVPs or compute Jacobian than multiply
        
        if nd <= nx  # do JVPs (for each dual)
            xnew = Vector{Float64}(undef, nx)
            for i = 1:nd
                xnew .= xv + h*xdot[:, i]  # directional derivative
                ydot[:, i] = (func(xnew, p) - yv)/h
            end
        
        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            for i = 1:nx
                delta = max(xv[i]*h, h)  # use relative step size unless it is too small
                xv[i] += delta
                J[:, i] = (func(xv, p) - yv)/delta
                xv[i] -= delta
            end
            ydot .= J * xdot
        end
    
    elseif mode == "cfd"  # central finite diff

        h = cbrt(eps(Float64))  # theoretical optimal (absolute) step size
    
        # check whether we should do a series of JVPs or compute Jacobian than multiply
        
        if nd <= nx  # do JVPs
            xnew = Vector{Float64}(undef, nx)
            for i = 1:nd
                xnew .= xv + h*xdot[:, i]  # directional derivative
                yp = func(xnew, p)
                xnew .= xv - h*xdot[:, i]
                ym = func(xnew, p)
                ydot[:, i] = (yp - ym)/(2*h)
            end
        
        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            for i = 1:nx
                delta = max(xv[i]*h, h)  # use relative step size unless it is too small
                xv[i] += delta
                yp = func(xv, p)
                xv[i] -= 2*delta
                ym = func(xv, p)
                J[:, i] = (yp - ym)/(2*delta)
                xv[i] += delta  # reset
            end
            ydot .= J * xdot
        end

    elseif mode == "cs"  # complex step
        
        h = 1e-30
        xnew = Vector{ComplexF64}(undef, nx)
    
        # check whether we should do a series of JVPs or compute Jacobian than multiply
        
        if nd <= nx  # do JVPs (for each dual)
            for i = 1:nd
                xnew .= xv + h*im*xdot[:, i]  # directional derivative
                ydot[:, i] = imag(func(xnew, p))/h
            end
        
        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            xnew .= xv
            for i = 1:nx
                xnew[i] += h*im
                J[:, i] = imag(func(xnew, p))/h
                xnew[i] -= h*im
            end
            ydot .= J * xdot
        end

    elseif mode == "jacobian"
        J = jacobian(xv, p)
        ydot .= J * xdot

    elseif mode == "vp"  # jacobian vector product
        for i = 1:nd
            ydot[:, i] = jvp(xv, p, xdot[:, i])
        end

    else
        error("invalid mode")
    end

    return pack_dual(yv, ydot, T)
end


function ChainRulesCore.rrule(::typeof(provide_rule), func, x, p, mode, jacobian, jvp, vjp)

    # evaluate function
    y = func(x, p)
    nx = length(x)
    ny = length(y)
    
    if mode == "vp"

        function vppullback(ybar)
            xbar = vjp(x, p, ybar)
            return NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        return y, vppullback
    end
        
    J = Matrix{Float64}(undef, ny, nx)

    if mode == "ffd"

        h = sqrt(eps(Float64))
        for i = 1:nx
            delta = max(x[i]*h, h)  # use relative step size unless it is too small
            x[i] += delta
            J[:, i] = (func(x, p) - y)/delta
            x[i] -= delta
        end

    elseif mode == "cfd"

        h = cbrt(eps(Float64))
        for i = 1:nx
            delta = max(x[i]*h, h)  # use relative step size unless it is too small
            x[i] += delta
            yp = func(x, p)
            x[i] -= 2*delta
            ym = func(x, p)
            J[:, i] = (yp - ym)/(2*delta)
            x[i] += delta  # reset
        end

    elseif mode == "cs"

        h = 1e-30
        xnew = Vector{ComplexF64}(undef, nx)
        xnew .= x
        for i = 1:nx
            xnew[i] += h*im
            J[:, i] = imag(func(xnew, p))/h
            xnew[i] -= h*im
        end

    elseif mode == "jacobian"
        
        J .= jacobian(x, p)

    else
        error("invalid mode")
    end

    function pullback(ybar)
        xbar = J'*ybar
        return NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end

ReverseDiff.@grad_from_chainrules provide_rule(func, x::TrackedArray, p, mode, jacobian, jvp, vjp)
ReverseDiff.@grad_from_chainrules provide_rule(func, x::AbstractArray{<:TrackedReal}, p, mode, jacobian, jvp, vjp)

end
