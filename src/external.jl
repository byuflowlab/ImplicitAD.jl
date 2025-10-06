"""
    provide_rule(func, x, p=(); mode="ffd", jacobian=nothing, jvp=nothing, vjp=nothing)

Provide partials rather than rely on AD.  For cases where AD is not available
or to provide your own rule, or to use mixed mode, etc.

# Arguments
- `func::function`, `x::vector{float}`, `p::tuple`:  function of form: y = func(x, p), where x are variables and p are fixed parameters.
- `mode::string`:
    - "ffd": forward finite difference
    - "cfd": central finite difference
    - "cs": complex step
    - "jacobian": provide your own jacobian (see jacobian function)
    - "vp": provide jacobian vector product and vector jacobian product (see jvp and vjp)
- `jacobian::function`: only used if mode="jacobian". J = jacobian(x, p) provide J_ij = dy_i / dx_j
- `jvp::function`: only used if mode="vp" and in forward mode. ydot = jvp(x, p, v) provide Jacobian vector product J*v
- `vjp::function`: only used if mode="vp" and in revese mode. xbar = vjp(x, p, v) provide vector Jacobian product v'*J
"""
provide_rule(func, x, p=(); mode="ffd", jacobian=nothing, jvp=nothing, vjp=nothing) = _provide_rule(func, x, p, mode, jacobian, jvp, vjp)

_provide_rule(func, x, p, mode, jacobian, jvp, vjp) = func(x, p)

function _provide_rule(func, x::AbstractVector{<:ForwardDiff.Dual{T,V,N}}, p, mode, jacobian, jvp, vjp) where {T,V,N}

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

    return pack_dual(yv, ydot, T, Val(N))
end


function ChainRulesCore.rrule(::typeof(_provide_rule), func, x, p, mode, jacobian, jvp, vjp)

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

ReverseDiff.@grad_from_chainrules _provide_rule(func, x::ReverseDiff.TrackedArray, p, mode, jacobian, jvp, vjp)
ReverseDiff.@grad_from_chainrules _provide_rule(func, x::AbstractArray{<:ReverseDiff.TrackedReal}, p, mode, jacobian, jvp, vjp)


"""
    derivativesetup(func, x, p, ad, compiletape=false)

Set up a derivative function to make it easier to repeatedly differentiate (e.g., in an optimization).
Primary use case is to pass derivatives to Python or another language.

# Arguments
- `func::Function`: The function to differentiate. Must have signature `f = func(x, p)`
    where `x` are the variables to differentiate with respect to and `p` are additional parameters.
- `x::AbstractVector`: a typical input vector, used to size the derivative arrays.
- `p::Any`: Additional parameters to pass to `func`.  If this input changes, a new function must be setup.
- `ad::String`: The type of automatic differentiation to use. Options are:
    - `"fjacobian"`: Forward mode AD to compute the full Jacobian df/dx.
    - `"rjacobian"`: Reverse mode AD to compute the full Jacobian df/dx.
    - `"jvp"`: Forward mode AD to compute the Jacobian-vector product df/dx * v.
    - `"vjp"`: Reverse mode AD to compute the vector-Jacobian product v' * df/dx.
- `compiletape::Bool`: (optional, default=false) If using "rjacobian", whether to compile the tape for faster execution (assumes no branching behavior).

# Returns
A function that computes the requested derivative. The signature depends on the type of derivative:
- For `"fjacobian"` and `"rjacobian"`: `df(J, x)` where `J` is a preallocated array to hold the Jacobian.
- For `"jvp"`: `djvp(ydot, x, xdot)` where `ydot` is a preallocated array to hold the output and `xdot` is the input direction vector.
- For `"vjp"`: `dvjp(xbar, x, ybar)` where `xbar` is a preallocated array to hold the output and `ybar` is the input direction vector.
"""
function derivativesetup(func, x, p, ad, compiletape=false)

    fwrap(x) = func(x, p)
    return derivativesetupinternal(fwrap, x, ad, compiletape)

end

# internal function to provide shared function for julia and python outer functions
# f = func(x)
function derivativesetupinternal(func, x, ad, compiletape=false)

    # forward mode Jacobian
    if ad == "fjacobian"
        cfg = ForwardDiff.JacobianConfig(func, x)
        dfwd(J, x) = ForwardDiff.jacobian!(J, func, x, cfg)
        return dfwd

    # reverse mode Jacobian
    elseif ad == "rjacobian"
        if compiletape
            tape = ReverseDiff.JacobianTape(func, x, ReverseDiff.JacobianConfig(x))
            ctape = ReverseDiff.compile(tape)
            drevcompile(J, x) = ReverseDiff.jacobian!(J, ctape, x)
            return drevcompile
        else
            cfg = ReverseDiff.JacobianConfig(x)
            drev(J, x) = ReverseDiff.jacobian!(J, func, x, cfg)
            return drev
        end

    # (forward) Jacobian vector product
    elseif ad == "jvp"
        djvp(ydot, x, xdot) = ForwardDiff.derivative!(ydot, t -> func(x + t*xdot), 0.0)
        return djvp

    # (reverse) vector Jacobian product
    elseif ad == "vjp"
        cfg = ReverseDiff.GradientConfig(x)
        dvjp(xbar, x, ybar) = ReverseDiff.gradient!(xbar, xt -> ybar' * func(xt), x, cfg)
        return dvjp

    else
        error("Unsupported AD method: $ad")
    end

end
