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

function _provide_rule(func, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, mode, jacobian, jvp, vjp) where {T}

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

ReverseDiff.@grad_from_chainrules _provide_rule(func, x::TrackedArray, p, mode, jacobian, jvp, vjp)
ReverseDiff.@grad_from_chainrules _provide_rule(func, x::AbstractArray{<:TrackedReal}, p, mode, jacobian, jvp, vjp)
