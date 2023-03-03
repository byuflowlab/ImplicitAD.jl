# implicit unsteady
function jvp(residual, y, yprevd, t, tprev, xd, p)

    # evaluate residual function
    rd = residual(y, yprevd, t, tprev, xd, p)  # constant y

    # extract partials
    b = -fd_partials(rd)

    return b
end

# implicit unsteady
function vjp(residual, y, yprev, t, tprev, x, p, v)
    return ReverseDiff.gradient((yprevtilde, xtilde) -> v' * residual(y, yprevtilde, t, tprev, xtilde, p), (yprev, x))
end


# implicit unsteady
function drdy_forward(residual, y::AbstractVector, yprev, t, tprev, x, p)
    A = ForwardDiff.jacobian(ytilde -> residual(ytilde, yprev, t, tprev, x, p), y)
    return A
end

function drdy_forward(residual, y::Number, yprev, t, tprev, x, p)  # 1D case
    A = ForwardDiff.derivative(ytilde -> residual(ytilde, yprev, t, tprev, x, p), y)
    return A
end



# ------ Overloads for explicit_unsteady ----------

"""
    explicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make an explicit time-marching analysis AD compatible (specifically with ForwardDiff and
ReverseDiff).

# Arguments:
 - `solve::function`: function `y, t = solve(x, p)`.  Perform a time marching analysis,
    and return the matrix `y = [y[1] y[2] y[3] ... y[N]] where `y[i]` is the state vector at time step `i` (a rows is a state, a columms is a timesteps)
    and vector t = [t[1], t[2], t[3] ... t[N]]` where `t[i]` is the corresponding time,
    for input variables `x`, and fixed variables `p`.
 - `perform_step::function`: Either `y[i] = perform_step(y[i-1], t[i], t[i-1], x, p)` or
    in-place `perform_step!(y[i], y[i-1], t[i], t[i-1], x, p)`. Set the next set of state
    variables `y[i]` (scalar or vector), given the previous state `y[i-1]` (scalar or vector),
    current time `t[i]`, previous time `t[i-1]`, variables `x` (vector), and fixed parameters `p` (tuple).
 - `x`: evaluation point
 - `p`: fixed parameters, default is empty tuple.
"""
function explicit_unsteady(solve, perform_step, x, p=())

    # ---- check for in-place version and wrap as needed -------
    if applicable(perform_step, 1.0, 1.0, 1.0, 1.0, 1.0)  # out-of-place
        return _outofplace_explicit_unsteady(solve, perform_step, x, p)
    else
        return _inplace_explicit_unsteady(solve, perform_step, x, p)
    end

end

# If no AD, just solve normally.
_outofplace_explicit_unsteady(solve, perform_step, x, p) = solve(x, p)
_inplace_explicit_unsteady(solve, perform_step, x, p) = solve(x, p)

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function _outofplace_explicit_unsteady(solve, perform_step, x::AbstractVector{<:ForwardDiff.Dual{T,V,N}}, p) where {T,V,N}

    # wrap out-of-place function as in-place
    perform_step! = (yw, yprevw, tw, tprevw, xw, pw) -> begin
        yw .= perform_step(yprevw, tw, tprevw, xw, pw)
        return yw
    end

    # now call in-place version
    return _inplace_explicit_unsteady(solve, perform_step!, x, p)
end

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function _inplace_explicit_unsteady(solve, perform_step!, x::AbstractVector{<:ForwardDiff.Dual{T,V,N}}, p) where {T,V,N}

    # find values
    xv = fd_value(x)
    yv, tv = solve(xv, p)

    # get solution dimensions
    ny, nt = size(yv)

    # initialize output
    yd = similar(yv, ForwardDiff.Dual{T,V,N}, ny, nt)

    # function barrier for actual computations
    return _inplace_explicit_unsteady!(yd, yv, tv, perform_step!, x, p)
end

# function barrier for actual computations
function _inplace_explicit_unsteady!(yd, yv, tv, perform_step!, x::AbstractVector{<:ForwardDiff.Dual{T,V,N}}, p) where {T,V,N}

    # --- Initial Time Step --- #

    # # solve for Jacobian-vector product
    # perform_step!(view(yd,:,1), view(yv,:,1), tv[1], tv[1], x, p)

    ydotd = similar(x, size(yd, 1))

    # combine values and partials
    perform_step!(ydotd, yv[:,1], tv[1], tv[1], x, p)
    ydot = fd_partials(ydotd)

    @views yd[:, 1] = pack_dual(yv[:, 1], ydot, T)

    # for iy = 1:size(yd, 1)
    #     yd[iy, 1] = ForwardDiff.Dual{T,V,N}(yv[iy,1], yd[iy,1].partials)
    # end

    # --- Additional Time Steps --- #

    for it = 2:size(yd, 2)

        # # solve for Jacobian-vector product
        # perform_step!(view(yd,:,it), view(yd,:,it-1), tv[it], tv[it-1], x, p)

        # combine values and partials
        perform_step!(ydotd, yd[:,it-1], tv[it], tv[it-1], x, p)
        ydot = fd_partials(ydotd)

        @views yd[:, it] = pack_dual(yv[:, it], ydot, T)

        # # combine values and partials
        # for iy = 1:size(yd, 1)
        #     yd[iy, it] = ForwardDiff.Dual{T,V,N}(yv[iy,it], yd[iy,it].partials)
        # end
    end

    return yd, tv
end

# ReverseDiff needs single array output so unpack before returning to user
_inplace_explicit_unsteady(solve, perform_step!, x::ReverseDiff.TrackedArray, p) = ieu_unpack_reverse(solve, perform_step!, x, p)
_outofplace_explicit_unsteady(solve, perform_step, x::AbstractVector{<:ReverseDiff.TrackedReal}, p) = oeu_unpack_reverse(solve, perform_step, x, p)

function _inplace_explicit_unsteady_reverse(solve, perform_step!, x, p)

    # wrap in-place function as out-of-place
    perform_step = (yprevw, tw, tprevw, xw, pw) -> begin
        T = promote_type(eltype(yprevw), typeof(tw), typeof(tprevw), eltype(xw))
        yw = zeros(T, length(yprevw))
        perform_step!(yw, yprevw, tw, tprevw, xw, pw)
        return yw
    end

    # now call out-of-place version
    return _outofplace_explicit_unsteady_reverse(solve, perform_step, x, p)
end

# just declaring dummy function for below
function _outofplace_explicit_unsteady_reverse(solve, perform_step, x, p) end

# unpack for user
function ieu_unpack_reverse(solve, perform_step!, x, p)
    yt = _inplace_explicit_unsteady_reverse(solve, perform_step!, x, p)
    return yt[1:end-1, :], yt[end, :]
end

function oeu_unpack_reverse(solve, perform_step, x, p)
    yt = _outofplace_explicit_unsteady_reverse(solve, perform_step, x, p)
    return yt[1:end-1, :], yt[end, :]
end

# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_outofplace_explicit_unsteady_reverse), solve, perform_step, x, p)

    # evaluate solver
    yv, tv = solve(x, p)

    # get solution dimensions
    ny, nt = size(yv)

    # create local copy of the output to guard against values getting overwritten
    yv = copy(yv)
    tv = copy(tv)

    # construct function for vector jacobian product (assume parameters are constant)
    fvjp = (yprev, t, tprev, x, λ) -> λ' * perform_step(yprev, t[1], tprev[1], x, p)

    # construct sample inputs
    gyprev = similar(yv, ny)
    gt = similar(yv, 1)
    gtprev = similar(yv, 1)
    gx = similar(x)
    gλ = similar(yv, ny)

    # construct tape
    tape = ReverseDiff.GradientTape(fvjp, (gyprev, gt, gtprev, gx, gλ))

    # compile tape (optional)
    # if compile
    #     tape = ReverseDiff.compile(tape)
    # end

    # construct vector-jacobian product function
    vjp = let gt=gt, gtprev=gtprev, gλ=gλ

        function outofplace_explicit_unsteady_vjp(yprev, t, tprev, x, λ)

            ReverseDiff.gradient!((gyprev, gt, gtprev, gx, gλ), tape, (yprev, [t], [tprev], x, λ))

            return gyprev, gx
        end

    end

    function pullback(ytbar)

        @views ybar = ytbar[1:end-1, :]

        if nt > 1

            # --- Final Time Step --- #
            @views λ = ybar[:, nt]
            @views Δybar, Δxbar = vjp(yv[:, nt-1], tv[nt], tv[nt-1], x, λ)
            xbar = Δxbar
            @views ybar[:, nt-1] += Δybar

            # --- Additional Time Steps --- #
            for i = nt-1:-1:2
                @views λ = ybar[:, i]
                @views Δybar, Δxbar = vjp(yv[:, i-1], tv[i], tv[i-1], x, λ)
                xbar += Δxbar
                @views ybar[:, i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            @views Δybar, Δxbar = vjp(yv[:, 1], tv[1], tv[1], xbar, λ)
            xbar += Δxbar

        else

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            @views Δybar, Δxbar = vjp(yv[:, 1], tv[1], tv[1], xbar, λ)
            xbar = Δxbar

        end

        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent()
    end

    return [yv; tv'], pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _outofplace_explicit_unsteady_reverse(solve, perform_step!, x::TrackedArray, p)
ReverseDiff.@grad_from_chainrules _outofplace_explicit_unsteady_reverse(solve, perform_step!, x::AbstractVector{<:TrackedReal}, p)

# ------ Overloads for implicit_unsteady ----------

"""
    implicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make a implicit time-marching analysis AD compatible (specifically with ForwardDiff and
ReverseDiff).

# Arguments:
- `solve::function`: function `y, t = solve(x, p)`.  Perform a time marching analysis,
    and return the matrix `y = [y[1] y[2] y[3] ... y[N]] where `y[i]` is the state vector at time step `i` (a rows is a state, a columms is a timesteps)
    and vector t = [t[1], t[2], t[3] ... t[N]]` where `t[i]` is the corresponding time,
    for input variables `x` and fixed variables `p`.
 - `residual::function`: Either `r[i] = residual(y[i], y[i-1], t[i], t[i-1], x, p)` or
    in-place `residual!(r[i], y[i], y[i-1], t[i], t[i-1], x, p)`. Set current residual
    `r[i]` (scalar or vector), given current state `y[i]` (scalar or vector),
    previous state `y[i-1]` (scalar or vector),
    current and preivous time variables `t[i]` and `t[i-1]`,
    `x` (vector), and fixed parameters `p` (tuple).
 - `x`: evaluation point
 - `p`: fixed parameters, default is empty tuple.
 - `drdy`: drdy(residual, y, x, p). Provide (or compute yourself): ∂ri/∂yj.  Default is
    forward mode AD.
 - `lsolve::function`: `lsolve(A, b)`. Linear solve `A x = b` (where `A` is computed in
    `drdy` and `b` is computed in `jvp`, or it solves `A^T x = c` where `c` is computed
    in `vjp`). Default is backslash operator.
"""
function implicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

    # ---- check for in-place version and wrap as needed -------
    new_residual = residual
    if applicable(residual, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # in-place

        function unsteady_residual_wrap(yw, yprevw, tw, tprevw, xw, pw)  # wrap residual function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(typeof(tw), typeof(tprevw), eltype(yw), eltype(yprevw), eltype(xw))
            rw = zeros(T, length(yw))  # match type of input variables
            residual(rw, yw, yprevw, tw, tprevw, xw, pw)
            return rw
        end
        new_residual = unsteady_residual_wrap
    end

    return _implicit_unsteady(solve, new_residual, x, p, drdy, lsolve)
end

# If no AD, just solve normally.
_implicit_unsteady(solve, residual, x, p, drdy, lsolve) = solve(x, p)

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function _implicit_unsteady(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}

    # evaluate solver
    xv = fd_value(x)
    yv, tv = solve(xv, p)

    # get solution dimensions
    ny, nt = size(yv)

    # initialize output
    yd = similar(yv, ForwardDiff.Dual{T}, ny, nt)

    # --- Initial Time Step --- #

    # solve for Jacobian-vector products
    @views b = jvp(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], x, p)

    # compute partial derivatives
    @views A = drdy(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], xv, p)

    # linear solve
    ydot = lsolve(A, b)

    # # repack in ForwardDiff Dual
    @views yd[:, 1] = pack_dual(yv[:, 1], ydot, T)

    # --- Additional Time Steps --- #

    for i = 2:nt

        # solve for Jacobian-vector product
        @views b = jvp(residual, yv[:, i], yd[:, i-1], tv[i], tv[i-1], x, p)

        # compute partial derivatives
        @views A = drdy(residual, yv[:, i], yv[:, i-1], tv[i], tv[i-1], xv, p)

        # linear solve
        ydot = lsolve(A, b)

        # repack in ForwardDiff Dual
        @views yd[:, i] = pack_dual(yv[:, i], ydot, T)

    end

    return yd, tv
end

# ReverseDiff needs single array output so unpack before returning to user
_implicit_unsteady(solve, residual, x::ReverseDiff.TrackedArray, p, drdy, lsolve) = iu_unpack_reverse(solve, residual, x, p, drdy, lsolve)
_implicit_unsteady(solve, residual, x::AbstractVector{<:ReverseDiff.TrackedReal}, p, drdy, lsolve) = iu_unpack_reverse(solve, residual, x, p, drdy, lsolve)

# just declaring dummy function for below
function _implicit_unsteady_reverse(solve, residual, x, p, drdy, lsolve) end

# unpack for user
function iu_unpack_reverse(solve, residual, x, p, drdy, lsolve)
    yt = _implicit_unsteady_reverse(solve, residual, x, p, drdy, lsolve)
    return yt[1:end-1, :], yt[end, :]
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_implicit_unsteady_reverse), solve, residual, x, p, drdy, lsolve)

    # evaluate solver
    yv, tv = solve(x, p)

    # get solution dimensions
    ny, nt = size(yv)

    # create local copy of the output to guard against values getting overwritten
    yv = copy(yv)
    tv = copy(tv)

    function pullback(ytbar)

        @views ybar = ytbar[1:end-1, :]

        if nt > 1

            # --- Final Time Step --- #
            @views A = drdy(residual, yv[:, nt], yv[:, nt-1], tv[nt], tv[nt-1], x, p)
            @views λ = lsolve(A', ybar[:, nt])
            @views Δybar, Δxbar = vjp(residual, yv[:, nt], yv[:, nt-1], tv[nt], tv[nt-1], x, p, -λ)
            xbar = Δxbar
            @views ybar[:, nt-1] += Δybar

            # --- Additional Time Steps --- #
            for i = nt-1:-1:2
                @views A = drdy(residual, yv[:, i], yv[:, i-1], tv[i], tv[i-1], x, p)
                @views λ = lsolve(A', ybar[:, i])
                @views Δybar, Δxbar = vjp(residual, yv[:, i], yv[:, i-1], tv[i], tv[i-1], x, p, -λ)
                xbar += Δxbar
                @views ybar[:, i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views A = drdy(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], x, p)
            @views λ = lsolve(A', ybar[:, 1])
            @views Δybar, Δxbar = vjp(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], xbar, p, -λ)
            xbar += Δxbar

        else

            # --- Initial Time Step --- #
            @views A = drdy(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], x, p)
            @views λ = lsolve(A', ybar[:, 1])
            @views Δybar, Δxbar = vjp(residual, yv[:, 1], yv[:, 1], tv[1], tv[1], xbar, p, -λ)
            xbar = Δxbar

        end

        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return [yv; tv'], pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _implicit_unsteady_reverse(solve, residual, x::TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules _implicit_unsteady_reverse(solve, residual, x::AbstractVector{<:TrackedReal}, p, drdy, lsolve)