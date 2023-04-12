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

# convert function to in-place if written in out-of-place manner.
# allocations occur elsewhere
function _make_onestep_inplace(onestep)
    if applicable(onestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # out-of-place if only 6 arguments
       return (y, yprev, t, tprev, xd, xci, p) -> begin
            y .= perform_step(yprev, t, tprev, xd, xci, p)
        end
    else
        return onestep
    end
end


"""
    explicit_unsteady(initialize, onestep, solve, t, xd, xc, p=(); cache=nothing)

Make reverse-mode efficient for explicit ODEs
Builds tape over each time step separately and analytically propagates between,
rather than recording a long tape over the entire time sequence.

# Arguments:
- `initialize::function`: Return initial state variables.  `y0 = initialize(t0, xd, xc0, p)`. May or may not depend
    on t0 (initial time), xd (design variables), xc0 (initial control variables), p (fixed parameters)
- `onestep::function`: define how states are updated (assuming one-step methods). `y = onestep(yprev, t, tprev, xd, xci, p)`
    or in-place `onestep!(y, yprev, t, tprev, xd, xci, p)`.  Set the next set of state variables `y`, given the previous state `yprev`,
    current time `t`, previous time `tprev`, design variables `xd`, current control variables `xc`, and fixed parameters `p`.
- `t::vector{float}`: time steps that simulation runs across
- `xd::vector{float}`: design variables, don't change in time, but do change from one run to the next (otherwise they would be parameters)
- `xc::matrix{float}`, size nxc x nt: control variables. xc[:, i] are the control variables at the ith time step.
- `p::tuple`: fixed parameters, i.e., they are always constant and so do not affect derivatives.  Default is empty tuple.

# Keyword Arguments:
 - `cache=nothing`: see `explicit_unsteady_cache`.  If computing derivatives more than once, you should compute the
    cache beforehand the save for later iterations.  Otherwise, it will be created internally.
"""
explicit_unsteady(initialize, onestep, t, xd, xc, p=(); cache=nothing) = _explicit_unsteady(initialize, _make_onestep_inplace(onestep), t, xd, xc, p, cache)


# solve the ODE, stepping in time, using the provided onestep function
function odesolve(initialize, onestep!, t, xd, xc, p)

    # initialization step
    @views y0 = initialize(t[1], xd, xc[:, 1], p)

    # allocate array
    y = zeros(length(y0), length(t))
    @views y[:, 1] .= y0

    # step through
    nt = length(t)
    for i = 2:nt
        @views onestep!(y[:, i], y[:, i-1], t[i], t[i-1], xd, xc[:, i], p)
    end

    return y
end


# If no AD, or forward, just solve normally.
_explicit_unsteady(initialize, onestep!, t, xd, xc, p, cache) = odesolve(initialize, onestep!, t, xd, xc, p)


# ReverseDiff cases

"""
    explicit_unsteady_cache(initialize, onestep!, ny, xd, xc, p=(); compile=false)

Initialize arrays and functions needed for explicit_unsteady reverse mode

# Arguments
- `initialize::function`: Return initial state variables.  `y0 = initialize(t0, xd, xc0, p)`. May or may not depend
    on t0 (initial time), xd (design variables), xc0 (initial control variables), p (fixed parameters)
- `onestep::function`: define how states are updated (assuming one-step methods). `y = onestep(yprev, t, tprev, xd, xci, p)`
    or in-place `onestep!(y, yprev, t, tprev, xd, xci, p)`.  Set the next set of state variables `y`, given the previous state `yprev`,
    current time `t`, previous time `tprev`, design variables `xd`, current control variables `xc`, and fixed parameters `p`.
- `ny::int`: number of states
- `nxd::int`: number of design variables
- `nxc::int`: number of control variables (at a given time step)
- `p::tuple`: fixed parameters, i.e., they are always constant and so do not affect derivatives.  Default is empty tuple.
- `compile::bool`: indicates whether the tape for the function `onestep` can be
   prerecorded.  Will be much faster but should only be `true` if `onestep` does not contain any branches.
   Otherwise, ReverseDiff may return incorrect gradients.
"""
function explicit_unsteady_cache(initialize, onestep, ny, nxd, nxc, p=(); compile=false)

    # convert to inplace
    onestep! = _make_onestep_inplace(onestep)

    # allocate y variable for one time step
    T = eltype(ReverseDiff.track([1.0]))  # TODO: is this always the same?
    yone = Vector{T}(undef, ny)

    # vector jacobian product
    function fvjp(yprev, t, tprev, xd, xci, λ)
        onestep!(yone, yprev, t[1], tprev[1], xd, xci, p)
        return λ' * yone
    end

    # allocate space for gradients
    gyprev = zeros(ny)
    gt = zeros(1)
    gtprev = zeros(1)
    gxd = zeros(nxd)
    gxci = zeros(nxc)
    gλ = zeros(ny)
    input = (gyprev, gt, gtprev, gxd, gxci, gλ)
    cfg = ReverseDiff.GradientConfig(input)

    # version with no tape
    vjp_step = (yprev, t, tprev, xd, xci, λ) -> begin
        ReverseDiff.gradient!((gyprev, gt, gtprev, gxd, gxci, gλ), fvjp, (yprev, [t], [tprev], xd, xci, λ), cfg)
        return gyprev, gxd, gxci
    end

    # --- repeat for initialize function -----
    fvjp_init(t, xd, xc1, λ) = λ' * initialize(t[1], xd, xc1, p)

    input_init = (gt, gxd, gxci, gλ)
    cfg_init = ReverseDiff.GradientConfig(input_init)

    vjp_init = (t1, xd, xc1, λ) -> begin
        ReverseDiff.gradient!((gt, gxd, gxci, gλ), fvjp_init, ([t1], xd, xc1, λ), cfg_init)
        return gxd, gxci
    end
    # --------------

    # ----- compile tape
    if compile
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(fvjp, input, cfg))

        vjp_step = (yprev, t, tprev, xd, xci, λ) -> begin
            ReverseDiff.gradient!((gyprev, gt, gtprev, gxd, gxci, gλ), tape, (yprev, [t], [tprev], xd, xci, λ))
            return gyprev, gxd, gxci
        end

        tape_init = ReverseDiff.compile(ReverseDiff.GradientTape(fvjp_init, input_init, cfg_init))

        vjp_init = (t1, xd, xc1, λ) -> begin
            ReverseDiff.gradient!((gt, gxd, gxci, gλ), tape_init, ([t1], xd, xc1, λ))
            return gxd, gxci
        end
    end
    # --------------

    return vjp_step, vjp_init
end

# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_explicit_unsteady), initialize, onestep!, t, xd, xc, p, cache)

    # evaluate solver
    yv = odesolve(initialize, onestep!, t, xd, xc, p)

    # get solution dimensions
    ny, nt = size(yv)

    # unpack cache
    if isnothing(cache)
        cache = explicit_unsteady_cache(initialize, onestep!, ny, xd, xc, p, compile=false)
    end
    vjp_step, vjp_init = cache

    function explicit_unsteady_pullback(ybar)

        # initialize outputs
        xdbar = zeros(length(xd))
        xcbar = zeros(size(xc))

        if nt > 1

            # --- Additional Time Steps --- #
            for i = nt:-1:2
                @views λ = ybar[:, i]
                @views Δybar, Δxdbar, Δxcibar = vjp_step(yv[:, i-1], t[i], t[i-1], xd, xc[:, i], λ)
                xdbar .+= Δxdbar
                @views xcbar[:, i] .= Δxcibar
                @views ybar[:, i-1] .+= Δybar
            end

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] .= Δxcibar

        else

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] .= Δxcibar

        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), xdbar, xcbar, NoTangent(), NoTangent()
    end

    return yv, explicit_unsteady_pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _explicit_unsteady(initialize, onestep!, t, xd::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, xc::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, p, cache)
ReverseDiff.@grad_from_chainrules _explicit_unsteady(initialize, onestep!, t, xd::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, xc, p, cache)
ReverseDiff.@grad_from_chainrules _explicit_unsteady(initialize, onestep!, t, xd, xc::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, p, cache)



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