# ---- common functions -----------

# solve the ODE, stepping in time, using the provided onestep function
function odesolve(initialize, onestep, t, xd, xc, p)

    onestep! = _make_onestep_inplace(onestep)

    # initialization step
    @views y0 = initialize(t[1], xd, xc[:, 1], p)

    # allocate array
    T = promote_type(eltype(xd), eltype(xc))
    y = zeros(T, length(y0), length(t))
    @views y[:, 1] = y0

    # step through
    nt = length(t)
    for i = 2:nt
        @views onestep!(y[:, i], y[:, i-1], t[i], t[i-1], xd, xc[:, i], p)
    end

    return y
end

# convert function to in-place if written in out-of-place manner.
# allocations occur elsewhere
function _make_onestep_inplace(onestep)
    if applicable(onestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # out-of-place if only 6 arguments
       return (y, yprev, t, tprev, xd, xci, p) -> begin
            y .= onestep(yprev, t, tprev, xd, xci, p)
        end
    else
        return onestep
    end
end

function _make_residual_inplace(residual)
    if applicable(residual, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        return (r, y, yprev, t, tprev, xd, xci, p) -> begin
            r .= residual(y, yprev, t, tprev, xd, xci, p)
        end
    else
        return residual
    end
end



# ------ Overloads for explicit_unsteady ----------


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



# If no AD, or forward, just solve normally.
_explicit_unsteady(initialize, onestep!, t, xd, xc, p, cache) = odesolve(initialize, onestep!, t, xd, xc, p)


# ReverseDiff cases

"""
    explicit_unsteady_cache(initialize, onestep!, ny, nxd, nxc, p=(); compile=false)

Initialize arrays and functions needed for explicit_unsteady reverse mode

# Arguments
- `initialize::function`: Return initial state variables.  `y0 = initialize(t0, xd, xc0, p)`. May or may not depend
    on t0 (initial time), xd (design variables), xc0 (initial control variables), p (fixed parameters)
- `onestep::function`: define how states are updated (assuming one-step methods). `y = onestep(yprev, t, tprev, xd, xci, p)`
    or in-place `onestep!(y, yprev, t, tprev, xd, xci, p)`.  Set the next set of state variables `y`, given the previous state `yprev`,
    current time `t`, previous time `tprev`, design variables `xd`, current control variables for that time step `xci`, and fixed parameters `p`.
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
        cache = explicit_unsteady_cache(initialize, onestep!, ny, length(xd), length(xc[:, 1]), p, compile=false)
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
                @views xcbar[:, i] = Δxcibar
                @views ybar[:, i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] = Δxcibar

        else

            # --- Initial Time Step --- #
            @views λ = ybar[:, 1]
            Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] = Δxcibar

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


# default ∂r/∂y (square) uses forward mode
function drdy_forward(residual, r, y::AbstractVector, yprev, t, tprev, xd, xci, p)
    return ForwardDiff.jacobian((rt, ytilde) -> residual(rt, ytilde, yprev, t, tprev, xd, xci, p), r, y)
end

function drdy_forward(residual, rf, y::Number, yprev, t, tprev, xd, xci, p)  # 1D case
    return ForwardDiff.derivative((rt, ytilde) -> residual(rt, ytilde, yprev, t, tprev, xd, xci, p), r, y)
end




"""
    implicit_unsteady(initialize, onestep, residual, t, xd, xc, p=(); cache=nothing, drdy=drdy_forward, lsolve=linear_solve)

Make reverse-mode efficient for implicit ODEs (solvers compatible with AD).

# Arguments:
- `initialize::function`: Return initial state variables.  `y0 = initialize(t0, xd, xc0, p)`. May or may not depend
    on t0 (initial time), xd (design variables), xc0 (initial control variables), p (fixed parameters)
- `onestep::function`: define how states are updated (assuming one-step methods) including any solver. `y = onestep(yprev, t, tprev, xd, xci, p)`
    or in-place `onestep!(y, yprev, t, tprev, xd, xci, p)`.  Set the next set of state variables `y`, given the previous state `yprev`,
    current time `t`, previous time `tprev`, design variables `xd`, current control variables for that time step `xci`, and fixed parameters `p`.
- `residual::function`: define residual that is solved in onestep.  Either `r = residual(y, yprev, t, tprev, xd, xci, p)` where variables are same as above or
    `residual!(r, y, yprev, t, tprev, xd, xci, p)`.
- `t::vector{float}`: time steps that simulation runs across
- `xd::vector{float}`: design variables, don't change in time, but do change from one run to the next (otherwise they would be parameters)
- `xc::matrix{float}`, size nxc x nt: control variables. xc[:, i] are the control variables at the ith time step.
- `p::tuple`: fixed parameters, i.e., they are always constant and so do not affect derivatives.  Default is empty tuple.

# Keyword Arguments:
- `cache=nothing`: see `implicit_unsteady_cache`.  If computing derivatives more than once, you should compute the
    cache beforehand the save for later iterations.  Otherwise, it will be created internally.
- `drdy`: drdy(residual, r, y, yprev, t, tprev, xd, xci, p). Provide (or compute yourself): ∂ri/∂yj.  Default is
    forward mode AD.
- `lsolve::function`: `lsolve(A, b)`. Linear solve `A x = b` (where `A` is computed in
   `drdy` and `b` is computed in `jvp`, or it solves `A^T x = c` where `c` is computed
   in `vjp`). Default is backslash operator.
"""
implicit_unsteady(initialize, onestep, residual, t, xd, xc, p=(); cache=nothing, drdy=drdy_forward, lsolve=linear_solve) =
    _implicit_unsteady(initialize, _make_onestep_inplace(onestep), _make_residual_inplace(residual), t, xd, xc, p, cache, drdy, lsolve)


# If no AD, just solve normally.
_implicit_unsteady(initialize, onestep!, residual!, t, xd, xc, p, cache, drdy, lsolve) = odesolve(initialize, onestep!, t, xd, xc, p)

# forward mode, overload each onestep!

_implicit_unsteady(initialize, onestep!, residual!, t, xd::AbstractVector{<:ForwardDiff.Dual{T, V, N}}, xc::AbstractVector{<:ForwardDiff.Dual{T}}, p, cache, drdy, lsolve) where {T, V, N} = _implicit_unsteady_forward(initialize, onestep!, residual!, t, xd, xc, p, drdy, lsolve, T, V, N)
_implicit_unsteady(initialize, onestep!, residual!, t, xd, xc::AbstractVector{<:ForwardDiff.Dual{T, V, N}}, p, cache, drdy, lsolve) where {T, V, N} = _implicit_unsteady_forward(initialize, onestep!, residual!, t, xd, xc, p, drdy, lsolve, T, V, N)
_implicit_unsteady(initialize, onestep!, residual!, t, xd::AbstractVector{<:ForwardDiff.Dual{T, V, N}}, xc, p, cache, drdy, lsolve) where {T, V, N} = _implicit_unsteady_forward(initialize, onestep!, residual!, t, xd, xc, p, drdy, lsolve, T, V, N)

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function _implicit_unsteady_forward(initialize, onestep!, residual!, t, xd, xc, p, drdy, lsolve, T, V, N)

    # initialize
    @views yd0 = initialize(t[1], xd, xc[:, 1], p)

    # extract values
    xdv = fd_value(xd)
    xcv = fd_value(xc)
    y0 = fd_value(yd0)

    # get solution dimensions
    ny = length(y0)
    nt = length(t)

    # initialize output
    yv = similar(y0, ny, nt)
    yd = similar(yd0, ForwardDiff.Dual{T, V, N}, ny, nt)

    # initialize jvp's
    rd = Vector{ForwardDiff.Dual{T, V, N}}(undef, ny)
    function jvp_step(residual, y, yprevd, t, tprev, xdd, xcd, p)

        # evaluate residual function
        residual(rd, y, yprevd, t, tprev, xdd, xcd, p)  # constant y

        # extract partials
        b = -fd_partials(rd)

        return b
    end

    # allocate for drdy
    rf = Vector{Float64}(undef, ny)

    # --- Initial Time Step --- #
    yv[:, 1] .= y0
    yd[:, 1] .= yd0

    # --- Additional Time Steps --- #
    for i = 2:nt

        # update state using values (no duals)
        @views onestep!(yv[:, i], yv[:, i-1], t[i], t[i-1], xdv, xcv[:, i], p)

        # solve for Jacobian-vector product
        @views b = jvp_step(residual!, yv[:, i], yd[:, i-1], t[i], t[i-1], xd, xc[:, i], p)

        # compute partial derivatives
        @views A = drdy(residual!, rf, yv[:, i], yv[:, i-1], t[i], t[i-1], xdv, xcv[:, i], p)

        # linear solve
        ydot = lsolve(A, b)

        # repack in ForwardDiff Dual
        @views yd[:, i] = pack_dual(yv[:, i], ydot, T)
    end

    return yd
end


"""
    implicit_unsteady_cache(initialize, residual, ny, nxd, nxc, p=(); compile=false)

Initialize arrays and functions needed for implicit_unsteady reverse mode

# Arguments
- `initialize::function`: Return initial state variables.  `y0 = initialize(t0, xd, xc0, p)`. May or may not depend
    on t0 (initial time), xd (design variables), xc0 (initial control variables), p (fixed parameters)
- `residual::function`: define residual that is solved in onestep.  Either `r = residual(y, yprev, t, tprev, xd, xci, p)` where variables are same as above
    `residual!(r, y, yprev, t, tprev, xd, xci, p)`.
- `ny::int`: number of states
- `nxd::int`: number of design variables
- `nxc::int`: number of control variables (at a given time step)
- `p::tuple`: fixed parameters, i.e., they are always constant and so do not affect derivatives.  Default is empty tuple.
- `compile::bool`: indicates whether the tape for the function `onestep` can be
   prerecorded.  Will be much faster but should only be `true` if `onestep` does not contain any branches.
   Otherwise, ReverseDiff may return incorrect gradients.
"""
function implicit_unsteady_cache(initialize, residual, ny, nxd, nxc, p=(); compile=false)

    # convert to inplace
    residual! = _make_residual_inplace(residual)

    # allocate r variable for one time step
    T = eltype(ReverseDiff.track([1.0]))  # TODO: is this always the same?
    r = Vector{T}(undef, ny)

    # vector jacobian product
    function fvjp(y, yprev, t, tprev, xd, xci, λ)
        residual!(r, y, yprev, t[1], tprev[1], xd, xci, p)
        return λ' * r
    end

    # allocate space for gradients
    gy = zeros(ny)
    gyprev = zeros(ny)
    gt = zeros(1)
    gtprev = zeros(1)
    gxd = zeros(nxd)
    gxci = zeros(nxc)
    gλ = zeros(ny)
    input = (gy, gyprev, gt, gtprev, gxd, gxci, gλ)
    cfg = ReverseDiff.GradientConfig(input)

    # version with no tape
    vjp_step = (y, yprev, t, tprev, xd, xci, λ) -> begin
        ReverseDiff.gradient!((gy, gyprev, gt, gtprev, gxd, gxci, gλ), fvjp, (y, yprev, [t], [tprev], xd, xci, λ), cfg)
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

        vjp_step = (y, yprev, t, tprev, xd, xci, λ) -> begin
            ReverseDiff.gradient!((gy, gyprev, gt, gtprev, gxd, gxci, gλ), tape, (y, yprev, [t], [tprev], xd, xci, λ))
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
function ChainRulesCore.rrule(::typeof(_implicit_unsteady), initialize, onestep!, residual!, t, xd, xc, p, cache, drdy, lsolve)

    # evaluate solver
    yv = odesolve(initialize, onestep!, t, xd, xc, p)

    # get solution dimensions
    ny, nt = size(yv)

    # unpack cache
    if isnothing(cache)
        cache = implicit_unsteady_cache(initialize, residual!, ny, length(xd), length(xc[:, 1]), p, compile=false)
    end
    vjp_step, vjp_init = cache

    # initialize
    r = Vector{Float64}(undef, ny)
    A = Matrix{Float64}(undef, ny, ny)
    λ = Vector{Float64}(undef, ny)

    function pullback(ybar)

        xdbar = zeros(length(xd))
        xcbar = zeros(size(xc))

        if nt > 1

            # --- Additional Time Steps --- #
            for i = nt:-1:2
                @views A .= drdy(residual!, r, yv[:, i], yv[:, i-1], t[i], t[i-1], xd, xc[:, i], p)
                @views λ .= lsolve(A', ybar[:, i])
                @views Δybar, Δxdbar, Δxcibar = vjp_step(yv[:, i], yv[:, i-1], t[i], t[i-1], xd, xc[:, i], -λ)
                xdbar .+= Δxdbar
                @views xcbar[:, i] = Δxcibar
                @views ybar[:, i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views λ .= -ybar[:, 1]
            @views Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], -λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] = Δxcibar

        else

            # --- Initial Time Step --- #
            @views λ .= -ybar[:, 1]
            @views Δxdbar, Δxcibar = vjp_init(t[1], xd, xc[:, 1], -λ)
            xdbar .+= Δxdbar
            @views xcbar[:, 1] = Δxcibar

        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), xdbar, xcbar, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return yv, pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _implicit_unsteady(initialize, onestep!, residual!, t, xd::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, xc::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, p, cache, drdy, lsolve)
ReverseDiff.@grad_from_chainrules _implicit_unsteady(initialize, onestep!, residual!, t, xd::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, xc, p, cache, drdy, lsolve)
ReverseDiff.@grad_from_chainrules _implicit_unsteady(initialize, onestep!, residual!, t, xd, xc::Union{ReverseDiff.TrackedArray, AbstractVector{<:ReverseDiff.TrackedReal}}, p, cache, drdy, lsolve)