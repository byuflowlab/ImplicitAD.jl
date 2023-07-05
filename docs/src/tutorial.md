# Quick Start

We can generically represent the solver that converges the residuals and computes the corresponding state variables as:

`y = solve(x, p)`

Where x are variables and p are fixed parameters.  Our larger code may then have a mix of explicit and implicit functions.

```julia
function example(a)
    b = 2*a
    x = @. exp(b) + a
    y = solve(x)
    z = sqrt.(y)
    return z
end
```

To make this function compatible we only need to replace the call to `solve` with an overloaded function `implicit` defined in this module:
```julia
using ImplicitAD

function example(a)
    b = 2*a
    x = @. exp(b) + a
    y = implicit(solve, residual, x)
    z = sqrt.(y)
    return z
end
```
Note that the only new piece of information we need to expose is the residual function so that partial derivatives can be computed.  The new function is now compatible with ForwardDiff or ReverseDiff, for any solver, and efficiently provides the correct derivatives without differentiating inside the solver.

The residuals can either be returned:
`r = residual(y, x, p)`
or modified in place:
`residual!(r, y, x, p)`.

The input x should be a vector, and p is a tuple of fixed parameters.  The state and corresponding residuals, y and r, can be vectors or scalars (for 1D residuals).  The subfunctions are overloaded to handle both cases efficiently.

Limitation: ReverseDiff does not currently support compiling the tape for custome rules.  See this issue in ReverseDiff: <https://github.com/JuliaDiff/ReverseDiff.jl/issues/187>

## Basic Usage

Let's go through a complete example now. Assume we have two nonlinear implicit equations:
```math
r_1(x, y) = (y_1 + x_1) (y_2^3 - x_2) + x_3 = 0
r_2(x, y) = \sin(y_2 \exp(y_1) - 1) x_4 = 0
```

We will use the NLsolve package to solve these equations (refer to the first example in their documentation if not familiar with NLsolve).  We will also put explict operations before and after the solve just to show how this will work in the midst of a larger program.  In this case we use the in-place form of the residual function.

```@example basic
using NLsolve

function residual!(r, y, x, p)
    r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
    r[2] = sin(y[2]*exp(y[1])-1)*x[4]
end

function solve(x, p)
    rwrap(r, y) = residual!(r, y, x[1:4], p)  # closure using some of the input variables within x just as an example
    res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
    return res.zero
end

function program(x)
    z = 2.0*x
    w = z + x.^2
    y = solve(w)
    return y[1] .+ w*y[2]
end
nothing # hide
```

Now if we tried to run `ForwardDiff.jacobian(program, x)` it will not work (actually it will work if we change the type of our starting point ``x_0`` to also be a Dual).  But even for solvers where we can propagate AD through it would typically be an inefficient way to compute the derivatives.  We now need to modify this script to use our package.  Here is what the modified `program` function will look like.

```@example basic
using ImplicitAD

function modprogram(x)
    z = 2.0*x
    w = z + x.^2
    y = implicit(solve, residual!, w)
    return y[1] .+ w*y[2]
end
nothing # hide
```

It is now both ForwardDiff and ReverseDiff compatible.

```@example basic
using ForwardDiff
using ReverseDiff

x = [1.0; 2.0; 3.0; 4.0; 5.0]

J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)
println(J1)
println("max abs difference = ", maximum(abs.(J1 - J2)))
```


## Overloading Subfunctions

If the user can provide (or lazily compute) their own partial derivatives for ∂r/∂y then they can provide their own subfunction:
`∂r∂y = drdy(residual, y, x, p)` (where `r = residual(y, x, p)`).  The default implementation computes these partials with ForwardDiff. Additionally the user can override the linear solve:
`x = lsolve(A, b)`.  The default is the backslash operator.

Some examples where one may wish to override these behaviors are for cases with significant sparsity (e.g., using SparseDiffTools), to preallocate the Jacobian, to provide a specific matrix factorization, or if the number of states is large overriding both methods will often be beneficial so that you can use iterative linear solvers (matrix-free Krylov methods) and thus provide efficient Jacobian vector products rather than a Jacobian.

The other partials, ∂r/∂x, are not computed directly, but rather are used in efficient Jacobian vector (or vector Jacobian) products.

As an example, let's continue the same problem from the previous section.  We note that we can provide the Jacobian ∂r/∂y analytically and so we will skip the internal ForwardDiff implementation. We provide our own function for `drdy`, and we will preallocate so we can modify the Jacobian in place:

```@example basic
function drdy(residual, y, x, p, A)
    A[1, 1] = y[2]^3-x[2]
    A[1, 2] = 3*y[2]^2*(y[1]+x[1])
    u = exp(y[1])*cos(y[2]*exp(y[1])-1)*x[4]
    A[2, 1] = y[2]*u
    A[2, 2] = u
    return A
end
nothing # hide
```

We can now pass this function in with a keyword argument to replace the default implementation for this subfunction.

```@example basic
function modprogram2(x)
    z = 2.0*x
    w = z + x.^2
    A = zeros(2, 2)
    my_drdy(residual, y, x, p) = drdy(residual, y, x, p, A)
    p = () # no parameters in this case
    y = implicit(solve, residual!, w, p, drdy=my_drdy)
    return y[1] .+ w*y[2]
end

J3 = ForwardDiff.jacobian(modprogram2, x)
println(maximum(abs.(J1 - J3)))

```


## Linear residuals

If the residuals are linear (i.e., Ay = b) we could still use the above nonlinear formulation but it will be inefficient or require more work from the user.  Instead, we can provide the partial derivatives analytically for the user.  In this case, the user need only provide the inputs A and b.

Let's consider a simple example.

```@example linear
using SparseArrays: sparse

function program(x)
    Araw = [x[1]*x[2] x[3]+x[4];
        x[3]+x[4] 0.0]
    b = [2.0, 3.0]
    A = sparse(Araw)
    y = A \ b
    z = y.^2
    return z
end
nothing # hide
```

This function is actually not compatible with ForwardDiff because of the use of a sparse matrix (obviously unnecessary with such a small matrix, just for illustration).  We now modify this function using this package, with a one line change using `implicit_linear`, and can now compute the Jacobian.

Let's consider a simple example.

```@example linear
using ImplicitAD
using ForwardDiff
using ReverseDiff

function modprogram(x)
    Araw = [x[1]*x[2] x[3]+x[4];
        x[3]+x[4] 0.0]
    b = [2.0, 3.0]
    A = sparse(Araw)
    y = implicit_linear(A, b)
    z = y.^2
    return z
end

x = [1.0; 2.0; 3.0; 4.0]

J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)

println(J1)
println(maximum(abs.(J1 - J2)))
```

For `implicit_linear` there are two keywords for custom subfunctions:

1) `lsolve(A, b)`: same purpose as before: solve ``A x = b`` where the default is the backslash operator.
2) `fact(A)`: provide a matrix factorization of ``A``, since two linear solves are performed (for the primal and dual values).  default is `factorize` defined in `LinearAlgebra`.

## Eigenvalue Problems

Like the linear case, we can provide analytic derivatives for eigenvalue problems (many of which are not overridden for AD anyway).  These are problems of the form:
```math
A v = \lambda B v
```
For standard eigenvalue problems B is the identity matrix.  The user just needs to provide the matrices A, B, and some function to solve the eigenvalue problem (which could use any method).  The solver should be in the following form: `λ, V, U = eigsolve(A, B)` where λ is a vector of eigenvalues, V is a matrix with corresponding eigenvectors in the columns (i.e., λ[i] corresponds to V[:, i]), and U is a matrix whose columns contain the left eigenvectors (u^H A = λ u^H B).  The left eigenvectors must be in the same order as the right eigenvectors (i.e., U' * B * V must be diagonal).  U need not be normalized as we do that internally.  Note that if A and B are symmetric/Hermitian then U = V.  Currently only eigenvalue derivatives are provided (not eigenvector derivatives).
Let's now see an example.

```@example eigen
using ImplicitAD
using ForwardDiff
using ReverseDiff
using LinearAlgebra: eigvals, eigvecs

function eigsolve(A, B)
    λ = eigvals(A, B)
    V = eigvecs(A, B)
    U = eigvecs(A', B')

    return λ, V, U
end

function test(x)
    A = [x[1] x[2]; x[3] x[4]]
    B = [x[5] x[6]; x[7] x[8]]
    λ = ImplicitAD.implicit_eigval(A, B, eigsolve)  # replaced from λ, _, _ = eigsolve(A, B)
    z = [real(λ[1]) + imag(λ[1]); real(λ[2]) + imag(λ[2])]  # just some dummy output
    return z
end

x = [-4.0, -17.0, 2.0, 2.0, 2.5, 5.6, -4.0, 1.1]
J1 = ForwardDiff.jacobian(test, x)
J2 = ReverseDiff.jacobian(test, x)

println(J1)
println(maximum(abs.(J1 - J2)))
```

## Ordinary Differential Equations

This package supports both explicit and implicit ODEs. For explicit ODEs only the reverse case is overloaded: it compiles a tape for a single time step, then reuses that tape at each subsequent time step while analytically propagating derivatives between time steps.  The precomputation and reuse of the tape can significantly reduce memory and improve speed as compared to recording the tape over the whole time series, particularly as the tape gets longer.  Forward mode is not overloaded, it just does regular forward mode AD through the whole time series, since there is nothing to exploit in an explicit case.

For implicit ODEs both reverse and forward mode are overloaded.  Reverse mode takes advantage of the same reduced tape size.  Additionally, both forward and reverse mode use implicit differentiation (direct or adjoint) for the solve that occurs at each time step.

Currently, we only support one-step methods. The underlying methodology works for multi-step methods, it is only the function signature that is currently limited to one-step.  This choice was made, at least for now, just to keep the API simple while covering the vast majority of uses cases.

Both out-of-place and in-place functions are supported.  For both ODE types, an out-of-place time step has the signature:

`y = onestep(yprev, t, tprev, xd, xci, p)`

whereas in-place looks like:

`onestep!(y, yprev, t, tprev, xd, xci, p)`

These functions update the state `y` given:
- `yprev`: the previous state
- `t`: the current time
- `tprev`: the previous time
- `xd`: design variables. design variables are fixed in time but change across design iterations.
- `xci`: control variables. control variables change at each time step (and likely across design iterations), and `xci` refers to the control variables at this time step.
- `p`: parameters. parameters are fixed in time and across design iterations.

### Explicit ODEs

We start with an explicit ODE, and use an out-of-place function for simplicity.  Recall that we need to pass in an ODE method with the following signature: `y = onestep(yprev, t, tprev, xd, xci, p)`

In this case we choose an explicit forward Euler method.  This is not generally an effective method, but is chosen just to keep the example brief. Again, any one-step method can be used.  The unit tests show a more complex example using a 5/4 Runge-Kutta method.

```@example explicit
using ImplicitAD
using ReverseDiff

function fwdeuler(odefun, yprev, t, tprev, xd, xci, p)

    dt = t - tprev
    y = yprev .+ dt*odefun(yprev, t, xd, xci, p)

    return y
end
nothing # hide
```

Note that we've written it generically so that a user can pass in any ODE function with the signature: `odefun(yprev, t, xd, xci, p)`.  In our case, let's choose a simple set of ODEs, the Lotka–Volterra equations.  We set the variables of the Lotka–Volterra equations as design variables, so there are no parameters and no control variables:

```@example explicit
function lotkavolterra(y, t, xd, xci, p)
    return [xd[1]*y[1] - xd[2]*y[1]*y[2];
        -xd[3]*y[2] + xd[4]*y[1]*y[2]]
end
nothing # hide
```

We can now combine the two to create our one-step method for this specific problem:

```@example explicit
onestep(yprev, t, tprev, xd, xci, p) = fwdeuler(lotkavolterra, yprev, t, tprev, xd, xci, p)
nothing # hide
```

The package also expects an initialization function of the form: `y0 = initialize(t0, xd, xc0, p)`.  In our case, we just pass in a simple starting point:

```@example explicit
initialize(t0, xd, xc0, p) = [1.0, 1.0]
nothing # hide
```

The remaining inputs are the time steps (we choose 10 seconds with a spacing of 0.1), design variables, control variables (a matrix of size number of control variables by number of time steps), and parameters (a tuple).

```@example explicit
t = range(0.0, 10.0, step=0.1)
xd = [1.5, 1.0, 3.0, 1.0]
xc = Matrix{Float64}(undef, 0, length(t))
p = ()
nothing # hide
```

Let's now put this into a program.  For simplicity, let's assume that this is our entire model, it just takes the design variables in as inputs, and the overall output is the sum of our states at the last time step. If we weren't using the features of this package, we would just now initialize, then iterate through each time step to update the states.  This is a simple function to write, but there is a built-in function called `odesolve` so we'll just use it.

```@example explicit
function program(x)
    y = ImplicitAD.odesolve(initialize, onestep, t, x, xc, p)
    return sum(y[:, end])
end
nothing # hide
```

However, this is not what we want.  We want to reuse the tape across time steps and analyticly propagate of derivatives between time steps, so that we don't have to record a long tape with reverse mode AD.  In this case the function just requires a name change `explicit_unsteady`:

```@example explicit
function modprogram_nocache(x)
    y = explicit_unsteady(initialize, onestep, t, x, xc, p)
    return sum(y[:, end])
end
nothing # hide
```

However, to really benefit we should preallocate our array storage, and compile the reverse-mode tape.

!!! warning

    Compiling should only be done if there is no branching in your ODE (e.g., conditional statements).  Otherwise, the derivatives may not be correct since you will compile for a branch that you might not follow at a later evaluation.

In this case we can safely compile the tape.
```@example explicit
ny = 2  # number of states
nxd = length(xd)  # number of design vars
nxc = 0  # number of control variables
cache = explicit_unsteady_cache(initialize, onestep, ny, nxd, nxc, p; compile=true)
nothing # hide
```

We now revise the function to use this cache.

```@example explicit
function modprogram(x)
    y = explicit_unsteady(initialize, onestep, t, x, xc, p; cache)
    return sum(y[:, end])
end
nothing # hide
```

Finally, let's compute the gradients.  First, by creating a long tape with the standard approach (note that in this simple example we don't spend the effort compile this tape and preallocate the storage, but in the benchmarks shown in the paper we do this):

```@example explicit
g1 = ReverseDiff.gradient(program, xd)
```

Or by compiling across just one-time step (avoiding the memory penalties that occur for large problems) and analytically propagating between steps.

```@example explicit
g2 = ReverseDiff.gradient(modprogram, xd)
```

We can see that these give the same results.
```@example explicit
println(g1 - g2)
```

This problem is so small we won't see any real time difference.  The paper linked in the README shows how these benefits become significant as the number of time steps increase.

This can also be done with in-place functions.  This often requires a little more care on types, depending on the one-step method used.  The unit tests show more extensive examples, and we'll do an in-place example with implicit ODEs next.

### Implicit ODEs

Let's now try an implicit ODE time stepping method.  Here we expect to see more benefit because we can take advantage of the shorter tape compilation as in the explicit case, but we can also take advantage of adjoints applied to the solves at each time step.  We'll do this one in-place although again both out-of-place and in-place forms are accepted.

Recall that the signature for in-place is: `onestep!(y, yprev, t, tprev, xd, xci, p)`.

The only new piece of information we need for implicit methods is, like the nonlinear solver case, the residual at a given time step.  For this example, we'll solve the Robertson function that is defined as follows, written in residual form.

```@example implicit
using ImplicitAD
using ReverseDiff
using NLsolve

function robertson(r, dy, y, t, xd, xci, p)
    r[1] = -xd[1]*y[1]               + xd[2]*y[2]*y[3] - dy[1]
    r[2] = xd[1]*y[1] - xd[3]*y[2]^2 - xd[2]*y[2]*y[3] - dy[2]
    r[3] = y[1] + y[2] + y[3] - xd[4]
end
nothing # hide
```

To update each state we use an implicit Euler method for simplicity:

```@example implicit
residual!(r, y, yprev, t, tprev, xd, xci, p) = robertson(r, (y .- yprev)/(t - tprev), y, t, xd, xci, p)
nothing # hide
```

To solve these sets of residuals, we use the default trust-region method in NLsolve as our solver.  This now provides the expected form of our one-step update.  Note that it looks just like the explicit case since the solver occurs inside.

```@example implicit
function onestep!(y, yprev, t, tprev, xd, xci, p)
    f!(r, yt) = residual!(r, yt, yprev, t, tprev, xd, xci, p)
    sol = nlsolve(f!, yprev, autodiff=:forward, ftol=1e-12)
    y .= sol.zero
    return nothing
end
nothing # hide
```

We again need an initialization function and to set the time steps, design variables, control variables, and parameters:
```@example implicit
function initialize(t0, xd, xc0, p)
    y0 = [1.0, 0.0, 0.0]
    return y0
end

t = range(1e-6, 1e5, length=100)
xd = [0.04, 1e4, 3e7, 1.0]
xc = Matrix{Float64}(undef, 0, 100)
p = ()
nothing # hide
```

We again need to set the cache, but notice that it requires passing in the residual function with the signature: `residual!(r, y, yprev, t, tprev, xd, xci, p)`

```@example implicit
ny = 3
nxd = 4
nxc = 0
cache = ImplicitAD.implicit_unsteady_cache(initialize, residual!, ny, nxd, nxc, p; compile=true)
nothing # hide
```

Our program will return all the states at the last time step.  Again, we start with the canonical case where we don't use adjoints, and record the tape across the entire time series.
```@example implicit
function program(x)
    y = ImplicitAD.odesolve(initialize, onestep!, t, x, xc, p)
    return y[:, end]
end
nothing # hide
```

The modified version only requires one new piece of information, the residuals, so that we can perform the adjoint.

```@example implicit
function modprogram(x)
    y = implicit_unsteady(initialize, onestep!, residual!, t, x, xc, p; cache)
    return y[:, end]
end
nothing # hide
```

Finally, we compute the Jacobian. The original program is actually not compatible with ReverseDiff because of the internals of NLSolve.  Fortunately, the adjoint doesn't need to propagate through the solver so we use our modified version.

```@example implicit
J = ReverseDiff.jacobian(modprogram, xd)
```

These examples are complete but brief.  Longer examples are available in the unit tests (see `tests` folder) and in the cases from the paper (see `examples` folder).

## Custom Rules

Consider now explicit (or potentially implicit) functions of the form: `y = func(x, p)` where `x` are variables and `p` are fixed parameters.  For cases where `func` is not compatible with AD, or for cases where we have a more efficient rule, we will want to insert our own derivatives into the AD chain.  This functionality could also be used for mixed-mode AD.  For example, by wrapping some large section of code in a function that we reply reverse mode AD on, then using that as a custom rule for the overall code that might be operating in forward mode.  More complex nestings are of course possible.

One common use case for a custom rule is when an external function call is needed, i.e., a function from another programming language is used within a larger Julia code.

We provide five options for injecting the derivatives of `func`.  You can provide the Jacobian `J = dy/dx`, or the JVPs/VJPs ``J v `` and ``v^T J``.  Alternatively, you can allow the package to estimate the derivatives using forward finite differencing, central finite differencing, or complex step.  In forward operation (with the finite differencing options) the package will choose between computing the Jacobian first or computing JVPs directly in order to minimize the number of calls to `func`.

Below is a simple example.  Let's first create a function, we call external, meant to represent a function that we cannot pass AD through (but of course can in this simple example).

```@example custom
function external(x, p)
    y = x.^2
    z = [x; y]
    return z
end
nothing # hide
```

Let's now call this function from our larger program that we wish to pass AD through:

```@example custom
function program(x)
    y = sin.(x)
    p = ()
    z = external(y, p)
    w = 5 * z
    return w
end
nothing # hide
```

Again, we assume that external is not AD compatible, so we modify this function with the `provide_rule` function provided in this package.

```@example custom
using ImplicitAD

function modprogram(x)
    y = sin.(x)
    p = ()
    z = provide_rule(external, y, p; mode="ffd")
    w = 5 * z
    return w
end
nothing # hide
```

The last argument we provided is the mode, which can be either:
- "ffd": forward finite differencing
- "cfd": central finite differencing
- "cs": complex step
- "jacobian": you provide `J = jacobian(x, p)`, use also keyword jacobian
- "vp": you provide Jacobian vector product `jvp(x, p, v)` and vector Jacobian product `vjp(x, p, v)` see keywords `jvp` and `vjp`

We can now use ForwardDiff or ReverseDiff and just the external code will be finite differenced (since we chose "ffd" above), and inserted into the AD chain.  Since this example is actually AD compatible everywhere we compare to using ForwardDiff through everything.

```@example custom
using ForwardDiff
using ReverseDiff

x = [1.0; 2.0; 3.0]
Jtrue = ForwardDiff.jacobian(program, x)
J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)

println(Jtrue)
println(maximum(abs.(Jtrue - J1)))
println(maximum(abs.(Jtrue - J2)))
```

Central difference and complex step work similarly.  The example, below shows how to provide the Jacobian.

```@example custom
using LinearAlgebra: diagm, I

function jacobian(x, p)
    dydx = diagm(2*x)
    dzdx = [I; dydx]
    return dzdx
end

function modprogram(x)
    y = sin.(x)
    p = ()
    z = provide_rule(external, y, p; mode="jacobian", jacobian)
    w = 5 * z
    return w
end

J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)
println(maximum(abs.(Jtrue - J1)))
println(maximum(abs.(Jtrue - J2)))
```

Finally, we show how to provide JVPs and VJPs.

```@example custom

function jvp(x, p, v)
    nx = length(x)
    return [v; 2*x.*v]
end

function vjp(x, p, v)
    nx = length(x)
    return v[1:nx] .+ 2*x.*v[nx+1:end]
end

function modprogram(x)
    y = sin.(x)
    p = ()
    z = provide_rule(external, y, p; mode="vp", jvp, vjp)
    w = 5 * z
    return w
end

J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)
println(maximum(abs.(Jtrue - J1)))
println(maximum(abs.(Jtrue - J2)))
```