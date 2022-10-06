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

Limitation: ReverseDiff does not currently support compiling the tape for custome rules.  See this issue in ReverseDiff: https://github.com/JuliaDiff/ReverseDiff.jl/issues/187

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

Now if we tried to run `ForwardDiff.jacobian(program, x)` it will not work.  It's not compatible with the internals of NLSolve, but even if it were it would be an inefficient way to compute the derivatives.  We now need to modify this script to use our package.  Here is what the modified `program` function will look like.

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

If the user can provide (or lazily compute) their own partial derivatives for ``\partial{r}/\partial{y}`` then they can provide their own subfunction:
`∂r∂y = drdy(residual, y, x, p)` (where `r = residual(y, x, p)`).  The default implementation computes these partials with ForwardDiff. Some examples where one may wish to override this behavior are for cases significant sparsity (e.g., using SparseDiffTools), for a large number of residuals (e.g., preallocating this Jacobian), or to provide a specific matrix factorization.

Additionally the user can override the linear solve:
`x = lsolve(A, b)`.  The default is the backslash operator.  One example where a user may wish to override is to use matrix-free Krylov methods for large systems (in connection with the computation for ∂r∂y).

The other partials, ∂r/∂x, are not computed directly, but rather are used in efficient Jacobian vector (or vector Jacobian) products.

As an example, let's continue the same problem from the previous section.  We note that we can provide the Jacobian ``\partial r/\partial y`` analytically and so we will skip the internal ForwardDiff implementation. We provide our own function for `drdy`, and we will preallocate so we can modify the Jacobian in place:

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