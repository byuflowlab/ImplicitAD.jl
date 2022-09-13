# ImplicitAD

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/dev/)
[![Build Status](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Introduction

Many engineering analyses use implicit functions.  We can represent any such implicit function generally as:
```math
r(x, y(x)) = 0
```
where $r$ are the residual functions we wish to drive to zero, $x$ are inputs, and $y$ are the state variables, which are also outputs once the system of equations is solved.  In other words, $y$ is an implicit function of $x$. Note that all of these are vector quantities.  $x$ is of size $n_x$, and $r$ and $y$ are of the same size $n_r$ (must have equal number of unknowns and equations for a solution to exist). Graphically we could depict this relationship as:

x --> [ r(x, y) ] --> y

From a differentiation perspective, we would like to compute $dy/dx$.  One can often use algorithmic differentiation (AD) in the same way one would for any explicit function.  Once we unroll the iterations of the solver the set of instructions is explicit.  However, this is at best inefficient and at worse inaccurate or not possible (at least not without a lot more effort).  To obtain accurate derivatives by propgating AD through a solver, the solver must be solved to a tight tolerance.  Generally tighter than is required to converge the primal values.  Sometimes this is not feasible.  Additionally, many operations inside the solvers are not overloaded for AD, this is especially true if calling solvers in other languages.  But even if we can do it (tight convergence is possible and everything under the hood is overloaded) we usually still shouldn't do it, as it would be computationally inefficient.  Instead we can use implicit differentiation, and pass the resulting partials back in the midst of an AD chain, to allow for AD to work seemlessly with implicit functions without having to differentiate through them.


## Usage

We can generically represent the solver that converges the residuals and computes the corresponding state variables as:

``y = solve(x)``

Our larger code may then have a mix of explicit and implicit functions

```julia
function example(a)
    b = 2*a
    x = @. exp(b) + a
    y = solve(x)
    z = sqrt.(y)
    return z
end
```

To make this function compatible we only need to replace the call to ``solve`` with an overloaded function ``implicit_function`` defined in this module:
```julia
using ImplicitAD

function example(a)
    b = 2*a
    x = @. exp(b) + a
    y = implicit_function(solve, residual, x)
    z = sqrt.(y)
    return z
end
```
Note that we must provide the solve function: ``y = solve(x)`` and also the residual function: ``r = residual(x, y)``.  The resulting code is now compatible with `ForwardDiff` or `ReverseDiff`, computes the derivatives correctly, and without wasteful AD propgation inside the solver.  This approach will work with any solver, and the implementation of the solver need not be AD compatible since AD does not occur internal to the solver.

If one or both of the partial derivative Jacobians: $\partial r / \partial y$ and $\partial r / \partial x$ is known, then the user can specify those with the keywords arguments `drdy=drdy` and `drdx=drdx` where the function of the form `∂r_i/∂y_j = drdy(x, y)` (same for drdx).

Internally, the package computes Jacobian-vector products (or vector-Jacobian products for reverse mode) and assumes dense partial derivatives.  However, the user can overload any of the `jvp`, `vjp`, `computeA`, and `lsolve`, `tlsolve` functions for cases where: memory is preallocated, sparsity is significant, or specific linear factorizations or specific linear solvers would be more efficient.

See the unit tests for a few examples.

## Theory




We can get to the derivatives we are after by using implicit differentiation. Recall that we are solving:
$r(x, y(x)) = 0$ and that we want $dy/dx$.  Using the chain rule we find that:
```math
\frac{dr}{dx} = \frac{\partial r}{\partial x} + \frac{\partial r}{\partial y} \frac{dy}{dx} = 0
```
Noting that because $r = 0$ at the solution, which is a constant, its total derivative to a change in input $dr/dx$ must also be zero.

The Jacobian we are after we will call $J = \frac{dy}{dx}$.  The other two partial derivative Jacobians we will call $A = \frac{\partial r}{\partial y}$ and $B = \frac{\partial r}{\partial x}$ just for convenience.  The above equation then gives:
```math
A J = -B
```
Note that our desired Jacobian depends only on partial derivatives at the solution of our residual equations (and thus not on the solver path).  We can compute or provide these partial derivatives by various means, though typically we will leverage AD.

#### Forward Mode

If we are using forward mode algorithmic differentiation, once we reach this implicit function we already know the derivatives of x with respect to our variables of interest.  We will call this $dx$.  The above equation described how we can compute the Jacobian $J$, but what we really want is not $J$ but rather the Jacobian vector product $J dx$, which gives us the derivatives of y with respect to our variables of interest, and we continue forward along the AD path.

We multiply both sides by $dx$
```math
A J dx = -B dx
```
or 
```math
A dy = -B dx
```
where dy is what we are after.

First we deal with the right hand side computing:
```math
b = -B dx
```
We can compute the resulting Jacobian vector product (Jacobian B times some vector v) without actually forming the matrix B, which is more efficient.  Recall that $B = \partial r/\partial x$.  If we want the Jacobian product $B v$, where $v = -dx$ in this case, we see that we are after a weighted sum of the columns $\sum_i \partial r/\partial x_i v_i$ where $v_i$ are the weights.  Normally we pick off one of these partials in forward mode AD one at a time with different seeds $e_i = [0, 0, 1, 0]$, but if we set the initial seed to $v_i$ then the partial derivatives we compute are precisely this weighted sum.  In other words, we initialize $x$ as a dual number with its values as the input value $x$, and its derivatives as $v$, we evalute the residual function, then extract the partial derivatives which now give the desired Jacobian vector product $b = B v$ (where $v = -dx$).

With the right-hand side complete, we just need to compute the square Jacobian $A = \partial r / \partial y$ and solve the linear system.  
```math
A dy = b
```
Because A is square, forward mode AD is usually preferable for computing these partial derivatives. If we know the structure of A, we would want to provide an appropriate factorization of A.  Often, this Jacobian is sparse, and so using graph coloring is desirable.  Also for very large systems we can use matrix-free methods (e.g., Krylov methods) to solve the linear system without actually constructing A.

In this package the default is just dense forward-mode AD (not-in place). But the function can easily be overloaded for your use case where you can pre-allocate, use sparse matrices, graph coloring (see SparseDiffTools.jl), and your own desired linear solver.


#### Reverse Mode

If using reverse mode AD we have the derivatives of $dy$ and need to propgate backwards to get $dx$.  Again, we're not really interested in $J$, but rather in $dx = J^T dy$.  First we take the transpose of our governing equation:
```math
J^T A^T = -B^T
```
Let's multiply both sides by some as yet unknown vector $u$.
```math
J^T A^T u = -B^T u
```
We now would like to solve for the value of $u$ such that
```math
A^T u = dy
```
Doing so will give us the desired Jacobian vector product $J^T dy$:
```math
J^T A^T u = -B^T u
```
```math
J^T dy = -B^T u 
```
```math
dx = -B^T u 
```

In this case we start with solving the linear system:
```math
A^T u = dy
```
given input $dy$.  Again, we will want to use an appropriate factorization for A, sparsity, and appropriate linear solver.  The default implementation again uses dense forward mode AD, and the linear solve uses matrix right division so that the factorization for A, if provided, can be used directly and avoid transposing the matrix.

With u now known we compute the vector-Jacobian product
$dx = -B^T u = -(u^T B)^T$.  Like the Jacobian-vector product of the previous section, we can compute this more efficiently by avoiding explicitly constructing the matrix B.  The vector-Jacobian product $v^T \partial r/ \partial x$ can be computing with reverse-mode AD where the intial seed (adjoint vector) is set with the weights $v$ (where $v = -u$ in this case).

In this implementation, rather than go under the hood and manipulate the seed (since this function is not explicitly available), we can instead simply compute the gradient of:
```math
\frac{\partial }{\partial x}\left( v^T r(x, y) \right) = v^T \frac{\partial r}{\partial x}
```
which the expansion shows gives us the desired product.
To repeat, we just compute the gradient of $(v^T r)$  with respect to $x$ (note it is a gradient and not a Jacobian since we have a scalar output). This gives the desired vector, i.e., the derivatives dx.

