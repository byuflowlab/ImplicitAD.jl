# Examples

## Basic Usage

Let's go through a complete example now. Assume we have two nonlinear implicit equations:
```math
r_1(x, y) = (y_1 + x_1) (y_2^3 - x_2) + x_3 = 0
r_2(x, y) = \sin(y_2 \exp(y_1) - 1) x_4 = 0
```

We will use the NLsolve package to solve these equations (refer to the first example in their documentation if not familiar with NLsolve).  We will also put explict operations before and after the solve just to show how this will work in the midst of a larger program.  

```@example basic

using NLsolve

function residual!(r, x, y)
    r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
    r[2] = sin(y[2]*exp(y[1])-1)*x[4]
end

function solve(x)
    rwrap(r, y) = residual!(r, x[1:4], y)  # closure using some of the input variables within x just as an example
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
    y = implicit_function(solve, residual!, w)
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

As outlined in the math derivation, the forward mode consists of three main operations and custom implementations can be passed in.

1) `jvp(residual, x, y, v)`:  Compute the Jacobian vector product `b = B*v` where ``B_{ij} = \partial r_i/\partial x_j``.  The default implementation uses forward mode AD where the Jacobian is not explicitly constructed (hence a Jacobian vector product).  This requires just evaluating the residual explicitly with dual numbers.

2) `drdy(residual, x, y)`: Provide/compute or lazily instantiate ``\partial r_i/\partial y_j``.  The default is forward mode AD.

3) `lsolve(A, b)`: Solve linear system ``A x = b`` where ``A`` is computed in `drdy` and ``b`` is computed in `jvp`.  The default is the backsplash operator.

In the reverse mode the operations are:

1) `drdy(residual, x, y)`: same as above.

2) `lsolve(A, b)`: same as above (although the passed in `A` is now the transpose of the matrix computed in `drdy` and ``b`` is a provided input).

3) `vjp(residual, x, y, v)`:  Compute the vector Jacobian product ``c = B^T v = (v^T B)^T`` where ``B_{ij} = \partial r_i/\partial x_j``.  The default implementation uses reverse mode AD where the Jacobian is not explicitly constructed.  Instead only a gradient call is needed.

Note that in all of these subfunctions `residual` is of the explicit form: `r = residual(x, y)`.  Since two of the functions are repeated, there are 4 functions that can be overriden if desired. Perhaps the most common would be to override `drdy` for cases where the Jacobian has significant sparsity, or is large so memory preallocation is important, or to apply a specific matrix factorization.  The linear solver `lsolve` might be overriden to use a Krylov method (in connection with using JVPs rather than an explicit `drdy`).  The functions `jvp` and `vjp` would be less commonly overriden, as they are efficient, but are available as needed.  There is also a keyword `drdx` where one can pass in a function of the same signature of `drdy` (but to compute ``\partial{r}/\partial{x}``).  This is less commonly useful as both `jvp` and `vjp` would then use explicit matrix multiplication, but may be beneficial in some cases.

As an example of custom subfunctions, let's continue the same problem from the previous section.  We note that we can provide the Jacobian ``\partial r/\partial y`` analytically and so we will skip the internal ForwardDiff implementation. We provide our own function for `drdy`, and we will preallocate so we can modify the Jacobian in place:

```@example basic
function drdy(residual, x, y, A)
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
    my_drdy(residual, x, y) = drdy(residual, x, y, A)
    y = implicit_function(solve, residual!, w, drdy=my_drdy)
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

This function is actually not compatible with ForwardDiff because of the use of a sparse matrix (obviously unnecessary with such a small matrix, just for illustration).  We now modify this function using this package, with a one line change using `implicit_linear_function`, and can now compute the Jacobian.

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
    y = implicit_linear_function(A, b)
    z = y.^2
    return z
end

x = [1.0; 2.0; 3.0; 4.0]
    
J1 = ForwardDiff.jacobian(modprogram, x)
J2 = ReverseDiff.jacobian(modprogram, x)

println(J1)
println(maximum(abs.(J1 - J2)))
```

For `implicit_linear_function` there are two keywords for custom subfunctions: 

1) `lsolve(A, b)`: same purpose as before: solve ``A x = b`` where the default is the backslash operator.
2) `fact(A)`: provide a matrix factorization of A, since two linear solves are performed (for the primal and dual values).  default is `factorize` defined in `LinearAlgebra`.