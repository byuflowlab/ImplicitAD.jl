# Theory

We can get to the derivatives we are after by using implicit differentiation. Recall that we are solving:
``r(x, y(x)) = 0`` and that we want ``dy/dx``.  Using the chain rule we find that:
```math
\frac{dr}{dx} = \frac{\partial r}{\partial x} + \frac{\partial r}{\partial y} \frac{dy}{dx} = 0
```
To be a properly solution, the residuals must remain at ``r = 0`` with changes in the input.  Thus, the total derivative ``dr/dx`` must also be zero.

The Jacobian we are after we will call ``J = \frac{dy}{dx}``.  The other two partial derivative Jacobians we will call ``A = \frac{\partial r}{\partial y}`` and ``B = \frac{\partial r}{\partial x}`` just for convenience.  The above equation then gives:
```math
A J = -B
```
Note that our desired Jacobian depends only on partial derivatives at the solution of our residual equations (and thus not on the solver path).  We can compute or provide these partial derivatives by various means, though typically we will leverage AD.

#### Forward Mode

If we are using forward mode algorithmic differentiation, once we reach this implicit function we already know the derivatives of x with respect to our variables of interest.  We will call this ``\dot{x}``.  The above equation described how we can compute the Jacobian ``J``, but what we really want is not ``J`` but rather the Jacobian vector product ``J \dot{x}``, which gives us the derivatives of y with respect to our variables of interest, and we continue forward along the AD path.

We multiply both sides by ``\dot{x}``
```math
A J \dot{x} = -B \dot{x}
```
or 
```math
A \dot{y} = -B \dot{x}
```
where ``\dot{y}`` is what we are after.

First we deal with the right hand side computing:
```math
b = -B \dot{x}
```
We can compute the resulting Jacobian vector product (Jacobian B times some vector v) without actually forming the matrix B, which is more efficient.  Recall that ``B = \partial r/\partial x``.  If we want the Jacobian product ``B v``, where ``v = -\dot{x}`` in this case, we see that we are after a weighted sum of the columns ``\sum_i \partial r/\partial x_i v_i`` where ``v_i`` are the weights.  Normally we pick off one of these partials in forward mode AD one at a time with different seeds ``e_i = [0, 0, 1, 0]``, but if we set the initial seed to ``v_i`` then the partial derivatives we compute are precisely this weighted sum.  In other words, we initialize ``x`` as a dual number with its values as the input value ``x``, and its derivatives as ``v``, we evalute the residual function, then extract the partial derivatives which now give the desired Jacobian vector product ``b = B v`` (where ``v = -\dot{x}``).

With the right-hand side complete, we just need to compute the square Jacobian ``A = \partial r / \partial y`` and solve the linear system.  
```math
A \dot{y} = b
```
Because A is square, forward mode AD is usually preferable for computing these partial derivatives. If we know the structure of A, we would want to provide an appropriate factorization of A.  Often, this Jacobian is sparse, and so using graph coloring is desirable.  Also for very large systems we can use matrix-free methods (e.g., Krylov methods) to solve the linear system without actually constructing A.

In summary the operations are:
1) Compute the JVP: ``b = -B \dot{x}`` for the upstream input ``\dot{x}`` where ``B = \partial r/\partial x``.
2) Solve the linear system: ``A \dot{y} = b`` where ``A = \partial r / \partial y``.

In this package the default is just dense forward-mode AD (not-in place). But the function can easily be overloaded for your use case where you can pre-allocate, use sparse matrices with graph coloring (see SparseDiffTools.jl), and your own desired linear solver.


#### Reverse Mode

If using reverse mode AD we have the derivatives of out outputs of interest with respect to y ``\bar{y}`` and need to propgate backwards to get ``\bar{x}``.  Again, we're not really interested in ``J``, but rather in ``\bar{x} = J^T \bar{y}``.  First we take the transpose of our governing equation:
```math
J^T A^T = -B^T
```
Let's multiply both sides by some as yet unknown vector ``u``.
```math
J^T A^T u = -B^T u
```
We now would like to solve for the value of ``u`` such that
```math
A^T u = \bar{y}
```
Doing so will give us the desired Jacobian vector product ``J^T \bar{y}``:
```math
J^T A^T u = -B^T u
```
```math
J^T \bar{y} = -B^T u 
```
```math
\bar{x} = -B^T u 
```

In this case we start with solving the linear system:
```math
A^T u = \bar{y}
```
given input ``\bar{y}``.  Again, we will want to use an appropriate factorization for A, sparsity, and appropriate linear solver.  The default implementation ses dense forward mode AD, and the linear solve is the same as before.

With u now known we compute the vector-Jacobian product
``\bar{x} = -B^T u = -(u^T B)^T``.  Like the Jacobian-vector product of the previous section, we can compute this more efficiently by avoiding explicitly constructing the matrix B.  The vector-Jacobian product ``v^T \partial r/ \partial x`` can be computing with reverse-mode AD where the intial seed (adjoint vector) is set with the weights ``v`` (where ``v = -u`` in this case).

In this implementation, rather than go under the hood and manipulate the seed (since this function is not explicitly available), we can instead simply compute the gradient of:
```math
\frac{\partial }{\partial x}\left( v^T r(x, y) \right) = v^T \frac{\partial r}{\partial x}
```
which the expansion shows gives us the desired product.
To repeat, we just compute the gradient of ``(v^T r)``  with respect to ``x`` (note it is a gradient and not a Jacobian since we have a scalar output). This gives the desired vector, i.e., the derivatives ``\bar{x}``.

In summary the operations are:
1) Solve the linear system: ``A^T u = \bar{y}`` for the upstream input ``\bar{y}`` where ``A = \partial r / \partial y``.
2) Compute the VJP: ``\bar{x} = -B^T u`` where ``B = \partial r/\partial x``.

### Linear Equations

For linear residuals the above nonlinear formulation will of course work, but we can provide partial derivatives symbolically.  This will be more efficient rather than relying on AD to compute these partials, or will make it easier on the user to not have to manually provide these.

#### Forward Mode

Consider the linear system: ``A y = b`` where ``A`` and or ``b`` is a function of our input ``x`` and ``y`` is our state variables.  For our purposes ``A`` and ``b`` are the inputs to our function, the derivatives of ``A`` and ``b`` with respect to ``x`` are explicit and will already be computed as part of the forward mode AD.  The solution to the linear system is represented mathematically as:
```math
y = A^{-1} b
```
and we need ``\dot{y}``, the derivatives of ``y`` with respect to our inputs of interest.  Using the chain rule we find:
```math
\dot{y} = \dot{A}^{-1} b + A^{-1} \dot{b}
```
We now need the derivative of a matix inverse, which we can find by considering its definition relative to an identity matrix:
```math
A A^{-1} = I
```
We now differentiate both sides:
```math
\dot{A} A^{-1} + A \dot{A}^{-1} = 0
```
and we can now solve for the derivative of the matrix inverse:
```math
\begin{aligned}
A \dot{A}^{-1} &= - \dot{A} A^{-1} \\
 \dot{A}^{-1} &= - A^{-1} \dot{A} A^{-1} 
\end{aligned}
```
We substitute this result into our above equation..
```math
\dot{y} = - A^{-1} \dot{A} A^{-1}  b + A^{-1} \dot{b}
```
We simplify this expression by noting that ``A^{-1}  b = y`` and factoring out the matrix inverse.
```math
\begin{aligned}
\dot{y} &= - A^{-1} \dot{A} y + A^{-1} \dot{b}\\
\dot{y} &= A^{-1} (-\dot{A} y + \dot{b})
\end{aligned}
```
This is the desired result:
```math
\dot{y} = A^{-1} (\dot{b} -\dot{A} y)\\
```

Both ``\dot{A}`` and ``\dot{b}`` are already known from the previous step in the forward AD process.  We simply extract those dual numbers from the inputs.  Or more conveniently we extract the partials from the vector: ``(b - Ay)`` since ``y`` is a constant, the solution to the linear system, and thus does not contain any dual numbers. Note that we should save the factorization of ``A`` from the primal linear solve to reuse in the equation above.


#### Reverse Mode

Reverse mode is a little trickier.  We need the derivatives of outputs of interest with respect to ``A`` and ``b``.  For convenience, we consider one output at a time (though it works the same for multiple), and call this output ``\xi``.  Notationally this is
```math
\bar{A} = \frac{d \xi}{dA},\, \bar{b} = \frac{d \xi}{db}
```
In this step of the reverse AD chain we would know ``\bar{y}`` the derivative of our output of interest with respect to ``y``. 

Computing the desired derivatives is easier in index notation since it produces third order tensors (e.g., derivative of a vector with respect to a matrix).  Let's go after the first derivative using the chain rule
```math
\bar{A}_{ij} = \frac{d \xi}{d A_{ij}} = \frac{d \xi}{d y_k} \frac{d y_k}{d A_{ij}}
```
We now need the derivative of the vector ``y`` with respect to ``A``.  To get there we take derivatives of our governing equation (``Ay = b``):
```math
\frac{d}{d A_{ij}}  \left( A_{lm} y_m = b_l  \right)
```
The vector ``b`` is independent of ``A`` so we have:
```math
\begin{aligned}
\frac{d A_{lm}}{d A_{ij}} y_m + A_{lm} \frac{d y_m}{d A_{ij}}   & = 0 \\
\delta_{li}\delta_{mj} y_m + A_{lm} \frac{d y_m}{A_{ij}}   & = 0 \\
\delta_{li} y_j + A_{lm} \frac{d y_m}{A_{ij}}   & = 0 \\
A_{nl}^{-1} \delta_{li} y_j + A_{nl}^{-1} A_{lm} \frac{d y_m}{A_{ij}}   & = 0 \\
A_{nl}^{-1} \delta_{li} y_j + \delta_{nm} \frac{d y_m}{A_{ij}}   & = 0 \\
A_{ni}^{-1}  y_j +  \frac{d y_n}{A_{ij}}   & = 0 
\end{aligned}
```
Thus:
```math
\frac{d y_k}{A_{ij}} = -A_{ki}^{-1}  y_j  
```

We now substitute this result back into our above equation:
```math
\begin{aligned}
\bar{A}_{ij} &= \bar{y}_k \frac{d y_k}{d A_{ij}} \\
\bar{A}_{ij} &= -\bar{y}_k A_{ki}^{-1}  y_j   \\
\bar{A}_{ij} &= - (A_{ik}^{-T}\bar{y}_k)  y_j   \\
\end{aligned}
```
Which we can recognize as an outer product
```math
\bar{A} - (A^{-T} \bar{y}) y^T
```

We now repeat the procedure for the derivatives with respect to ``b``
```math
\bar{b}_i = \frac{d \xi}{d b_i} = \frac{d \xi}{d y_j} \frac{d y_j}{d b_i} = \bar{y}_j \frac{d y_j}{d b_i}
```
We can easily get this last derivative
```math
\frac{d y_j}{d b_i} = \frac{d (A^{-1}_{jk} b_k)}{d b_i} = A^{-1}_{jk} \frac{d b_k}{d b_i} = A^{-1}_{jk} \delta_{ki} = A^{-1}_{ji} 
```
Thus:
```math
\bar{b}_i = \bar{y}_j A^{-1}_{ji} 
```
We now have the desired result
```math
\bar{b} = A^{-T} \bar{y}
```

From these results we see that we must first solve the linear system
```math
A^T u = \bar{y}
```
from which we easily get the desired derivatives for the revsere mode pullback operation
```math
\begin{aligned}
\bar{A} &= -u y^T\\
\bar{b} &= u
\end{aligned}
```
Note that we should again save the factorization for ``A`` from the primal solve to reuse in this second linear solve.

