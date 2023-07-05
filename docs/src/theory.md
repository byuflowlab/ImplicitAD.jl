# Theory

The main theory and some examples (source code available in the `examples` folder) are available in this [paper](https://arxiv.org/pdf/2306.15243.pdf).

Some supplementary cases that have analytic solutions (linear solvers and eigenvalues) were not shown in the paper but are described below for completeness.

## Linear Equations

For linear residuals the nonlinear formulation shown in the paper will of course work, but we can provide partial derivatives directly, or rather provide the solution analytically.

### Forward Mode

Consider the linear system: ``A y = b``, where ``A`` and/or ``b`` is a function of our input ``x``, and ``y`` is our state variables.  For our purposes, ``A`` and ``b`` are the inputs to our function.  The derivatives of ``A`` and ``b`` with respect to our input variables of interest (``\dot{A}`` and ``\dot{b}``) would have been computed already as part of the forward mode AD.  The solution to the linear system is represented mathematically as:
```math
y = A^{-1} b \tag{1}
```
and we need ``\dot{y}``, the derivatives of ``y`` with respect to our inputs of interest.  Using the chain rule we find:
```math
\dot{y} = \dot{A}^{-1} b + A^{-1} \dot{b} \tag{2}
```
We now need the derivative of a matix inverse, which we can find by considering its definition relative to an identity matrix:
```math
A A^{-1} = I
```
We differentiate both sides:
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
We substitute this result into Eq. (2)
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
\boxed{\dot{y} = A^{-1} (\dot{b} -\dot{A} y)}
```

Both ``\dot{A}`` and ``\dot{b}`` are already known from the previous step in the forward AD process.  We simply extract those dual numbers from the inputs.  Or more conveniently we extract the partials from the vector: ``(b - Ay)`` since ``y`` is a constant, the solution to the linear system, and thus does not contain any dual numbers. Note that we should save the factorization of ``A`` from the primal linear solve (Eq. 1) to reuse in the equation above.


### Reverse Mode

Reverse mode is a bit more work.  We seek the derivatives of the overall outputs of interest with respect to ``A`` and ``b``.  For convenience, we consider one output at a time, and again call this output ``\xi``.  Notationally this is
```math
\bar{A} = \frac{d \xi}{dA},\, \bar{b} = \frac{d \xi}{db}
```
In this step of the reverse AD chain, we would know ``\bar{y}``, the derivative of our output of interest with respect to ``y``.

Computing the desired derivatives is easier in index notation since it produces third order tensors (e.g., derivative of a vector with respect to a matrix).  We compute the first derivative using the chain rule
```math
\bar{A}_{ij} = \frac{d \xi}{d A_{ij}} = \frac{d \xi}{d y_k} \frac{d y_k}{d A_{ij}} = \bar{y}_k \frac{d y_k}{d A_{ij}}
\tag{3}
```
We now need the derivative of the vector ``y`` with respect to ``A``.  To get there, we take derivatives of our governing equation (``Ay = b``):
```math
\frac{d}{d A_{ij}}  \left( A_{lm} y_m = b_l  \right)
```
The vector ``b`` is independent of ``A``; so we have:
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

We now substitute this result back into Eq. (3):
```math
\begin{aligned}
\bar{A}_{ij} &= -\bar{y}_k A_{ki}^{-1}  y_j   \\
\bar{A}_{ij} &= - (A_{ik}^{-T}\bar{y}_k)  y_j   \\
\end{aligned}
```
which we can recognize as an outer product
```math
\bar{A} = - (A^{-T} \bar{y}) y^T
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
\boxed{A^T \lambda = \bar{y}}
```
from which we easily get the desired derivatives
% for the reverse mode pullback operation
```math
    \boxed{
\begin{aligned}
\bar{A} &= -\lambda y^T\\
\bar{b} &= \lambda
\end{aligned}
    }
```
Note that we should again save the factorization for ``A`` from the primal solve to reuse in this second linear solve.


## Eigenvalues

A generalized eigenvalue problem solves the following equation for the pair ``\lambda_i, v_i`` where ``\lambda_i`` is the ``i^\text{th}`` eigenvalue and ``v_i`` the corresponding eigenvector, given the square matrices ``A`` and ``B``:
```math
A v_i = \lambda_i B v_i
\tag{4}
```
For a standard eigenvalue problem, ``B`` is the identity matrix.
Or in terms of left eigenvalues:
```math
u_i^H A = \lambda_i u_i^H B
\tag{5}
```
For Hermitian (or real symmetric) matrices we can see that the left and right eigenvectors would be the same.

Since any eigenvector can be scaled by an arbitrary value and still satisfy the above equation, the solution is made unique by using the standard normalization:

```math
u_i^H B v_i = 1
\tag{6}
```

which we see is equivalent to ``u_i^H A v_i = \lambda_i``.

### Forward Mode

We take derivatives of Eq. 4, using the chain rule, with respect to our inputs of interest.
```math
\dot{A} v_i + A \dot{v_i} = \dot{\lambda_i} B v_i + \lambda_i \dot{B} v_i + \lambda_i B \dot{v_i}
```
We rearrange the equation collecting like terms:
```math
\dot{\lambda_i} B v_i  = (\dot{A} - \lambda_i \dot{B} ) v_i + (A - \lambda_i B) \dot{v_i}
```
We now multiply the whole equation by ``u_i^H`` on the left:
```math
    \dot{\lambda_i} u_i^H B v_i  = u_i^H (\dot{A} - \lambda_i \dot{B} ) v_i + u_i^H (A - \lambda_i B) \dot{v_i}
```
From the left eigenvalue definition (Eq. 5) the last term cancels out, and from the normalization (Eq. 6) the first term simplifies giving the desired result for the derivatives of our eigenvalues:
```math
    \boxed{
        \dot{\lambda_i} = u_i^H (\dot{A} - \lambda_i \dot{B} ) v_i
    }
```

As an implementation trick we can directly use the following operation:
`λ = λ + ui' * (A - λ * B) * vi`
The primal value is zero for the last term so the primal is unchanged.  For the partial term, the last term gives the desired derivatives since `λ` contains no dual information (only ``A`` and ``B`` contain dual numbers).

### Reverse Mode

We follow a similar procedure to that of the linear system of equations.  Again, we consider one output at a time called ``\xi``.  The inputs are ``A`` and ``B`` and we consider only the eigenvalues as outputs ``\lambda``.  Thus, we need
```math
\bar{A} = \frac{d \xi}{dA},\, \bar{B} = \frac{d \xi}{dB}
```
given
```math
\bar{\lambda} = \frac{d \xi}{d \lambda}
```

Using index notation we need:
```math
\bar{A}_{ij} = \frac{d \xi}{d A_{ij}} = \frac{d \xi}{d \lambda_k} \frac{d \lambda_k}{d A_{ij}}
\tag{7}
```
We now need the derivative of the vector ``\lambda`` with respect to ``A``, which we get from taking derivatives of our governing equation (Eq. 4).  For simplicity, we will consider just one eigenvalue (dropping the index denoting separate eigenvalues/vectors), then will add it back in when reassembling the above equation.  The derivative of Eq. 4, in index notation, for a single eigenvalue/vector pair is:
```math
\frac{d}{d A_{ij}}  \left( A_{kl} v_l = \lambda B_{km} v_m  \right)
```
We propagate the derivative through:
```math
\delta_{ik}\delta_{jl} v_l + A_{kl}\frac{d v_l}{d A_{ij}}  = \frac{d \lambda}{d A_{ij}} B_{km} v_m + \lambda B_{km} \frac{d v_m}{d A_{ij}}
```
The last term on each side can be combined as we recognize that ``l`` and ``m`` are both dummy indices (we will change the last term to use ``l`` instead)
```math
\delta_{ik}\delta_{jl} v_l + (A_{kl} - \lambda B_{kl})\frac{d v_l}{d A_{ij}}  = \frac{d \lambda}{d A_{ij}} B_{km} v_m
```
We now multiply through by the left eigenvector ``u_k``
```math
\delta_{ik}\delta_{jl} u_k v_l + u_k (A_{kl} - \lambda B_{kl})\frac{d v_l}{d A_{ij}}  = \frac{d \lambda}{d A_{ij}} u_k B_{km} v_m
```
We recognize that the term ``u_k (A_{kl} - \lambda B_{kl})`` is zero from Eq. 5.  We also see that the term ``u_k B_{km} v_m `` is one from Eq. 6.  The above then simplifies to
```math
\delta_{ik}\delta_{jl} u_k v_l  = \frac{d \lambda}{d A_{ij}}
```
We now apply the Kronecker deltas:
```math
\frac{d \lambda}{d A_{ij}}  = u_i v_j
```
or in vector notation (for a single eigenvector/value pair):
```math
\frac{d \lambda}{dA} = u v^T
```
We can now compute the desired derivative from Eq. 7, where we now reinsert the summation since there are multiple eigenvalue/vectors.
```math
    \boxed{\bar{A} = \sum_k \bar{\lambda}_k u_k v_k^T}
```

We repeat the same process for the derivatives with respect to ``B``.
```math
\begin{align*}
\frac{d}{d B_{ij}}  ( A_{kl} v_l & = \lambda B_{km} v_m  )\\
  A_{kl} \frac{d v_l}{d B_{ij}} & = \frac{d \lambda}{d B_{ij}} B_{km} v_m  + \lambda \frac{d B_{km}}{d B_{ij}} v_m  + \lambda B_{km} \frac{d v_m}{d B_{ij}} \\
  (A_{kl} - \lambda B_{kl}) \frac{d v_l}{d B_{ij}} & = \frac{d \lambda}{d B_{ij}} B_{km} v_m  + \lambda \delta_{ik}\delta_{jm} v_m  \\
  u_k (A_{kl} - \lambda B_{kl}) \frac{d v_l}{d B_{ij}} & = \frac{d \lambda}{d B_{ij}} u_k B_{km} v_m  + \lambda u_k  \delta_{ik}\delta_{jm} v_m  \\
  0 & = \frac{d \lambda}{d B_{ij}} + \lambda u_k  \delta_{ik}\delta_{jm} v_m  \\
  \frac{d \lambda}{d B_{ij}} & =  - \lambda u_i v_j
\end{align*}
```
or in index notation:
```math
\frac{d \lambda}{d B} = -\lambda u v^T
```
We then reassemble the summation across multiple eigenvalues:
```math
\boxed{\bar{B} = -\sum_k \bar{\lambda}_k \lambda_k u_k v_k^T}
```

