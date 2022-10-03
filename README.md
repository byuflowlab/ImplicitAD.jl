# ImplicitAD.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/dev/)
[![Build Status](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Summary**: Make implicit functions compatible with algorithmic differentiation (specifically ForwardDiff and ReverseDiff) without differenting inside the solvers.

**Author**: Andrew Ning

**Motivation**:

Many engineering analyses use implicit functions.  We can represent any such implicit function generally as:
```math
r(y; x) = 0
```
where ``r`` are the residual functions we wish to drive to zero, ``x`` are inputs, and ``y`` are the state variables, which are also outputs once the system of equations is solved.  In other words, ``y`` is an implicit function of ``x``. Note that all of these are vector quantities.  ``x`` is of size ``n_x``, and ``r`` and ``y`` are of the same size ``n_r`` (must have equal number of unknowns and equations for a solution to exist). Graphically we could depict this relationship as:

x --> [ r(y; x) ] --> y

We then chose some appropriate solver to converge these residuals.  From a differentiation perspective, we would like to compute ``dy/dx``.  One can often use algorithmic differentiation (AD) in the same way one would for any explicit function.  Once we unroll the iterations of the solver the set of instructions is explicit.  However, this is at best inefficient and at worse inaccurate or not possible (at least not without a lot more effort).  To obtain accurate derivatives by propgating AD through a solver, the solver must be solved to a tight tolerance.  Generally tighter than is required to converge the primal values.  Sometimes this is not feasible because operations inside the solvers may not overloaded for AD, this is especially true if calling solvers in other languages.  But even if we can do it (tight convergence is possible and everything under the hood is overloaded) we usually still shouldn't, as it would be computationally inefficient.  Instead we can use implicit differentiation, to allow for AD to work seemlessly with implicit functions without having to differentiate through them.

This package provides an implementation so that a simple one-line change can be applied to allow AD to be propgated around any solver.  Note that the implementation of the solver need not be AD compatible since AD does not not occur inside the solver.  This package is overloaded for [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).  There are also optional inputs so that subfunction behavior can be customized (e.g., preallocation, custom linear solvers, custom factorizations, custom jacobian vector products, etc.)

**Documentation**:

- Start with the [quick start tutorial](tutorial.md) to learn basic usage.
- For more advanced usage see some additional [examples](examples.md).
- The API is described in the [reference](reference.md) page.
- The math is particularly helpful for those wanting to provide their own custom subfunctions. See the [theory](theory.md) page.

**Run Unit Tests**:

```julia
pkg> activate .
pkg> test
```
