# ImplicitAD.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/ImplicitAD.jl/dev/)
[![Build Status](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/byuflowlab/ImplicitAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Summary**: Automate adjoints: make implicit functions compatible with algorithmic differentiation without differenting inside the solvers. Also allow for custom rules with explicit functions (e.g., calling external code, mixed mode AD).

**Author**: Andrew Ning and Taylor McDonnell

**Features**:

- Compatible with ForwardDiff and ReverseDiff
- Compatible with any solver (no differentiation occurs inside the solver)
- Simple drop in functionality
- Customizable subfunctions to accommodate different use cases like iterative linear solvers, custom Jacobian vector products, etc.
- Version for ordinary differentiation equations (i.e., discrete unsteady adjoint)
- Analytic overrides for linear systems
- Analytic overrides for eigenvalue problems
- Can provide custom rules to be inserted into the AD chain. Provides finite differencing and complex step defaults for cases where AD is not available (e.g., calling another language).

**Implicit Motivation**:

Many engineering analyses use implicit functions.  We can represent any such implicit function generally as:
```math
r(y; x) = 0
```
where ``r`` are the residual functions we wish to drive to zero, ``x`` are inputs, and ``y`` are the state variables, which are also outputs once the system of equations is solved.  In other words, ``y`` is an implicit function of ``x`` (``x -> r(y; x) -> y``).

We then chose some appropriate solver to converge these residuals.  From a differentiation perspective, we would like to compute ``dy/dx``.  One can often use algorithmic differentiation (AD) in the same way one would for any explicit function.  Once we unroll the iterations of the solver the set of instructions is explicit.  However, this is at best inefficient and at worse inaccurate or not possible (at least not without a lot more effort).  To obtain accurate derivatives by propgating AD through a solver, the solver must be solved to a tight tolerance.  Generally tighter than is required to converge the primal values.  Sometimes this is not feasible because operations inside the solvers may not be overloaded for AD, this is especially true when calling solvers in other languages.  But even if we can do it (tight convergence is possible and everything under the hood is overloaded) we usually still shouldn't, as it would be computationally inefficient.  Instead we can use implicit differentiation, to allow for AD to work seemlessly with implicit functions without having to differentiate through them.

This package provides an implementation so that a simple one-line change can be applied to allow AD to be propgated around any solver.  Note that the implementation of the solver need not be AD compatible since AD does not not occur inside the solver.  This package is overloaded for [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).  There are also optional inputs so that subfunction behavior can be customized (e.g., preallocation, custom linear solvers, custom factorizations).

**Custom Rule Motivation**:

A different but related need is to propagate AD through functions that are not-AD compatible. A common example would be a call to a subfunction is another language that is part of a larger AD compatible function. This packages provides a simple wrapper to estimate the derivatives of the subfunction with finite differencing (forward or central) or complex step.  Those derivatives are then inserted into the AD chain so that the overall function seamlessly works with ForwardDiff or ReverseDiff.

That same functionality is useful also in cases where a function is already AD compatible but where a more efficient rule is available.  We can provide the Jacobian or the Jacobian vector / vector Jacobian products directly.  One common example is mixed mode AD.  In this case we may have a subfunction that is most efficiently differentiated in reverse mode, but the overall function is differentiated in forward mode.  We can provide a custom rule for the subfunction which will then be inserted into the forward mode chain. 

**Documentation**:

- Start with the tutorial to learn usage.  
- The API is described in the reference page.
- The math is particularly helpful for those wanting to provide their own custom subfunctions. See the theory page.

**Run Unit Tests**:

```julia
pkg> activate .
pkg> test
```

**Other Packages**:

[Nonconvex.jl](https://julianonconvex.github.io/Nonconvex.jl/stable/gradients/implicit/) and [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl) are other prior implementations of the nonlinear case. [SciML](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#sensitivity_diffeq) provides support for continuous unsteady adjoints of ODEs.  They have also recently added an implementation for the [nonlinear case](https://docs.sciml.ai/SciMLSensitivity/stable/manual/nonlinear_solve_sensitivities/).
