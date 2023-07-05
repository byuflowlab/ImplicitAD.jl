# ImplicitAD Documentation

**Summary**: Automate steady and unsteady adjoints.

Make implicit functions compatible with algorithmic differentiation (AD) without differentiating inside the solvers (discrete adjoint). Even though one can sometimes propagate AD through a solver, this is typically inefficient and less accurate.  Instead, one should use adjoints or direct (forward) methods. However, implementing adjoints is often cumbersome. This package allows for a one-line change to automate this process.  End-users can then use your package with AD normally, and utilize adjoints automatically.

We've also enabled methods to efficiently compute derivatives through explicit and implicit ODE solvers (unsteady discrete adjoint).  For the implicit solve at each time step we can apply the same methodology.  However, both still face memory challenges for long time-based simulations.  We analytically propagate derivatives between time steps so that reverse mode AD tapes only need to extend across a single time step. This allows for arbitrarily long time sequences without increasing memory requirements.

As a side benefit the above functionality easily allows one to define custom AD rules.  This is perhaps most useful when calling code from another language.  We provide fall backs for utilizing finite differencing and complex step efficiently if the external code cannot provide derivatives (ideally via Jacobian vector products).  This functionality can also be used for mixed-mode AD.

**Author**: Andrew Ning and Taylor McDonnell

**Features**:

- Compatible with ForwardDiff and ReverseDiff (or any ChainRules compliant reverse mode AD package)
- Compatible with any solver (no differentiation occurs inside the solver)
- Simple drop in functionality
- Customizable subfunctions to accommodate different use cases (e.g., custom linear solvers, factorizations, matrix-free operators)
- Version for ordinary differentiation equations (i.e., discrete unsteady adjoint)
- Analytic overrides for linear systems (more efficient)
- Analytic overrides for eigenvalue problems (more efficient)
- Can provide custom rules to be inserted into the AD chain. Provides finite differencing and complex step defaults for cases where AD is not available (e.g., calling another language).

**Documentation**:

- Start with the [tutorial](tutorial.md) to learn usage.
- The API is described in the [reference](reference.md) page.
- The math is particularly helpful for those wanting to provide their own custom subfunctions. See the theory and also some scaling examples in this [PDF](https://arxiv.org/pdf/2306.15243.pdf).  A supplementary document deriving the linear and eigenvalue cases is available in the [theory](theory.md) section.

**Run Unit Tests**:

```julia
pkg> activate .
pkg> test
```

**Citing**:

For now, please cite the following preprint.  DOI: [10.48550/arXiv.2306.15243](https://doi.org/10.48550/arXiv.2306.15243)

**Other Packages**:

[Nonconvex.jl](https://julianonconvex.github.io/Nonconvex.jl/stable/gradients/implicit/) and [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl) are other prior implementations of the nonlinear portion of this package.  [SciML](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#sensitivity_diffeq) provides support for continuous unsteady adjoints of ODEs.  They have also recently added an implementation for the [nonlinear case](https://docs.sciml.ai/SciMLSensitivity/stable/manual/nonlinear_solve_sensitivities/).