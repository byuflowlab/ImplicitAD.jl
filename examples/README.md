# Examples

This folder contains the examples from the paper: <https://doi.org/10.48550/arXiv.2306.15243>.  There are three scripts: nonlinear.jl, explicit.jl, and implicit.jl.  All are written with for loops, solving problems of increasing size, but in practice I often ran them one size at a time as I found that produced more consistent timings.  The last case (implicit.jl) requires a patch to NLsolve described here: <https://github.com/JuliaNLSolvers/NLsolve.jl/issues/281>.  The new methodology, with ImplicitAD, will work fine without it since it doesn't propagate through the solver.  But if you want to compare using revese mode directly through NLsolve the patch is required.

