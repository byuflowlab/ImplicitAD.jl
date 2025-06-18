# Examples

There three scripts: nonlinear.jl, explicit.jl, and implicit.jl from the paper: <https://doi.org/10.48550/arXiv.2306.15243>.  All are written with for loops, solving problems of increasing size, but in practice I often ran them one size at a time as I found that produced more consistent timings.  The last case (implicit.jl) requires a patch to NLsolve described here: <https://github.com/JuliaNLSolvers/NLsolve.jl/issues/281>.  The new methodology, with ImplicitAD, will work fine without it since it doesn't propagate through the solver.  But if you want to compare using reverse mode directly through NLsolve the patch is required.

There is also an example using Python, where a custom rule is provided for the Julia side: nnwrap.jl and nnwrapper.py.  In this particular case we use the AD capabilities of PyTorch with a basic neural net.  The custom rule capabilities allow for the code to be in any language (with fallbacks for finite difference or complex step if AD is not possible in the other language).

