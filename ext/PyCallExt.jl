module PyCallExt

using ImplicitAD
using PythonCall: Py, pyconvert

"""
    derivativesetup(func, x, p, ad, compiletape=false)

Set up a derivative function to make it easier to repeatedly differentiate (e.g., in an optimization).
Primary use case is to pass derivatives to Python or another language.

# Arguments
- `func::Function`: The function to differentiate. Must have signature `f = func(x, p)`
    where `x` are the variables to differentiate with respect to and `p` are additional parameters.
- `x::AbstractVector`: a typical input vector, used to size the derivative arrays.
- `p::Any`: Additional parameters to pass to `func`.  If this input changes, a new function must be setup.
- `ad::String`: The type of automatic differentiation to use. Options are:
    - `"fjacobian"`: Forward mode AD to compute the full Jacobian df/dx.
    - `"rjacobian"`: Reverse mode AD to compute the full Jacobian df/dx.
    - `"jvp"`: Forward mode AD to compute the Jacobian-vector product df/dx * v.
    - `"vjp"`: Reverse mode AD to compute the vector-Jacobian product v' * df/dx.
- `compiletape::Bool`: (optional, default=false) If using "rjacobian", whether to compile the tape for faster execution (assumes no branching behavior).

# Returns
A function that computes the requested derivative. The signature depends on the type of derivative:
- For `"fjacobian"` and `"rjacobian"`: `df(J, x)` where `J` is a preallocated array to hold the Jacobian.
- For `"jvp"`: `djvp(ydot, x, xdot)` where `ydot` is a preallocated array to hold the output and `xdot` is the input direction vector.
- For `"vjp"`: `dvjp(xbar, x, ybar)` where `xbar` is a preallocated array to hold the output and `ybar` is the input direction vector.
"""
function ImplicitAD.derivativesetup(func::Py, x, p, ad, compiletape=false)

    fwrap(x) = pyconvert(Vector, func(x, p))

    return ImplicitAD.derivativesetupinternal(fwrap, x, ad, compiletape)
end

end