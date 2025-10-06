module PyCallExt

using ImplicitAD
using PythonCall: Py, pyconvert

# overload function for case where PythonCall is loaded
function ImplicitAD.derivativesetup(func::Py, x, p, ad, compiletape=false)

    fwrap(x) = pyconvert(Vector, func(x, p))

    return ImplicitAD.derivativesetupinternal(fwrap, x, ad, compiletape)
end

end