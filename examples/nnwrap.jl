using ImplicitAD
using ForwardDiff
using ReverseDiff

# Using my version of Python (must be done before calling PythonCall)
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/opt/miniconda3/bin/python"

using PythonCall

# need to add directory to PYTHONPATH
PythonCall.pyimport("sys").path.insert(0, pwd())

const NNWrapper = pyimport("nnwrapper").NNWrapper()


function runfun()

    function output(x, p)
        y = NNWrapper.eval(x)
        return pyconvert(Vector, y)
    end

    function jacobian(x, p)
        dydx =  NNWrapper.jacobian(x)
        return pyconvert(Matrix, dydx)
    end

    function jvp(x, p, v)
        ydot = NNWrapper.jvp(x, v)
        return pyconvert(Vector, ydot)
    end

    function vjp(x, p, v)
        xbar = NNWrapper.vjp(x, v)
        return pyconvert(Vector, xbar)
    end


    function example(x)
        y = 2.0 * x
        p = ()
        # z = provide_rule(output, y, p; mode="jacobian", jacobian)  # alternative option if you want to provide jacobian
        z = provide_rule(output, y, p; mode="vp", jvp, vjp)
        w = z .^2
        return w
    end

    x = [8, 307.0, 130.0, 3504., 12.0, 70, 1]
    out = example(x)
    println("output = ", out)

    dwdx = ForwardDiff.jacobian(example, x)
    dwdx2 = ReverseDiff.jacobian(example, x)
    println(dwdx)
    println(dwdx2)
    println(maximum(abs.(dwdx - dwdx2)))
    # println(size(dwdx))

end


runfun()