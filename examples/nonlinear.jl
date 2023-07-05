using ImplicitAD
using ForwardDiff
using ReverseDiff
using BenchmarkTools
using NLsolve
using FiniteDiff
using DiffResults

function residual!(r, y, x, p=())
    offset = 1

    r[1] = -4*x[1]*y[1]*(y[2] - y[1]^2) - 2*(offset - y[1])
    n = length(y)
    for i = 2:n-1
        r[i] = -4*x[i]*y[i]*(y[i+1] - y[i]^2) - 2*(offset - y[i]) + 2*x[i-1]*(y[i] - y[i-1]^2)
    end
    r[n] = 2*x[n]*(y[n] - y[n-1]^2)
    return nothing
end

function solve(x, p=())
    rwrap(r, y) = residual!(r, y, x, p)
    res = nlsolve(rwrap, 2*ones(eltype(x), length(x)), autodiff=:forward)
    return res.zero
end

function program!(y, x)
    y .= solve(x)
end

function modprogram!(y, x)
    y .= implicit(solve, residual!, x)
end

function runit()

    nvec = [2, 4, 8, 16, 32, 64, 128]
    nn = length(nvec)

    time1 = zeros(nn)
    time2 = zeros(nn)
    time3 = zeros(nn)
    time4 = zeros(nn)
    time5 = zeros(nn)

    for i = 1:length(nvec)
        n = nvec[i]
        x = 100*ones(n)

        f = zeros(n)
        J = DiffResults.JacobianResult(f, x)
        Jdiff = zeros(length(f), length(x))

        fwd_cache1 = ForwardDiff.JacobianConfig(program!, f, x)
        fwd_cache2 = ForwardDiff.JacobianConfig(modprogram!, f, x)

        f_tape1 = ReverseDiff.JacobianTape(program!, f, x)
        rev_cache1 = ReverseDiff.compile(f_tape1)
        f_tape2 = ReverseDiff.JacobianTape(modprogram!, f, x)
        rev_cache2 = ReverseDiff.compile(f_tape2)

        finite_cache = FiniteDiff.JacobianCache(x, Val{:central})

        t1 = @benchmark ForwardDiff.jacobian!($J, $program!, $f, $x, $fwd_cache1)
        t2 = @benchmark ForwardDiff.jacobian!($J, $modprogram!, $f, $x, $fwd_cache2)
        t3 = @benchmark ReverseDiff.jacobian!($J, $rev_cache1, $x)
        t4 = @benchmark ReverseDiff.jacobian!($J, $rev_cache2, $x)
        t5 = @benchmark FiniteDiff.finite_difference_jacobian!($Jdiff, $program!, $x, $finite_cache)


        time1[i] = median(t1).time * 1e-9
        time2[i] = median(t2).time * 1e-9
        time3[i] = median(t3).time * 1e-9
        time4[i] = median(t4).time * 1e-9
        time5[i] = median(t5).time * 1e-9

        println(time1[i])
        println(time2[i])
        println(time3[i])
        println(time4[i])
        println(time5[i])
        println()
    end
    return time1, time2, time3, time4, time5
end


time1, time2, time3, time4, time5 = runit()

