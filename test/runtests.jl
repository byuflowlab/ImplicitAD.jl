using ImplicitAD
using Test
using NLsolve
using ForwardDiff
using ReverseDiff
using FiniteDiff
using LinearAlgebra: Symmetric, factorize, diagm, eigvals, eigvecs, I
using SparseArrays: sparse
using FLOWMath: brent
using UnPack: @unpack
import OrdinaryDiffEq

@testset "residual" begin


    function residual!(r, y, x, p)
        r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r[2] = sin(y[2]*exp(y[1])-1)*x[4]
        return r
    end

    function solve(x, p)
        rwrap(r, y) = residual!(r, y, x[1:4], p)
        res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
        return res.zero
    end

    function program(x)
        z = 2.0*x
        w = z + x.^2
        y = solve(w)
        return y[1] .+ w*y[2]
    end

    function modprogram(x)
        z = 2.0*x
        w = z + x.^2
        y = implicit(solve, residual!, w, ())
        return y[1] .+ w*y[2]
    end

    x = [1.0; 2.0; 3.0; 4.0; 5.0]

    J1 = ForwardDiff.jacobian(modprogram, x)
    J2 = ReverseDiff.jacobian(modprogram, x)

    @test all(isapprox.(J1, J2, atol=1e-14))


    # -------- supply one of the jacobians -----------

    function drdy(residual, y, x, p)
        A = zeros(2, 2)
        A[1, 1] = y[2]^3-x[2]
        A[1, 2] = 3*y[2]^2*(y[1]+x[1])
        u = exp(y[1])*cos(y[2]*exp(y[1])-1)*x[4]
        A[2, 1] = y[2]*u
        A[2, 2] = u
        return A
    end

    function modprogram2(x)
        z = 2.0*x
        w = z + x.^2
        y = implicit(solve, residual!, w, (), drdy=drdy)
        return y[1] .+ w*y[2]
    end

    J3 = ForwardDiff.jacobian(modprogram2, x)
    J4 = ReverseDiff.jacobian(modprogram2, x)

    @test all(isapprox.(J1, J3, atol=1e-14))
    @test all(isapprox.(J1, J4, atol=1e-14))

end

@testset "not in place" begin


    function residual(y, x, p)
        r1 = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r2 = sin(y[2]*exp(y[1])-1)*x[4]
        return [r1; r2]
    end

    function solve(x, p)
        rwrap(y) = residual(y, x[1:4], p)
        res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
        return res.zero
    end

    function modprogram(x)
        z = 2.0*x
        w = z + x.^2
        y = implicit(solve, residual, w, ())
        return y[1] .+ w*y[2]
    end

    x = [1.0; 2.0; 3.0; 4.0; 5.0]

    J1 = ForwardDiff.jacobian(modprogram, x)
    J2 = ReverseDiff.jacobian(modprogram, x)

    @test all(isapprox.(J1, J2, atol=1e-14))

end

@testset "iterative" begin

    function residual!(r, y, x, p)
        r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r[2] = sin(y[2]*exp(y[1])-1)*x[4]
        return r
    end

    ystar = zeros(2)

    function solve(x, p)
        rwrap(r, y) = residual!(r, y, x[1:4], p)
        res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
        ystar .= res.zero
        return ystar
    end

    A = zeros(2, 2)

    function drdy(residual, y, x, p)
        A[1, 1] = y[2]^3-x[2]
        A[1, 2] = 3*y[2]^2*(y[1]+x[1])
        u = exp(y[1])*cos(y[2]*exp(y[1])-1)*x[4]
        A[2, 1] = y[2]*u
        A[2, 2] = u
        return A
    end

    function modprogram(x)
        z = 2.0*x
        w = z + x.^2
        y = implicit(solve, residual!, w, (), drdy=drdy)
        y = implicit(solve, residual!, [y[1], y[2], w[3], w[4]], (), drdy=drdy)
        return y[1] .+ w*y[2]
    end

    x = [1.0; 2.0; 3.0; 4.0; 5.0]

    J1 = ForwardDiff.jacobian(modprogram, x)
    J2 = ReverseDiff.jacobian(modprogram, x)

    @test all(isapprox.(J1, J2, atol=1e-14))

end

# @testset "linear" begin


#     function solvelin(x)
#         A = [x[1]*x[2] x[3]+x[4];
#             x[3]+x[4] 0.0]
#         b = [2.0, 3.0]
#         Af = factorize(Symmetric(sparse(A)))
#         return Af \ b
#     end

#     function Alin(residual, x, y)
#         return [x[1]*x[2] x[3]+x[4];
#             x[3]+x[4] 0.0]
#     end

#     # r = A * y - b
#     # drdx = dAdx * y
#     function Blin(residual, x, y)
#         B = [x[2]*y[1]  x[1]*y[1]  y[2]  y[2];
#             0.0  0.0  y[1]  y[1]]
#         return B
#     end

#     function test(x)
#         w = implicit_function(solvelin, nothing, x, drdy=Alin, drdx=Blin)
#         return 2*w
#     end

#     x = [1.0, 2.0, 3.0, 4.0]
#     J1 = ForwardDiff.jacobian(test, x)
#     J2 = ReverseDiff.jacobian(test, x)

#     @test all(isapprox.(J1, J2, atol=1e-14))
# end


@testset "another overload" begin

    function residual!(r, y, x, p)
        r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r[2] = sin(y[2]*exp(y[1])-1)*x[4]
        return r
    end

    function solve(x, p)
        rwrap(r, y) = residual!(r, y, x[1:4], p)
        res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
        return res.zero
    end

    # function program(x)
    #     z = 2.0*x
    #     w = z + x.^2
    #     y = solve(w)
    #     return y[1] .+ w*y[2]
    # end


    function mydrdy(y, x, p, A, r)

        wrap!(rt, yt) = residual!(rt, yt, x, p)

        ForwardDiff.jacobian!(A, wrap!, r, y)

        return A
    end

    function modprogram(x, drdy)
        z = 2.0*x
        w = z + x.^2
        y = implicit(solve, residual!, w, (), drdy=drdy)
        return y[1] .+ w*y[2]
    end


    function run()

        r = zeros(2)
        A = zeros(2, 2)

        drdy(residual, y, x, p) = mydrdy(y, x, p, A, r)

        wrap(x) = modprogram(x, drdy)

        x = [1.0; 2.0; 3.0; 4.0; 5.0]
        J1 = ForwardDiff.jacobian(wrap, x)

        # J2 = FiniteDiff.finite_difference_jacobian(wrap, x)
        J2 = ReverseDiff.jacobian(wrap, x)

        return J1, J2
    end

    J1, J2 = run()

    @test all(isapprox.(J1, J2, atol=1e-12))

end

@testset "linear2" begin

    function test(a)
        A = a[1]*[1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear(A, b)
        z = 2*x
        return z
    end

    function test2(a)
        A = [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear(A, b)
        z = 2*x
        return z
    end

    function test3(a)
        A = a[1] * [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * ones(3)
        x = implicit_linear(A, b)
        z = 2*x
        return z
    end

    a = [1.2, 2.3, 3.1, 4.3]
    J1 = ForwardDiff.jacobian(test, a)
    J2 = ReverseDiff.jacobian(test, a)

    @test all(isapprox.(J1, J2, atol=3e-12))

    J1 = ForwardDiff.jacobian(test2, a)
    J2 = ReverseDiff.jacobian(test2, a)

    @test all(isapprox.(J1, J2, atol=3e-12))

    J1 = ForwardDiff.jacobian(test3, a)
    J2 = ReverseDiff.jacobian(test3, a)

    @test all(isapprox.(J1, J2, atol=3e-12))
end

@testset "1d (also parameters)" begin

    residual(y, x, p) = y/x[1] + x[2]*cos(y)
    solve(x, p) = brent(y -> residual(y, x, p), p[1], p[2])[1]

    function test_orig(x)
        yl = -3.0
        yu = 3.0
        p = [yl, yu]
        x[1] *= 3
        y = solve(x, p)
        z = [y; (x[2]+3)^2]
        return z
    end

    function test(x)
        yl = -3.0
        yu = 3.0
        p = [yl, yu]
        x[1] *= 3
        y = implicit(solve, residual, x, p)
        z = [y; (x[2]+3)^2]
        return z
    end

    xv = [0.5, 1.0]
    # test(xv)

    J1 = ForwardDiff.jacobian(test_orig, xv)
    J2 = ForwardDiff.jacobian(test, xv)
    @test all(isapprox.(J1, J2, rtol=1e-15))

end

@testset "explicit_unsteady (in-place)" begin

    function lotkavolterra(dy, y, x, t)
        dy[1] = x[1]*y[1] - x[2]*y[1]*y[2]
        dy[2] = -x[3]*y[2] + x[4]*y[1]*y[2]
        return nothing
    end

    x0 = [1.5,1.0,3.0,1.0]; y0 = [1.0, 1.0]; tspan = (0.0, 10.0);
    prob = OrdinaryDiffEq.ODEProblem(lotkavolterra, y0, tspan, x0)
    alg = OrdinaryDiffEq.Tsit5()

    # times at which to evaluate the solution
    t = tspan[1]:0.1:tspan[2];

    # method for solving the ODEProblem
    function unsteady_solve(x, p=())
        _prob = OrdinaryDiffEq.remake(prob, p=x)
        sol = OrdinaryDiffEq.solve(_prob, alg, adaptive=false, tstops=t)
        return hcat(sol.u...), sol.t
    end

    # grab Tsit5 cache constants from DifferentialEquations
    cache = OrdinaryDiffEq.Tsit5ConstantCacheActual(Float64, Float64)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache

    # initialize for reuse
    k1 = nothing; k2 = nothing; k3 = nothing
    k4 = nothing; k5 = nothing; k6 = nothing

    # perform_step function
    function perform_step!(y, yprev, t, tprev, x, p)
        if t == tprev
            # initial conditions are prescribed
            y .= y0
        else

            # initialize k vectors on first call (saved for inplace after that)
            T = eltype(x)
            if isnothing(k1) || eltype(k1) != T
                k1 = similar(y, T)
                k2 = similar(y, T)
                k3 = similar(y, T)
                k4 = similar(y, T)
                k5 = similar(y, T)
                k6 = similar(y, T)
            end

            # adopted from perform_step! for the Tsit5 algorithm
            dt = t - tprev
            lotkavolterra(k1, yprev, x, t)
            lotkavolterra(k2, yprev+dt*a21*k1, x, t+c1*dt)
            lotkavolterra(k3, yprev+dt*(a31*k1+a32*k2), x, t+c2*dt)
            lotkavolterra(k4, yprev+dt*(a41*k1+a42*k2+a43*k3), x, t+c3*dt)
            lotkavolterra(k5, yprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4), x, t+c4*dt)
            lotkavolterra(k6, yprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), x, t+dt)
            y .= yprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
        end
    end

    # objective/loss function
    program(x) = sum(sum(unsteady_solve(x)[1]))

    # modified objective/loss function
    modprogram(x) = sum(sum(explicit_unsteady(unsteady_solve, perform_step!, x, ())[1]))

    # compute forward-mode sensitivities using ForwardDiff
    g1 = ForwardDiff.gradient(program, x0)

    # compute forward-mode sensitivities using ImplicitAD
    g2 = ForwardDiff.gradient(modprogram, x0)

    # compute reverse-mode sensitivities using ImplicitAD
    g3 = ReverseDiff.gradient(modprogram, x0)

    # test forward-mode ImplicitAD
    @test all(isapprox.(g1, g2, atol=1e-12))

    # test reverse-mode ImplicitAD
    @test all(isapprox.(g1, g3, atol=1e-11))

end

@testset "explicit_unsteady (out-of-place)" begin

    function lotkavolterra(y, x, t)
        dy1 = x[1]*y[1] - x[2]*y[1]*y[2]
        dy2 = -x[3]*y[2] + x[4]*y[1]*y[2]
        return [dy1, dy2]
    end

    x0 = [1.5,1.0,3.0,1.0]; y0 = [1.0, 1.0]; tspan = (0.0, 10.0);
    prob = OrdinaryDiffEq.ODEProblem(lotkavolterra, y0, tspan, x0)
    alg = OrdinaryDiffEq.Tsit5()

    # times at which to evaluate the solution
    t = tspan[1]:0.1:tspan[2];

    # method for solving the ODEProblem
    function unsteady_solve(x, p=())
        _prob = OrdinaryDiffEq.remake(prob, p=x)
        sol = OrdinaryDiffEq.solve(_prob, alg, adaptive=false, tstops=t)
        return hcat(sol.u...), sol.t
    end

    # grab Tsit5 cache constants from DifferentialEquations
    cache = OrdinaryDiffEq.Tsit5ConstantCacheActual(Float64, Float64)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache

    # perform_step function
    function perform_step(yprev, t, tprev, x, p)
        if t == tprev
            # initial conditions are prescribed
            y = y0
        else
            # adopted from perform_step! for the Tsit5 algorithm
            dt = t - tprev
            k1 = lotkavolterra(yprev, x, t)
            k2 = lotkavolterra(yprev+dt*a21*k1, x, t+c1*dt)
            k3 = lotkavolterra(yprev+dt*(a31*k1+a32*k2), x, t+c2*dt)
            k4 = lotkavolterra(yprev+dt*(a41*k1+a42*k2+a43*k3), x, t+c3*dt)
            k5 = lotkavolterra(yprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4), x, t+c4*dt)
            k6 = lotkavolterra(yprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), x, t+dt)
            y = yprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
        end
        return y
    end

    # objective/loss function
    program(x) = sum(sum(unsteady_solve(x)[1]))

    # modified objective/loss function
    modprogram(x) = sum(sum(explicit_unsteady(unsteady_solve, perform_step, x, (); compile=false)[1]))

    # modified objective/loss function (with compiled tape)
    compiled_modprogram(x) = sum(sum(explicit_unsteady(unsteady_solve, perform_step, x, (); compile=true)[1]))

    # compute forward-mode sensitivities using ForwardDiff
    g1 = ForwardDiff.gradient(program, x0)

    # compute forward-mode sensitivities using ImplicitAD
    g2 = ForwardDiff.gradient(modprogram, x0)

    # compute reverse-mode sensitivities using ImplicitAD
    g3 = ReverseDiff.gradient(modprogram, x0)

    # compute reverse-mode sensitivities using ImplicitAD (with compiled vjp)
    g4 = ReverseDiff.gradient(compiled_modprogram, x0)

    # test forward-mode ImplicitAD
    @test all(isapprox.(g1, g2, atol=1e-12))

    # test reverse-mode ImplicitAD
    @test all(isapprox.(g1, g3, atol=1e-12))

    # test reverse-mode ImplicitAD (with compiled vjp)
    @test all(isapprox.(g1, g4, atol=1e-12))

end

@testset "implicit_unsteady (in-place)" begin

    function robertson!(out,dy,y,x,t)
        out[1] = - x[1]*y[1]               + x[2]*y[2]*y[3] - dy[1]
        out[2] = + x[1]*y[1] - x[3]*y[2]^2 - x[2]*y[2]*y[3] - dy[2]
        out[3] = y[1] + y[2] + y[3] - x[4]
    end

    x0 = [0.04,1e4,3e7,1.0]; tspan=(1e-6,1e5); y0 = [1.0,0.0,0.0]; dy0 = [-0.04,0.04,0.0];
    prob = OrdinaryDiffEq.DAEProblem(robertson!, dy0, y0, tspan, x0, differential_vars = [true,true,false])
    alg = OrdinaryDiffEq.DImplicitEuler()

    # times at which to evaluate the solution
    t = range(tspan[1], tspan[2], length=100)

    # method for solving the DAEProblem
    function unsteady_solve(x, p=())
        _prob = OrdinaryDiffEq.remake(prob, p=x)
        sol = OrdinaryDiffEq.solve(_prob, alg, abstol=1e-9, reltol=1e-9, saveat=t, initializealg=OrdinaryDiffEq.NoInit())
        return hcat(sol.u...), sol.t
    end

    # residual function
    function residual!(r, y, yprev, t, tprev, x, p)
        if t == tspan[1]
            r .= y - y0
        else
            robertson!(r, (y - yprev)/(t - tprev), y, x, t)
        end
    end

    # objective/loss function
    program(x) = sum(sum(unsteady_solve(x)[1]))

    # modified objective/loss function
    modprogram(x) = sum(sum(implicit_unsteady(unsteady_solve, residual!, x, ())[1]))

    # compute forward-mode sensitivities using ForwardDiff
    g1 = ForwardDiff.gradient(program, x0)

    # compute forward-mode sensitivities using ImplicitAD
    g2 = ForwardDiff.gradient(modprogram, x0)

    # compute reverse-mode sensitivities using ImplicitAD
    g3 = ReverseDiff.gradient(modprogram, x0)

    # test forward-mode ImplicitAD
    @test all(isapprox.(g1, g2, atol=1e-12))

    # test reverse-mode ImplicitAD
    @test all(isapprox.(g1, g3, atol=1e-12))

end

@testset "implicit_unsteady (out-of-place)" begin

    function robertson(dy,y,x,t)
        out1 = - x[1]*y[1]               + x[2]*y[2]*y[3] - dy[1]
        out2 = + x[1]*y[1] - x[3]*y[2]^2 - x[2]*y[2]*y[3] - dy[2]
        out3 = y[1] + y[2] + y[3] - x[4]
        return [out1, out2, out3]
    end

    x0 = [0.04,1e4,3e7,1.0]; tspan=(1e-6,1e5); y0 = [1.0,0.0,0.0]; dy0 = [-0.04,0.04,0.0];
    prob = OrdinaryDiffEq.DAEProblem(robertson, dy0, y0, tspan, x0, differential_vars = [true,true,false])
    alg = OrdinaryDiffEq.DImplicitEuler()

    # times at which to evaluate the solution
    t = range(tspan[1], tspan[2], length=100)

    # method for solving the DAEProblem
    function unsteady_solve(x, p=())
        _prob = OrdinaryDiffEq.remake(prob, p=x)
        sol = OrdinaryDiffEq.solve(_prob, alg, abstol=1e-9, reltol=1e-9, saveat=t, initializealg=OrdinaryDiffEq.NoInit())
        return hcat(sol.u...), sol.t
    end

    # residual function
    function residual(y, yprev, t, tprev, x, p)
        if t == tspan[1]
            return promote_type(typeof(t), eltype(y), eltype(x)).(y - y0)
        else
            return robertson((y - yprev)/(t - tprev), y, x, t)
        end
    end

    # objective/loss function
    program(x) = sum(sum(unsteady_solve(x)[1]))

    # modified objective/loss function
    modprogram(x) = sum(sum(implicit_unsteady(unsteady_solve, residual, x, ())[1]))

    # compute forward-mode sensitivities using ForwardDiff
    g1 = ForwardDiff.gradient(program, x0)

    # compute forward-mode sensitivities using ImplicitAD
    g2 = ForwardDiff.gradient(modprogram, x0)

    # compute reverse-mode sensitivities using ImplicitAD
    g3 = ReverseDiff.gradient(modprogram, x0)

    # test forward-mode ImplicitAD
    @test all(isapprox.(g1, g2, atol=1e-12))

    # test reverse-mode ImplicitAD
    @test all(isapprox.(g1, g3, atol=1e-12))

end

@testset "provide partials" begin

    function yo(x, p)
        y = x.^2
        z = 3*y
        w = cos.(z)
        # w[1] += z[2]
        b = w .+ [[z[2]/10]; zeros(eltype(w), length(w)-1)]
        return b
    end

    function yo2(x, p)
        y = x.^3
        z = [sin.(y); y[1]/1000]
        w = exp.(z)
        return w
    end

    function jacobian(x, p)
        y = x.^3
        z = [sin.(y); y[1]/1000]
        w = exp.(z)
        dydx = diagm(3*x.^2)
        dzdy = [diagm(cos.(y)); 1.0/1000 zeros(eltype(y), length(y)-1)']
        dzdx = dzdy*dydx
        dwdz = diagm(exp.(z))
        dwdx = dwdz*dzdx

        return dwdx
    end

    function jvp(x, p, v)
        # dwdx*v = dwdz*dzdy*dydx*v
        y = x.^3
        z = [sin.(y); y[1]/1000]
        w = exp.(z)

        v = @. 3*x^2 * v  # dydx*v
        v = @. [cos(y) * v; v[1]/1000]  # dzdy*newv
        v = @. exp(z) * v #dwdz*newv

        return v
    end

    function vjp(x, p, v)
        # v'*dwdx = v'*dwdz*dzdy*dydx
        y = x.^3
        z = [sin.(y); y[1]/1000]
        w = exp.(z)

        v = @. v * exp(z)  # v'*dwdz
        v = [v[1]*cos(y[1]) + v[end]/1000; v[2:end-1] .* cos.(y[2:end])]  # newv'*dzdy
        v = @. v * 3*x^2  # newv'*dydx

        return v
    end

    function program(x)
        p = ()
        w = yo(x, p)
        w2 = [w; w; w]
        v = yo2(w2, p)
        return v .+ w[2]
    end

    function modprogram(x, mode, jacobian=nothing, jvp=nothing, vjp=nothing)
        p = ()
        w = yo(x, p)
        w2 = [w; w; w]
        v = provide_rule(yo2, w2, p; mode, jacobian, jvp, vjp)
        return v .+ w[2]
    end

    function program2(x)
        p = ()
        w = yo(x, p)
        w2 = [w[1]; w[2]]
        v = yo2(w2, p)
        return v .+ w[2]
    end

    function modprogram2(x, mode)
        p = ()
        w = yo(x, p)
        w2 = [w[1]; w[2]]
        v = provide_rule(yo2, w2, p; mode)
        return v .+ w[2]
    end


    x = [1.0, 2.0, 3.0]

    mode = "ffd"
    J1 = ForwardDiff.jacobian(program, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-4))

    J1 = ForwardDiff.jacobian(program2, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram2(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-4))

    J1 = ReverseDiff.jacobian(program, x)
    J2 = ReverseDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-4))

    mode = "cfd"
    J1 = ForwardDiff.jacobian(program, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-7))

    J1 = ForwardDiff.jacobian(program2, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram2(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-7))

    J1 = ReverseDiff.jacobian(program, x)
    J2 = ReverseDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-7))

    mode = "cs"
    J1 = ForwardDiff.jacobian(program, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    J1 = ForwardDiff.jacobian(program2, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram2(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    J1 = ReverseDiff.jacobian(program, x)
    J2 = ReverseDiff.jacobian(xt -> modprogram(xt, mode), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    mode = "jacobian"
    J1 = ForwardDiff.jacobian(program, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram(xt, mode, jacobian), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    J1 = ReverseDiff.jacobian(program, x)
    J2 = ReverseDiff.jacobian(xt -> modprogram(xt, mode, jacobian), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    mode = "vp"
    J1 = ForwardDiff.jacobian(program, x)
    J2 = ForwardDiff.jacobian(xt -> modprogram(xt, mode, jacobian, jvp), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))

    J1 = ReverseDiff.jacobian(program, x)
    J2 = ReverseDiff.jacobian(xt -> modprogram(xt, mode, jacobian, jvp, vjp), x)
    @test all(isapprox.(J1, J2, rtol=1e-15))


end


@testset "eigenvalues" begin

    # -- B is identity, eigenvalues/vectors are complex ---
    function eigsolve1(A, B)
        λ = eigvals(A)
        V = eigvecs(A)
        U = eigvecs(A')
        U = [U[:, 2] U[:, 1]]  # reorder to match left eigenvector order

        return λ, V, U
    end

    function test1(x)
        A = [x[1] x[2]; x[3] 2.0]
        B = Matrix(1.0I, 2, 2)
        λ = implicit_eigval(A, B, eigsolve1)
        z = [real(λ[1]) + imag(λ[1]); real(λ[2]) + imag(λ[2])]
        return z
    end

    x = [-4.0, -17.0, 2.0]
    J1 = ForwardDiff.jacobian(test1, x)
    Jfd = FiniteDiff.finite_difference_jacobian(test1, x, Val{:central})
    J2 = ReverseDiff.jacobian(test1, x)

    @test all(isapprox.(J1, J2, atol=1e-15))
    @test all(isapprox.(J1, Jfd, atol=1e-10))


    # --- B is identity, symmetric case (eigenvalues/vectors are all real) ---
    function eigsolve2(A, B)
        λ = eigvals(A)
        V = eigvecs(A)
        return λ, V, V
    end

    function test2(x)
        A = [x[1] x[2]; x[2] x[3]]
        B = Matrix(1.0I, 2, 2)
        λ = ImplicitAD.implicit_eigval(A, B, eigsolve2)
        z = [real(λ[1]) + imag(λ[1]); real(λ[2]) + imag(λ[2])]
        return z
    end

    x = [-4.0, -17.0, 2.0]
    J1 = ForwardDiff.jacobian(test2, x)
    Jfd = FiniteDiff.finite_difference_jacobian(test2, x, Val{:central})
    J2 = ReverseDiff.jacobian(test2, x)

    @test all(isapprox.(J1, J2, atol=1e-15))
    @test all(isapprox.(J1, Jfd, atol=1e-9))

    # --- A and B both random ------
    function eigsolve3(A, B)
        λ = eigvals(A, B)
        V = eigvecs(A, B)
        U = eigvecs(A', B')

        return λ, V, U
    end

    function test3(x)
        A = [x[1] x[2]; x[3] 2.0]
        B = [x[4] x[5]; x[6] x[7]]
        λ = ImplicitAD.implicit_eigval(A, B, eigsolve3)
        z = [real(λ[1]) + imag(λ[1]); real(λ[2]) + imag(λ[2])]
        return z
    end

    x = [-4.0, -17.0, 2.0, 2.5, 5.6, -4.0, 1.1]
    J1 = ForwardDiff.jacobian(test3, x)
    Jfd = FiniteDiff.finite_difference_jacobian(test3, x, Val{:central})
    J2 = ReverseDiff.jacobian(test3, x)

    @test all(isapprox.(J1, J2, atol=1e-15))
    @test all(isapprox.(J1, Jfd, atol=1e-9))

end