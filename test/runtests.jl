using ImplicitAD
using Test
using NLsolve
using ForwardDiff
using ReverseDiff
using FiniteDiff
using LinearAlgebra: Symmetric, factorize, diagm, eigvals, eigvecs, I
using SparseArrays: sparse
using FLOWMath: brent


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

@testset "linear_user_mmul" begin

    count = 0

    function my_multiply(A, x)
        (m, n) = size(A)
        T = promote_type(eltype(A), eltype(x))
        y = zeros(T, m)
        for j in 1:n
            for i in 1:m
                y[i] += A[i, j] * x[j]
            end
        end
        # Provide a way to make sure this function was called
        count += 1
        return y
    end

    function test(a)
        A = a[1] * [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear(A, b; mmul=my_multiply)
        z = 2 * x
        return z
    end

    function test2(a)
        A = [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear(A, b; mmul=my_multiply)
        z = 2 * x
        return z
    end

    function test3(a)
        A = a[1] * [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * ones(3)
        x = implicit_linear(A, b; mmul=my_multiply)
        z = 2 * x
        return z
    end

    a = [1.2, 2.3, 3.1, 4.3]
    J1 = ForwardDiff.jacobian(test, a)
    J2 = ReverseDiff.jacobian(test, a)

    @test count == 1
    @test all(isapprox.(J1, J2, atol=3e-12))

    J1 = ForwardDiff.jacobian(test2, a)
    J2 = ReverseDiff.jacobian(test2, a)

    @test count == 2
    @test all(isapprox.(J1, J2, atol=3e-12))

    J1 = ForwardDiff.jacobian(test3, a)
    J2 = ReverseDiff.jacobian(test3, a)

    @test count == 3
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


@testset "explicit_unsteady, out-of-place, design vars" begin

    # ODE solver (Tsit5 explicit)
    function odestep(odefun, yprev, t, tprev, xd, xci, p)

        # constants
        c1 = 0.161; c2 = 0.327; c3 = 0.9; c4 = 0.9800255409045097;
        a21 = 0.161;
        a31 = -0.008480655492356989; a32 = 0.335480655492357;
        a41 = 2.8971530571054935; a42 = -6.359448489975075; a43 = 4.3622954328695815;
        a51 = 5.325864828439257; a52 = -11.748883564062828; a53 = 7.4955393428898365; a54 = -0.09249506636175525;
        a61 = 5.86145544294642; a62 = -12.92096931784711; a63 = 8.159367898576159; a64 = -0.071584973281401; a65 = -0.028269050394068383;
        a71 = 0.09646076681806523; a72 = 0.01; a73 = 0.4798896504144996; a74 = 1.379008574103742; a75 = -3.290069515436081; a76 = 2.324710524099774

        dt = t - tprev

        k1 = odefun(yprev, t, xd, xci, p)
        ytemp2 = @. yprev+dt*a21*k1
        k2 = odefun(ytemp2, t+c1*dt, xd, xci, p)
        ytemp3 = @. yprev+dt*(a31*k1+a32*k2)
        k3 = odefun(ytemp3, t+c2*dt, xd, xci, p)
        ytemp4 = @. yprev+dt*(a41*k1+a42*k2+a43*k3)
        k4 = odefun(ytemp4, t+c3*dt, xd, xci, p)
        ytemp5 = @. yprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
        k5 = odefun(ytemp5, t+c4*dt, xd, xci, p)
        ytemp6 = @. yprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
        k6 = odefun(ytemp6, t+dt, xd, xci, p)

        y = @. yprev + dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)

        return y
    end

    # ODE definition
    function lotkavolterra(y, t, xd, xci, p)
        return [xd[1]*y[1] - xd[2]*y[1]*y[2];
            -xd[3]*y[2] + xd[4]*y[1]*y[2]]
    end

    # setup solver
    ny = 2
    onestep(yprev, t, tprev, xd, xci, p) = odestep(lotkavolterra, yprev, t, tprev, xd, xci, p)
    initialize(t0, xd, xc0, p) = [1.0, 1.0]
    t = range(0.0, 10.0, step=0.1)

    # setup variables
    xd = [1.5,1.0,3.0,1.0]
    xc = Matrix{Float64}(undef, 0, length(t))
    p = ()

    # setup cache (needs a reverse-mode compatible type for the intermediate variables of this particular solver)
    nxd = length(xd)
    nxc = 0
    cache = explicit_unsteady_cache(initialize, onestep, ny, nxd, nxc, p; compile=true)

    # without package (need to change type of intermediate variables)
    function program(x)
        y = ImplicitAD.odesolve(initialize, onestep, t, x, xc, p)
        return sum(y)
    end

    # with implicitAD
    function modprogram(x)
        y = explicit_unsteady(initialize, onestep, t, x, xc, p; cache)
        return sum(y)
    end

    # compute forward and reverse-mode sensitivities through the iterations
    g1 = ForwardDiff.gradient(program, xd)

    g2 = ReverseDiff.gradient(program, xd)

    # compute reverse-mode sensitivities using ImplicitAD
    g3 = ReverseDiff.gradient(modprogram, xd)

    # test
    @test all(isapprox.(g1, g3, atol=1e-12))
    @test all(isapprox.(g2, g3, atol=1e-12))
end

@testset "explicit_unsteady, in-place, control vars" begin

    struct Tsit5{TV}
        k1::TV
        k2::TV
        k3::TV
        k4::TV
        k5::TV
        k6::TV
        ytemp2::TV
        ytemp3::TV
        ytemp4::TV
        ytemp5::TV
        ytemp6::TV
    end


    function Tsit5(ny, T)

        k1 = Vector{T}(undef, ny)
        k2 = Vector{T}(undef, ny)
        k3 = Vector{T}(undef, ny)
        k4 = Vector{T}(undef, ny)
        k5 = Vector{T}(undef, ny)
        k6 = Vector{T}(undef, ny)
        ytemp2 = Vector{T}(undef, ny)
        ytemp3 = Vector{T}(undef, ny)
        ytemp4 = Vector{T}(undef, ny)
        ytemp5 = Vector{T}(undef, ny)
        ytemp6 = Vector{T}(undef, ny)

        return Tsit5(k1, k2, k3, k4, k5, k6, ytemp2, ytemp3, ytemp4, ytemp5, ytemp6)
    end

    function odestep!(tsit::Tsit5, odefun, y, yprev, t, tprev, xd, xci, p)

        # constants
        c1 = 0.161; c2 = 0.327; c3 = 0.9; c4 = 0.9800255409045097;
        a21 = 0.161;
        a31 = -0.008480655492356989; a32 = 0.335480655492357;
        a41 = 2.8971530571054935; a42 = -6.359448489975075; a43 = 4.3622954328695815;
        a51 = 5.325864828439257; a52 = -11.748883564062828; a53 = 7.4955393428898365; a54 = -0.09249506636175525;
        a61 = 5.86145544294642; a62 = -12.92096931784711; a63 = 8.159367898576159; a64 = -0.071584973281401; a65 = -0.028269050394068383;
        a71 = 0.09646076681806523; a72 = 0.01; a73 = 0.4798896504144996; a74 = 1.379008574103742; a75 = -3.290069515436081; a76 = 2.324710524099774

        dt = t - tprev

        odefun(tsit.k1, yprev, t, xd, xci, p)
        @. tsit.ytemp2 = yprev+dt*a21*tsit.k1
        odefun(tsit.k2, tsit.ytemp2, t+c1*dt, xd, xci, p)
        @. tsit.ytemp3 = yprev+dt*(a31*tsit.k1+a32*tsit.k2)
        odefun(tsit.k3, tsit.ytemp3, t+c2*dt, xd, xci, p)
        @. tsit.ytemp4 = yprev+dt*(a41*tsit.k1+a42*tsit.k2+a43*tsit.k3)
        odefun(tsit.k4, tsit.ytemp4, t+c3*dt, xd, xci, p)
        @. tsit.ytemp5 = yprev+dt*(a51*tsit.k1+a52*tsit.k2+a53*tsit.k3+a54*tsit.k4)
        odefun(tsit.k5, tsit.ytemp5, t+c4*dt, xd, xci, p)
        @. tsit.ytemp6 = yprev+dt*(a61*tsit.k1+a62*tsit.k2+a63*tsit.k3+a64*tsit.k4+a65*tsit.k5)
        odefun(tsit.k6, tsit.ytemp6, t+dt, xd, xci, p)

        @. y = yprev + dt*(a71*tsit.k1+a72*tsit.k2+a73*tsit.k3+a74*tsit.k4+a75*tsit.k5+a76*tsit.k6)

        return y
    end


    # -------- define ODE -------------
    function sigmoid(x)
        z = exp(-x)
        return one(z) / (one(z) + z)
    end

    function sigmoid_blend(f1x, f2x, x, xt, hardness=50)
        sx = sigmoid(hardness*(x-xt))
        return f1x + sx*(f2x-f1x)
    end

    function get_aero(M)
        CLa1 = 3.44 + 1/cosh((M - 1)/0.06)^2
        CLa2 = 3.44 + 1/cosh(0.15/0.06)^2 - 0.96/0.63*(M - 1.15)
        CLa =  sigmoid_blend(CLa1, CLa2, M, 1.15)

        CD1 = 0.013 + 0.0144*(1 + tanh((M - 0.98)/0.06))
        CD2 = 0.013 + 0.0144*(1 + tanh(0.17/0.06)) - 0.011*(M - 1.15)
        CD0 = sigmoid_blend(CD1, CD2, M, 1.15)

        kappa1 = 0.54 + 0.15*(1 + tanh((M - 0.9)/0.06))
        kappa2 = 0.54 + 0.15*(1 + tanh(0.25/0.06)) + 0.14*(M - 1.15)
        kappa = sigmoid_blend(kappa1, kappa2, M, 1.15)

        return CLa, CD0, kappa
    end

    struct StandardAtm{TF}
        R::TF
        gamma::TF
        Tsl::TF
        psl::TF
    end

    StandardAtm() = StandardAtm(287.053, 1.4, 288.15, 101325.0)

    function atm(sa::StandardAtm, hkm)
        # hkm = h/1e3

        T = sa.Tsl - 71.5 + 2.0*log(1 + exp(35.75 - 3.25*hkm) + exp(-3.0 + 0.0003*hkm^3))
        p = sa.psl*exp(-0.118*hkm - 0.0015*hkm^2/(1 - 0.018*hkm + 0.0011*hkm^2))

        rho = p/(sa.R*T)
        a = sqrt(sa.gamma * sa.R * T)

        return rho, a
    end

    struct ClimbConstants{TF, TM}
        Isp::TF
        S::TF
        g::TF
        Q::TM
    end

    ClimbConstants() = ClimbConstants(1600.0, 530.0*0.092903, 9.80665,
        [24.55416794296985 -0.28583808117639187 -0.02998468662626332 0.0010901749936383198 -1.0796289992336762e-5
        16.816653138159015 -0.9455088718102456  0.07650920826522727 -0.0028260953205047944 3.0950742821331396e-5
        -31.601553070196815 -0.9497558256343946  0.06539937416161391 7.934162473061309e-5 -1.5650745803961714e-5
        52.73909009635939 1.4241420345766118 -0.16511709949375472 0.003013085699850632 -1.2779625049498016e-5
        -25.548458356626316 0.2223300355871822  0.03971815888577348 -0.0010010526356029202 6.026042742844311e-6])

    function climb(dy, y, t, xd, xc, p)
        # unpack states (velocity, flight path angle, altitude, range, mass)
        V, γ, h, _, m = y

        # unpack parameters
        sa, cc = p

        # control variables
        alpha = xc[1]

        # atmosphere
        rho, a = atm(sa, h)
        M = V/a

        # thrust
        throttle = 1.0
        h10 = h * 3280.84 / 10e3  # convert to 10,000 ft
        Tmax = [1 M M^2 M^3 M^4] * cc.Q * [1; h10; h10^2; h10^3; h10^4] * 4.4482216153
        T = throttle * Tmax[1]

        # aero
        CLa, CD0, kappa = get_aero(M)

        q = 0.5 * rho * V^2
        L = CLa * alpha * q * cc.S
        D = (CD0 + kappa*CLa*alpha^2) * q * cc.S

        # dynamics
        dy[1] = T/m*cos(alpha) - D/m - cc.g*sin(γ)
        dy[2] = T/(m*V)*sin(alpha) + L/(m*V) - cc.g*cos(γ)/V
        dy[3] = V*sin(γ)
        dy[4] = V*cos(γ)
        dy[5] = -T/(cc.g * cc.Isp)
    end

    function initialize(t0, xd, xc0, p)
        V0 = 135.964
        γ0 = 0.0
        h0 = 100.0
        r0 = 0.0
        m0 = 19030.468
        return [V0, γ0, h0, r0, m0]
    end
    # ----------------------------

    # setup solver
    ny = 5
    tsit = Tsit5(ny, Float64)
    onestep!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsit, climb, y, yprev, t, tprev, xd, xci, p)
    t = range(0.0, 320.0, step=0.1)

    # setup variables
    xd = Float64[]
    xc = 2*pi/180*ones(length(t))
    xc = reshape(xc, 1, length(xc))  # make into matrix


    # setup cache (needs a reverse-mode compatible type for the intermediate variables of this particular solver)
    nxd = length(xd)
    nxc = length(xc[:, 1])
    TR = eltype(ReverseDiff.track([1.0]))
    tsitr = Tsit5(ny, TR)
    onestepr!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsitr, climb, y, yprev, t, tprev, xd, xci, p)
    params = (StandardAtm(), ClimbConstants())
    cache = explicit_unsteady_cache(initialize, onestepr!, ny, nxd, nxc, params; compile=true)


    # without package (need to change type of intermediate variables)
    tsit_og = Tsit5(ny, Float64)
    onestep_og!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsit_og, climb, y, yprev, t, tprev, xd, xci, p)
    function program(x)
        T = eltype(x)
        if eltype(tsit_og.k1) != T
            tsit_og = Tsit5(ny, T)
            onestep_og!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsit_og, climb, y, yprev, t, tprev, xd, xci, p)
        end

        y = ImplicitAD.odesolve(initialize, onestep_og!, t, xd, x, params)
        V, γ, h, r, m = y[:, end]
        rho, a = atm(params[1], h)
        M = V/a
        return [M, γ, h]  # target Mach, flight path angle, and altitude
    end


    # with implicitAD
    function modprogram(x)
        y = explicit_unsteady(initialize, onestep!, t, xd, x, params; cache)
        V, γ, h, r, m = y[:, end]
        rho, a = atm(params[1], h)
        M = V/a
        return [M, γ, h]  # target Mach, flight path angle, and altitude
    end

    # compute forward and reverse-mode sensitivities through the iterations
    J1 = ForwardDiff.jacobian(program, xc)
    J2 = ReverseDiff.jacobian(program, xc)

    # compute reverse-mode sensitivities using ImplicitAD
    J3 = ReverseDiff.jacobian(modprogram, xc)

    # test ImplicitAD
    @test all(isapprox.(J1, J3, atol=2e-12))
    @test all(isapprox.(J2, J3, atol=2e-12))

end

@testset "implicit_unsteady, out of place" begin

    function robertson(dy, y, t, xd, xci, p)
        return [- xd[1]*y[1]                + xd[2]*y[2]*y[3] - dy[1];
                  xd[1]*y[1] - xd[3]*y[2]^2 - xd[2]*y[2]*y[3] - dy[2];
                  y[1] + y[2] + y[3] - xd[4]]
    end


    # implicit euler
    residual(y, yprev, t, tprev, xd, xci, p) = robertson((y .- yprev)/(t - tprev), y, t, xd, xci, p)


    function onestep(yprev, t, tprev, xd, xci, p)
        f(yt) = residual(yt, yprev, t, tprev, xd, xci, p)
        sol = nlsolve(f, yprev, autodiff=:forward, ftol=1e-12)
        return sol.zero
    end

    function initialize(t0, xd, xc0, p)
        y0 = [1.0, 0.0, 0.0]
        return y0
    end

    function runit()
        xd = [0.04, 1e4, 3e7, 1.0]
        xc = Matrix{Float64}(undef, 0, 100)
        p = ()
        t = range(1e-6, 1e5, length=100)

        ny = 3
        nxd = 4
        nxc = 0
        cache = ImplicitAD.implicit_unsteady_cache(initialize, residual, ny, nxd, nxc, p; compile=true)

        function program(x)
            y = ImplicitAD.odesolve(initialize, onestep, t, x, xc, p)
            return y[:, end]
        end

        function modprogram(x)
            y = implicit_unsteady(initialize, onestep, residual, t, x, xc, p; cache)
            return y[:, end]
        end

        J1 = ForwardDiff.jacobian(program, xd)

        J3 = ReverseDiff.jacobian(modprogram, xd)

        return J1, J3
    end

    J1, J3 = runit()

    @test all(isapprox.(J1, J3, atol=5e-10))

end

@testset "implicit_unsteady, in place" begin

    function robertson(r, dy, y, t, xd, xci, p)
        r[1] = - xd[1]*y[1]              + xd[2]*y[2]*y[3] - dy[1]
        r[2] = xd[1]*y[1] - xd[3]*y[2]^2 - xd[2]*y[2]*y[3] - dy[2]
        r[3] = y[1] + y[2] + y[3] - xd[4]
    end


    # implicit euler
    residual!(r, y, yprev, t, tprev, xd, xci, p) = robertson(r, (y .- yprev)/(t - tprev), y, t, xd, xci, p)


    function onestep!(y, yprev, t, tprev, xd, xci, p)
        f!(r, yt) = residual!(r, yt, yprev, t, tprev, xd, xci, p)
        sol = nlsolve(f!, yprev, autodiff=:forward, ftol=1e-12)
        y .= sol.zero
        return nothing
    end

    function initialize(t0, xd, xc0, p)
        y0 = [1.0, 0.0, 0.0]
        return y0
    end

    function runit()
        xd = [0.04, 1e4, 3e7, 1.0]
        xc = Matrix{Float64}(undef, 0, 100)
        p = ()
        t = range(1e-6, 1e5, length=100)

        ny = 3
        nxd = 4
        nxc = 0
        cache = ImplicitAD.implicit_unsteady_cache(initialize, residual!, ny, nxd, nxc, p; compile=true)

        function program(x)
            y = ImplicitAD.odesolve(initialize, onestep!, t, x, xc, p)
            return y[:, end]
        end

        function modprogram(x)
            y = implicit_unsteady(initialize, onestep!, residual!, t, x, xc, p; cache)
            return y[:, end]
        end

        J1 = ForwardDiff.jacobian(program, xd)

        J2 = ForwardDiff.jacobian(modprogram, xd)

        J3 = ReverseDiff.jacobian(modprogram, xd)

        return J1, J2, J3
    end

    J1, J2, J3 = runit()

    @test all(isapprox.(J1, J2, atol=5e-10))
    @test all(isapprox.(J1, J3, atol=5e-10))

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


# mainly for use via python, though we test in julia here just to make sure functionality is correct
@testset "derivative setup" begin

    function actuatordisk(a, A, rho, Vinf)
        q = 0.5 * rho * Vinf^2
        CT = 4 * a * (1 - a)
        CP = CT * (1 - a)
        T = q * A * CT
        P = q * A * CP * Vinf
        return T, P
    end

    actuatorwrapper(x, p) = collect(actuatordisk(x...))

    x = [0.3, 1.0, 1.1, 8.0]
    p = ()

    # forward jacobian
    jacobian = derivativesetup(actuatorwrapper, x, p, "fjacobian")

    J = zeros(2, length(x))
    jacobian(J, x)

    # reverse jacobian
    Jr = zeros(2, length(x))
    rjacobian = derivativesetup(actuatorwrapper, x, p, "rjacobian")

    rjacobian(Jr, x)

    # check equality
    @test all(isapprox.(J, Jr, rtol=1e-15))

    # spot checks (analytic)
    a = x[1]
    T, P = actuatorwrapper(x, p)
    @test isapprox(J[1, 1], T/(a*(1 - a))*(1 - 2*a), rtol=1e-15)  # dT/da
    @test isapprox(J[2, 3], P/x[3], rtol=1e-15)  # dP/drho

    # compiled reverse jacobian
    Jrc = zeros(2, length(x))
    rjacobiancomp = derivativesetup(actuatorwrapper, x, p, "rjacobian", true)

    rjacobiancomp(Jrc, x)
    @test all(isapprox.(Jr, Jrc, rtol=1e-15))

    # jacobian vector product
    jvp = derivativesetup(actuatorwrapper, x, p, "jvp")
    xdot = ones(length(x))
    fdot = zeros(2)
    jvp(fdot, x, xdot)
    @test all(isapprox.(fdot, J * xdot, rtol=1e-15))

    vjp = derivativesetup(actuatorwrapper, x, p, "vjp")
    xbar = zeros(length(x))
    fbar = ones(2)
    vjp(xbar, x, fbar)
    @test all(isapprox.(xbar, J' * fbar, rtol=1e-15))

end