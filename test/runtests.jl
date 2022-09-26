using ImplicitAD
using Test
using NLsolve
using ForwardDiff
using ReverseDiff
using FiniteDiff
using LinearAlgebra: Symmetric, factorize
using SparseArrays: sparse

@testset "residual" begin
    

    function residual(x, y)
        T = promote_type(eltype(x), eltype(y))
        r = zeros(T, 2)
        r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r[2] = sin(y[2]*exp(y[1])-1)*x[4]
        return r
    end

    function solve(x)
        rwrap(y) = residual(x[1:4], y)
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
        y = implicit_function(solve, residual, w)
        return y[1] .+ w*y[2]
    end

    x = [1.0; 2.0; 3.0; 4.0; 5.0]
    
    J1 = ForwardDiff.jacobian(modprogram, x)
    J2 = ReverseDiff.jacobian(modprogram, x)

    @test all(isapprox.(J1, J2, atol=1e-14))


    # -------- supply one of the jacobians -----------

    function drdy(residual, x, y)
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
        y = implicit_function(solve, residual, w, drdy=drdy)
        return y[1] .+ w*y[2]
    end

    J3 = ForwardDiff.jacobian(modprogram2, x)
    J4 = ReverseDiff.jacobian(modprogram2, x)

    @test all(isapprox.(J1, J3, atol=1e-14))
    @test all(isapprox.(J1, J4, atol=1e-14))

end



@testset "linear" begin
    

    function solvelin(x)
        A = [x[1]*x[2] x[3]+x[4];
            x[3]+x[4] 0.0]
        b = [2.0, 3.0]
        Af = factorize(Symmetric(sparse(A)))
        return Af \ b
    end

    function Alin(residual, x, y)
        return [x[1]*x[2] x[3]+x[4];
            x[3]+x[4] 0.0]
    end

    # r = A * y - b
    # drdx = dAdx * y
    function Blin(residual, x, y)
        B = [x[2]*y[1]  x[1]*y[1]  y[2]  y[2];
            0.0  0.0  y[1]  y[1]]
        return B
    end

    function test(x)
        w = implicit_function(solvelin, nothing, x, drdy=Alin, drdx=Blin)
        return 2*w
    end

    x = [1.0, 2.0, 3.0, 4.0]
    J1 = ForwardDiff.jacobian(test, x)
    J2 = ReverseDiff.jacobian(test, x)
    
    @test all(isapprox.(J1, J2, atol=1e-14))
end


@testset "overload" begin

    function residual(r, x, y)
        T = promote_type(eltype(x), eltype(y))
        r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
        r[2] = sin(y[2]*exp(y[1])-1)*x[4]
        return r
    end

    function solve(x)
        rwrap(r, y) = residual(r, x[1:4], y)
        res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
        return res.zero
    end

    function program(x)
        z = 2.0*x
        w = z + x.^2
        y = solve(w)
        return y[1] .+ w*y[2]
    end
       
   
    struct MyData end

    function myjvp(residual, x, y, v, rd)
        println("here")

        # dual
        xd = ImplicitAD.pack_dual(x, v, MyData)
        
        # evaluate residual function
        residual(rd, xd, y)  # constant y
        
        # extract partials
        _, b = ImplicitAD.unpack_dual(rd) 

        return b
    end

    function mydrdy(x, y, A, r)

        wrap!(rt, yt) = residual(rt, x, yt)
   
        ForwardDiff.jacobian!(A, wrap!, r, y)    

        return A
    end

    function modprogram(x, jvp, drdy)
        z = 2.0*x
        w = z + x.^2
        y = implicit_function(solve, residual, w, jvp=jvp, drdy=drdy)
        return y[1] .+ w*y[2]
    end


    function run()
        
        rd = zeros(ForwardDiff.Dual{MyData}, 2)
        r = zeros(2)
        A = zeros(2, 2)

        jvp(residual, x, y, v) = myjvp(residual, x, y, v, rd)
        drdy(residual, x, y) = mydrdy(x, y, A, r)

        wrap(x) = modprogram(x, jvp, drdy)
        
        x = [1.0; 2.0; 3.0; 4.0; 5.0]
        J1 = ForwardDiff.jacobian(wrap, x)  # can now change x and compute more while data stays allocated.
        
        J2 = FiniteDiff.finite_difference_jacobian(wrap, x)
        
        return J1, J2
    end

    J1, J2 = run()

    @test all(isapprox.(J1, J2, atol=2e-6))

end

@testset "linear" begin

    function test(a)
        A = a[1]*[1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear_function(A, b)
        z = 2*x
        return z
    end

    function test2(a)
        A = [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * a[2:4]
        x = implicit_linear_function(A, b)
        z = 2*x
        return z
    end

    function test3(a)
        A = a[1] * [1.0 2.0 3.0; 4.1 5.3 6.4; 7.4 8.6 9.7]
        b = 2.0 * ones(3)
        x = implicit_linear_function(A, b)
        z = 2*x
        return z
    end

    a = [1.2, 2.3, 3.1, 4.3]
    J1 = ForwardDiff.jacobian(test, a)
    J2 = FiniteDiff.finite_difference_jacobian(test, a, Val{:central})

    @test all(isapprox.(J1, J2, atol=1e-6))

    J1 = ForwardDiff.jacobian(test2, a)
    J2 = FiniteDiff.finite_difference_jacobian(test2, a, Val{:central})

    @test all(isapprox.(J1, J2, atol=2e-6))

    J1 = ForwardDiff.jacobian(test3, a)
    J2 = FiniteDiff.finite_difference_jacobian(test3, a, Val{:central})

    @test all(isapprox.(J1, J2, atol=2e-6))
end