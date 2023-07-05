using ImplicitAD
using ForwardDiff
using ReverseDiff
using BenchmarkTools
using NLsolve


function residual!(r, y, yprev, t, tprev, xd, xci, p)
    plate(r, y, t, xd, xci, p)  # using r as a placeholder for dy
    r .-= (y .- yprev)/(t - tprev)
end

function onestep!(y, yprev, t, tprev, xd, xci, p)
    f!(r, yt) = residual!(r, yt, yprev, t, tprev, xd, xci, p)
    j!(J, yt) = dTdy(J, yt, t, tprev, p)
    T = eltype(xci)
    if T <: ReverseDiff.TrackedReal
        T = eltype(ReverseDiff.track([1.0]))
    end
    sol = nlsolve(f!, j!, T.(yprev), ftol=1e-12)
    y .= sol.zero
    return nothing
end


function Q(T, p)
    (; hc, Ta, ϵ, σ) = p
    Qc = hc*(T - Ta)
    Qr = ϵ*σ*(T^4 - Ta^4)
    return Qc + Qr
end

function plate(dy, y, t, xd, xci, p)

    (; k, rho, Cp, δ, tz, n, T, dT) = p
    alpha = k/(rho*Cp*δ^2)
    beta = 2/(rho*Cp*tz)

    # set temperature grid
    T[2:n-1, 2:n-1] .= reshape(y, n-2, n-2)

    # Direchlet b.c. on bottom
    T[end, :] .= xci

    # Neuman b.c. on sides and top
    @views for i = 2:n-1
        T[i, 1] = T[i, 2]  # left
        T[i, end] = T[i, end-1]  # right
    end
    @views for j = 1:n
        T[1, j] = T[2, j] # top
    end

    # update interior points
    @views for i = 2:n-1
        for j = 2:n-1
            Tij = T[i, j]
            dT[i, j] = alpha*(T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*Tij) - beta*(Q(Tij, p))
        end
    end

    dy .= dT[2:n-1, 2:n-1][:]
end

function dTdy(J, y, t, tprev, p)

    (; k, rho, Cp, δ, tz, n, hc, ϵ, σ) = p
    alpha = k/(rho*Cp*δ^2)
    beta = 2/(rho*Cp*tz)

    J .= 0.0

    indices = LinearIndices((2:n-1, 2:n-1))
    ni, nj = size(indices)
    for i = 1:ni
        for j = 1:nj
            ij = indices[i, j]
            J[ij, ij] = -alpha*4 - beta*(hc + ϵ*σ*4*y[ij]^3)
            if i < ni
                J[ij, indices[i+1, j]] = alpha
            end
            if i > 1
                J[ij, indices[i-1, j]] = alpha
            end
            if i == 1
                J[ij, ij] += alpha
            end
            if j < nj
                J[ij, indices[i, j+1]] = alpha
            end
            if j == nj
                J[ij, ij] += alpha
            end
            if j > 1
                J[ij, indices[i, j-1]] = alpha
            end
            if j == 1
                J[ij, ij] += alpha
            end
            J[ij, ij] -= 1.0 / (t - tprev)
        end
    end

end




function initialize(t0, xd, xc0, p)
    (; n) = p
    return 300*ones((n-2)*(n-2))
end

function runit(n, nt)

    # problem constants
    p = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(n, n), dT=zeros(n, n), n=n, δ=1/(n-1))

    t = range(0.0, 5000, nt)
    nt = length(t)

    xc = reverse(range(600, 1000, n))*ones(nt)' # 1000*ones(n, nt)
    xd = Float64[]


    pf = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(n, n), dT=zeros(n, n), n=n, δ=1/(n-1))
    function program(xc)
        xcm = reshape(xc, n, nt)

        TF = eltype(xc)
        if eltype(pf.T) != TF
            pf = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(TF, n, n), dT=zeros(TF, n, n), n=n, δ=1/(n-1))
        end

        y = ImplicitAD.odesolve(initialize, onestep!, t, xd, xcm, pf)

        return y[1, end]  # top left corner temperature at last time
    end

    # ------ implicitAD ----------

    TR = eltype(ReverseDiff.track([1.0]))
    pr = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(TR, n, n), dT=zeros(TR, n, n), n=n, δ=1/(n-1))
    cache = ImplicitAD.implicit_unsteady_cache(initialize, residual!, (n-2)*(n-2), 0, n, pr; compile=true)

    Jdrdy = zeros((n-2)*(n-2), (n-2)*(n-2))
    function drdy(residual, r, y, yprev, t, tprev, xd, xci, p)
        dTdy(Jdrdy, y, t, tprev, p)
        return Jdrdy
    end

    function modprogram(xc)
        xcm = reshape(xc, n, nt)

        y = implicit_unsteady(initialize, onestep!, residual!, t, xd, xcm, p; cache, drdy)

        return y[1, end]
    end

    xcv = xc[:]

    fwd_cache = ForwardDiff.GradientConfig(program, xcv)
    f_tape1 = ReverseDiff.GradientTape(modprogram, xcv)
    rev_cache1 = ReverseDiff.compile(f_tape1)
    f_tape2 = ReverseDiff.GradientTape(program, xcv)
    rev_cache2 = ReverseDiff.compile(f_tape2)

    g1 = zeros(length(xcv))
    g2 = zeros(length(xcv))
    g3 = zeros(length(xcv))


    # approximate cost of finite diff
    program(xcv)
    t1 = @benchmark $program($xcv)
    time1 = median(t1).time * 1e-9 * (2*length(xcv) + 1)

    # forward
    t2 = @benchmark ForwardDiff.gradient!($g1, $program, $xcv, $fwd_cache)
    time2 = median(t2).time * 1e-9

    # reverse diff
    t3 = @benchmark ReverseDiff.gradient!($g2, $rev_cache2, $xcv)
    time3 = median(t3).time * 1e-9

    # reverse w/ implicitad
    t4 = @benchmark ReverseDiff.gradient!($g3, $rev_cache1, $xcv)
    time4 = median(t4).time * 1e-9  # reverse implicit diff

    println(time1, " ", time2, " ", time3, " ", time4)

    return nothing
    # return time1, time2, time3, time4

end

nt = 100
nvec = [3, 5, 7, 9, 11, 13, 15, 17, 19]
# nvec = [9]
nn = length(nvec)
t1 = zeros(nn)
t2 = zeros(nn)
t3 = zeros(nn)
t4 = zeros(nn)
states = zeros(nn)

for i = 1:nn
    n = nvec[i]
    runit(n, nt)
    states[i] = (n-2)^2
end
