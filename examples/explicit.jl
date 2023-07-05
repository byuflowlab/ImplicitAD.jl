using ImplicitAD
using ForwardDiff
using ReverseDiff
using BenchmarkTools

struct Tsit5{TV, TF}
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
    c1::TF
    c2::TF
    c3::TF
    c4::TF
    a21::TF
    a31::TF
    a32::TF
    a41::TF
    a42::TF
    a43::TF
    a51::TF
    a52::TF
    a53::TF
    a54::TF
    a61::TF
    a62::TF
    a63::TF
    a64::TF
    a65::TF
    a71::TF
    a72::TF
    a73::TF
    a74::TF
    a75::TF
    a76::TF
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

    # constants
    c1 = 0.161; c2 = 0.327; c3 = 0.9; c4 = 0.9800255409045097;
    a21 = 0.161;
    a31 = -0.008480655492356989; a32 = 0.335480655492357;
    a41 = 2.8971530571054935; a42 = -6.359448489975075; a43 = 4.3622954328695815;
    a51 = 5.325864828439257; a52 = -11.748883564062828; a53 = 7.4955393428898365; a54 = -0.09249506636175525;
    a61 = 5.86145544294642; a62 = -12.92096931784711; a63 = 8.159367898576159; a64 = -0.071584973281401; a65 = -0.028269050394068383;
    a71 = 0.09646076681806523; a72 = 0.01; a73 = 0.4798896504144996; a74 = 1.379008574103742; a75 = -3.290069515436081; a76 = 2.324710524099774

    return Tsit5(k1, k2, k3, k4, k5, k6, ytemp2, ytemp3, ytemp4, ytemp5, ytemp6, c1, c2, c3, c4, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76)
end

function odestep!(tsit::Tsit5, odefun, y, yprev, t, tprev, xd, xci, p)

    dt = t - tprev

    odefun(tsit.k1, yprev, t, xd, xci, p)
    @. tsit.ytemp2 = yprev + dt*tsit.a21*tsit.k1
    odefun(tsit.k2, tsit.ytemp2, t+tsit.c1*dt, xd, xci, p)
    @. tsit.ytemp3 = yprev + dt*(tsit.a31*tsit.k1+tsit.a32*tsit.k2)
    odefun(tsit.k3, tsit.ytemp3, t+tsit.c2*dt, xd, xci, p)
    @. tsit.ytemp4 = yprev + dt*(tsit.a41*tsit.k1+tsit.a42*tsit.k2+tsit.a43*tsit.k3)
    odefun(tsit.k4, tsit.ytemp4, t+tsit.c3*dt, xd, xci, p)
    @. tsit.ytemp5 = yprev + dt*(tsit.a51*tsit.k1+tsit.a52*tsit.k2+tsit.a53*tsit.k3+tsit.a54*tsit.k4)
    odefun(tsit.k5, tsit.ytemp5, t+tsit.c4*dt, xd, xci, p)
    @. tsit.ytemp6 = yprev + dt*(tsit.a61*tsit.k1+tsit.a62*tsit.k2+tsit.a63*tsit.k3+tsit.a64*tsit.k4+tsit.a65*tsit.k5)
    odefun(tsit.k6, tsit.ytemp6, t+dt, xd, xci, p)

    @. y = yprev + dt*(tsit.a71*tsit.k1+tsit.a72*tsit.k2+tsit.a73*tsit.k3+tsit.a74*tsit.k4+tsit.a75*tsit.k5+tsit.a76*tsit.k6)

    return y
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


function initialize(t0, xd, xc0, p)
    (; n) = p
    return 300*ones((n-2)*(n-2))
end

function runit(n, nt)

    tsit = Tsit5((n-2)*(n-2), Float64)
    onestep!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsit, plate, y, yprev, t, tprev, xd, xci, p)

    # problem constants
    p = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(n, n), dT=zeros(n, n), n=n, δ=1/(n-1))

    t = range(0.0, 5000, nt)
    nt = length(t)

    xc = reverse(range(600, 1000, n))*ones(nt)'
    xd = Float64[]

    function program(xc)
        xcm = reshape(xc, n, nt)

        TF = eltype(xc)
        if eltype(tsit.k1) != TF
            tsit = Tsit5((n-2)*(n-2), TF)
            pf = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(TF, n, n), dT=zeros(TF, n, n), n=n, δ=1/(n-1))
            onestep!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsit, plate, y, yprev, t, tprev, xd, xci, pf)
        end

        y = ImplicitAD.odesolve(initialize, onestep!, t, xd, xcm, p)

        return y[1, end]  # top left corner temperature at last time
    end

    # ------ implicitAD ----------

    TR = eltype(ReverseDiff.track([1.0]))
    tsitf = Tsit5((n-2)*(n-2), Float64)
    tsitr = Tsit5((n-2)*(n-2), TR)
    pr = (k=400.0, rho=8960.0, Cp=386.0, tz=.01, σ=5.670373e-8, hc=1.0, Ta=300.0, ϵ=0.5, T=zeros(TR, n, n), dT=zeros(TR, n, n), n=n, δ=1/(n-1))
    onestepf!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsitf, plate, y, yprev, t, tprev, xd, xci, p)
    onestepr!(y, yprev, t, tprev, xd, xci, p) = odestep!(tsitr, plate, y, yprev, t, tprev, xd, xci, pr)
    cache = ImplicitAD.explicit_unsteady_cache(initialize, onestepr!, (n-2)*(n-2), 0, n, p; compile=true)

    function modprogram(xc)
        xcm = reshape(xc, n, nt)

        y = explicit_unsteady(initialize, onestepf!, t, xd, xcm, p; cache)

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


    # approximate cost of central diff
    t1 = @benchmark $program($xcv)
    time1 = median(t1).time * 1e-9 * (2*length(xcv) + 1)

    # forward
    t2 = @benchmark ForwardDiff.gradient!($g1, $program, $xcv, $fwd_cache)
    time2 = median(t2).time * 1e-9

    #reverse diff
    t3 = @benchmark ReverseDiff.gradient!($g2, $rev_cache2, $xcv)
    time3 = median(t3).time * 1e-9

    # reverse w/ implicitad
    t4 = @benchmark ReverseDiff.gradient!($g3, $rev_cache1, $xcv)
    time4 = median(t4).time * 1e-9  # reverse implicit diff

    println(time1, " ", time2, " ", time3, " ", time4)

    return time1, time2, time3, time4

end

nt = 1000
nvec = [3, 5, 7, 9, 11, 13, 15, 17, 19]
# nvec = [19]
nn = length(nvec)
t1 = zeros(nn)
t2 = zeros(nn)
t3 = zeros(nn)
t4 = zeros(nn)
states = zeros(nn)

for i = 1:nn
    n = nvec[i]
    t1[i], t2[i], t3[i], t4[i] = runit(n, nt)
    states[i] = (n-2)^2
end
