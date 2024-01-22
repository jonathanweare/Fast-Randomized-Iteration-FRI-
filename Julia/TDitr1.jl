using LinearAlgebra
using SparseArrays
using Random
using Distributions
using Plots
using StatsPlots; ; gr(dpi=600)
# pyplot()

Random.seed!(1)

# plt1 = plot(title="Mean First Passage Time", ylabel="l_2 error", yscale=:log10, minorgrid=true)
# plt2 = plot(title="Committor", ylabel="l_2 error", yscale=:log10, minorgrid=true)
# plt3 = plot(framestyle=:none, background_color=:transparent, ylabel=:none)
# plt3 = plot(ylabel="(relative avar)/n^3", minorgrid=true)
# plt4 = plot(minorgrid=true, ylabel="(relative avar)/n^3")

plt1 = plot()
plt2 = plot()

for d = 2:2

    nitr = 2000

    global plt1, plt2, plt3, plt4

    alpha = 1.0

    n = 10*2^d
    # n = 10

    m = 1000000
    nouts = 0

    M = 0.1*n^2

    p = zeros(n)
    sup = zeros(n-1)
    sub = zeros(n-1)
    dd = zeros(n)

    x = ((1:n).-0.5)./n

    p = exp.(n.*(cospi.(4 .*x).+1)./(4*pi))
    sup[1:n-1] = 1 ./ (2 .*(exp.(n .* ( cospi.( 4 .* x[1:n-1] ) .- cospi.( 4 .* x[2:n] ) ) ./ (4*pi) ) .+ 1))
    # sup[1:n-1] = 1 ./ (p[1:n-1] ./ p[2:n] .+ 1)
    sub[1:n-1] = 1 ./ (2 .*(1 .+ exp.(n .* ( cospi.( 4 .* x[2:n] ) .- cospi.( 4 .* x[1:n-1] ) ) ./ (4*pi) )))
    # sub[1:n-1] = 1 ./ (1 .+ p[2:n] ./ p[1:n-1])
    dd[1] = 1 - sup[1]
    dd[2:n-1] = 1 .- sup[2:n-1] .- sub[1:n-2]
    dd[n] = 1 - sub[n-1]
    P = spdiagm(-1=>sub,0=>dd,1=>sup)
    A = I - P


    rT = ones(n)
    rT[1] = 0
    rT[n] = 0

    T = zeros(n)
    T[2:n-1] = A[2:n-1,2:n-1]\rT[2:n-1]

    rQ = zeros(n)
    rQ[2:n-1] = Vector(P[2:n-1,n])
    rQ[n] = 1

    Q = zeros(n)
    Q[n] = 1
    Q[2:n-1] = A[2:n-1,2:n-1]\rQ[2:n-1]

    # plt1 = plot!(plt1, x[2:n-1], T[2:n-1], lw=4 )
    # plt2 = plot!(plt2, x[2:n-1], Q[2:n-1], lw=4 )

    QTD = rand(n)
    QTD[1] = 0
    QTD[n] = 1

    QTSD = rand(n)
    QTSD[1] = 0
    QTSD[n] = 1

    WTD = rand(n)
    WTD[1] = 0
    WTD[n] = 0

    eQTD = zeros(m)
    eQTSD = zeros(m)

    for k = 1:m
        J = sample((2:n-1))
        K = wsample((1:n),Vector(P[J,:]))

        alpha = 0.01
        # alpha = k^(-0.6)
        
        if J != K
            # I - alpha*(I-P)
            QTD[J] -= alpha * (QTD[J]-QTD[K])
            WTD[J] -= alpha * (WTD[J]-WTD[K])
        end

        res = rQ[2:n-1] .- A[2:n-1,2:n-1]*QTD[2:n-1]
        c = dot(WTD[2:n-1],res)
        z = dot(WTD[2:n-1],A[2:n-1,2:n-1]*WTD[2:n-1])

        QTSD[2:n-1] = QTD[2:n-1] .+ (c/z) .* WTD[2:n-1]

        # QTD = QTSD

        if k%1000 == 1
            nouts = Int((k-1)/1000)+1

            println(nouts, " ", c," ", z, " ", norm(WTD) )

            eQTD[nouts] = norm(QTD - Q)
            eQTSD[nouts] = norm(QTSD - Q)
            # println(Int((k-1)/100)+1, " ",eQTD[Int((k-1)/100)+1], " ", eQTSD[Int((k-1)/100)+1])
        end
    end

    # plt1 = plot!(plt1, x, Q )
    # plt1 = plot!(plt1, x, QTD)

    plt1 = plot!(plt1, x, Q )
    plt1 = plot!(plt1, x, QTSD)


    plt2 = plot!(plt2,  eQTD[1:nouts], label="eQTD", yscale=:log10 )
    plt2 = plot!(plt2,  eQTSD[1:nouts], label="eQTSD", yscale=:log10 )

    # RP = zeros(n,n)
    # for k = 1:n
    #     v = rand(Multinomial(Int(M), Vector(P[k,:])))
    #     RP[k,:] = v ./ M
    #     # println(RP[k,:])
    # end
    #
    # RA = I - RP
    # invRA = zeros(n,n)
    # invRA[2:n-1,2:n-1] = inv(Matrix(RA[2:n-1,2:n-1]))
    # for m = 1:1000
    #     invRA[2:n-1,2:n-1] = RP[2:n-1,2:n-1]*invRA[2:n-1,2:n-1]+I
    # end
    # invRA[1,1] = 1
    # invRA[n,n] = 1
    #
    #
    # rhsRT = ones(n)
    # rhsRT[1] = 0
    # rhsRT[n] = 0
    #
    # RT = zeros(n)
    # RT[2:n-1] = RA[2:n-1,2:n-1]\rhsRT[2:n-1]
    #
    # rhsRQ = zeros(n)
    # rhsRQ[2:n-1] = Vector(RP[2:n-1,n])
    # rhsRQ[n] = 1
    #
    # RQ = zeros(n)
    # RQ[n] = 1
    # RQ[2:n-1] = RA[2:n-1,2:n-1]\rhsRQ[2:n-1]

    # plt1 = plot!(plt1, x[2:n-1], RT[2:n-1], seriestype=:scatter, ms=4 )
    # plt2 = plot!(plt2, x[2:n-1], RQ[2:n-1], seriestype=:scatter, ms=4 )


    # efP = eigen(Matrix(P[2:n-1,2:n-1]))

    # plt4 = plot(efP.values)

    # Wex = efP.vectors[:,1]

    # plt3 = plot!(plt3, x[2:n-1], sqrt(n).*Wex)


    # # println(efA.values[1]," ",dot(Wex,A[2:n-1,2:n-1]*Wex))
    # # println(efP.values)

    # ABig = zeros(n-1,n-1)
    # ABig[1,1] = 1
    # ABig[2:n-1,1] = alpha .* rT[2:n-1]
    # ABig[2:n-1,2:n-1] = (1-alpha)*I + alpha .* P[2:n-1,2:n-1]
    #
    # ABigEigs = eigen(ABig)
    #
    #
    #
    #
    # TBig = rand(n-1,2)
    # TBig[1,1] = 1
    # TBig[1,2] = 0
    # # TBig[2:n-1,1] = rT[2:n-1]
    #
    # C1 = TBig'*ABig*TBig
    # C0 = TBig'*TBig
    # CEigs = eigen(C1,C0)
    #
    # TSI = TBig*CEigs.vectors[:,2]
    # TSI = TSI./TSI[1]
    # TRich = TBig[:,1]
    # TRich = TRich./TRich[1]
    #
    # eTRich = zeros(nitr)
    # eTSI = zeros(nitr)
    #
    # eTRich[1] = norm(TRich[2:n-1]-T[2:n-1])
    # eTSI[1] = norm(TSI[2:n-1]-T[2:n-1])
    #
    # for m = 1:(nitr-1)
    #     TBig = ABig*TBig
    #     TBig[:,1] = TBig[:,1]./TBig[1,1]
    #     TBig[:,2] = TBig[:,2]./norm(TBig[:,2])
    #
    #     u = TBig[2:n-1,1]
    #     res = zeros(n-2)
    #     res = rT[2:n-1] .- A[2:n-1,2:n-1]*u
    #     w = TBig[2:n-1,2]
    #
    #     c = dot(w,res)
    #     z = dot(w,A[2:n-1,2:n-1]*w)
    #
    #     # println(c," ",z)
    #
    #     u = u .+ (c/z).*w
    #
    #     # C1 = TBig'*ABig*TBig
    #     # C0 = TBig'*TBig
    #     # CEigs = eigen(C1,C0)
    #     #
    #     # TSI = TBig*CEigs.vectors[:,2]
    #     # TSI = TSI./TSI[1]
    #     TRich = TBig[:,1]
    #     TRich = TRich./TRich[1]
    #
    #     eTRich[m+1] = norm(TRich[2:n-1]-T[2:n-1])
    #     # eTSI[m+1] = norm(TSI[2:n-1]-T[2:n-1])
    #     eTSI[m+1] = norm( u - T[2:n-1] )
    #
    #     # println(eTRich[m+1]," ", eTSI[m+1])
    #
    # end
    #
    #
    # plt1 = plot!(plt1, (0:nitr-1), eTRich, lc=Int(2^(d-1)), lw=4, legend=:none)
    # plt1 = plot!(plt1, (0:nitr-1), eTSI, lc=Int(2^(d-1)), ls=:dot, lw=4, legend=:none)
    #
    # # println()
    #
    # ABig = zeros(n-1,n-1)
    # ABig[1,1] = 1
    # ABig[2:n-1,1] = alpha .* rQ[2:n-1]
    # ABig[2:n-1,2:n-1] = (1-alpha)*I + alpha .* P[2:n-1,2:n-1]
    #
    # # ABigEigs = eigen(ABig)
    #
    # # println(ABigEigs.values)
    # # println()
    # # println(ABigEigs.vectors[:,n-1])
    # #
    # # u = ABigEigs.vectors[2:n-1,n-1] ./ ABigEigs.vectors[1,n-1]
    # # println(norm(u .- Q[2:n-1]))
    # # println()
    #
    # QBig = rand(n-1,2)
    # QBig[1,1] = 1
    # QBig[1,2] = 0
    # # QBig[2:n-1,1] = rQ[2:n-1]
    #
    # C1 = QBig'*ABig*QBig
    # C0 = QBig'*QBig
    # CEigs = eigen(C1,C0)
    #
    # QSI = QBig*CEigs.vectors[:,2]
    # QSI = QSI./QSI[1]
    # QRich = QBig[:,1]
    # QRich = QRich./QRich[1]
    #
    # eQRich = zeros(nitr)
    # eQSI = zeros(nitr)
    #
    # eQRich[1] = norm(QRich[2:n-1]-Q[2:n-1])
    # eQSI[1] = norm(QSI[2:n-1]-Q[2:n-1])
    #
    # for m = 1:(nitr-1)
    #     QBig = ABig*QBig
    #     QBig[:,1] = QBig[:,1]./QBig[1,1]
    #     QBig[:,2] = QBig[:,2]./norm(QBig[:,2])
    #
    #     u = QBig[2:n-1,1]
    #     res = rQ[2:n-1] .- A[2:n-1,2:n-1]*u
    #     w = QBig[2:n-1,2]
    #
    #     c = dot(w,res)
    #     z = dot(w,A[2:n-1,2:n-1]*w)
    #
    #     # println(c," ",z)
    #
    #     u = u .+ (c/z).*w
    #
    #     # C1 = QBig'*ABig*QBig
    #     # C0 = QBig'*QBig
    #     # CEigs = eigen(C1,C0)
    #     #
    #     # QSI = QBig*CEigs.vectors[:,2]
    #     # QSI = QSI./QSI[1]
    #     QRich = QBig[:,1]
    #     QRich = QRich./QRich[1]
    #
    #     eQRich[m+1] = norm(QRich[2:n-1]-Q[2:n-1])
    #     # eQSI[m+1] = norm(QSI[2:n-1]-Q[2:n-1])
    #     eQSI[m+1] = norm( u - Q[2:n-1] )
    #     # println(eQSI[m+1])
    #
    #
    # end
    #
    #
    # plt2 = plot!(plt2, (0:nitr-1), eQRich, lc=Int(2^(d-1)), lw=4,background_color_legend = RGBA(1.0,1.0,1.0,0.7),
    # label="n=$(n)", legendfont=font(14), legend=:topright)
    # plt2 = plot!(plt2, (0:nitr-1), eQSI, lc=Int(2^(d-1)),ls=:dot, lw=4,  label=:none)

    # plt3 = plot!(plt3, (-1:-1)',(-1:-1)', lc=Int(2^(d-1)), lw=4,  label="n=$(n), Richardson",
    # legendfont=font(14), legend=:topleft, margin=:none)
    # plt3 = plot!(plt3, (-1:-1)',(-1:-1)', lc=Int(2^(d-1)),ls=:dot, lw=4, label="n=$(n), Subspace Iteration",
    # legendfont=font(14), legend=:topleft, margin=:none)

    # plt1 = plot!(plt1, x[2:n-1], T[2:n-1], legend=:none, lw=4)
    # plt2 = plot!(plt2, x[2:n-1], Q[2:n-1], legend=:none, lw=4)
    # plt3 = plot!(plt3, x[2:n-1], ravarT[2:n-1], label="n=$(n)", lw=4, legendfont=font(14))
    # plt4 = plot!(plt4, x[2:n-1], ravarQ[2:n-1], label="n=$(n)", lw=4, legendfont=font(14))

end

plot( plt2 )

# plt = plot(plt1, plt2, size=(1200,400), layout=(1,2), guidfont=font(14), margin=8*Plots.mm,  xlabel="iteration count",
# xlabelfont=font(14), ylabelfont=font(14), ytickfonts=font(14), xtickfonts=font(14))
#
# savefig(plt, "iterconvergence.pdf")


# pT = plot(plt1,plt3, layout=(1,2), size=(1200,400), guidefont=font(14), margin=8*Plots.mm,
# xlabelfont=font(14), ytickfonts=font(14), xtickfonts=font(14), plot_title="Mean First Passage Time", xlabel="(i-0.5)/n")
# savefig(pT, "avarTplot.pdf")
#
# pQ = plot(plt2,plt4, layout=(1,2), size=(1200,400), guidefont=font(14), margin=8*Plots.mm,
# xlabelfont=font(14), ytickfonts=font(14), xtickfonts=font(14), plot_title="Committor", xlabel="(i-0.5)/n")
# savefig(pQ, "avarQplot.pdf")
