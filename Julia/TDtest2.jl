using LinearAlgebra
using SparseArrays
using Random
using Distributions
using Plots
# pyplot()

Random.seed!(2)

plt1 = plot(ylabel="time", yscale=:log10, minorgrid=true)
plt2 = plot(ylabel="probability", yscale=:log10, minorgrid=true)
plt3 = plot()
# plt3 = plot(ylabel="(relative avar)/n^3", minorgrid=true)
# plt4 = plot(minorgrid=true, ylabel="(relative avar)/n^3")

for d = 1:3

    nitr = 10000

    global plt1, plt2, plt3, plt4

    alpha = 1.0

    n = 10*2^(d-1)
    # n = 10

    M = 0.1*n^2

    p = zeros(n)
    sup = zeros(n-1)
    sub = zeros(n-1)
    dd = zeros(n)

    x = ((1:n).-0.5)./n

    p = exp.(n.*(cospi.(4 .*x).+1)./(4*pi))
    sup[1:n-1] = 1 ./ (3 .*(exp.(n .* ( cospi.( 4 .* x[1:n-1] ) .- cospi.( 4 .* x[2:n] ) ) ./ (4*pi) ) .+ 1))
    # sup[1:n-1] = 1 ./ (p[1:n-1] ./ p[2:n] .+ 1)
    sub[1:n-1] = 1 ./ (3 .*(1 .+ exp.(n .* ( cospi.( 4 .* x[2:n] ) .- cospi.( 4 .* x[1:n-1] ) ) ./ (4*pi) )))
    # sub[1:n-1] = 1 ./ (1 .+ p[2:n] ./ p[1:n-1])
    dd[1] = 1 - sup[1]
    dd[2:n-1] = 1 .- sup[2:n-1] .- sub[1:n-2]
    dd[n] = 1 - sub[n-1]
    P = spdiagm(-1=>sub,0=>dd,1=>sup)
    A = I - P
    invA = zeros(n,n)
    invA[2:n-1,2:n-1] = inv(Matrix(A[2:n-1,2:n-1]))
    for m = 1:1000
        invA[2:n-1,2:n-1] = P[2:n-1,2:n-1]*invA[2:n-1,2:n-1]+I
    end
    invA[1,1] = 1
    invA[n,n] = 1


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

    ABig = zeros(n-1,n-1)
    ABig[1,1] = 1
    ABig[2:n-1,1] = alpha .* rT[2:n-1]
    ABig[2:n-1,2:n-1] = (1-alpha)*I + alpha .* P[2:n-1,2:n-1]

    ABigEigs = eigen(ABig)

    TBig = rand(n-1,2)
    TBig[1,2] = 0
    # TBig[2:n-1,1] = rT[2:n-1]

    C1 = TBig'*ABig*TBig
    C0 = TBig'*TBig
    CEigs = eigen(C1,C0)

    TSI = TBig*CEigs.vectors[:,2]
    TSI = TSI./TSI[1]
    TRich = TBig[:,1]
    TRich = TRich./TRich[1]

    eTRich = zeros(nitr)
    eTSI = zeros(nitr)

    eTRich[1] = norm(TRich[2:n-1]-T[2:n-1])
    eTSI[1] = norm(TSI[2:n-1]-T[2:n-1])

    for m = 1:(nitr-1)
        TBig = ABig*TBig
        TBig[:,1] = TBig[:,1]/TBig[1,1]
        TBig[:,2] = TBig[:,2]./norm(TBig[:,2])

        C1 = TBig'*ABig*TBig
        C0 = TBig'*TBig
        CEigs = eigen(C1,C0)

        TSI = TBig*CEigs.vectors[:,2]
        TSI = TSI./TSI[1]
        TRich = TBig[:,1]
        TRich = TRich./TRich[1]

        eTRich[m+1] = norm(TRich[2:n-1]-T[2:n-1])
        eTSI[m+1] = norm(TSI[2:n-1]-T[2:n-1])

    end


    plt1 = plot!(plt1, (0:nitr-1), eTRich, lw=3)
    plt1 = plot!(plt1, (0:nitr-1), eTSI, lw=3)

    # println()

    ABig = zeros(n-1,n-1)
    ABig[1,1] = 1
    ABig[2:n-1,1] = alpha .* rQ[2:n-1]
    ABig[2:n-1,2:n-1] = (1-alpha)*I + alpha .* P[2:n-1,2:n-1]

    ABigEigs = eigen(ABig)

    # println(ABigEigs.values)
    # println()
    # println(ABigEigs.vectors[:,n-1])
    #
    # u = ABigEigs.vectors[2:n-1,n-1] ./ ABigEigs.vectors[1,n-1]
    # println(norm(u .- Q[2:n-1]))
    # println()

    QBig = rand(n-1,2)
    QBig[1,2] = 0
    # QBig[2:n-1,1] = rQ[2:n-1]

    C1 = QBig'*ABig*QBig
    C0 = QBig'*QBig
    CEigs = eigen(C1,C0)

    QSI = QBig*CEigs.vectors[:,2]
    QSI = QSI./QSI[1]
    QRich = QBig[:,1]
    QRich = QRich./QRich[1]

    eQRich = zeros(nitr)
    eQSI = zeros(nitr)

    eQRich[1] = norm(QRich[2:n-1]-Q[2:n-1])
    eQSI[1] = norm(QSI[2:n-1]-Q[2:n-1])

    for m = 1:(nitr-1)
        QBig = ABig*QBig
        QBig[:,1] = QBig[:,1]/QBig[1,1]
        QBig[:,2] = QBig[:,2]./norm(QBig[:,2])

        C1 = QBig'*ABig*QBig
        C0 = QBig'*QBig
        CEigs = eigen(C1,C0)

        QSI = QBig*CEigs.vectors[:,2]
        QSI = QSI./QSI[1]
        QRich = QBig[:,1]
        QRich = QRich./QRich[1]

        eQRich[m+1] = norm(QRich[2:n-1]-Q[2:n-1])
        eQSI[m+1] = norm(QSI[2:n-1]-Q[2:n-1])

    end


    plt2 = plot!(plt2, (0:nitr-1), eQRich, lw=4)
    plt2 = plot!(plt2, (0:nitr-1), eQSI, lw=4)

    # plt1 = plot!(plt1, x[2:n-1], T[2:n-1], legend=:none, lw=4)
    # plt2 = plot!(plt2, x[2:n-1], Q[2:n-1], legend=:none, lw=4)
    # plt3 = plot!(plt3, x[2:n-1], ravarT[2:n-1], label="n=$(n)", lw=4, legendfont=font(14))
    # plt4 = plot!(plt4, x[2:n-1], ravarQ[2:n-1], label="n=$(n)", lw=4, legendfont=font(14))

end

plot(plt1)

# pT = plot(plt1,plt3, layout=(1,2), size=(1200,400), guidefont=font(14), margin=8*Plots.mm,
# xlabelfont=font(14), ytickfonts=font(14), xtickfonts=font(14), plot_title="Mean First Passage Time", xlabel="(i-0.5)/n")
# savefig(pT, "avarTplot.pdf")
#
# pQ = plot(plt2,plt4, layout=(1,2), size=(1200,400), guidefont=font(14), margin=8*Plots.mm,
# xlabelfont=font(14), ytickfonts=font(14), xtickfonts=font(14), plot_title="Committor", xlabel="(i-0.5)/n")
# savefig(pQ, "avarQplot.pdf")
