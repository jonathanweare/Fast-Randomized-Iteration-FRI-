using LinearAlgebra
using SparseArrays
using Random
using Plots

Random.seed!(1)

plt1 = plot(ylabel="time", yscale=:log10, minorgrid=true, xlabel="(i-0.5)/n")
plt2 = plot(ylabel="probability", yscale=:log10, minorgrid=true, xlabel="(i-0.5)/n")
plt3 = plot(ylabel="(relative avar)/n^3", xlabel="(i-0.5)/n", minorgrid=true)
plt4 = plot(xlabel="(i-0.5)/n", minorgrid=true, ylabel="(relative avar)/n^3")

for m = 1:3

    global plt1, plt2, plt3, plt4

    n = 10*2^m

    p = zeros(n)
    sup = zeros(n-1)
    sub = zeros(n-1)
    dd = zeros(n)

    x = ((1:n).-0.5)./n


    p = exp.(n.*(cospi.(4 .*x).+1)./(4*pi))
    sup[1:n-1] = 1 ./ (exp.(n .* ( cospi.( 4 .* x[1:n-1] ) .- cospi.( 4 .* x[2:n] ) ) ./ (4*pi) ) .+ 1)
    # sup[1:n-1] = 1 ./ (p[1:n-1] ./ p[2:n] .+ 1)
    sub[1:n-1] = 1 ./ (1 .+ exp.(n .* ( cospi.( 4 .* x[2:n] ) .- cospi.( 4 .* x[1:n-1] ) ) ./ (4*pi) ))
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


    ravarT = zeros(n)
    ravarQ = zeros(n)
    for i = 2:n-1
        for k = 2:n-1
            l = k-1
            ET = T[l]-T[k]
            EQ = Q[l]-Q[k]
            ravarT[i] += invA[i,k]*invA[i,k]*P[k,l]*ET*ET
            ravarQ[i] += invA[i,k]*invA[i,k]*P[k,l]*EQ*EQ

            l = k+1
            ET = T[l]-T[k]
            EQ = Q[l]-Q[k]
            ravarT[i] += invA[i,k]*invA[i,k]*P[k,l]*ET*ET
            ravarQ[i] += invA[i,k]*invA[i,k]*P[k,l]*EQ*EQ
        end
        ravarT[i] /= T[i]*T[i]*n*n
        ravarQ[i] /= Q[i]*Q[i]*n*n
    end



    plt1 = plot!(plt1, x[2:n-1], T[2:n-1], legend=:none, lw=2)
    plt2 = plot!(plt2, x[2:n-1], Q[2:n-1], legend=:none, lw=2)
    plt3 = plot!(plt3, x[2:n-1], ravarT[2:n-1], label="n=$(n)", lw=2)
    plt4 = plot!(plt4, x[2:n-1], ravarQ[2:n-1], label="n=$(n)", lw=2)

end

pT = plot(plt1,plt3, layout=(1,2), plot_title="Mean First Passage Time")
savefig(pT, "avarTplot.pdf")

pQ = plot(plt2,plt4, layout=(1,2), plot_title="Committor")
savefig(pQ, "avarQplot.pdf")
