using LinearAlgebra
using SparseArrays
using Random

include("compress.jl")

Random.seed!(1)

# function user_sparse_matvec(x::Array{Float64})
#
#     n = length(x)
#
#     ee = ones(n-1)
#     dd = range(1,length=n)
#     dd = dd./n
#
#     A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)
#
#     return A*x
#
# end


# n = 1000
# ee = ones(n-1)
# dd = range(1,length=n)
# dd = dd./n
# A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)
# b = randn(n)
# b = b./norm(b)

n = 10000
λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,n) + diagm(λ)
b = randn(n)

# N = 16
# n = N^3
# A = spdiagm(-1=>fill(-1.0, N - 1), 0=>fill(3.0, N), 1=>fill(-2.0, N - 1))
# # Id = speye(N)
# Id = copy(sparse(1.0*I, N, N));
# A = kron(A, Id) + kron(Id, A)
# A = kron(A, Id) + kron(Id, A)
# x = ones(n)
# # x = zeros(n)
# # x[1] = 1
# b = A * x


xtrue = A\b


d = 100
p = 2*d
m = 1000
nsmpl = 1

h = 0.1


x0 = zeros(n)
r0 = b - A*x0



S = randn(p,n)
SAB = zeros(Float64,p,d)


Bave = zeros(Float64,n,d)

ABave = zeros(Float64,n,d)

# println("k = 0")
println("  norm(r) = $(norm(r0))")
println("  norm(r)/norm(b) = $(norm(r0)/norm(b))")

for j = 1:nsmpl
    global x, r, B, AB, SAB, Bave, ABave

    B = zeros(Float64,n,d)

    AB = zeros(Float64,n,d)

    r = copy(r0)
    x = copy(x0)

    for k=1:d

        B[:,k] = copy(r)./norm(r,1)

        AB[:,k] = A*B[:,k]
        z = AB[:,1:k]\r
        # z = AB[:,1:k]\r0
        # z = (S*AB[:,1:k])\(S*r)
        # z = (S*AB[:,1:k])\(S*r0)

        # SAB[:,k] = S*AB[:,k]
        # z = SAB[:,1:k]\Sr

        # println(z)

        s = B[:,1:k]*z
        # s = x0 - x + B[:,1:k]*z

        pivotal_compress(s,m)

        rold = r

        x = x + h.*s

        r = r - h.*(A*s)

        println("k = $k")
        println("  norm(r) = $(norm(r))")
        println("  norm(b-Ax) = $(norm(b.-A*x))")
        println("  norm(b-Ax)/norm(b-Axold) = $(norm(r)/norm(rold))")
        println("  norm(b-Ax)/norm(b) = $(norm(b.-A*x)/norm(b))")

        Bave[:,k] = Bave[:,k]+B[:,k]
        ABave[:,k] = ABave[:,k]+AB[:,k]
    end
end

Bave = Bave./nsmpl
ABave = ABave./nsmpl

# z = (AB[:,1:k])\(r0)
zave = (S*ABave)\(S*r0)

y = x0+Bave*zave

println()
println("  norm(b-Ay) = $(norm(b.-A*y))")
println("  norm(b-Ay)/norm(b-Ax) = $(norm(b.-A*y)/norm(b-A*x))")

# if k>1
#     # w = (SB[:,1:k-1])\Sr
#     w = B[:,1:k-1]\B[:,k]
#     B[:,k] = B[:,k] - B[:,1:k-1]*w
#     AB[:,k] = AB[:,k] - AB[:,1:k-1]*w
# end
# SB[:,k] = S*B[:,k]
# nrm = norm(SB[:,k])
# nrm = norm(B[:,k],1)
# B[:,k] = B[:,k]./nrm
# AB[:,k] = AB[:,k]./nrm
# SB[:,k] = SB[:,k]./nrm

println("  cond(B,2) = $(cond(Bave))")

println()

# println("final")
# println(norm(b.-A*x))
# println(norm(b.-A*x)/norm(b))
