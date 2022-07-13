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
x0 = zeros(n)

d = 10
p = 2*d
m = 100


x = copy(x0)
r0 = b - A*x0

B = zeros(Float64,n,d)

AB = zeros(Float64,n,d)

S = randn(p,n)
SAB = zeros(Float64,p,d)

r = copy(r0)
x = copy(r0)
println("k = 0")
println("  norm(r) = $(norm(r))")
println("  norm(r)/norm(b) = $(norm(r)/norm(b))")

for k=1:d
        global x, r, B, AB, SAB

        # Sr = S*r
        B[:,k] = copy(r)./norm(r,1)

        # s = copy(r)

        # pivotal_compress(s,m)

        AB[:,k] = A*B[:,k]
        z = (S*AB[:,1:k])\(S*(b-A*x))

        # SAB[:,k] = S*AB[:,k]
        # z = SAB[:,1:k]\Sr

        # println(z)

        s = copy(B[:,k])

        # println(s)
        pivotal_compress(s,m)
        # println()
        # println(s)

        r = A*s

        x = x+B[:,1:k]*z

        println("k = $k")
        println("  norm(r) = $(norm(b.-A*x))")
        println("  norm(r)/norm(b) = $(norm(b.-A*x)/norm(b))")

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

        println("  cond(B,2) = $(cond(B[:,1:k]))")
end

# println("final")
# println(norm(b.-A*x))
# println(norm(b.-A*x)/norm(b))
