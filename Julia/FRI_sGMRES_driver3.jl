using LinearAlgebra
using SparseArrays
using Random

include("compress.jl")

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
A = triu(rand(n,n),1) + diagm(λ)
b = rand(n)

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

d = 2
p = 2*d
m = 100

Random.seed!(1)

r0 = b.-A*x0
x = copy(x0)
r = copy(r0)

S = randn(p,n)

B = zeros(Float64,n,d-1)
AB = zeros(Float64,n,d-1)
SAB = zeros(Float64,p,d-1)

SA = S*A
Sr0 = S*r0

println("k = 0")
println("  norm(r) = $(norm(r))")
println("  norm(r)/norm(b) = $(norm(r)/norm(b))")

for j = 1:nsmpl

for k=2:d
        global x, r, B, AB, SAB

        s = copy(r)./norm(r,1)

        B[:,k-1] = copy(s)

        pivotal_compress(s,m)

        # println("pass2")
        r = A*s

        AB[:,k-1] = r

        SAB[:,k-1] = SA*B[:,k-1]
        z = SAB[:,1:k-1]\Sr0

        # z = AB[:,1:k-1]\r0

        x = x0 + B[:,1:k-1]*z

        println("k = $k")
        println("  norm(r) = $(norm(r0-A*(x-x0)))")
        println("  norm(r)/norm(b) = $(norm(b-A*x)/norm(b))")
        println("  cond(B,2) = $(cond(B[:,1:k-1]))")

end

end
