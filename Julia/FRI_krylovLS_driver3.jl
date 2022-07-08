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

d = 10
m = 100

Random.seed!(1)

r0 = b.-A*x0
x = copy(x0)
r = copy(r0)


B = zeros(Float64,n,d-1)

AB = zeros(Float64,n,d-1)

println("k = 0")
println("  norm(r) = $(norm(r))")
println("  norm(r)/norm(b) = $(norm(r)/norm(b))")

for k=2:d
        global x, r, B, AB

        s = copy(r)./norm(r,1)

        # B[:,k-1] = copy(s)

        # println("pass1")

        # println(s)
        # println()
        pivotal_compress(s,m)
        # println(s)

        B[:,k-1] = copy(s)

        # println("pass2")

        AB[:,k-1] = A*s
        z = AB[:,1:k-1]\r0

        # println(norm(AB[:,1:k-1]*z - r0))

        # println(z)

        x = x0 + B[:,1:k-1]*z

        r = r0.-AB[:,1:k-1]*z
        # r = AB[:,k-1]
        # r = r.-AB[:,1:k-1]*z

        println("k = $k")
        println("  norm(r) = $(norm(r0-A*(x-x0)))")
        println("  norm(r)/norm(b) = $(norm(b-A*x)/norm(b))")
        println("  cond(B,2) = $(cond(B[:,1:k-1]))")

end
