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

# n = 10000
# λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
# b = rand(n)

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
x0 = copy(b)

d = 3
m = 1000

x = copy(x0)
y = A*x

B = zeros(Float64,n,d-1)
AB = zeros(Float64,n,d-1)


println("k = 0")
println("  norm(r) = $(norm(b-y))")
println("  norm(r)/norm(b) = $(norm(b-y)/norm(b))")

for k=1:d-1
        global x, y, B, AB

        B[:,k] = copy(x)./norm(x)
        AB[:,k] = copy(y)./norm(x)

        z = AB[:,1:k]\b

        println(z)

        q = B[:,1:k]*z - x

        println(norm(B[:,1:k]*z))

        println(norm(x))

        println(norm(q))

        # pivotal_compress(q,m)

        x = x + q
        y = y + A*q

        println("k = $k")
        println("  norm(r) = $(norm(b-y))")
        # println("  norm(b-Ax)/norm(b-Axold) = $(norm(r)/norm(rold))")
        println("  norm(r)/norm(b) = $(norm(b-y)/norm(b))")
        println("  cond(B,2) = $(cond(B[:,1:k]))")

end
