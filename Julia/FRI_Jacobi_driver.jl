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

# N = 32
# n = N^3
# A = spdiagm(-1=>fill(-1.0, N - 1), 0=>fill(3.0, N), 1=>fill(-2.0, N - 1))
# Id = copy(sparse(1.0*I, N, N));
# A = kron(A, Id) + kron(Id, A)
# A = kron(A, Id) + kron(Id, A)
# x = ones(n)
# # x = zeros(n)
# # x[1] = 1
# b = A * x


xtrue = A\b

d = 10
m = 100
h = 0.01

Random.seed!(1)


x0 = zeros(Float64,n)
r0 = b - A*x0

x = copy(x0)
r = copy(r0)

xave = zeros(Float64,n)

println("k = 0")
println("  norm(r) = $(norm(b-A*x))")

for k=1:d
    global x, r, xave

    s = copy(r)

    pivotal_compress(s,m)

    rold = copy(r)

    r = r - h.*(A*s)
    x = x + h.*s

    # pivotal_compress(r,m)
    #
    # x = x + h.*(transpose(A)*r)

    println("k = $k")
    println("  norm(r) = $(norm(b-A*x))")
    println("  norm(b-Ax)/norm(b-Axold) = $(norm(r)/norm(rold))")
    println("  norm(r)/norm(r0) = $(norm(b-A*x)/norm(b-A*x0))")

    if k > d/2
        xave = xave + x.*(2/d)
        println()
        println("  norm(r) = $(norm(b-A*xave))")
        println("  norm(r)/norm(r0) = $(norm(b-A*xave)/norm(b-A*x0))")
    end
end
