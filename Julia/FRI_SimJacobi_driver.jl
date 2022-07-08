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

n = 1000
λ = @. 10 + (1:n)
A = triu(rand(n,n),1) + diagm(λ)
b = rand(n)

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

d = 1000
p = 500

Random.seed!(1)




x = zeros(Float64,n)
r = copy(b)

println(norm(r))
println()

for k=1:d
    global x, r
    S = randn(n,p)
    B = A*S

    z = B\r
    # println(z)
    # println()

    x = x.+S*z

    r = copy(r.-B*z)

    println(k)
    println(norm(r))
    println(norm(b.-A*x))
end
