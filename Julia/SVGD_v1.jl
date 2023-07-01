using LinearAlgebra
using SparseArrays
using Random
using Plots
using StatsBase

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

n = 1000
d = 10
# λ = (1:n)
# λ = n*ones(n)
# λ[1] = 10
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,d)

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



q = 5
h = 0.01

x = randn(d)

y = zeros(n)

z = zeros(d)

B = zeros(n+d,n+d)
B[1:d,d+1:n+d] = A'
B[d+1:n+d,1:d] = A

# ef = eigen(Symmetric(A'*A), 1:d)
ef = eigen(Symmetric(B), 1:d+1)
@show ef.values
@show (1 .- h*ef.values)

ef = eigen(Symmetric(B), n:n+d)
@show ef.values
@show (1 .- h*ef.values)


# #
# ef = eigen(Symmetric(I-h*A'*A), (n-k-1):n)   #k smallest eigenvalues/vectors
# @show (1-ef.values[1])/(1-ef.values[end])



eigs = zeros(q)

println("q = 0")
println("  eig = $lam")


for s=1:q

    global x, y, z

    xnrm2 = norm(x)
    xnrm2 = xnrm2*xnrm2

    ynrm2 = norm(y)
    ynrm2 = ynrm2*ynrm2

    nrm2 = xnrm2+ynrm2

    # x = x - h*(A'*A*x - xnrm2*x)

    x = x + h*(z-nrm2*x)

    q = h*(A*x-nrm2*y)

    y = y + q

    z = z + A'*q

    lam = norm(A*x)/norm(x)
    lam = lam

    eigs[s] = lam

    println("q = $s")
    println("  eig = $lam")
end

plot([1:q], eigs)
