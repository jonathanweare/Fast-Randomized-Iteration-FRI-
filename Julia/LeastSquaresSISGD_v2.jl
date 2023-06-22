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
# λ = (1:n)
λ = zeros(n)
# λ = n*ones(n)
# λ[1] = 10
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

q = 100000
h = 0.0004
k = 50
d = 10

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

# ef = eigen(Symmetric(A'*A), n:n)
# @show (1.0 .- h.*(ef.values))
# #
# ef = eigen(Symmetric(I-h*A'*A), (n-k-1):n)   #k smallest eigenvalues/vectors
# @show (1-ef.values[1])/(1-ef.values[end])

println("q = 0")
println("  norm(r) = $(norm(b-A*x))")

r = b - A*x

r_nrm = zeros(q)
rc_nrm = zeros(q)

b_nrm = norm(b)


for s=1:q

    global x, r, Y, Yave

    smpl = sample((1:n), d, replace=false, ordered=true)
    Sr = b[smpl] - A[smpl,:]*x

    Q, R = qr(hcat(Y,A'[:,smpl]*Sr))

    B = Matrix(Q)

    Y = B[:,1:k]

    SAB = A[smpl,:]*B
    c = SAB\Sr

    x = x .+ B*c

    Y = Y - h.*(A'[:,smpl]*SAB[:,1:k])

    r_nrm[s] = norm(b-A*x)/b_nrm

    println("q = $s")
    println("  norm(b-Ax) = $(r_nrm[s])")
    println("  norm(b-Ax) = $(norm(b .- A*x)/b_nrm)")
end

plot([1:q], log.(r_nrm))
