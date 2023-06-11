using LinearAlgebra
using SparseArrays
using Random
using Plots

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
# λ = @. 10 + (1:n)
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

q = 10000
h = 0.0004
k = 50

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

# ef = eigen(Symmetric(A2), n:n)
# @show (1.0 .- h.*(ef.values))
#
# ef = eigen(Symmetric(A2), 1:k+1)   #k smallest eigenvalues/vectors
# @show (1.0 .- h.*(ef.values))

println("q = 0")
println("  norm(r) = $(norm(b-A*x))")

r = b - A*x

r_nrm = zeros(q)
rc_nrm = zeros(q)

for s=1:q

    global x, r, Y

    Q, R = qr(hcat(Y,A'*r))

    B = Matrix(Q)

    Y = B[:,1:k]

    AB = A*B
    c = AB\r

    x = x .+ B*c

    r = r .- AB*c

    Y = Y - h.*(A'*AB[:,1:k])

    r_nrm[s] = norm(r)

    println("q = $s")
    println("  norm(b-Ax) = $(norm(r))")
    println("  norm(b-Ax) = $(norm(b .- A*x))")
end

plot([1:q], log.(r_nrm))
