using LinearAlgebra
using SparseArrays
using Random
using Plots

Random.seed!(2)

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
# λ = zeros(n)
# λ = n*ones(n)
# λ[1] = 10
# A = triu(rand(n,n),1) + diagm(λ)
# A = diagm(λ)
A = randn(n,n) + diagm(λ)
# A = (A+A') ./2
# A = A'*A
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



q = 2000
h = 1.0/(10.0 + n)
k = 20

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

# ef = eigen(Symmetric(A), n:n)
# @show (1.0 .- h.*(ef.values))
#
# ef = eigen(Symmetric(A), 1:k+1)   #k smallest eigenvalues/vectors
# @show (1.0 .- h.*(ef.values))

println("q = 0")
println("  norm(r) = $(norm(b-A*x))")

r_nrm = zeros(q)
rc_nrm = zeros(q)
AB = zeros(n,k+1)
AB[:,1:k] = A*Y

b_nrm = norm(b)

for s=1:q
    global x, Y, AB

    r = b - A*x
    x = x + h.*r

    Y = Y - h.*AB[:,1:k]

    Q, R = qr(hcat(Y,x))

    B = Matrix(Q)

    Y = B[:,1:k]

    AB = A*B
    # c = (B'*AB)\(B'*b)
    c = AB\b

    # AB = A1*B
    # c = AB\b

    rc = b - AB*c

    r_nrm[s] = norm(r)/b_nrm
    rc_nrm[s] = norm(rc)/b_nrm

    println("q = $s")
    println("  norm(b-Ax) = $(r_nrm[s])")
    println("  norm(b-ABc) = $(rc_nrm[s])")
    println("  norm(b-ABc)/norm(b-Ax) = $(rc_nrm[s]/r_nrm[s])")
end

plot([1:q], log.(r_nrm))
plot!([1:q], log.(rc_nrm))
