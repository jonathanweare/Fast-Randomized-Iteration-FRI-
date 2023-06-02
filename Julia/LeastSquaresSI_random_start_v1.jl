using LinearAlgebra
using SparseArrays
using Random

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
A2 = A'*A
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

q = 1
h = 0.0004
k = 50

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

ef = eigen(Symmetric(A2), n:n)
@show (1.0 .- h.*(ef.values))

ef = eigen(Symmetric(A2), 1:k+1)   #k smallest eigenvalues/vectors
@show (1.0 .- h.*(ef.values))

println("q = 0")
println("  norm(r) = $(norm(b-A*x))")

for s=1:q
    global x, Y

    r = b - A*x
    x = x + h.*(A'*r)

    Y = Y - h.*(A2*Y)

    Q, R = qr(hcat(Y,x))

    B = Matrix(Q)

    Y = B[:,1:k]

    AB = A*B
    c = AB\b

    rc = b - AB*c

    println("q = $s")
    println("  norm(b-Ax) = $(norm(r))")
    println("  norm(b-ABc) = $(norm(rc))")
    println("  norm(b-ABc)/norm(b-Ax) = $(norm(rc)/norm(r))")
end
