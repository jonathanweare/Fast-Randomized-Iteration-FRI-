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

n = 1000
λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,n) + diagm(λ)
A2 = A'*A
b = rand(n)

# N = 64
# n = N^3
# A = spdiagm(-1=>fill(-1.0, N - 1), 0=>fill(3.0, N), 1=>fill(-2.0, N - 1))
# # Id = speye(N)
# Id = copy(sparse(1.0*I, N, N));
# A = kron(A, Id) + kron(Id, A)
# A = kron(A, Id) + kron(Id, A)
# # x = ones(n)
# x = zeros(n)
# x[1] = 1
# b = A * x


xtrue = A2\b

d = 5


x = copy(b)

B = zeros(Float64,n,d)
AB = zeros(Float64,n,d)

for k=1:d
    global x, B, AB

    B[:,k] = x
    if k>1
        w = B[:,1:k-1]\x
        B[:,k] = B[:,k] - B[:,1:k-1]*w
    end
    nrm = norm(B[:,k])
    B[:,k] = B[:,k]./nrm

    x = copy(B[:,k])

    x = A2*x

    AB[:,k] = x

end


y = AB\b

x = B*y

println(norm(b))
println(norm(A*x.-b))
println(norm(A*x.-b)/norm(b))
