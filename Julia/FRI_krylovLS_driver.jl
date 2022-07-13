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

n = 10000
λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,n) + diagm(λ)
b = randn(n)

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


xtrue = A\b

d = 5
p = 2*d

m = 1000
nsmpl = 10



# S = I
S = randn(p,n)
SA = S*A
Bave = zeros(Float64,n,d)

for s = 1:nsmpl
    global x, B, SB, Bave

    x = copy(b)

    # Sx = S*x

    B = zeros(Float64,n,d)
    # SB = zeros(Float64,p,d)
    # SAB = zeros(Float64,p,d)

    for k=1:d

        B[:,k] = x
        # if k>1
        #     w = (SB[:,1:k-1])\Sx
        #     B[:,k] = x - B[:,1:k-1]*w
        # end
        # SB[:,k] = S*B[:,k]
        r = norm(S*B[:,k])
        B[:,k] = B[:,k]./r
        # SB[:,k] = SB[:,k]./r

        x = copy(B[:,k])

        pivotal_compress(x,m)
        # x = user_sparse_matvec(x)
        x = A*x

        # Sx = S*x
        #
        # SAB[:,k] = SAB[:,k].+Sx
    end

    Bave = Bave.+B
end

Bave = Bave./nsmpl

SAB = SA*Bave

Sr = S*b

# SAB2 = S*A*B[:,1:d]

y = SAB\Sr

# println(SAB2*y.-Sr)

x = Bave[:,1:d]*y

# println(S*A*x.-Sr)

# residual = A*x.-b

println(norm(b))
println(norm(A*x.-b))
println(norm(A*x.-b)/norm(b))
println()
println(norm(Sr))
println(norm(SAB*y.-Sr))
println(norm(SAB*y.-Sr)/norm(Sr))
