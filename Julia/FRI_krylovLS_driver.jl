using LinearAlgebra
using SparseArrays

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


n = 100
d = 100

m = 100

ee = ones(n-1)
dd = range(1,length=n)
dd = dd./n

A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)

b = randn(n)
b = b./norm(b)

B = Array{Float64,2}(undef,n,d+1)
# S = randn(2*d,n)
S = I

SAB = Array{Float64,2}(undef,n,d)

residual = user_sparse_matvec(b)
residual = residual.-b

println(norm(residual))

x = copy(residual)
x = x./norm(x,1)

B[:,1] = copy(x)

for k=2:d+1
    global x

    pivotal_compress(x,m)
    # x = user_sparse_matvec(x)

    x = A*x

    SAB[:,k-1] = S*x

    x = x./norm(x,1)

    B[:,k] = copy(x)
end

Sr = S*residual

y = SAB\Sr

x = b.-B[:,1:d]*y

residual = user_sparse_matvec(x)
residual = residual.-b

println(norm(residual))
