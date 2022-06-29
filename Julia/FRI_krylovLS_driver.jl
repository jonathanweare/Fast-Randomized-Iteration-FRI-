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


n = 20
d = 20
p = 20

m = 20

ee = ones(n-1)
dd = range(1,length=n)
dd = dd./n

A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)

b = randn(n)
b = b./norm(b)

xtrue = A\b

S = I
# S = randn(p,n)
B = Array{Float64,2}(undef,n,d)
SAB = Array{Float64,2}(undef,p,d)

x = copy(b)

for k=1:d
    global x

    x = x./norm(x,1)

    B[:,k] = copy(x)

    # if k>1
    #     Q, R = qr(B[:,1:k])
    #     B[:,1:k] = Matrix(Q)
    #     x = B[:,k]
    # end

    pivotal_compress(x,m)
    # x = user_sparse_matvec(x)
    x = A*x

    SAB[:,k] = S*x
end

Sr = S*b

SAB2 = S*A*B[:,1:d]

y = SAB\Sr

println(SAB2*y.-Sr)

x = B[:,1:d]*y

println(S*A*x.-Sr)



residual = A*x.-b

println(norm(residual))
