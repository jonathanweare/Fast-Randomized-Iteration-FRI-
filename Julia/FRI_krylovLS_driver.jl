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


n = 1000
d = 5
p = 2*d

m = 100
nsmpl = 100000

# ee = ones(n-1)
# dd = range(1,length=n)
# dd = dd./n
# A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)
# b = randn(n)
# b = b./norm(b)

Random.seed!(1)

λ = @. 10 + (1:n)
A = triu(rand(n,n),1) + diagm(λ)
b = rand(n)

xtrue = A\b

# S = I
S = randn(p,n)
SAB = zeros(Float64,p,d)
Bave = zeros(Float64,n,d)

for s = 1:nsmpl
    global x, B, Bave, U

    x = copy(b)

    Sx = S*x

    B = zeros(Float64,n,d)

    U = zeros(Float64,p,d)

    for k=1:d

        if k == 1
            r = norm(Sx)
            B[:,1] = x./r
            U[:,1] = Sx./r
        elseif k>1
            B[:,k] = x - B[:,k-1]*(U[:,1:k-1]\Sx)
            U[:,k] = S*B[:,k]
            r = norm(U[:,k])
            B[:,k] = B[:,k]./r
            U[:,k] = U[:,k]./r
        end

        x = copy(B[:,k])

        pivotal_compress(x,m)
        # x = user_sparse_matvec(x)
        x = A*x

        Sx = S*x

        SAB[:,k] = SAB[:,k].+Sx
    end

    Bave = Bave.+B
end

SAB = SAB./nsmpl
Bave = Bave./nsmpl

Sr = S*b

# SAB2 = S*A*B[:,1:d]

y = SAB\Sr

# println(SAB2*y.-Sr)

x = Bave[:,1:d]*y

# println(S*A*x.-Sr)

# residual = A*x.-b

println(norm(A*x.-b))
println(norm(b))
println(norm(A*x.-b)/norm(b))
