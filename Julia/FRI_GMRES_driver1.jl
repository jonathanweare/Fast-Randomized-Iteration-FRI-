using LinearAlgebra
using SparseArrays
using Random

include("compress.jl")

Random.seed!(1)

##############  The example system ###########
n = 10000
λ = @. 10 + (1:n)
# A = randn(n,n) + diagm(λ)
b = randn(n)

A = I
##################################


d = 100  ####### number of iterations
m = 1000  ####### number of non-zero entries

xtrue = A\b  ####### exact solution
x0 = zeros(n)  ######## initial guess
r0 = b.-A*x0  ######## initial residual


######## misc. initialization stuff ##############
x = copy(x0)
r = copy(r0)
B = zeros(Float64,n,d)
AB = zeros(Float64,n,d)
#############################




println("k = 0")
println("  norm(r) = $(norm(r))")  ######## print the norm of the residual
println("  norm(r)/norm(b) = $(norm(r)/norm(b))")


for k=1:d
        global x, r, B, AB

        s = copy(r)./norm(r,1) ######## s is a normalized copy of the residual

        pivotal_compress(s,m) ####### compress s

        B[:,k] = copy(s)  ###### add s to our basis
        AB[:,k] = A*s   ##### update A*basis

        z = AB[:,1:k]\r  #### find linear combination of basis vectors
                         #### so that x plus that linear combination
                         #### minimizes the current residual

        # println(z)
        q = B[:,1:k]*z  #### assemble the linear combination
        Aq = AB[:,1:k]*z  #### assemble A times the linear combination

        x = x + q       ##### update x by adding the linear combination

        rold = copy(r)  #### keep track of the last residual
        r = r - Aq      #### update the current residual

        println("k = $k")
        println("  norm(r) = $(norm(r0-A*(x-x0)))")
        println("  norm(b-Ax)/norm(b-Axold) = $(norm(r)/norm(rold))")
        println("  norm(r)/norm(b) = $(norm(b-A*x)/norm(b))")
        println("  cond(B,2) = $(cond(B[:,1:k]))")

end
