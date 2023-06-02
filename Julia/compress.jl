using DataStructures

function pivotal_sample(p::Array{Float64})
    i = 1
    j = 2
    k = 3
    d = length(p)
    a = p[i]
    b = p[j]
    Tol = 1e-12

    while k<=d
        if k<=d && (a<Tol || a>1-Tol)
            a = p[k]
            i = k
            k += 1
        end
        if k<=d && (b<Tol || b>1-Tol)
            b = p[k]
            j = k
            k += 1
        end
        u = rand()
        q = a + b
        if q>1 && q<2
            if u<(1-b)/(2-q)
                b = q - 1.0
                a = 1.0
            else
                a = q - 1.0
                b = 1.0
            end
        elseif q>0 && q<=1
            if u<b/q
                b = q
                a = 0
            else
                a = q
                b = 0
            end
        end

        if a<Tol
            a = 0
        elseif a>1-Tol
            a = 1.0
        end
        if b<Tol
            b = 0
        elseif b>1-Tol
            b = 1.0
        end

        p[i] = a
        p[j] = b
    end
end



function pivotal_compress(x::Array{Float64}, m::Int)

    Tol = 1e-12

    d = length(x)

    if m>=d
        return
    end

    p_prs = zeros(d)
    p_spl = abs.(x)
    pnorm = sum(p_spl)

    nnz = 0
    for i = 1:d
        if p_spl[i] <= Tol*pnorm
            x[i] = 0
            p_spl[i] = 0
        else
            nnz = nnz+1
        end
    end

    if nnz<m
        return
    end

    pnorm = sum(p_spl)

    plist = collect(enumerate(-p_spl))

    pheap = BinaryHeap(Base.By(last),plist)

    pmax = -(first(pheap))[2]
    k = 0

    while pmax>=pnorm/(m-k)
        p_spl[(first(pheap))[1]] = 0
        p_prs[(first(pheap))[1]] = pmax
        pnorm -= pmax
        pop!(pheap)
        pmax = -(first(pheap))[2]
        k+=1
    end

    s = (m-k)/sum(p_spl)
    p_spl = p_spl.*s

    pivotal_sample(p_spl)

    for i=1:d
        x[i] = (p_prs[i] + p_spl[i]/s)*sign(x[i])
    end

end
