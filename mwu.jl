"""
A Julia code for the MWU algorithm in paper:
On distributional discrepancy for experimental design with general assignment probabilities,
by Anup B. Rao, and Peng Zhang
"""

# Julia packages
using LinearAlgebra, Random

"""
The MWU algroithm 
Input: 
V: a matrix whose columns has norm at most 1
z0: the expected value of the returned assignment vector (aka, (z0+1)/2 is the vector of assignment probabilities)
oracle: an oracle function
ϵ: accuracy parameter
T: # of z_t's generated: z_1, ..., z_T
N: # of sampled used to estimate covariance matrix for each z_t

Output:
a random assignment vector z with expected value z0 and small ||cov(Vz)||
"""

function mwu(V, z0, oracle::Function, ϵ, T, N)
    d, _ = size(V)
    αs = []
    Ws = []
    
    W = eye(d)
    S = zeros(d,d)

    for t in 1:T
        
        # convert W to a diagonal matrix for speed-up
        W /= tr(W)
        F = svd(W)
        V_rotated = (F.V)' * V
        w_rotated = F.S
        V_square = V_rotated.^2

        # estimate cov(Bz_t)
        M = zeros(d,d)
        for _ in 1:N
            z = oracle(V_rotated, V_square, w_rotated, z0, ϵ)
            bv = V * (z - z0)
            M += bv * bv'
        end
        M /= N

        α = ϵ / (6*opnorm(M))
        S = S + M * α
        W = exp(S)
        push!(Ws, W)
        push!(αs, α)
    end
    
    # sample z
    ps = αs / sum(αs)
    cs = cumsum(ps)
    r = rand()
    i = 1
    while i < length(ps) && cs[i] < r
        i += 1
    end
    
    return oracle(V, Ws[i], z0, ϵ)
end

"""
Oracle for tall matrices (d >= n)
Input: 
V: a matrix whose columns has norm at most 1
V_square: the matrix obtained by squaring each entry of V
w: weight matrix's diagonal (aka, W = diag(w))
z0: the expected value of the returned assignment vector (aka, (z0+1)/2 is the vector of assignment probabilities)
ϵ: accuracy parameter

Output: 
a random assignment vector z with expected value z0 and small <cov(Vz), W>
"""

function min_trace_oracle(V, W, z0, ϵ)
    # convert W to a diagonal matrix
    W /= tr(W)
    F = svd(W)
    V_rotated = (F.V)' * V
    w_rotated = F.S
    V_square = V_rotated.^2

    return min_trace_oracle(V_rotated, V_square, w_rotated, z0, ϵ)
end

function min_trace_oracle(V, V_square, w, z0, ϵ) 
    d,n = size(V)
    
    z = copy(z0)
    alive = findall(abs.(z) .< 1.0) # alive entries
    
    while ~isempty(alive)

        row_nrms = [sum(V_square[i, alive]) for i in 1:d]
        row_ind_sorted = sortperm(row_nrms, rev=true) # decreasing order

        last_ind_of_big_rows = convert(Int64, ceil((ϵ/(1+ϵ)) * length(alive))) - 1
        big_rows = row_ind_sorted[1:last_ind_of_big_rows]
        small_rows = setdiff(1:d, big_rows)

        Vtb = V[big_rows, alive]
        Vtl = V[small_rows, alive]
        wt = w[small_rows]

        Dt = Diagonal([dot(wt, V_square[small_rows,i]) for i in alive])
        Mt = (1+ϵ)*Dt - Vtl'*Diagonal(wt)*Vtl

        Ut = nullspace(Vtb)
        A = Ut'*Mt*Ut
        
        F = eigen(A)
        zz = F.vectors[:,end]
        zz = real.(zz)
        y = Ut * zz
        @assert (y'*Mt*y)[1] >= 0
        
        z[alive] = update_one_step(z[alive], y)
        
        alive = findall(abs.(z) .< 1.0)
        @assert maximum(abs.(z)) <= 1.0 + 1e-10
    end
    
    return z
end


"""
function; update for the random walk
"""

function update_one_step(z, y)
    t1 = (1 .- z) ./ y
    t2 = -(1 .+ z) ./ y
    α_p = minimum(max.(t1,t2))
    α_m = -maximum(min.(t1,t2))
    
    if rand() <= α_m / (α_m+α_p)
        z += α_p * y
    else
        z -= α_m * y
    end

    return z
end

# some other functions
eye(n) = Matrix{Float64}(I, n, n)


# test example
function test()
    d,n = 10,10
    V = randn(d,n)
    max_nrms = sqrt(maximum(sum(V.^2, dims=1)))
    V /= max_nrms
    z0 = zeros(n)

    return mwu(V, z0, min_trace_oracle, 0.2, 100, 50)
end