using Clustering, Distributions, Hungarian, LinearAlgebra, Random, SpecialFunctions, StatsBase, Tables
include("TDA_Tools.jl")

# Function to catch cholesky factorization failures (isposdef doesn't seem to work)
function chol_trial(mtx)
    e = nothing
    try
        cholesky(mtx)
    catch e
    end
    return e
end

# Function to keep trying InverseWishart
function stubborn_IW(sz, mtx)
    try
        return rand(InverseWishart(sz, mtx))
    catch;
    end
end


# Reordering function for initial KMeans clustering (so clusters are named in order they appear in data)
# X is state data, K is total number of states
# Returns a dictionary that can be used to reorder states

function reorder(X, K)
    seen = []
    dict = Dict()

    for i in 1:K
        dict[i] = nothing
    end

    curr = 1
    idx = 1
    while idx <= length(X) && nothing in values(dict)
        if X[idx] in seen
            idx += 1
        else
            push!(seen, X[idx])
            dict[curr] = X[idx]
            idx += 1
            curr += 1
        end
    end

    if nothing in values(dict)
        remaining = [ _ for _ in 1:K ]
        seen = sort(seen, rev=true)

        for x in seen
            deleteat!(remaining, x)
        end
        remaining = sort(remaining, rev=true)

        while nothing in values(dict)
            dict[curr] = pop!(remaining)
            curr += 1
        end
    end

    return Dict(value => key for (key, value) in dict)
end

# function to initialize HMM (with KMeans)
function Init_HMM_w_Kmeans(T, pd_dim, PI_data, KMax, L, eps, gamma, nu0, rho1, rho2, overall_mu, overall_cov; comp=:lik)
    K, Z = PI_KMeans(T, pd_dim, PI_data, KMax)

    #reorder states in order of appearance in data
    dct = reorder(Z,K)
    Z = [ dct[z] for z in Z ]

    #Initialize Zcnts and Ys
    Zcnts = zeros(Int64, K)
    Ys = Dict(:sums => [ [zeros(size(PI_data[1][i])) for i in 1:pd_dim] for _ in 1:K], :sqrs => [ [zeros((size(PI_data[1][i])[1], size(PI_data[1][i])[1])) for i in 1:pd_dim] for _ in 1:K])

    for t in 1:T
        Zcnts[Z[t]] += 1
        Ys[:sums][Z[t]] .+= PI_data[t]
    end
    Means = Ys[:sums] ./ Zcnts

    for t in 1:T
        Ys[:sqrs][Z[t]] .+= [ PI_data[t][i]  * transpose(PI_data[t][i]) for i in 1:pd_dim]
    end
    Covs = [[Ys[:sqrs][k][i] / Zcnts[k] - Means[k][i] * transpose(Means[k][i]) for i in 1:pd_dim] for k in 1:K]

    # complete set of initial Zcnts, Means, Covs, and Ys
    for _ in (K+1):L
        push!(Zcnts, 0)

        tmp = []
        for i in 1:pd_dim
            sz = size(overall_cov[i])[1]
            mtx = (nu0-1)*overall_cov[i] 
            while chol_trial(mtx) != nothing #! isposdef(mtx)
                mtx += diagm([eps for j in 1:sz])
                if ! issymmetric(mtx)
                    mtx = (mtx + transpose(mtx))/2
                end
            end
            
            new_mtx = nothing
            while new_mtx == nothing
                new_mtx = stubborn_IW(nu0+sz, mtx)
            end
            while chol_trial(new_mtx) != nothing #! isposdef(mtx)
                new_mtx += diagm([eps for j in 1:sz])
                if ! issymmetric(new_mtx)
                    new_mtx = (new_mtx + transpose(new_mtx))/2
                end
            end

            push!(tmp, new_mtx)
        end
        push!(Covs, tmp)

        tmp = []
        for i in 1:pd_dim
            sz = size(overall_mu[i])
            new = rand(MvNormal(vec(overall_mu[i]), Covs[end][i]))
            push!(tmp, reshape(new, sz))
        end
        push!(Means, tmp)

        push!(Ys[:sums], [zeros(size(PI_data[1][i])) for i in 1:pd_dim])
        push!(Ys[:sqrs], [zeros((size(PI_data[1][i])[1], size(PI_data[1][i])[1])) for i in 1:pd_dim])
    end


    #initialize beta
    Betas = zeros(L)
    while 0.0 in Betas
        Betas = rand(Dirichlet(L,gamma/L))
    end

    #initialize Kappas 
    Kappas = [rand(Beta(rho1, rho2)) for i in 1:L]

    #initialize Ws
    W = zeros(Int64, T)
    N_mtx = zeros(Int64,(L,L))

    for t in 2:T
        i = Z[t-1]
        j = Z[t]
        if i == j
            W[t] = rand(Binomial(1,Kappas[j]))
        end
        if W[t] == 0
            N_mtx[i,j] += 1
        end
    end

    return K, Z, W, Zcnts, N_mtx, Betas, Kappas, Means, Covs, Ys
end

# Function to implement scaled backward message passing
function backward_messaging(L, T, pd_dim, X, Pi_bar_mtx, Kappas, Means, Covs; comp=:lik)
    G = zeros(T)
    m_hat = zeros(T,L)
    m_tilde = zeros(T,L)
    
    #initialize
    G[T] = L
    m_hat[T,:] .= 1/L
    m_tilde[T,:] .= 1
    
    #recurse
    for t in T-1:-1:1
        m_tilde[t,:] = [sum([(Pi_bar_mtx[k,j]*(1-Kappas[k]) + ==(j,k)*Kappas[k])*PI_comp[comp](pd_dim, X[t+1],Means[j],Covs[j])*m_hat[t+1,j] for j in 1:L]) for k in 1:L]
        G[t] = sum(m_tilde[t,:])
        m_hat[t,:] = m_tilde[t,:] / G[t]
    end
        
    #rescale after recursion
    return m_hat  
end

# Function to update states based on Backward Messages
function update_states(L, T, pd_dim, X, Pi_bar_mtx, Kappas, Means, Covs, b_msgs; comp=:lik) 
    #get number of dimensions and PI sizes
    sizes_tmp = [size(X[1][i]) for i in 1:pd_dim]
    sizes = []
    for i in 1:length(sizes_tmp)
        if length(sizes_tmp[i]) == 1
            push!(sizes, (sizes_tmp[i][1],1))
        else
            push!(sizes, sizes_tmp[i])
        end
    end
   
    
    #set up states, Z, W
    states = [ k for k in 1:L ]
    Z = zeros(Int64, T)
    W = zeros(Int64,T)

    #set up Y for sum and sum of squares
    Ys = Dict(:sums => [ [ zeros(sz) for sz in sizes ] for _ in 1:L ], :sqrs => [ [ zeros(sz[1]*sz[2], sz[1]*sz[2]) for sz in sizes ] for _ in 1:L ])
    
    # Update first state
    wghts = [ PI_comp[comp](pd_dim, X[1], Means[k], Covs[k])*b_msgs[1,k] for k in 1:L ]
    Z[1] = sample(states, Weights(wghts))
    for i in 1:pd_dim
        Ys[:sums][Z[1]][i] += X[1][i]
        Ys[:sqrs][Z[1]][i] += X[1][i] * transpose(X[1][i])
    end

    pairs = []
    for i in zip(states,zeros(Int64,L))
        push!(pairs,i)
    end
    for i in zip(states,ones(Int64,L))
        push!(pairs,i)
    end
    
    # Update all other states
    for t in 2:T
        wghts0 = [ Pi_bar_mtx[Z[t-1],k]*(1-Kappas[Z[t-1]])*PI_comp[comp](pd_dim, X[t], Means[k], Covs[k])*b_msgs[t,k] for k in 1:L ]
        wghts1 = [==(k,Z[t-1])*Kappas[Z[t-1]]*PI_comp[comp](pd_dim, X[t], Means[k], Covs[k])*b_msgs[t,k] for k in 1:L]
        wghts = vcat(wghts0,wghts1)
        smpl = sample(pairs, Weights(wghts))
        Z[t] = smpl[1]
        W[t] = smpl[2]
        for i in 1:pd_dim
            Ys[:sums][Z[t]][i] += X[t][i]
            Ys[:sqrs][Z[t]][i] += X[t][i] * transpose(X[t][i])
        end
    end
    
    return Z, W, Ys
end

# function to initialize HMM (without KMeans)
function Init_HMM(T, pd_dim, PI_data, KMax, L, eps, gamma, nu0, rho1, rho2, overall_mu, overall_cov; comp=:lik)
    
    #initialize beta
    Betas = zeros(L)
    while 0.0 in Betas
        Betas = rand(Dirichlet(L,gamma/L))
    end

    #initialize Kappas 
    Kappas = [rand(Beta(rho1, rho2)) for i in 1:L]

    #Initialize Pi_bar_mtx
    Pi_bar_mtx = zeros(L,L)

    for k in 1:L
        init_pi = rand(Dirichlet(alpha*Betas))
        init_pi = (1 - Kappas[k])*init_pi
        init_pi[k] += Kappas[k]
        Pi_bar_mtx[k,:] = init_pi
    end

    # set initial Means and Covs
    Means = []
    Covs = []
    for _ in 1:L
        tmp = []
        for i in 1:pd_dim
            sz = size(overall_cov[i])[1]
            mtx = (nu0-1)*overall_cov[i] 
            while chol_trial(mtx) != nothing #! isposdef(mtx)
                mtx += diagm([eps for j in 1:sz])
                if ! issymmetric(mtx)
                    mtx = (mtx + transpose(mtx))/2
                end
            end
            new_mtx = nothing
            while new_mtx == nothing
                new_mtx = stubborn_IW(nu0+sz, mtx)
            end
            while chol_trial(new_mtx) != nothing #! isposdef(mtx)
                new_mtx += diagm([eps for j in 1:sz])
                if ! issymmetric(new_mtx)
                    new_mtx = (new_mtx + transpose(new_mtx))/2
                end
            end

            push!(tmp, new_mtx)
        end
        push!(Covs, tmp)

        tmp = []
        for i in 1:pd_dim
            sz = size(overall_mu[i])
            new = rand(MvNormal(vec(overall_mu[i]), Covs[end][i]))
            push!(tmp, reshape(new, sz))
        end
        push!(Means, tmp)
    end
    

    # Initial state assignment
    b_msgs = backward_messaging(L, T, pd_dim, PI_data, Pi_bar_mtx, Kappas, Means, Covs, comp=comp)
    Z, W, Ys = update_states(L, T, pd_dim, PI_data, Pi_bar_mtx, Kappas, Means, Covs, b_msgs, comp=comp) 
    Zcnts = [sum(Z .== l) for l in 1:L]

    # Compute K
    K = sum(Zcnts .> 0)

    # Compute N_mtx
    N_mtx = zeros(Int64, (L,L))
    for t in 2:T
        i = Z[t-1]
        j = Z[t]
        if W[t] == 0
            N_mtx[i,j] += 1
        end
    end

    return K, Z, W, Zcnts, N_mtx, Betas, Kappas, Means, Covs, Ys
end


# function to update kappas for self transitions
function update_kappas(L, T, Z, W, rho1, rho2)
    Kappas = zeros(L)

    tots1 = zeros(Int64, L)
    tots2 = zeros(Int64, L)
    for t in 2:T
        idx = Z[t]
        tots1[idx] += W[t]
        tots2[idx] += (1 - W[t])
    end

    for l in 1:L
        Kappas[l] = rand(Beta(rho1 + tots1[l], rho2 + tots2[l]))
    end

    return Kappas, tots1, tots2
end

# function to update betas
function update_betas(L, Z, N_mtx, Betas, alpha, gamma)
    M = zeros(Int64, (L,L))
    for i in 1:L
        for j in 1:L
            if N_mtx[i,j] != 0
                c = 0
                for k in 1:N_mtx[i,j]
                    x = rand(Binomial(1,alpha * Betas[j]/(c + alpha * Betas[j])))
                    M[i,j] += x
                    c+=1
                end
            end
        end
    end
    M[1,1]+=1 #For first time point

    wghts = [sum(M[:,i]) for i in 1:L] .+ gamma/L
    newBetas = zeros(L)
    while 0.0 in newBetas
        newBetas = rand(Dirichlet(wghts))
    end

    return newBetas, M
end


# function to update emission parameters
function update_emissions(L, dim, Ys, nu0, eps, Zcnts, Thetas)
    szs_means = [ size(vec) for vec in Thetas[1][1] ]
    szs_covs = [ size(mtx) for mtx in Thetas[2][1] ]
    new_means = [ [ zeros(sz) for sz in szs_means ] for i in 1:L ]
    new_covs = [ [ zeros(sz) for sz in szs_covs ] for i in 1:L ]

    for k in 1:L 
        if Zcnts[k] > 0
            for d in 1:dim
                old_mean = Thetas[1][k][d]
                old_cov = Thetas[2][k][d]
                # Ensure old_cov is symmetric, positive definite
                #if ! issymmetric(old_cov) 
                    #old_cov = (old_cov + transpose(old_cov))/2
                #end
                #while 1 in (eigvals(old_cov) .< 0)
                    #old_cov += diagm([eps for _ in 1:size(old_cov)[1]])
                #end

                nu = nu0 + szs_covs[d][1] + Zcnts[k]
                nu_delta = (nu0 + szs_covs[d][1])*old_cov + Ys[:sqrs][k][d] - Ys[:sums][k][d]*transpose(Ys[:sums][k][d])/Zcnts[k]
                
                # Ensure nu_delta is symmetric, positive definite
                while chol_trial(nu_delta) != nothing
                    nu_delta += diagm([eps for _ in 1:size(nu_delta)[1]])
                    if ! issymmetric(nu_delta) 
                        nu_delta = (nu_delta + transpose(nu_delta))/2
                    end
                end
                
                new_cov = nothing
                while new_cov == nothing
                    new_cov = stubborn_IW(nu,nu_delta)
                end
                while chol_trial(new_cov) != nothing 
                    new_cov += diagm([eps for j in 1:size(new_cov)[1]])
                    if ! issymmetric(new_cov)
                        new_cov = (new_cov + transpose(new_cov))/2
                    end
                end
                new_covs[k][d] = new_cov
                
                sample_cov = inv(inv(old_cov) + Zcnts[k]*inv(new_cov))
                sample_mean = sample_cov*(inv(old_cov)*old_mean + inv(new_cov)*(Ys[:sums][k][d])) 

                # Ensure sample_cov is symmetric, positive definite
                while ! isposdef(sample_cov)
                    sample_cov += diagm([eps for _ in 1:size(sample_cov)[1]])
                    if ! issymmetric(sample_cov) 
                        sample_cov = (sample_cov + transpose(sample_cov))/2
                    end
                end
            
            
                new_mean = rand(MvNormal(vec(sample_mean), sample_cov))
                new_means[k][d] = reshape(new_mean, szs_means[d])
            end
        else
            new_means[k] = Thetas[1][k]
            new_covs[k] = Thetas[2][k]
        end       
    end
    
    return new_means, new_covs
end

# Gibbs Sampler for alpha
function update_alpha(a, b, L, N_mtx, M, alpha, Betas)
    S = Float64[]
    R = Float64[]
    for k in 1:L
        n = max(sum(N_mtx[k,:]),0.1)
        push!(S,rand(Bernoulli(n/(n + alpha))))
        push!(R,rand(Beta(alpha+1,n)))
    end
    p1 = max(a + sum(M) - sum(S), 0.1)
    p2 = max(b - sum(log.(R)),0.1)

    alpha_new = rand(Gamma(p1,p2))

    trial = rand(Dirichlet(alpha_new*Betas))
    while any(isnan.(trial)) || (0.0 in trial) 
        alpha_new *= 1.25
        trial = rand(Dirichlet(alpha_new*Betas))
    end

    return alpha_new
end



# Gibbs Sampler for gamma
function update_gamma(e, f, eps, K, M, gamma)

    eta = rand(Beta(gamma+1, sum(M)))

    piM = (e + K - 1) / (e + K - 1 + sum(M)*(f - log(eta + eps)))
    if piM > 1
        piM = 0.999
    end
    if piM < 0
        piM = 0.001
    end
    ind = rand(Binomial(1,piM))

    if ind > 0.5
        return rand(Gamma(e + K, 1/(f - log(eta + eps))))
    else
        return rand(Gamma(e + K - 1, 1/(f - log(eta + eps))))
    end

end

# Gibbs Sampler for the rhos
function log_post(r1, r2, L, tots1, tots2)
    lp = L*(loggamma(r1+r2) - loggamma(r1) - loggamma(r2)) 
    lp += sum(loggamma.(r1 .+ tots1)) + sum(loggamma.(r2 .+ tots2))
    lp -= sum(loggamma.(rho1 .+ rho2 .+ tots1 .+ tots2))
    return lp
end

function update_rhos(rho1, rho2, L, tots1, tots2)
    phi_range = [0.01, 0.99]
    eta_range = [0.01, 2]
    phi_grid_sz = 100
    eta_grid_sz = 100

    phi_grid = range(phi_range[1], stop = phi_range[2], length = phi_grid_sz)
    eta_grid = range(eta_range[1], stop = eta_range[2], length = eta_grid_sz)

    post_grid = zeros(phi_grid_sz*eta_grid_sz)
    for (i, phi) in enumerate(phi_grid)
        for (j, eta) in enumerate(eta_grid)
            r1 = phi/(eta^3)
            r2 = (1-phi)/(eta^3)
            post_grid[(i-1)*eta_grid_sz + j] = log_post(r1, r2, L, tots1, tots2)
        end
    end

    max = maximum(post_grid)
    post_grid = exp.(post_grid .- max)
    norm = sum(post_grid)
    post_grid = post_grid/norm

    smpl = rand(Multinomial(1, post_grid))
    idx = findall(>(0),smpl)[1]
    j = idx % eta_grid_sz
    i = convert(Int64, (idx - j)/eta_grid_sz + 1)
    phi = phi_grid[i]
    eta = eta_grid[j]

    return phi/(eta^3), (1 - phi)/(eta^3)
end


# Function to remove empty states
function empties_remover(T, Z, Zcnts, N_mtx, Betas, Kappas, Means, Covs)
    K = length(Zcnts)
    empties = []
    for k in 1:K
        if Zcnts[k] == 0
            push!(empties, k)
        end
    end
    
    # Remove and relabel in reverse order!
    while length(empties) > 0
    
        st = pop!(empties)
        #println("Removing State $(st)")
        
        #decrement all states larger than st
        for t in 1:T
            if Z[t] > st
                Z[t] -= 1
            end
        end
        
        #drop all entries associated to st
        Zcnts = vcat(Zcnts[1:st-1], Zcnts[st+1:end])
        Means = vcat(Means[1:st-1], Means[st+1:end])
        Covs = vcat(Covs[1:st-1], Covs[st+1:end]) 
        Kappas = vcat(Kappas[1:st-1], Kappas[st+1:end]) 
        Betas =  vcat(Betas[1:st-1], Betas[st+1:end]) 
        N_mtx = N_mtx[1:end .!= st, 1:end .!= st]
    end

    Betas = Betas / sum(Betas)
    
    return Z, Zcnts, N_mtx, Betas, Kappas, Means, Covs

end

function match_to_known(Z1, Z2)
    T = length(Z2)
    if length(Z1) != T
        T= minimum([T, length(Z1)])
    end

    dim = maximum([length(Set(Z2)), length(Set(Z1))])
    cost_mat = zeros(dim,dim)
    for i in 1:dim
        tmp1 = [==(i,Z1[k]) for k in 1:T]
        for j in 1:dim
            tmp2 = [==(j,Z2[k]) for k in 1:T]
            tmp = sum(abs.(tmp1 .- tmp2))
            cost_mat[i,j] = tmp
        end
    end

    assgn, cost = hungarian(cost_mat)
    reorder_dict = Dict{Int64,Int64}()
    for i in 1:dim
        reorder_dict[i] = assgn[i]
    end

    return dim, reorder_dict, cost
end

function log_likelihood(L, pd_dim, Z, Pi_bar_mtx, Means, Covs, X; comp=:lik)
    T = length(Z)
    A_mtx = zeros(T+1,L)
    C_vec = zeros(T)
    
    A_mtx[1,:] = [PI_comp[comp](pd_dim, X[1],Means[i],Covs[i]) for i in 1:L]
    A_mtx[1,:] /= sum(A_mtx[1,:])
    for t in 1:T
        A_mtx[t+1,:] = [sum(A_mtx[t,:] .* Pi_bar_mtx[:,i])*PI_comp[comp](pd_dim, X[t], Means[i], Covs[i]) for i in 1:L]
        C_vec[t] = sum(A_mtx[t+1,:])
        A_mtx[t+1,:] /= C_vec[t]
    end
    lik = sum(log.(C_vec))/T
    
    return lik, A_mtx
end

# Function to compare state changes in sequences
#Z1 is assumed to be the correct sequence
#Z2 is assumed to be the predicted sequence
#Returns: Matched change points, missed changes in Z1, and extra changes in Z2
function comp_change_pts(Z1,Z2)
    L = minimum([length(Z1), length(Z2)])
    Z1 = Z1[1:L]
    Z2 = Z2[1:L]
    change_pts1 = []
    change_pts2 = []
    for i in 2:L
        if Z1[i-1] != Z1[i]
            push!(change_pts1, i)
        end

        if Z2[i-1] != Z2[i]
            push!(change_pts2, i)
        end
    end

    L1 = length(change_pts1)
    L2 = length(change_pts2)
    L_min = minimum([L1,L2])
    pair_mtx = zeros(Int64, (L1,L2))

    for i in 1:L1
        for j in 1:L2
            pair_mtx[i,j] = abs(change_pts1[i] - change_pts2[j])
        end
    end

    prs = []
    ChP1_mtchd = []
    ChP2_mtchd = []
    vals = sort(collect(Set(pair_mtx)),rev=true)
    new_min = pop!(vals)
    min_pairs = findall( ==(new_min), pair_mtx)

    while length(prs) < L_min
        if length(min_pairs) == 0
            new_min = pop!(vals)
            min_pairs = findall( ==(new_min), pair_mtx)
        end

        tmp = pop!(min_pairs)
        if !(tmp[1] in ChP1_mtchd) && !(tmp[2] in ChP2_mtchd)
            push!(prs, (tmp[1],tmp[2]))
            push!(ChP1_mtchd, tmp[1])
            push!(ChP2_mtchd, tmp[2])
        end

    end

    Z_pairs = [(change_pts1[itm[1]], change_pts2[itm[2]]) for itm in prs]
    Z_pairs = sort(Z_pairs)

    #remove bad pairs
    remove_idxs = []
    for i in 1:length(Z_pairs)
        m = minimum(Z_pairs[i])
        M = maximum(Z_pairs[i])
        if length(Z_pairs[findall( i -> (i[1]>m && i[2]>m && i[1]<M && i[2]<M), Z_pairs)]) > 0
            push!(remove_idxs, i)
        end
    end

    while length(remove_idxs) > 0
        idx = pop!(remove_idxs)
        Z_pairs = Z_pairs[1:end .!= idx]
    end


    missed = collect(setdiff(Set(change_pts1), Set([itm[1] for itm in Z_pairs])))
    extra = collect(setdiff(Set(change_pts2), Set([itm[2] for itm in Z_pairs])))

    return Z_pairs, missed, extra

end

#function to print counts of state matches between a known sequence (Z_known) and a predicted sequence (Z_pred)
function match_comp(Z_known, Z_pred)
	K_known = maximum(Z_known)
	K_pred = maximum(Z_pred)
	pairs = collect(zip(Z_known, Z_pred))

	counts = Dict()

	for i in 1:K_known
		for j in 1:K_pred
			counts[(i,j)] = 0
		end
	end

	for pr in pairs
		counts[pr] += 1
	end

	counts = [[ky, counts[ky]] for ky in keys(counts)]
	sort!(counts, by = x -> x[2], rev=true)

	for itm in counts
		if itm[2] > 0
			println(itm)
		end
	end
end