using CSV, DataFrames, DelimitedFiles, Printf, Serialization
include("ds_HDP_HMM_AUX.jl")

# Decide if PIs will be compared by likelihood (:lik) or inverse distance (:sim)
comp = :sim
# Set which topological dimensions will be used for comparison
dims = [0,1]

# Insert training and test datafiles here
# data is assumed to be a time series of persistence images (pre-computed) of the same size and number of topological dimensions
# state numbers are assumed to be positive integers.

#training data
train_jls_file = "flare_data_2_cut.jls"
datafile_Z = "flare_states_2.csv"

#Z_Score Flag: set to 1 if persistence images need to be normalized
zScore = 0


#test data (will be loaded after training)
test_flg = 0 #Set to 1 if test data is provided
test_jls_file = ""
testfile_Z = ""
# Set number of iterations
iters = 2000

# Decide how often empty states are resampled (stored when step % memo_freq == 0)
resmpl_freq = 20

# Set Burn-In Period
burn = 0

# Decide how often results are stored (stored when step % memo_freq == 0)
memo_freq = 1

# Set Upper bound on total number of states
L = 20

# Decide whether KMeans is to be used to accelerate learning.
kmeans_flg = 0
KMax = 10 # Upper bound on KMeans algorithm for setup (only used when flag = 1)


# HYPERPARAMETERS

# eps is used to keep nearly singular covariance matrices invertible (eps added to each main diagonal entry)
eps = 10^(-6) 

#a and b are used for a prior on alpha
a = 1
b = 0.01

#c and d are used for a prior on gamma
c = 2
d = 1

#nu0 is for the inverse Wishart distribution (should be greater than 1)
nu0 = 2
delta0 = 10

############################# END OF USER SUPPLIED DATA #############################
pd_dim = length(dims)

#Make sure KMax <= L
if kmeans_flg != 0 && KMax > L
    L = KMax+1
end

#Load Training Data
train_data = deserialize(train_jls_file)
PI_data_train = []
for dct in train_data
    push!(PI_data_train, [ vec(dct[:pi][i+1]) for i in dims ])
end

Z_train_known = readdlm(datafile_Z, ',', Int64)

T_train = length(train_data)

train_data=nothing  #enforce garbage collection of train_data (to save memory)

orig_overall_mu, orig_overall_cov = PI_stats(PI_data_train)
if zScore == 1
    global PI_data_train
    sigmas = [sqrt(mtx) for mtx in orig_overall_cov]
    sigmaDs = [ [mtx[i,i] for i in 1:size(mtx)[1]] for mtx in sigmas]
    new_data = []
    for itm in PI_data_train
        tmp = []
        for i in 1:length(dims)
            push!(tmp, (itm[i] - orig_overall_mu[i]) ./ sigmaDs[i])
        end
        push!(new_data, tmp)
    end
    PI_data_train = new_data
end



#Compute overall stats
overall_mu, overall_cov = PI_stats(PI_data_train)

#Make overall_cov symm. pos.def
for d in 1:pd_dim
    if ! issymmetric(overall_cov[d])
        overall_cov[d] += transpose(overall_cov[d])
        overall_cov[d] /= 2
    end

    while 1 in (eigvals(overall_cov[d]) .< 0) 
        overall_cov[d] += diagm([ eps for _ in 1:size(overall_cov[d])[1]])
    end
end

#Flag to note when burn-in is finished
burn_flg = 0

# Initialize hyperparameters
alpha = rand(Gamma(a,1/b))
gamma = rand(Gamma(c,1/d))
phi = rand(Uniform(0,1))
eta = rand(Uniform(0,2)) #phi and eta are used to give rho1 and rho2
rho1 = phi/(eta^3)
rho2 = (1-phi)/(eta^3)

# Memo and counts to store results after burn-in period
memo = []
counts = Dict()
for i in 1:L
    counts[i] = [0,0.0]
end

# Initialize HMM
pd_dim = length(dims)
if kmeans_flg == 0
    K, Z, W, Zcnts, N_mtx, Betas, Kappas, Means, Covs, Ys = Init_HMM(T_train, pd_dim, PI_data_train, KMax, L, eps, gamma, nu0, rho1, rho2, overall_mu, overall_cov, comp=comp)
else
    K, Z, W, Zcnts, N_mtx, Betas, Kappas, Means, Covs, Ys = Init_HMM_w_Kmeans(T_train, pd_dim, PI_data_train, KMax, L, eps, gamma, nu0, rho1, rho2, overall_mu, overall_cov, comp=comp)
end
println("Initialization Complete.\nNumber of Initial Populated States = $(K).\n")

#Make sure Covs are symmetric and positive definite
for l in 1:L
    for d in 1:pd_dim
        if ! issymmetric(Covs[l][d])
            Covs[l][d] += transpose(Covs[l][d])
            Covs[l][d] /= 2
        end

        while 1 in (eigvals(Covs[l][d]) .< 0) 
            Covs[l][d] += diagm([ eps for _ in 1:size(Covs[l][d])[1]])
        end
    end
end

#Save Initial Means and Covs
Thetas = [deepcopy(Means), deepcopy(Covs)]

#Initialize Pi_bar_mtx
Pi_bar_mtx = zeros(L,L)

for k in 1:L
    init_pi = rand(Dirichlet(alpha*Betas))
    init_pi = (1 - Kappas[k])*init_pi
    init_pi[k] += Kappas[k]
    Pi_bar_mtx[k,:] = init_pi
end

# Save initialization to memo
Ztmp, Zcntstmp, N_mtxtmp, Betastmp, Kappastmp, Meanstmp, Covstmp = empties_remover(T_train, Z, Zcnts, N_mtx, Betas, Kappas, Means, Covs)

Pi_bar_mtx_tmp = zeros(K,K)

for k in 1:K
    init_pi = rand(Dirichlet(alpha*Betastmp))
    init_pi = (1 - Kappastmp[k])*init_pi
    init_pi[k] += Kappastmp[k]
    Pi_bar_mtx_tmp[k,:] = init_pi
end
lik, _ = log_likelihood(K, pd_dim, Ztmp, Pi_bar_mtx_tmp, Meanstmp, Covstmp, PI_data_train, comp=comp)
push!(memo,[0, lik, Ztmp, Zcntstmp, N_mtxtmp, Betastmp, Kappastmp, Pi_bar_mtx_tmp, Meanstmp, Covstmp])
Stps = [0]
LLs = [lik]

#Variables to keep track of iteration with best log-likelihood
idx = 1
ll = lik
#non_zero_states = K

pbar = ProgressBar(1:iters)
for step in pbar
    global K, L, burn_flg, rho1, rho2, gamma, alpha, pd_dim, idx, ll, LLs, Stps, non_zero_states
    global Z, Zcnts, Ys, W, b_msgs, Betas, Kappas, N_mtx, Pi_bar_mtx, Means, Covs
    global Ztmp, Zcntstmp, N_mtxtmp, Betastmp, Kappastmp, Meanstmp, Covstmp, Pi_bar_mtx_tmp, lik, memo, counts

    if (burn_flg == 0) && (step >= burn)
        burn_flg = 1
        println(pbar, "Burn-In Period Complete.")
    end

    # Backward Messaging
    b_msgs = backward_messaging(L, T_train, pd_dim, PI_data_train, Pi_bar_mtx, Kappas, Means, Covs, comp=comp)

    # Update State Assignments 
    Z, W, Ys = update_states(L, T_train, pd_dim, PI_data_train, Pi_bar_mtx, Kappas, Means, Covs, b_msgs, comp=comp)
    Zcnts = zeros(Int64,L)
    for t in 1:T_train
        Zcnts[Z[t]] += 1
    end

    # Compute N_mtx (forgot in original version)
    N_mtx = zeros(Int64, (L,L))
    for t in 2:T_train
        i = Z[t-1]
        j = Z[t]
        if W[t] == 0
            N_mtx[i,j] += 1
        end
    end

    # Update Kappas
    Kappas, tots1, tots2 = update_kappas(L, T_train, Z, W, rho1, rho2)

    # Update Betas and Pi_bar_mtx
    Betas, M = update_betas(L, Z, N_mtx, Betas, alpha, gamma)


    Pi_bar_mtx = zeros(L,L)
    for k in 1:L
        init_pi = rand(Dirichlet(alpha*Betas + N_mtx[k,:]))
        init_pi = (1 - Kappas[k])*init_pi
        init_pi[k] += Kappas[k]
        Pi_bar_mtx[k,:] = init_pi
    end

    # Update Means and Covs
    Means, Covs = update_emissions(L, pd_dim, Ys, nu0, eps, Zcnts, Thetas)

    # Update Hyperparameters
    alpha = update_alpha(a, b, L, N_mtx, M, alpha, Betas) 
    gamma = update_gamma(c, d, eps, L, M, gamma)
    rho1, rho2 = update_rhos(rho1, rho2, L, tots1, tots2)

    # For empty states, resample Mean and Cov
    if (step % resmpl_freq) == 0
        for k in 1:L
            if Zcnts[k] == 0
                tmp = []
                for i in 1:pd_dim
                    sz = size(overall_cov[i])[1]
                    mtx = (nu0-1)*overall_cov[i] 
                    while chol_trial(mtx) != nothing 
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
                Thetas[2][k] = tmp
                Covs[k] = tmp

                tmp = []
                for i in 1:pd_dim
                    sz = size(overall_mu[i])
                    new = rand(MvNormal(vec(overall_mu[i]), Covs[k][i]))
                    push!(tmp, reshape(new, sz))
                end
                Thetas[1][k] = tmp
                Means[k] = tmp
            end
        end
    end

    #COMMS
    if (step % memo_freq == 0 && burn_flg == 1) || step == iters

        # Save data to memo
        Ztmp, Zcntstmp, N_mtxtmp, Betastmp, Kappastmp, Meanstmp, Covstmp = empties_remover(T_train, Z, Zcnts, N_mtx, Betas, Kappas, Means, Covs)
        Knew = length(Zcntstmp)
        if Knew != K
            println(pbar, "Number of states changed to $(Knew).")
            K = Knew
        end

        Pi_bar_mtx_tmp = zeros(K,K)

        for k in 1:K
            init_pi = rand(Dirichlet(alpha*Betastmp))
            init_pi = (1 - Kappastmp[k])*init_pi
            init_pi[k] += Kappastmp[k]
            Pi_bar_mtx_tmp[k,:] = init_pi
        end
        
        lik, _ = log_likelihood(K, pd_dim, Ztmp, Pi_bar_mtx_tmp, Meanstmp, Covstmp, PI_data_train, comp=comp)
        push!(memo,[step, lik, Ztmp, Zcntstmp, N_mtxtmp, Betastmp, Kappastmp, Pi_bar_mtx_tmp, Meanstmp, Covstmp])
        push!(Stps, step)
        push!(LLs, lik)

        if lik > ll
            ll = lik
            idx = length(memo)
        end

        counts[Knew][1]+=1
        counts[Knew][2]+=lik
        
    end
end

#Sort Counts by frequency and log-likelihood
state_data = []
for l in 1:L
    push!(state_data, (l, counts[l][1], counts[l][2]/counts[l][1]))
end
sort!(state_data, by = x -> (0, x[2], x[3]), rev=true)
println("Top State Counts and Log-Likelihoods:")
for itm in state_data[1:5]
    println("$(itm[1]) States (count = $(itm[2]); avg. LL = $(itm[3]))")
end
println("")

step, lik, Z, Zcnts, N_mtx, Betas, Kappas, Pi_bar_mtx, Means, Covs = memo[idx]
K = length(Zcnts)
println("Max Log-likelihood on Training Data = $(ll).")

#Compare to known states using Munkres Hungarian Algorithm and reorder states
K_new, reorder_dict, cost = match_to_known(Z, Z_train_known)
Z = [reorder_dict[z] for z in Z]
Zcnts = [ sum( Z .== k ) for k in 1:K_new]

old_N_mtx, old_Betas, old_Kappas, old_Pi_bar_mtx, old_Means, old_Covs = deepcopy.([N_mtx, Betas, Kappas, Pi_bar_mtx, Means, Covs])
Betas = zeros(K_new)
Kappas = zeros(K_new)
Means = [overall_mu for _ in 1:K_new]
Covs = [overall_cov for _ in 1:K_new]
N_mtx = zeros(Int64, (K_new, K_new))
Pi_bar_mtx = zeros((K_new, K_new))
for s in 1:K
    Betas[reorder_dict[s]] = old_Betas[s]
    Kappas[reorder_dict[s]] = old_Kappas[s]
    Means[reorder_dict[s]] = old_Means[s]
    Covs[reorder_dict[s]] = old_Covs[s]
end
for i in 1:K
    for j in 1:K
        N_mtx[reorder_dict[i], reorder_dict[j]] = old_N_mtx[i,j]
        Pi_bar_mtx[reorder_dict[i], reorder_dict[j]] = old_Pi_bar_mtx[i,j]
    end
end

K = K_new

# Compute state accuracy on training data
tot = 0
for t in 1:T_train
    if Z[t] != Z_train_known[t]
         global tot += 1
    end
end
pct = round((T_train-tot)/T_train*100, sigdigits=5)
println("State Match percentage on training data = $(pct)%.")

# Compute change point accuracy on training data
Z_pairs_train, missed_train, extra_train = comp_change_pts(Z_train_known,Z)
mtch_pts = length(Z_pairs_train)
mssd_pts = length(missed_train)
extr_pts = length(extra_train)
recall_pct = mtch_pts / (mtch_pts + mssd_pts)
prec_pct = mtch_pts / (mtch_pts + extr_pts)
f1_pct = 200/(1/recall_pct + 1/prec_pct)
recall_pct *= 100
prec_pct *= 100

avg = 0.0
std = 0.0
for i in 1:mtch_pts
    global avg += abs(Z_pairs_train[i][1] - Z_pairs_train[i][2])
    global std += (Z_pairs_train[i][1] - Z_pairs_train[i][2])^2
end
avg /= mtch_pts
std /= mtch_pts
std -= avg^2
std = sqrt(std)
println("Training Change Point Recall: $(recall_pct)%.")
println("Training Change Point Precision: $(prec_pct)%.")
println("Training Change Point F1 Score: $(f1_pct)%")
println("Change Point Accuracy: $(avg) +/- $(std).")



#PLOT LOG-LIKELIHOOD
layout = Layout(;title="Log-Likelihood for Training Data")
p1 = plot(scatter(x=Stps,y=LLs,mode="lines"),layout)

#Plot Zs
L = minimum([length(Z), length(Z_train_known)])
X = 1:L
plt1 = scatter(x=X, y=Z_train_known[1:L], name="Actual States")
plt2 = scatter(x=X,y=Z[1:L], name="Predicted States")
layout = Layout(;title="Comparison of State Transitions for Training Data")
p2 = plot([plt1,plt2], layout)

if test_flg == 0
    #Display
    display(p1)
    display(p2)
else
    println("")
    #Load Test Data
    test_data = deserialize(test_jls_file)
    PI_data_test = []
    for dct in test_data
        push!(PI_data_test, [ vec(dct[:pi][i+1]) for i in dims ])
    end

    Z_test_known = readdlm(testfile_Z, ',', Int64)

    T_test = length(test_data)

    test_data=nothing  #enforce garbage collection of test_data (to save memory)

    if zScore == 1
        global PI_data_test
        sigmas = [sqrt(mtx) for mtx in orig_overall_cov]
        sigmaDs = [ [mtx[i,i] for i in 1:size(mtx)[1]] for mtx in sigmas]
        new_data = []
        for itm in PI_data_test
            tmp = []
            for i in 1:length(dims)
                push!(tmp, (itm[i] - orig_overall_mu[i]) ./ sigmaDs[i])
            end
            push!(new_data, tmp)
        end
        PI_data_test = new_data
    end

    b_msgs = backward_messaging(K, T_test, pd_dim, PI_data_test, Pi_bar_mtx, Kappas, Means, Covs, comp=comp)
    Z_test, W_test, Ys_test = update_states(K, T_test, pd_dim, PI_data_test, Pi_bar_mtx, Kappas, Means, Covs, b_msgs, comp=comp)
    lik_test, _ = log_likelihood(K, pd_dim, Z_test, Pi_bar_mtx, Means, Covs, PI_data_test, comp=comp)
    println("Log-likelihood on Test Data = $(lik_test).")

    tot = 0
    for t in 1:T_test
        if Z_test[t] != Z_test_known[t]
            global tot += 1
        end
    end
    pct = round((T_test-tot)/T_test*100, sigdigits=5)
    println("State Match percentage on test data = $(pct)%.")

    Z_pairs_test, missed_test, extra_test = comp_change_pts(Z_test_known,Z_test)
    mtch_pts = length(Z_pairs_test)
    mssd_pts = length(missed_test)
    extr_pts = length(extra_test)
    recall_pct = mtch_pts / (mtch_pts + mssd_pts)
    prec_pct = mtch_pts / (mtch_pts + extr_pts)
    f1_pct = 200/(1/recall_pct + 1/prec_pct)
    recall_pct *= 100
    prec_pct *= 100

    avg = 0.0
    std = 0.0
    for i in 1:mtch_pts
        global avg += abs(Z_pairs_test[i][1] - Z_pairs_test[i][2])
        global std += (Z_pairs_test[i][1] - Z_pairs_test[i][2])^2
    end
    avg /= mtch_pts
    std /= mtch_pts
    std -= avg^2
    std = sqrt(std)
    println("Test Set Change Point Recall: $(recall_pct)%.")
    println("Test Set Change Point Precision: $(prec_pct)%.")
    println("Test Set Change Point F1 Score: $(f1_pct)%")
    println("Change Point Accuracy: $(avg) +/- $(std)")
    

    #Plot Zs
    L = minimum([length(Z_test), length(Z_test_known)])
    X = 1:L
    plt3 = scatter(x=X, y=Z_test_known[1:L], name="Actual States")
    plt4 = scatter(x=X,y=Z_test[1:L], name="Predicted States")
    layout = Layout(;title="Comparison of State Transitions for Test Data")
    p3 = plot([plt3,plt4], layout)

    display(p1)
    display(p2)
    display(p3)

end



