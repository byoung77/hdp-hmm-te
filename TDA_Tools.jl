using Clustering, CSV, DataFrames, Distributions, LinearAlgebra, PersistenceDiagrams, PlotlyJS, ProgressBars, Ripserer, Serialization, StatsBase

# function to convert time series data to persistence images
    # time_series is an array of points (as tuples)
    # output_file is where the data will be stored (as a .jls file)
    # window_sz is the number of data points that will be used to create the persistence image
    # slide is how much the window will advance each iteration
    # jump allows you to pick how many samples in the window to use for computation (jump = 10 would take every 10th sample)
    # max_dim is the maximum dimension used in the persistene image (increasing this will greatly slow down the computation)
    # pi_sz is the size of the (typically) square matrix representing the persistence image.
        # If this is an int, all dimensions will have the same base size.
        # You can specify an array of sizes for each dimenion (array should be length max_dim + 1).
    # stack=true will stack all PIs into a single N X 1 vector

function timeSeries_to_PIs(time_series, output_file, window_sz, slide; jump=1, max_dim=1, pi_sz=5, stack=false, converter=nothing)
    T = length(time_series)
    if typeof(pi_sz) == Int64
        pi_sz = [pi_sz for i in 1:(max_dim+1)]
    elseif length(pi_sz) != (max_dim + 1) 
        # Maybe do genuine error flagging here
        println("Length of persistence image size array does not equal (max_dim + 1).")
        return nothing
    end

    data = []

    println("Computing Persistence Diagrams:")
    all_pds = [[] for i in 1:(max_dim+1)]
    for i in ProgressBar(1:slide:(T - window_sz+1))
        dct = Dict{Symbol,Any}(:time => [i,i+window_sz-1], :pd => [], :pi => [])
        pd = ripserer(time_series[i:jump:(i+window_sz-1)], dim_max=max_dim)
        dct[:pd] = pd
        push!(data, dct)
        if converter == nothing
            for j in 1:(max_dim+1)
                push!(all_pds[j], pd[j])
            end
        end
    end

    if converter == nothing
        PI_convert = [PersistenceImage(all_pds[i], size=pi_sz[i]) for i in 1:(max_dim+1)]
    else
        PI_convert = converter
    end

    println("Computing Persistence Images:")
    for dct in ProgressBar(data)
        pd = dct[:pd]
        pis = []

        for i in 1:(max_dim+1)
            push!(pis, PI_convert[i](pd[i]))
        end

        if stack
            pis = [vcat(vec.(pis)...)]
        end

        dct[:pi] = pis
    end

    serialize(output_file, data)
    println("Data converted and saved as $(output_file).")
    return PI_convert
end


# Function to compute mean and covariance for persistence images
# Means are returned in the same shape as the group of PIs
# Covariance matrices are based on vectorized PIs (i.e. vec(pi) for each image)

function PI_stats(PIs)
    pd_dim = length(PIs[1])
    sep_PIs = [ [PI[i] for PI in PIs] for i in 1:pd_dim ]
    avg = mean.(sep_PIs)
    sg2 = cov.(sep_PIs)
    return avg, sg2
end


# Function to generate a random PI from average and covariance
# eps is a tolerance to help invert nearly singular covariance matrices

function PI_smpl(avg, sg2; eps=0)
    sz_avg = size(avg)
    sz_cov = size(sg2)[1]
    avg = vec(avg)
    if eps > 0
        sg2 = sg2 + diagm([eps for i in 1:sz_cov])
    end

    ret = rand(MvNormal(avg,sg2))

    #PIs cannot have negative entries
    while 1 in (ret .< 0)
        ret = rand(MvNormal(avg,sg2))
    end

    return reshape(ret,sz_avg)
end


# Function to compute similarity score between two persistence images (basically 1/distance)
# M1 and M2 are the persistence images (assumed to be the same shape)
# p is the index of the norm
# eps is a cut-off for the norm (so the maximum similarity is 1/eps)

function cutoff_PI_sim(M1, M2; p=1, eps=0.00001)
    tmp = norm(M1-M2,p)
    if tmp < eps
        return 1/eps
    end
    return 1/tmp
end


# Function to compute total similarity between two sets of PIs.
# Each dimension in the PI is assumed to be independent.
# The return value is the product of the individual similarities for each dimension.
# PI and Given are the persistence images to be compared 
# p and eps are the same as in cutoff_PI_sim.
# Dummy variable is for wrapper dictionary with likelihood.


function PI_overall_similarity(pd_dim, PI, Given, dummy; p=2, eps=0.000001)
    ret = 1
    for i in 1:pd_dim
        ret *= cutoff_PI_sim(PI[i], Given[i], p=p, eps=eps)
    end
    return ret
end

# functions to compute likelihoods for PI given mean and covariance (PI dimensions assumed independent)
# eps sets the lower bound for values (so that very small values don't return 0)
function PI_Normal_pdf(PI, Mu, Cov; eps=10^(-10))
    X = vec(PI)
    Mu = vec(Mu)

    while !(isposdef(Cov))
        Cov += diagm([eps for _ in 1:length(Mu)])
    end

    lh =  pdf(MvNormal(Mu,Cov),X)
    if lh < eps
        return eps
    end
    return lh
end

# Overall MvNormal likelihood for PIs (optional arguments are dummies for the wrapper dictionary below)
function PI_overall_likelihood(pd_dim, PI, Mus, Covs; p=1, eps=0.000001)
    ret = 1
    for i in 1:pd_dim
        ret *= PI_Normal_pdf(PI[i], Mus[i], Covs[i])
    end
    return ret
end

# Wrapper dictionary to contain similarity and likelihood comparisons
PI_comp = Dict(:sim => PI_overall_similarity, :lik => PI_overall_likelihood)


# Function to run KMeans on Persistence Images
# All PIs are stacked to an N X 1 vector and the standard Euclidean metric is used.
# All Ks from 2 to KMax are tried.  The K with best score is returned with the clustering data.

function PI_KMeans(T, pd_dim, PIs, Kmax; scoring=:silhouettes)
    # Generate data matrix
    smpls = length(PIs)
    dim = size(vcat(vec.(PIs[1])...))[1]
    X = zeros(dim, smpls)
    for i in 1:smpls
        X[:,i] = vcat(vec.(PIs[i])...)
    end

    # Run KMeans.  
    clusterings = [kmeans(X, k, maxiter=200) for k in 2:Kmax]
    scores = clustering_quality.(Ref(X), clusterings, quality_index=scoring)
    max_score, idx = findmax(scores)
    K = idx+1

    # Warn if K is Kmax
    if K == Kmax
        println("Warning: K = Kmax.  Consider increasing Kmax!")
    end

    Z = clusterings[idx].assignments

    return K, Z
end



# Function to convert persistance diagram (birth, death) given by pd to tilted persistence diagram (birth, persistence).
# Returns an array:  first components are the tilted PDs for each dimension, last component is a [max_x, max_y] for plotting.

function tilted_PD(pd)
    ret = []
    max_x = 0
    max_y = 0

    #Handle 0-dim components
    dim_data = pd[1]
    pts = []
    for j in 1:length(dim_data)-1
        pt = dim_data[j][2]
        if pt > max_y
            max_y = pt
        end
        push!(pts,[0,pt])
    end
    push!(ret,pts)

    #Handle higher dimensional components
    for i in 2:length(pd)
        dim_data = pd[i]
        pts = []
        for j in 1:length(dim_data)
            pt1 = dim_data[j][1]
            pt2 = dim_data[j][2] - pt1

            if pt1 > max_x
                max_x = pt1
            end

            if pt2 > max_y
                max_y = pt2
            end

            push!(pts,[pt1,pt2])
        end
        push!(ret,pts)
    end

    #Handle last point
    push!(ret[1],[0,max_y])
    push!(ret,[max_x,max_y])

    return ret
end

#Function to plot tilted persistence diagram.
# tilt is a tilted PD array as output by tilted_PD above/
# max_x, max_y set the dimensions of the diagram.  If not specified, dimensions are taken from tilt.
# titl is the optional title of the PD
# lgnd=false will suppress the legend on the PD

function tilted_PD_plotter(tilt; max_x=nothing, max_y=nothing, titl=nothing, lgnd=true)

    if max_x == nothing
        max_x = ceil(tilt[end][1])
    end

    if max_y == nothing
        max_y = ceil(tilt[end][2])
    end

    if titl != nothing
        layout = Layout(;title=titl, xaxis_range = [-0.05, max_x], yaxis_range = [-0.05, max_y])
    else
        layout = Layout(xaxis_range = [-0.05, max_x], yaxis_range = [-0.05, max_y])
    end

    #plot dim 0
    dim_data = tilt[1]
    X = [pt[1] for pt in dim_data]
    Y = [pt[2] for pt in dim_data]
    plts = [scatter(x=X,y=Y, mode = "markers", name = "Dim 0")]
   
    for j in 1:length(tilt[2:end-1])
        dim_data = tilt[j+1]
        X = [pt[1] for pt in dim_data]
        Y = [pt[2] for pt in dim_data]
        push!(plts, scatter(x=X,y=Y, mode = "markers", name = "Dim $(j)"))
    end
    plt = plot(plts, layout)

    if ! lgnd
        plot!(legend=false)
    end

    return plt

end

