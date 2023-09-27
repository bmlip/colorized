using Distributions, Plots

function plotGMM(X::Matrix, clusters::Vector, γ::Matrix, title)
    # Plot data set and (fitted) mixture model consisting of two Gaussian distributions
    # X contains a 2-d data set (every column holds a data point)
    # clusters holds the 2 Gaussian elements of the mixture model
    # γ contains p(cluster|X), and should contain NaN elements if not yet known

    # Plot contours of the element distributions
    K = length(clusters)
    result = plot(title=title)
    for k=1:K
        X1=Matrix{Float64}(undef,50,50); X2=Matrix{Float64}(undef,50,50); d=Matrix{Float64}(undef,50,50)
        # Create bounding box for thse contour plot
        lims = [-2*sqrt(cov(clusters[k])[1,1]) 2*sqrt(cov(clusters[k])[1,1]);
                -2*sqrt(cov(clusters[k])[2,2]) 2*sqrt(cov(clusters[k])[2,2])] + repeat(mean(clusters[k]), 1, 2)
        X1 = LinRange(lims[1,1], lims[1,2], 50)
        X2 = LinRange(lims[2,1], lims[2,2], 50)

        contour!(X1, X2, (x, y) -> pdf(clusters[k], [x, y]))
    end


    # Plot data points
    if isnan(γ[1,1])
        scatter!(X[1,:], X[2,:], label="observations")
    else
        scatter!(X[1,:], X[2,:], label="observations")
    end

    return result
end
