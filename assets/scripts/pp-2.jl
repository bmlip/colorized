"""
Utility functions for Probabilistic Programming - 2
Wouter Kouw
Last update: 07-01-2020
"""

import Turing: logaddexp
import Distributions: MvNormal, logpdf
import MCMCChains: Chains
using Plots
pyplot()

function prob_cluster(x, y, ϕ, μ1, Σ1)
    "Weighted probability under Gaussian"
    return ϕ*pdf(MvNormal(μ1, Σ1), [x, y])
end

function plot_posteriors(X, μ1, μ2; Σ1=[1. 0.;0. 1.], Σ2=[1. 0.;0. 1.], ϕ=[0.5, 0.5], x1=range(-5, stop=3), x2=range(-5, stop=4), size=(500, 500))
    "Scatter data and overlay posterior probabilities via contour plot."

    # Show the data
    scatter(X[:,1], X[:,2], 
            label="observations", 
            color="black", 
            xlims=[x1[1] x1[end]], 
            ylims=[x2[1] x2[end]], 
            xlabel=L"$x_1$", 
            ylabel=L"$x_2$", 
            grid=false, 
            size=size)

    # Overlay the posterior probabilities for each cluster
    contour!(x1, x2, (x, y) -> prob_cluster(x, y, ϕ[1], μ1, Σ1), color="red", label="C1", cbar=nothing)
    contour!(x1, x2, (x, y) -> prob_cluster(x, y, ϕ[2], μ2, Σ2), color="blue", label="C2", cbar=nothing)
end

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"
    return vec(convert(Array{Float64,2}, mean(chain[param].value.data, dims=[1, 3])[:,:,1]))
end;
