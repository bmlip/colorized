"""
Utility functions for Probabilistic Programming - 2
Wouter Kouw
Last update: 07-01-2020
"""

import Turing: logaddexp
import Distributions: MvNormal, pdf, logpdf
import MCMCChains: Chains
import Plots: scatter, contour, contour!, xlabel!, ylabel!
using LaTeXStrings


function prob_cluster(x, y, μ, Σ)
    "Probability under Gaussian"

    # Ensure matrix is Hermitian
    Σ = enforce_Hermitian!(Σ)

    # Ensure matrix is non-negative
    Σ = enforce_nonneg!(Σ)

    # Return weighted probability
    return pdf(MvNormal(μ, Σ), [x, y])
end

function enforce_nonneg!(M)
    "Enforce non-negative elements"
    M = max.(0, M)
end

function enforce_Hermitian!(M)
    "Enforce Hermitian"
    M = (M + M')/2
end

function plot_clusters(observations;
                       μ=[],
                       Σ=[[1. 0.;0. 1.], [1. 0.;0. 1.]],
                       x1=range(-5, stop=3),
                       x2=range(-5, stop=4),
                       colorlist=["red", "blue", "green"],
                       size=(500, 500))
    "Scatter data and overlay Gaussian clusters."

    # Show the data
    plt = scatter(observations[:,1],
                  observations[:,2],
                  label="observations",
                  color="black",
                  xlims=[x1[1] x1[end]],
                  ylims=[x2[1] x2[end]],
                  xlabel=L"$x_1$",
                  ylabel=L"$x_2$",
                  grid=false,
                  size=size)

    # Loop over clusters
    for k = 1:length(μ)

        # Overlay the posterior probabilities for k-th cluster
        contour!(plt, x1, x2,
                 (x, y) -> prob_cluster(x, y, μ[k], Σ[k]),
                 color=colorlist[k],
                 label="C"*string(k),
                 cbar=nothing)
    end
    plot(plt)
end

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"
    return vec(convert(Array{Float64,2}, mean(chain[param].value.data, dims=[1, 3])[:,:,1]))
end;
