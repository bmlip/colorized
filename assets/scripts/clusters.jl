import Distributions: MvNormal, pdf
import Plots: scatter, contour, contour!, xlabel!, ylabel!
using ColorSchemes


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
                       xlims=(-3,3),
                       ylims=(-3,3),
                       colors=["reds", "blues", "greens"],     
                       xlabel="x₁",
                       ylabel="x₂",
                       fontsize=15,
                       linewidth=3,
                       grid=false,
                       figsize=(500, 500))
    "Scatter data and overlay Gaussian clusters."

    # Show the data
    plt = scatter(observations[:,1],
                  observations[:,2],
                  label="",
                  color="black",
                  xlims=xlims,
                  ylims=ylims,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  labelfontsize=fontsize-2,
                  guidefontsize=fontsize,
                  grid=grid,
                  size=figsize)

    # Number of components              
    K = length(μ)

    # Ranges for contour plots
    xr = range(xlims[1], step=0.01, stop=xlims[2])
    yr = range(ylims[1], step=0.01, stop=ylims[2])

    # Loop over clusters
    for k = 1:K

        # Overlay the posterior probabilities for k-th cluster
        contour!(plt, xr, yr,
                 (x, y) -> prob_cluster(x, y, μ[k], Σ[k]),
                 color=colors[k],
                 fontsize=fontsize,
                 linewidth=linewidth,
                 cbar=nothing)
    end
    plot(plt)
end

