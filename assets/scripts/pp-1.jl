"""
Utility functions for Probabilistic Programming - 1
Wouter Kouw
Last update: 09-01-2020
"""

import SpecialFunctions: gamma
import Distributions: Normal, Gamma
import DataFrames: DataFrame, select!, insertcols!
import MCMCChains: Chains
import Plots: contour, contour!, xlabel!, ylabel!


function prob_normalgamma(μ::Number, τ::Number; m=0, l=1, a=1, b=1)
    "Compute probability of sample under Normal-Gamma distribution."
    return b^a*sqrt(l)/(gamma(a)*sqrt(2*pi)) * τ^(a - 1/2) * exp(-b*τ) * exp( -(l*τ*(μ - m)^2)/2)
end

function prob_normal_gamma(μ::Number, τ::Number; m=0, v=1, a=1, b=1)
    "Compute probability of sample under a bivariate [Normal, Gamma] distribution."

    # Probablity under Normal distribution (Distributions.jl uses mean-standard-deviation parameterization)
    pμ = 1/(sqrt(2*pi*v))*exp(-(μ - m)^2/(2*v))

    # Probablity under Gamma distribution (Distributions.jl uses shape-scale parameterization)
    pτ = b^a / gamma(a)*τ^(a-1)*exp(-b*τ)

    # Return product of marginal distributions
    return pμ * pτ
end

function plot_normalgamma(m, l, a, b;
                          label=nothing,
                          x1=range(-5, stop=5),
                          x2=range(0.1, stop=5),
                          overlay=false)
    "Contour plot of Normal-Gamma distribution with given parameters."

    if overlay
        # Overlay the posterior probability
        contour!(x1, x2, (x, y) -> prob_normalgamma(x, y, m=m, l=l, a=a, b=b), label=label, cbar=nothing)
    else
        # Overlay the posterior probability
        contour(x1, x2, (x, y) -> prob_normalgamma(x, y, m=m, l=l, a=a, b=b), label=label, cbar=nothing)
    end

    # Axis properties
    xlabel!("μ")
    ylabel!("τ")
end

function plot_normal_gamma(m_μ, v_μ, a_τ, b_τ;
                           label=nothing,
                           color=nothing,
                           x1=range(-5, stop=5),
                           x2=range(0.1, stop=5),
                           overlay=false)
    "Contour plot of bivariate [Normal, Gamma] distribution with given parameters."

    if overlay
        # Overlay the posterior probability
        contour!(x1, x2,
                 (x, y) -> prob_normal_gamma(x, y, m=m_μ, v=v_μ, a=a_τ, b=b_τ),
                 label=label,
                 color=color,
                 cbar=nothing)
    else
        # Overlay the posterior probability
        contour(x1, x2,
                (x, y) -> prob_normal_gamma(x, y, m=m_μ, v=v_μ, a=a_τ, b=b_τ),
                label=label,
                color=color,
                cbar=nothing)
    end

    # Axis properties
    xlabel!("μ")
    ylabel!("τ")
end

function plot_decisionboundary(weights;
                               label=nothing,
                               color="black",
                               x1=range(-4, step=0.01, stop=4),
                               x2=range(-4, step=0.01, stop=4),
                               overlay=true)
    "Contour plot of linear function"

    # Logistic function
    ϕ(x) = 1/(1 + exp(-x))

    if overlay
        contour!(x1, x2,
                 (x,y) -> ϕ(weights[1]*x + weights[2]*y + weights[3]) > 0.5,
                 label=label,
                 color=color,
                 cbar=nothing)
    else
        contour(x1, x2,
                (x,y) -> ϕ(weights[1]*x + weights[2]*y + weights[3]) > 0.5,
                label=label,
                color=color,
                cbar=nothing)
    end
end

function dfcol_str2int(df::DataFrame, colname::Symbol)
    "Map column of dataframe from type string to type integer."

    # Create dummy array
    dummy = zeros(size(df,1),)

    # Map string elements to integer
    for (i, element) in enumerate(unique(df[!, colname]))
        dummy[df[!,colname] .== element] .= i-1
    end

    # Remove column
    select!(df, Not(colname))

    # Insert dummy var as column
    insertcols!(df, size(df,2), (colname => dummy))

    return df
end

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"

    # Mean of chain
    mchain = mean(chain[param].value.data, dims=[1, 3])

    # Return as matrix
    return vec(convert(Array{Float64,2}, mchain[:,:,1]))
end
