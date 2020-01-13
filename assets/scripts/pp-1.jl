"""
Utility functions for Probabilistic Programming - 1
Wouter Kouw
Last update: 09-01-2020
"""

using SpecialFunctions: gamma
using Distributions: Normal, Gamma
using Plots
pyplot()

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

function plot_normalgamma(m, l, a, b; label=nothing, x1=range(-5, stop=5), x2=range(0.1, stop=5), overlay=false)
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

function plot_normal_gamma(m_μ, v_μ, a_τ, b_τ; label=nothing, color=nothing, x1=range(-5, stop=5), x2=range(0.1, stop=5), overlay=false)
    "Contour plot of bivariate [Normal, Gamma] distribution with given parameters."

    if overlay
        # Overlay the posterior probability
        contour!(x1, x2, (x, y) -> prob_normal_gamma(x, y, m=m_μ, v=v_μ, a=a_τ, b=b_τ), label=label, color=color, cbar=nothing)
    else
        # Overlay the posterior probability
        contour(x1, x2, (x, y) -> prob_normal_gamma(x, y, m=m_μ, v=v_μ, a=a_τ, b=b_τ), label=label, color=color, cbar=nothing)
    end        
    
    # Axis properties
    xlabel!("μ")
    ylabel!("τ")
end

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"
    return vec(convert(Array{Float64,2}, mean(chain[param].value.data, dims=[1, 3])[:,:,1]))
end;
