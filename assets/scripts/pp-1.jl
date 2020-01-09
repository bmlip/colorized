"""
Utility functions for Probabilistic Programming - 1
Wouter Kouw
Last update: 09-01-2020
"""

using SpecialFunctions: gamma
using Plots
pyplot()

function prob_normalgamma(μ::Number, τ::Number; m=0, l=1, a=1, b=1)
    "Compute probability of sample under Normal-Gamma distribution."
    return b^a*sqrt(l)/(gamma(a)*sqrt(2*pi)) * τ^(a - 1/2) * exp(-b*τ) * exp( -(l*τ*(μ - m)^2)/2)
end

function plot_normalgamma(m, l, a, b; x1=range(-5, stop=5), x2=range(0.1, stop=5), overlay=false)
    "Contour plot of distribution with given parameters."

    if overlay
        # Overlay the posterior probability
        contour!(x1, x2, (x, y) -> prob_normalgamma(x, y, m=m, l=l, a=a, b=b), cbar=nothing)
    else
        # Overlay the posterior probability
        contour(x1, x2, (x, y) -> prob_normalgamma(x, y, m=m, l=l, a=a, b=b), cbar=nothing)
    end        
    
    # Axis properties
    xlabel!("μ")
    ylabel!("τ")
end
