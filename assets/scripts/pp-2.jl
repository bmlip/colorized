"""
Utility functions for Probabilistic Programming - 2
Wouter Kouw
Last update: 07-01-2020
"""

import Turing: logaddexp
import Distributions: MvNormal, logpdf
using Plots
pyplot()

function predict(x, y, pi, μ1, μ2, Σ1, Σ2)
    "Compute posterior probabilities of samples under model."

    # Probability under cluster 1
    logp1 = log(pi[1]) + logpdf(MvNormal(μ1, Σ1), [x, y])

    # Probability under cluster 2
    logp2 = log(pi[2]) + logpdf(MvNormal(μ2, Σ2), [x, y])

    # Add probabilities using log-sum-exp trick for numeric stability.
    return logaddexp(logp1, logp2)
end

function plot_posteriors(X, μ1, μ2, Σ1, Σ2; x1=range(-5, stop=3), x2=range(-5, stop=4))
    "Scatter data and overlay posterior probabilities via contour plot."

    # Show the data
    scatter(X[:,1], X[:,2], legend=false, title="Synthetic Dataset")

    # Overlay the posterior probability
    contour!(x1, x2, (x, y) -> predict(x, y, [0.5, 0.5], μ1, μ2, Σ1, Σ2))
end
