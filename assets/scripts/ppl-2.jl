import Turing: logaddexp
import Distributions: MvNormal, logpdf
using Plots
pyplot()

function predict(x, y, w, μ1, μ2)
    "Compute posterior probabilities of samples under model"

    # Probability under cluster 1
    logp1 = log(w[1]) + logpdf(MvNormal(μ1, 1.), [x, y])

    # Probability under cluster 2
    logp2 = log(w[2]) + logpdf(MvNormal(μ2, 1.), [x, y])

    # Add probabilities using log-sum-exp trick for numeric stability.
    return logaddexp(logp1, logp2)
end

function plot_posteriors(X, μ1_estimated, μ2_estimated)

    # Show the data
    scatter(X[1,:], X[2,:], legend=false, title="Synthetic Dataset")

    # Overlay the posterior probability
    x1 = range(-5, stop=3)
    x2 = range(-5, stop=4)
    contour!(x1, x2, (x, y) -> predict(x, y, [0.5, 0.5], μ1_estimated, μ2_estimated))
    # colorbar()
end
