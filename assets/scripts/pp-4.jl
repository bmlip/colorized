"""
Utility functions for Probabilistic Programming - 4
Wouter Kouw
Last update: 27-01-2020
"""

using MCMCChains: Chains
using ForneyLab
using Plots
pyplot()

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"
    return mean(chain[param].value.data, dims=[1])
end

function plot_messages(state_prediction::ProbabilityDistribution{Univariate, GaussianMeanVariance},
                       measurement_likelihood::ProbabilityDistribution{Univariate, GaussianMeanVariance},
                       corrected_state_prediction::ProbabilityDistribution{Univariate, GaussianMeanVariance})
    
    # Extract parameters
    m_p = mean(state_prediction)
    v_p = var(state_prediction)
    m_l = mean(measurement_likelihood)
    v_l = var(measurement_likelihood)
    m_c = mean(corrected_state_prediction)
    v_c = var(corrected_state_prediction)
    
    # Set plot range
    range_p = [m_p - 8*sqrt(v_p), m_p + 8*sqrt(v_p)]
    range_l = [m_l - 8*sqrt(v_l), m_l + 8*sqrt(v_l)]
    range_c = [m_c - 8*sqrt(v_c), m_c + 8*sqrt(v_c)]    
    plot_range = range(minimum([range_p[1], range_l[1], range_c[1]]), 
                       maximum([range_p[2], range_l[2], range_c[2]]), length=300)
    
    # Probabilities over range
    probs_p = [pdf(state_prediction, a) for a in plot_range]
    probs_l = [pdf(measurement_likelihood, a) for a in plot_range]
    probs_c = [pdf(corrected_state_prediction, a) for a in plot_range]
    
    # Start plotting
    plot(plot_range, probs_p, color="red", label="state prediction", size=(800,500))
    plot!(plot_range, probs_l, color="blue", label="measurement likelihood")
    plot!(plot_range, probs_c, color="green", label="corrected state prediction")
    xlabel!("x-coordinate")
    ylabel!("Probability")
end

function pdf(dist::ProbabilityDistribution{Univariate, GaussianMeanVariance}, x::Number)
    return exp(-0.5*(log(2*pi) + log(dist.params[:v]) + (x - dist.params[:m])^2 / dist.params[:v]))
end;
