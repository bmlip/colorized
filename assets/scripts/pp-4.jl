"""
Utility functions for Probabilistic Programming - 4
Wouter Kouw
Last update: 27-01-2020
"""

using MCMCChains: Chains
using Plots
pyplot()

function mean_chain(chain::Chains, param::Symbol)
    "Quick function to extract sample average from MCMCChains object"
    return mean(chain[param].value.data, dims=[1])
end;
