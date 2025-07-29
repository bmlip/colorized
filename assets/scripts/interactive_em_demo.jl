# Interactive Expectation Maximization demo for the 'old faithful' data set
# This script should be executed from the commandline

using DataFrames, CSV
include("gmm_plot.jl")

function continueOrExit()
    # Helper function: wait for keypress, quit on q
    if (readline() == "q") exit() end
end

old_faithful = CSV.read("../datasets/old_faithful.csv")
X = convert(Matrix{Float64}, [old_faithful[1] old_faithful[2]]') # Every column is a data point
N = size(X, 2)

# Initialization
#clusters = MvNormal[MvNormal([3.;50.].*rand(2)+[2.;40.], diagm([.5;10^2])); MvNormal([3.;50.].*rand(2)+[2.;40.], diagm([.5;10^2]))]
clusters = [MvNormal([4.;60.], diagm([.5;10^2]));
            MvNormal([2.;80.], diagm([.5;10^2]))]
π_hat = [0.5;0.5]
γ = fill!(Matrix{Float64}(2,N), NaN)

ion()
plotGMM(X, clusters, γ)
print("Press [enter] for E-step or [q] to quit: "); continueOrExit()

while(true)
    # E-step: update γ
    norm = [pdf(clusters[1], X) pdf(clusters[2], X)] * π_hat
    γ[1,:] = (π_hat[1] * pdf(clusters[1],X) ./ norm)'
    γ[2,:] = 1 - γ[1,:]
    clf(); plotGMM(X, clusters, γ)
    print("Press [enter] for M-step or [q] to quit: "); continueOrExit()

    # M-step: update cluster parameters and π_hat
    m = sum(γ, 2)
    π_hat = m / N
    μ_hat = (X * γ') ./ m'
    for k=1:2
        μ_k = μ_hat[:,k]
        Z = (X .- μ_k)
        Σ_k = Hermitian(((Z .* (γ[k,:])') * Z') / m[k])
        clusters[k] = MvNormal(μ_k, convert(Matrix, Σ_k))
    end
    clf(); plotGMM(X, clusters, γ)
    print("Press [enter] for E-step or [q] to quit: "); continueOrExit()
end