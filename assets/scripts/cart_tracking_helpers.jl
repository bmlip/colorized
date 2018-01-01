include("linear_transition_model.jl")
using Distributions, PyPlot

function generateNoisyMeasurements( z_start::Vector{Float64},
                                    u::Vector{Float64},
                                    A::Matrix{Float64},
                                    b::Vector{Float64},
                                    Σz::Matrix{Float64},
                                    Σx::Matrix{Float64})
    # Simulate linear state-space model
    # z[t] = A*z[t-1] + b*u[t] + N(0,Σz)
    # x[t] = N(z[t],Σx)

    # Return noisy observations [x[1],...,x[n]]

    n = length(u)
    d = length(z_start)

    # Sanity checks
    (n>0) || error("u should contain at least one value")
    (d>0) || error("z_start has an invalid dimensionality")
    (size(A) == (d,d)) || error("Transition matrix A does not have correct dimensions")
    (length(b) == d) || error("b does not have the correct dimensionality")
    (size(Σz) == (d,d)) || error("Covariance matrix Σz does not have correct dimensions")
    (size(Σx) == (d,d)) || error("Covariance matrix Σx does not have correct dimensions")

    x = Vector[] # x will be a list of n vectors
    z = copy(z_start)
    proc_noise = MvNormal(zeros(d), Σz)
    for i=1:n
        z = A*z + b*u[i] + rand(proc_noise)
        push!(x, rand(MvNormal(z, Σx)))
    end

    return x
end

function plotCartPrediction(prediction::MvGaussian{2},
                            measurement::MvGaussian{2},
                            corr_prediction::MvGaussian{2})
    # Make a fancy plot of the Kalman-filtered cart position
    p = Normal(mean(prediction)[1], var(prediction)[1])
    m = Normal(mean(measurement)[1], var(measurement)[1])
    c = Normal(mean(corr_prediction)[1], var(corr_prediction)[1])

    plot_range = linspace(mean(c)-8*sqrt(var(p)), mean(c)+8*sqrt(var(p)), 300)
    PyPlot.figure(figsize=(15,5))
    bg_img = imread("figures/cart-bg.png")
    height = floor((plot_range[end] - plot_range[1])/3)
    imshow(bg_img, zorder=0, extent=[plot_range[1], plot_range[end], 0., height])
    plot(plot_range, height*Distributions.pdf(p, plot_range), "r-")
    plot(plot_range, height*Distributions.pdf(m, plot_range), "b-")
    plot(plot_range, height*Distributions.pdf(c, plot_range), "g-")
    legend(["Prediction "*L"p(z[n]|z[n-1],u[n])",
            "Noisy measurement "*L"p(z[n]|x[n])",
            "Corrected prediction "*L"p(z[n]|z[n-1],u[n],x[n])"], prop=Dict("size"=>14), loc=1)
    fill_between(plot_range, 0, height*Distributions.pdf(p, plot_range), color="r", alpha=0.1)
    fill_between(plot_range, 0, height*Distributions.pdf(m, plot_range), color="b", alpha=0.1)
    fill_between(plot_range, 0, height*Distributions.pdf(c, plot_range), color="g", alpha=0.1)
    xlim([plot_range[1],plot_range[end]]); ylim([0.,height])
    ax=gca()
    ax[:yaxis][:set_visible](false)
    xlabel("Position")
end