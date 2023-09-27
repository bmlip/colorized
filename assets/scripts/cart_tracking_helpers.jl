using Plots, Distributions, Images, LaTeXStrings

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
    chol_Σz = cholesky(Σz)
    chol_Σx = cholesky(Σx)
    proc_noise = convert(Matrix,chol_Σz) * randn(d)
    for i=1:n
        proc_noise = convert(Matrix,chol_Σz) * randn(d)
        z = A*z + b*u[i] + proc_noise
        obs_noise = convert(Matrix,chol_Σx) * randn(length(z))
        push!(x, z + obs_noise)
    end

    return x
end

function plotCartPrediction(predictive_mean, predictive_cov, mean, cov,measurement_position,measurement_cov)
    # Make a fancy plot of the Kalman-filtered cart position
    p = Normal(predictive_mean, predictive_cov)
    m = Normal(measurement_position, measurement_cov)
    c = Normal(mean, cov)
    plot_range = range(mean-8*sqrt(predictive_cov), mean+8*sqrt(predictive_cov), length=300)
    bg_img = load("figures/cart-bg.png")
    height = floor((plot_range[end] - plot_range[1])/3)
    sz = size(bg_img)
    x, y = LinRange(plot_range[begin], plot_range[end], sz[1]), collect(LinRange(0, 5, sz[2]))
    result = plot(x, y, bg_img, xlabel="Position")
    plot!(plot_range, height*pdf.(p, plot_range), label="Prediction "*L"p(z[n]|z[n-1],u[n])", fill=(0, .1))
    plot!(plot_range, height*pdf.(m, plot_range), label="Noisy measurement "*L"p(z[n]|x[n])", fill=(0, .1))
    plot!(plot_range, height*pdf.(c, plot_range), label="Corrected prediction "*L"p(z[n]|z[n-1],u[n],x[n])", xrange = (plot_range[begin], plot_range[end]), yrange=(-3, height), fill=(0, .1))
  
    return result
end
