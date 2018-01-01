# Helper function for lesson 8 lecture notes.
using Distributions

function genDataset(N::Int64)
    # Generate dataset {(x1,y1),...,(xN,yN)}
    # x is a 2d feature vector [x1;x2]
    # y ∈ {false,true} is a binary class label
    # p(x|y) is multi-modal (mixture of uniform and Gaussian distributions)
    X = Matrix{Float64}(2,N); y = Vector{Bool}(N)
    for n=1:N
        if (y[n]=(rand()>0.6)) # p(y=true) = 0.6
            # Sample class 1 conditional distribution
            if rand()<0.5
                X[:,n] = [6.0; 0.5] .* rand(2) + [3.0; 6.0]
            else
                X[:,n] = sqrt(0.5) * randn(2) + [5.5, 0.0]
            end
        else
            # Sample class 2 conditional distribution
            X[:,n] = randn(2) + [1., 4.]
        end
    end

    return (X, y)
end

function buildGenerativeDiscriminationBoundary(X::Matrix, y::Vector{Bool})
    # Generate discrimination boundary function x[2] = boundary(x[1]) for a Gaussian generative model:
    # X = [x_1,...,x_N]
    # y = [y_1;...;y_N]
    # x is a 2-d real (feature) vector
    # y ∈ {false,true}
    # x|y ~ 𝓝(x|μ_y, Σ_y)
    # We find the class-conditional Gaussian distributions by MLE
    # See lesson 7 (generative classification) for more details
    (size(X,1)==2) || error("The columns of X should have length 2")

    # MLE of p(y)
    p_1_est = sum(y.==true) / length(y)
    π_hat = [p_1_est; 1-p_1_est]

    # MLE of class-conditional multivariate Gaussian densities
    X_cls1 = X[:,y.==true]
    X_cls2 = X[:,y.==false]
    d1 = fit_mle(FullNormal, X_cls1)  # MLE density estimation d1 = N(μ₁, Σ₁)
    d2 = fit_mle(FullNormal, X_cls2)  # MLE density estimation d2 = N(μ₂, Σ₂)
    Σ = π_hat[1]*cov(d1) + π_hat[2]*cov(d2) # Combine Σ₁ and Σ₂ into Σ

    conditionals = [MvNormal(mean(d1), Σ); MvNormal(mean(d2), Σ)] # p(x|C)

    # Discrimination boundary of the posterior (p(apple|x;D) = p(peach|x;D) = 0.5)
    β(k) = inv(Σ)*mean(conditionals[k])
    γ(k) = -0.5 * mean(conditionals[k])' * inv(Σ) * mean(conditionals[k]) + log(π_hat[k])
    function discriminant_x2(x1)
        # Solve discriminant equation for x2
        β12 = β(1) - β(2)
        γ12 = (γ(1) - γ(2))[1,1]
        return -1*(β12[1]*x1 + γ12) ./ β12[2]
    end

    return discriminant_x2
end