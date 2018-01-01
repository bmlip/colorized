# Helper function for the PCA demo

function readDataSet(filename)
    f = open(filename);
    data = nothing
    for line in eachline(f)
        row = map((s) -> parse(Int, s), split(replace(line, "  ", " "), " "))
        data = (data == nothing) ? row' : vcat(data, row')
    end
    close(f)

    return data
end

type pPCAParams
    W::Matrix{Float64}
    m::Vector{Float64}
    v::Float64
end

function pPCA(X::Matrix, M::Int64)
    # Implement probabilistic PCA using the EM algorithm
    # X collects the data points on its columns
    # X may contain missing values, indicated by NaN

    # Probabilistic model:
    # x[n] = W*z[n] + m + ε, where W is a matrix, m is a vector and ε is a noise term
    # z[n] ~ N(0,I), where I is the identity matrix
    # ε ~ N(0, v*I), where v is a scalar

    N = size(X, 2); D = size(X, 1)
    missing_values = isnan(X)
    has_missing_values = sum(missing_values) > 0

    # Find row means of X, ignoring missing values
    if has_missing_values
        X[missing_values] = 0.0
        m = ( sum(X, 2) ./ (N*ones(D) - sum(missing_values,2)) )[:,1] # [:,1] makes sure this is a vector
        X[missing_values] = NaN
        observed_d = [find(!missing_values[:,n]) for n=1:N] # observed dimensions for every data vector
        observed_n = [find(!missing_values[d,:]) for d=1:D] # observed value indexes per dimension
        x_observed = [X[observed_d[n],n] for n=1:N]  # data vectors without the missing data
    else
        m = ( sum(X, 2) ./ N )[:,1] # [:,1] makes sure this is a vector
        _X = X .- m # shift data to be zero-mean
    end

    # Initialize variables
    θ = pPCAParams(Matrix{Float64}(D,M), m, 1.0) # θ holds the parameters
    Mz = randn(M,N)              # Mz[:,n] = MEAN[z[n]]
    Σz = [zeros(M,M) for n=1:N]  # Σz[n] = COV[z[n]]

    function eStep!(Mz, Σz)
        # Update Mz and Σz
        W = θ.W; v = θ.v

        if has_missing_values
            for n=1:N
                _W = W[observed_d[n],:]
                _x = x_observed[n] - θ.m[observed_d[n]]
                M_inv = inv(_W'*_W + v*eye(M))
                Σz[n] = v * M_inv
                Mz[:,n] = M_inv * _W' * _x
            end
        else
            M_inv = inv(W'*W + v*eye(M))
            Mz[:] = M_inv * W' * _X
            Σz[1] = v * M_inv # COV[z[n]] is equal for all n. We only update the one for n=1
        end
    end

    function mStep!(θ)
        # Update W, m, v
        if has_missing_values
            m = zero(θ.m)
            for n=1:N
                m[observed_d[n]] += x_observed[n] - (θ.W[observed_d[n],:] * Mz[:,n])[:,1]
            end
            m = m ./ (N - sum(missing_values, 2)[:,1])
            θ.m = m

            for d=1:D
                A = inv(Mz[:,observed_n[d]]*Mz[:,observed_n[d]]' + sum(Σz[observed_n[d]]))
                b = zeros(M)
                for n in observed_n[d]
                    b += Mz[:,n]*(X[d,n] - m[d])
                end
                θ.W[d,:] = ( A * b )'
            end
            W = θ.W

            θ.v = 0.0
            for n=1:N
                for d in observed_d[n]
                    θ.v +=  (X[d,n] - (W[d,:]'*Mz[:,n])[1] - m[d])^2 + (W[d,:]' * Σz[n] * W[d,:])[1]
                end
            end
            θ.v = θ.v / (N*D-sum(missing_values))

        else
            # no need to update m
            W = _X * Mz' * inv(Mz*Mz' + N*Σz[1])
            θ.W = W
            θ.v = sum((_X - W*Mz).^2)/(N*D) + trace(W*Σz[1]*W')/D
        end
    end

    for i=1:30 # Here should be a decent convergence check, but hey, this also works...
        mStep!(θ)
        eStep!(Mz, Σz)
    end

    return (θ, Mz) # Mz is actually redundant, but we return it for convenient plotting
end