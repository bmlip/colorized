### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 23c689fc-d294-11ef-086e-47c4f871bed2
md"""
# Generative Classification

"""

# ╔═╡ 23c6997e-d294-11ef-09a8-a50563e5975b
md"""
## Preliminaries

Goal 

  * Introduction to linear generative classification with a Gaussian-categorical generative model

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 196-202 (section 4.2 focusses on binary classification, whereas in these lecture notes we describe generative classification for multiple classes).

"""

# ╔═╡ 23c6afea-d294-11ef-1264-f7af4807f23f
md"""
## Challenge: an apple or a peach?

**Problem**: You're given numerical values for the skin features roughness and color for 200 pieces of fruit, where for each piece of fruit you also know if it is an apple or a peach. Now you receive the roughness and color values for a new piece of fruit but you don't get its class label (apple or peach). What is the probability that the new piece is an apple?

"""

# ╔═╡ 23c6b99a-d294-11ef-3072-83e3233746f7
md"""
**Solution**: To be solved later in this lesson.

Let's first generate a data set (see next slide).

"""

# ╔═╡ 23c70206-d294-11ef-3a4e-e59279edb052
using Plots, Distributions

N = 250; p_apple = 0.7; Σ = [0.2 0.1; 0.1 0.3]
p_given_apple = MvNormal([1.0, 1.0], Σ)                         # p(X|y=apple)
p_given_peach = MvNormal([1.7, 2.5], Σ)                         # p(X|y=peach)
X = Matrix{Float64}(undef,2,N); y = Vector{Bool}(undef,N)       # true corresponds to apple
for n=1:N
    y[n] = (rand() < p_apple)                                   # Apple or peach?
    X[:,n] = y[n] ? rand(p_given_apple) : rand(p_given_peach)   # Sample features
end
X_apples = X[:,findall(y)]'; X_peaches = X[:,findall(.!y)]'     # Sort features on class
x_test = [2.3; 1.5]                                             # Features of 'new' data point

scatter(X_apples[:,1], X_apples[:,2], label="apples", marker=:x, markerstrokewidth=3)       # apples
scatter!(X_peaches[:,1], X_peaches[:,2], label="peaches", marker=:+,  markerstrokewidth=3)  # peaches
scatter!([x_test[1]], [x_test[2]], label="unknown")                                         # 'new' unlabelled data

# ╔═╡ 23c719bc-d294-11ef-26ff-71350fa27678
md"""
## Generative Classification Problem Statement

Given is a data set  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}``

  * inputs ``x_n \in \mathbb{R}^M`` are called **features**.
  * outputs ``y_n \in \mathcal{C}_k``, with ``k=1,\ldots,K``; The **discrete** targets ``\mathcal{C}_k`` are called **classes**.

"""

# ╔═╡ 23c7236c-d294-11ef-2388-254850d0e76e
md"""
We will again use the 1-of-``K`` notation for the discrete classes. Define the binary **class selection variable**

```math
y_{nk} = \begin{cases} 1 & \text{if  } \, y_n \in \mathcal{C}_k\\
0 & \text{otherwise} \end{cases}
```

(Hence, the notations ``y_{nk}=1`` and ``y_n \in \mathcal{C}_k`` mean the same thing.)

"""

# ╔═╡ 23c73302-d294-11ef-0c12-571686b202a9
md"""
The plan for generative classification: build a model for the joint pdf ``p(x,y)= p(x|y)p(y)`` and use Bayes to infer the posterior class probabilities 

```math
p(y|x) = \frac{p(x|y) p(y)}{\sum_{y^\prime} p(x|y^\prime) p(y^\prime)} \propto p(x|y)\,p(y)
```

"""

# ╔═╡ 23c73b54-d294-11ef-0ef8-8d9159139a1b
md"""
## 1 - Model specification

#### Likelihood

Assume Gaussian **class-conditional distributions** with **equal covariance matrix** across the classes,

```math
 p(x_n|\mathcal{C}_{k}) = \mathcal{N}(x_n|\mu_k,\Sigma)
 
```

with notational shorthand: ``\mathcal{C}_{k} \triangleq (y_n \in \mathcal{C}_{k})``.

"""

# ╔═╡ 23c74748-d294-11ef-2170-bf45b6379e4d
md"""
#### Prior

We use a categorical distribution for the class labels ``y_{nk}``: 

```math
p(\mathcal{C}_{k}) = \pi_k
```

"""

# ╔═╡ 23c74f18-d294-11ef-3058-a53b3f1482fb
md"""
Hence, using the one-hot coding formulation for ``y_{nk}``, the generative model ``p(x_n,y_n)`` can be written as

```math
\begin{align*}
 p(x_n,y_n) &= \prod_{k=1}^K p(x_n,y_{nk}=1)^{y_{nk}} \\
   &= \prod_{k=1}^K \left( \pi_k \cdot\mathcal{N}(x_n|\mu_k,\Sigma)\right)^{y_{nk}}
\end{align*}
```

"""

# ╔═╡ 23c75dc8-d294-11ef-3c57-614e75f06d8f
md"""
We will refer to this model as the **Gaussian-Categorical Model** ($(HTML("<span id='GCM'>GCM</span>"))). 

  * N.B. In the literature, this model (with possibly unequal ``\Sigma_k`` across classes) is often called the Gaussian Discriminant Analysis  model and the special case with equal covariance matrices ``\Sigma_k=\Sigma`` is also called Linear Discriminant Analysis. We think these names are a bit unfortunate as it may lead to confusion with the [discriminative method for classification](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Discriminative-Classification.ipynb).

"""

# ╔═╡ 23c763ce-d294-11ef-015b-736be1a5e9d6
md"""
As usual, once the model has been specified, the rest (inference for parameters and model prediction) through straight probability theory.

"""

# ╔═╡ 23c77196-d294-11ef-379b-cdf1f31a0994
md"""
## Computing the log-likelihood

The $(HTML("<span id='generative-classification-llh'>log-likelihood</span>")) given the full data set ``D=\{(x_n,y_n), n=1,2,\ldots,N\}`` is then

```math
\begin{align*}
\log\, p(D|\theta) &\stackrel{\text{IID}}{=} \sum_n \log \prod_k p(x_n,y_{nk}=1\,|\,\theta)^{y_{nk}}  \\
  &=  \sum_{n,k} y_{nk} \log p(x_n,y_{nk}=1\,|\,\theta) \\
     &=  \sum_{n,k} y_{nk}  \log p(x_n|y_{nk}=1)  +  \sum_{n,k} y_{nk} \log p(y_{nk}=1) \\
   &=  \sum_{n,k} y_{nk}  \log\mathcal{N}(x_n|\mu_k,\Sigma)  +  \sum_{n,k} y_{nk} \log \pi_k \\
   &=  \sum_{n,k} y_{nk} \underbrace{ \log\mathcal{N}(x_n|\mu_k,\Sigma) }_{ \text{see Gaussian lecture} } + \underbrace{ \sum_k m_k \log \pi_k }_{ \text{see multinomial lecture} } 
\end{align*}
```

where we used ``m_k \triangleq \sum_n y_{nk}``.

"""

# ╔═╡ 23c7779a-d294-11ef-2e2c-6ba6cadb1381
md"""
## 2 -  Parameter Inference for Classification

We'll do Maximum Likelihood estimation for ``\theta = \{ \pi_k, \mu_k, \Sigma \}`` from data ``D``.

"""

# ╔═╡ 23c78316-d294-11ef-3b6e-d1bdd24620d0
md"""
Recall (from the previous slide) the log-likelihood (LLH)

```math
\log\, p(D|\theta) =  \sum_{n,k} y_{nk} \underbrace{ \log\mathcal{N}(x_n|\mu_k,\Sigma) }_{ \text{Gaussian} } + \underbrace{ \sum_k m_k \log \pi_k }_{ \text{multinomial} } 
```

"""

# ╔═╡ 23c78d3e-d294-11ef-0309-ff10f58f0252
md"""
Maximization of the LLH for the GDA model breaks down into

  * **Gaussian density estimation** for parameters ``\mu_k, \Sigma``, since the first term contains exactly the log-likelihood for MVG density estimation. We've already done this, see the [Gaussian distribution lesson](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Gaussian-Distribution.ipynb#ML-for-Gaussian).
  * **Multinomial density estimation** for class priors ``\pi_k``, since the second term holds exactly the log-likelihood for multinomial density estimation, see the [Multinomial distribution lesson](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Multinomial-Distribution.ipynb#ML-for-multinomial).

"""

# ╔═╡ 23c798ce-d294-11ef-0190-f342f30e2266
md"""
The ML for multinomial class prior (we've done this before!)

```math
\begin{align*}   
\hat \pi_k = \frac{m_k}{N} 
\end{align*}
```

"""

# ╔═╡ 23c7a54c-d294-11ef-0252-ef7a043e995c
md"""
Now group the data into separate classes and do MVG ML estimation for class-conditional parameters (we've done this before as well):

```math
\begin{align*}
 \hat \mu_k &= \frac{ \sum_n y_{nk} x_n} { \sum_n y_{nk} } = \frac{1}{m_k} \sum_n y_{nk} x_n \\
 \hat \Sigma  &= \frac{1}{N} \sum_{n,k} y_{nk} (x_n-\hat \mu_k)(x_n-\hat \mu_k)^T \\
  &= \sum_k \hat \pi_k \cdot \underbrace{ \left( \frac{1}{m_k} \sum_{n} y_{nk} (x_n-\hat \mu_k)(x_n-\hat \mu_k)^T  \right) }_{ \text{class-cond. variance} } \\
  &= \sum_k \hat \pi_k \cdot \hat \Sigma_k
\end{align*}
```

where ``\hat \pi_k``, ``\hat{\mu}_k`` and ``\hat{\Sigma}_k`` are the sample proportion, sample mean and sample variance for the ``k``th class, respectively.

"""

# ╔═╡ 23c7ab20-d294-11ef-1926-afae49e79923
md"""
Note that the binary class selection variable ``y_{nk}`` groups data from the same class.

"""

# ╔═╡ 23c7baa4-d294-11ef-22c1-31b0d86f5586
md"""
## 3 - Application: Class prediction for new Data

Let's apply the trained model to predict the class for given a 'new' input ``x_\bullet``:

```math
\begin{align*}
p(\mathcal{C}_k|x_\bullet,D ) &= \int p(\mathcal{C}_k|x_\bullet,\theta ) \underbrace{p(\theta|D)}_{\text{ML: }\delta(\theta - \hat{\theta})} \mathrm{d}\theta \\
&= p(\mathcal{C}_k|x_\bullet,\hat{\theta} ) \\
&\propto p(\mathcal{C}_k)\,p(x_\bullet|\mathcal{C}_k) \\
&= \hat{\pi}_k \cdot \mathcal{N}(x_\bullet | \hat{\mu}_k, \hat{\Sigma}) \\
  &\propto \hat{\pi}_k \exp \left\{ { - {\frac{1}{2}}(x_\bullet - \hat{\mu}_k )^T \hat{\Sigma}^{ - 1} (x_\bullet - \hat{\mu}_k )} \right\}\\
  &=\exp \Big\{ \underbrace{-\frac{1}{2}x_\bullet^T \hat{\Sigma}^{ - 1} x_\bullet}_{\text{not a function of }k} + \underbrace{\hat{\mu}_k^T \hat{\Sigma}^{-1}}_{\beta_k^T} x_\bullet \underbrace{- {\frac{1}{2}}\hat{\mu}_k^T \hat{\Sigma}^{ - 1} \hat{\mu}_k  + \log \hat{\pi}_k }_{\gamma_k} \Big\}  \\
  &\propto  \frac{1}{Z}\exp\{\beta_k^T x_\bullet + \gamma_k\} \\
  &\triangleq \sigma\left( \beta_k^T x_\bullet + \gamma_k\right)
\end{align*}
```

where  ``\sigma(a_k) \triangleq \frac{\exp(a_k)}{\sum_{k^\prime}\exp(a_{k^\prime})}`` is $(HTML("<span id='softmax'>called a</span>")) [**softmax**](https://en.wikipedia.org/wiki/Softmax_function) (a.k.a. **normalized exponential**) function, and

```math
\begin{align*}
\beta_k &= \hat{\Sigma}^{-1} \hat{\mu}_k \\
\gamma_k &= - \frac{1}{2} \hat{\mu}_k^T \hat{\Sigma}^{-1} \hat{\mu}_k  + \log \hat{\pi}_k \\
Z &= \sum_{k^\prime}\exp\{\beta_{k^\prime}^T x_\bullet + \gamma_{k^\prime}\}\,. \quad \text{(normalization constant)} 
\end{align*}
```

"""

# ╔═╡ 23c7c920-d294-11ef-1b6d-d98dd54dcbe3
md"""
The softmax function is a smooth approximation to the max-function. Note that we did not a priori specify a softmax posterior, but rather it followed from applying Bayes rule to the prior and likelihood assumptions. 

"""

# ╔═╡ 23c7d700-d294-11ef-1268-c1441a3301a4
md"""
Note the following properties of the softmax function ``\sigma(a_k)``:

  * ```math
    \sigma(a_k)
    ```

    is monotonicaly ascending function and hence it preserves the order of ``a_k``. That is, if ``a_j>a_k`` then ``\sigma(a_j)>\sigma(a_k)``.
  * ```math
    \sigma(a_k)
    ```

    is always a proper probability distribution, since ``\sigma(a_k)>0`` and ``\sum_k \sigma(a_k) = 1``.

"""

# ╔═╡ 23c7e4a0-d294-11ef-16e9-6f96a41baf97
md"""
## Discrimination Boundaries

The class log-posterior ``\log p(\mathcal{C}_k|x) \propto \beta_k^T x + \gamma_k`` is a linear function of the input features.

"""

# ╔═╡ 23c7f170-d294-11ef-1340-fbdf4ce5fd44
md"""
Thus, the contours of equal probability (**discriminant functions**) are lines (hyperplanes) in the feature space

```math
\log \frac{{p(\mathcal{C}_k|x,\theta )}}{{p(\mathcal{C}_j|x,\theta )}} = \beta_{kj}^T x + \gamma_{kj} = 0
```

where we defined ``\beta_{kj} \triangleq \beta_k - \beta_j`` and similarly for ``\gamma_{kj}``.

"""

# ╔═╡ 23c82154-d294-11ef-0945-c9c94fc2a44d
md"""
How to classify a new input ``x_\bullet``? The Bayesian answer is a posterior distribution $ p(\mathcal{C}*k|x*\bullet)$. If you must choose, then the class with maximum posterior class probability

```math
\begin{align*}
k^* &= \arg\max_k p(\mathcal{C}_k|x_\bullet) \\
  &= \arg\max_k \left( \beta _k^T x_\bullet + \gamma_k \right)
\end{align*}
```

is an appealing decision. 

"""

# ╔═╡ 23c82e10-d294-11ef-286a-ff6fee0f2805
md"""
## $(HTML("<span id='code-generative-classification-example'>Code Example</span>")):  Working out the "apple or peach" example problem

We'll apply the above results to solve the "apple or peach" example problem.

"""

# ╔═╡ 23c846e8-d294-11ef-3be4-c79885446a0a
# Make sure you run the data-generating code cell first
using Distributions, Plots
# Multinomial (in this case binomial) density estimation
p_apple_est = sum(y.==true) / length(y)
π_hat = [p_apple_est; 1-p_apple_est]

# Estimate class-conditional multivariate Gaussian densities
d1 = fit_mle(FullNormal, X_apples')  # MLE density estimation d1 = N(μ₁, Σ₁)
d2 = fit_mle(FullNormal, X_peaches') # MLE density estimation d2 = N(μ₂, Σ₂)
Σ = π_hat[1]*cov(d1) + π_hat[2]*cov(d2) # Combine Σ₁ and Σ₂ into Σ
conditionals = [MvNormal(mean(d1), Σ); MvNormal(mean(d2), Σ)] # p(x|C)

# Calculate posterior class probability of x∙ (prediction)
function predict_class(k, X) # calculate p(Ck|X)
    norm = π_hat[1]*pdf(conditionals[1],X) + π_hat[2]*pdf(conditionals[2],X)
    return π_hat[k]*pdf(conditionals[k], X) ./ norm
end
println("p(apple|x=x∙) = $(predict_class(1,x_test))")

# Discrimination boundary of the posterior (p(apple|x;D) = p(peach|x;D) = 0.5)
β(k) = inv(Σ)*mean(conditionals[k])
γ(k) = -0.5 * mean(conditionals[k])' * inv(Σ) * mean(conditionals[k]) + log(π_hat[k])
function discriminant_x2(x1)
    # Solve discriminant equation for x2
    β12 = β(1) .- β(2)
    γ12 = (γ(1) .- γ(2))[1,1]
    return -1*(β12[1]*x1 .+ γ12) ./ β12[2]
end

scatter(X_apples[:,1], X_apples[:,2], label="apples", marker=:x, markerstrokewidth=3)   # apples
scatter!(X_peaches[:,1], X_peaches[:,2], label="peaches", marker=:+,  markerstrokewidth=3) # peaches
scatter!([x_test[1]], [x_test[2]], label="unknown")           # 'new' unlabelled data point

x1 = range(-1,length=10,stop=3)
plot!(x1, discriminant_x2(x1), color="black", label="") # Plot discrimination boundary
plot!(x1, discriminant_x2(x1), fillrange=-10, alpha=0.2, color=:blue, label="")
plot!(x1, discriminant_x2(x1), fillrange=10, alpha=0.2, color=:red, xlims=(-0.5, 3), ylims=(-1, 4), label="")

# ╔═╡ 23c85d90-d294-11ef-375e-7101d4d3cbfa
md"""
## Why Be Bayesian?

A student in one of the previous years posed the following question at Piazza: 

> " After re-reading topics regarding generative classification, this question popped into my mind: Besides the sole purpose of the lecture, which is getting to know the concepts of generative classification and how to implement them, are there any advantages of using this instead of using deep neural nets? DNNs seem simpler and more powerful."


The following answer was provided: 

  * If you are only are interested in approximating a function, say ``y=f_\theta(x)``, and you have lots of examples ``\{(x_i,y_i)\}`` of desired behavior, then often a non-probabilistic DNN is a fine approach.
  * However, if you are willing to formulate your models in a probabilistic framework, you can improve on the deterministic approach in many ways, eg,

> 1. Bayesian evidence for model performance assessment. This means you can use the whole data set for training without an ad-hoc split into testing and training data sets.


> 2. Uncertainty about parameters in the model is a measure that allows you to do *active learning*, ie, choose data that is most informative (see also the [lesson on intelligent agents](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Intelligent-Agents-and-Active-Inference.ipynb)). This will allow you to train on small data sets, whereas the deterministic DNNs generally require much larger data sets.


> 3. Prediction with uncertainty/confidence bounds.


> 4. Explicit specification and separation of your assumptions.


> 5. A framework that supports scoring both accuracy and model complexity in the same currency (probability). How are you going to penalize the size of your network in a deterministic framework?


> 6. Automatic learning rates, no tuning parameters. For instance, the Kalman gain is a data-dependent, optimal learning rate. How will *you* choose your learning rates in a deterministic framework? Trial and error?


> 7. Principled absorption of different sources of knowledge. Eg, outcome of one set of experiments can be captured by a posterior distribution that serves as a prior distribution for the next set of experiments.


> Admittedly, it's not easy to understand the probabilistic approach, but it is worth the effort.


"""

# ╔═╡ 23c8698e-d294-11ef-2ae8-83bebd89d6c0
md"""
## Recap Generative Classification

Gaussian-Categorical Model specification:  

```math
p(x,\mathcal{C}_k|\,\theta) = \pi_k \cdot \mathcal{N}(x|\mu_k,\Sigma)
```

"""

# ╔═╡ 23c87654-d294-11ef-3aaf-595b207054a5
md"""
If the class-conditional distributions are Gaussian with equal covariance matrices across classes (``\Sigma_k = \Sigma``), then   the discriminant functions are hyperplanes in feature space.

"""

# ╔═╡ 23c88284-d294-11ef-113b-f57800a10e5d
md"""
ML estimation for ``\{\pi_k,\mu_k,\Sigma\}`` in the GCM model breaks down to simple density estimation for Gaussian and multinomial/categorical distributions.

"""

# ╔═╡ 23c88ec8-d294-11ef-3e0d-8de1377a14bf
md"""
Posterior class probability is a softmax function

```math
 p(\mathcal{C}_k|x,\theta ) \propto \exp\{\beta_k^T x + \gamma_k\}
```

where ``\beta _k= \Sigma^{-1} \mu_k`` and ``\gamma_k=- \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k  + \log \pi_k``.

"""

# ╔═╡ 23c8a0d6-d294-11ef-1d1b-2dca4796f411
open("../../styles/aipstyle.html") do f
    display("text/html", read(f,String))
end

# ╔═╡ Cell order:
# ╟─23c689fc-d294-11ef-086e-47c4f871bed2
# ╟─23c6997e-d294-11ef-09a8-a50563e5975b
# ╟─23c6afea-d294-11ef-1264-f7af4807f23f
# ╟─23c6b99a-d294-11ef-3072-83e3233746f7
# ╠═23c70206-d294-11ef-3a4e-e59279edb052
# ╟─23c719bc-d294-11ef-26ff-71350fa27678
# ╟─23c7236c-d294-11ef-2388-254850d0e76e
# ╟─23c73302-d294-11ef-0c12-571686b202a9
# ╟─23c73b54-d294-11ef-0ef8-8d9159139a1b
# ╟─23c74748-d294-11ef-2170-bf45b6379e4d
# ╟─23c74f18-d294-11ef-3058-a53b3f1482fb
# ╟─23c75dc8-d294-11ef-3c57-614e75f06d8f
# ╟─23c763ce-d294-11ef-015b-736be1a5e9d6
# ╟─23c77196-d294-11ef-379b-cdf1f31a0994
# ╟─23c7779a-d294-11ef-2e2c-6ba6cadb1381
# ╟─23c78316-d294-11ef-3b6e-d1bdd24620d0
# ╟─23c78d3e-d294-11ef-0309-ff10f58f0252
# ╟─23c798ce-d294-11ef-0190-f342f30e2266
# ╟─23c7a54c-d294-11ef-0252-ef7a043e995c
# ╟─23c7ab20-d294-11ef-1926-afae49e79923
# ╟─23c7baa4-d294-11ef-22c1-31b0d86f5586
# ╟─23c7c920-d294-11ef-1b6d-d98dd54dcbe3
# ╟─23c7d700-d294-11ef-1268-c1441a3301a4
# ╟─23c7e4a0-d294-11ef-16e9-6f96a41baf97
# ╟─23c7f170-d294-11ef-1340-fbdf4ce5fd44
# ╟─23c82154-d294-11ef-0945-c9c94fc2a44d
# ╟─23c82e10-d294-11ef-286a-ff6fee0f2805
# ╠═23c846e8-d294-11ef-3be4-c79885446a0a
# ╟─23c85d90-d294-11ef-375e-7101d4d3cbfa
# ╟─23c8698e-d294-11ef-2ae8-83bebd89d6c0
# ╟─23c87654-d294-11ef-3aaf-595b207054a5
# ╟─23c88284-d294-11ef-113b-f57800a10e5d
# ╟─23c88ec8-d294-11ef-3e0d-8de1377a14bf
# ╠═23c8a0d6-d294-11ef-1d1b-2dca4796f411
