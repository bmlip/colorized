### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a759653c-0da4-40b7-9e9e-1e3d2e4df4ea
using Random

# ╔═╡ ae2a65fa-1322-43b1-80cf-ee3ad1c47312
using Plots, LaTeXStrings

# ╔═╡ 6a20aa94-e2fa-45ab-9889-62d44cbfc1ba
using Optim # Optimization library

# ╔═╡ 616e84d7-063d-4d9d-99e4-56aecf3c7ee4
using Distributions

# ╔═╡ 25eefb10-d294-11ef-0734-2daf18636e8e
md"""
# Discriminative Classification

"""

# ╔═╡ 25ef12bc-d294-11ef-1557-d98ba829a804
md"""
## Preliminaries

Goal 

  * Introduction to discriminative classification models

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 213 - 217 (Laplace approximation)
      * Bishop pp. 217 - 220 (Bayesian logistic regression)
      * [T. Minka (2005), Discriminative models, not discriminative training](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Minka-2005-Discriminative-models-not-discriminative-training.pdf)

"""

# ╔═╡ 25ef2806-d294-11ef-3cb6-0f3e76b9177e
md"""
## Challenge: difficult class-conditional data distributions

Our task will be the same as in the preceding class on (generative) classification. But this time, the class-conditional data distributions look very non-Gaussian, yet the linear discriminative boundary looks easy enough:

"""

# ╔═╡ ad6b7f43-ccae-4b85-bd6e-051cd4d771cd
# Generate dataset {(x1,y1),...,(xN,yN)}
# x is a 2-d feature vector [x_1;x_2]
# y ∈ {false,true} is a binary class label
# p(x|y) is multi-modal (mixture of uniform and Gaussian distributions)

N = 200;

# ╔═╡ c5777ae3-e499-46a6-998a-05b97693b3e1
X_test = [3.75; 1.0] # Features of 'new' data point

# ╔═╡ 25ef6ece-d294-11ef-270a-999c8d457b24
md"""
## Main Idea of Discriminative Classification

Again, a data set is given by  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}`` with ``x_n \in \mathbb{R}^M`` and ``y_n \in \mathcal{C}_k``, with ``k=1,\ldots,K``.

"""

# ╔═╡ 25ef7f54-d294-11ef-3f05-0d85fe6e7a17
md"""
Sometimes, the precise assumptions of the (Gaussian-Categorical) generative model 

```math
p(x_n,y_n\in\mathcal{C}_k|\theta) =  \pi_k \cdot \mathcal{N}(x_n|\mu_k,\Sigma)
```

clearly do not match the data distribution.

"""

# ╔═╡ 25efa2fe-d294-11ef-172f-9bb09277f59e
md"""
Here's an **IDEA**! Let's model the posterior 

```math
p(y_n\in\mathcal{C}_k|x_n)
```

*directly*, without any assumptions on the class densities.

"""

# ╔═╡ 25efbe42-d294-11ef-3e4e-cfea366757da
md"""
Similarly to regression, we will assume that the inputs ``x`` are given, so we wil not add a model ``p(x)`` for input uncertainties.

"""

# ╔═╡ 25efd6b6-d294-11ef-3b21-6363ef531eb5
md"""
## Model Specification for Bayesian Logistic Regression

We will work this idea out for a 2-class problem. Assume a data set is given by  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}`` with ``x_n \in \mathbb{R}^M`` and ``y_n \in \{0,1\}``.

"""

# ╔═╡ 25f02ac6-d294-11ef-26c4-f142b8ac4b5f
md"""
What model should we use for the posterior distribution ``p(y_n \in \mathcal{C}_k|x_n)``?

"""

# ╔═╡ 25f0adde-d294-11ef-353e-4b4773df9ff5
md"""
#### Likelihood

We will take inspiration from the [generative classification](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Generative-Classification.ipynb#softmax) approach, where we derived the class posterior 

```math
p(y_{nk} = 1\,|\,x_n,\beta_k,\gamma_k) = \sigma(\beta_k^T x_n + \gamma_k)
```

as a **softmax** function of a linear map of the input.  

Here, in logistic regression, we *choose* the 2-class softmax function (which is called the [**logistic** function](https://en.wikipedia.org/wiki/Logistic_function)) with linear discrimination bounderies for the posterior class probability:

```math
p(y_n =1 \,|\, x_n, w) = \sigma(w^T x_n) \,.
```

where 

```math
\sigma(a) = \frac{1}{1+e^{-a}}
```

is the *logistic* function.

Clearly, it follows from this assumption that ``p(y_n =0 \,|\, x_n, w) = 1- \sigma(w^T x_n)``.

"""

# ╔═╡ 25f0f618-d294-11ef-0d94-bf80c8e2957b
md"""
![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/Figure4.9.png?raw=true)

(Bishop fig.4.9). The logistic function ``\sigma(a) = 1/(1+e^{-a})`` (red), together with the $(HTML("<span id='scaled-probit'>scaled probit function</span>")) ``\Phi(\lambda a)``, for ``\lambda^2=\pi/8`` (in blue). We will use this approximation later in the [Laplace approximation](#gaussian-cdf).

"""

# ╔═╡ 25f12528-d294-11ef-0c65-97c61935e9c2
md"""
Adding the other class (``y_n=0``) leads to the following posterior class distribution:

```math
\begin{align*}
p(y_n \,|\, x_n, w) &= \mathrm{Bernoulli}\left(y_n \,|\, \sigma(w^T x_n) \right) \\
&= \sigma(w^T x_n)^{y_n} \left(1 - \sigma(w^T x_n)\right)^{(1-y_n)} \tag{B-4.89} \\
  &= \sigma\left( (2y_n-1) w^T x_n\right)
\end{align*}
```

Note that for the 3rd equality, we have made use of the fact that ``\sigma(-a) = 1-\sigma(a)``.

Each of these three models in B-4.89 are **equivalent**. We mention all three notational options since they all appear in the literature.  

"""

# ╔═╡ 25f1390c-d294-11ef-364d-17e4c93b9a57
md"""
For the data set ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}``, the **likelihood function** for the parameters ``w`` is then given by

```math
p(D|w) = \prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)
```

"""

# ╔═╡ 25f14226-d294-11ef-369f-e545d5fe2700
md"""
This choice for the class posterior is called **logistic regression**, in analogy to **linear regression**:

```math
\begin{align*}
p(y_n|x_n,w) &= \mathcal{N}(y_n|w^T x_n,\beta^{-1}) \quad &&\text{for linear regression} \\
p(y_n|x_n,w) &= \sigma\left( (2y_n-1) w^T x_n\right) &&\text{for logistic regression}
\end{align*}
```

"""

# ╔═╡ 25f14f82-d294-11ef-02fb-2dc632b8f118
md"""
In the discriminative approach, the parameters ``w`` are **not** structured into ``\{\mu,\Sigma,\pi \}``. In principle they are "free" parameters for which we can choose any value that seems appropriate. This provides discriminative approach with more flexibility than the generative approach. 

"""

# ╔═╡ 25f15e0a-d294-11ef-3737-79a68c9b3c61
md"""
#### Prior

In *Bayesian* logistic regression, we often add a **Gaussian prior on the weights**: 

```math
\begin{align*}
p(w) = \mathcal{N}(w \,|\, m_0, S_0) \tag{B-4.140}
\end{align*}
```

"""

# ╔═╡ 25f19230-d294-11ef-2dfd-6d4927e86f57
md"""
## Some Notes on the Model

Note that for generative classification, for the sake of simplicity, we used maximum likelihood estimation for the model parameters. In this lesson on discriminative classification, we specify both a prior and likelihood function for the parameters ``w``, which allows us to compute a Bayesian posterior for the weights. In principle, we could have used Bayesian parameter estimation for the generative classification model as well (but the math is not suited for a introductory lesson).  

In the optional paper by [T. Minka (2005)](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Minka-2005-Discriminative-models-not-discriminative-training.pdf), you can read how the model assumptions for discriminative classification can be re-interpreted as a special generative model (this paper not for exam). 

As an exercise, please check that for logistic regression with ``p(y_n =1 \,|\, x_n, w) = \sigma(w^T x_n)``, the **discrimination boundary**, which can be computed by

```math
\frac{p(y_n\in\mathcal{C}_1|x_n)}{p(y_n\in\mathcal{C}_0|x_n)} \overset{!}{=} 1
```

is a straight line, see [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Classification.ipynb). 

"""

# ╔═╡ 25f19ed8-d294-11ef-3298-efa16dda1dde
md"""
## $(HTML("<span id='logistic-regression-posterior'>Parameter Inference</span>"))

After model specification, the rest follows by application of probability theory.

The posterior for the weights follows by Bayes rule

```math
\begin{align*}
\underbrace{p(w \,|\, D)}_{\text{posterior}} &=  \frac{p(w) p(D|w)}{\int p(w) p(D|w) \mathrm{d}w} \\ &= \frac{\overbrace{\mathcal{N}(w \,|\, m_0, S_0)}^{\text{prior}} \cdot \overbrace{\prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)}^{\text{likelihood}}}{\underbrace{\int \mathcal{N}(w \,|\, m_0, S_0) \prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right) \mathrm{d}w}_{\text{evidence}}} \tag{B-4.142}
\end{align*}
```

In principle, Bayesian inference is done now! 

Unfortunately, the posterior ``p(w \,|\, D)`` is not Gaussian and the evidence ``p(D)`` is also not analytically computable. (We will deal with this later).

"""

# ╔═╡ 25f1ab08-d294-11ef-32ed-493792e121b7
md"""
## Application: the predictive distribution

For a new data point ``x_\bullet``, the predictive distribution for ``y_\bullet`` is given by 

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &= \int p(y_\bullet = 1 \,|\, x_\bullet, w) \, p(w\,|\, D) \,\mathrm{d}w \\
  &= \int \sigma(w^T x_\bullet) \, p(w\,|\, D) \,\mathrm{d}w \tag{B-4.145}
\end{align*}
```

"""

# ╔═╡ 25f1b404-d294-11ef-1c3a-a5a8142bb202
md"""
After substitution of ``p(w | D)`` from B-4.142, we have closed-form expressions for both the posterior ``p(w|D)`` and the predictive distribution ``p(y_\bullet = 1 \mid x_\bullet, D)``. Unfortunately, these expressions contain integrals that are not analytically computable. 

"""

# ╔═╡ 25f1c2a0-d294-11ef-009c-69b64e87e5fb
md"""
Many methods have been developed to approximate the integrals in order to get analytical or numerical solutions. Here, we present the **Laplace approximation**, which is one of the simplest methods with broad applicability to Bayesian calculations.

"""

# ╔═╡ 25f1d29a-d294-11ef-0ae2-bf73a66952c6
md"""
## The Laplace Approximation

The central idea of the Laplace approximation is to approximate a (possibly unnormalized) distribution ``f(z)`` by a Gaussian distribution ``q(z)``. 

"""

# ╔═╡ 25f1e12c-d294-11ef-156a-874d9183f620
md"""
We first give the result, followed by the derivation. The **Laplace approximation** of the distribution ``f(z)`` is given by:

```math
\begin{align*}
f(z) &\approx f(z_0) \exp\left( - \frac{1}{2} (z-z_0)^T A (z-z_0)\right) \\
\text{where: } z_0 &= \arg\max_z \left( \log f(z)\right) \\
A &= - \nabla \nabla \left. \log f(z) \right|_{z=z_0}
\end{align*}
```

"""

# ╔═╡ 25f1f202-d294-11ef-3e01-bf5ac8cb92d5
md"""
The Laplace approximation usually serves one or both of the following two purposes: 

1. To approximate a posterior distribution without closed-form expression by a Gaussian distribution.
2. To approximate (part of) the integrand in an integral with purpose to get an analytical solution for the integral.

"""

# ╔═╡ 25f20242-d294-11ef-1a8c-a52791807d86
md"""
#### $(HTML("<span id='Laplace-example'>Example</span>"))

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/Figure4.14a.png?raw=true)

(Bishop fig.4.14a). Laplace approximation (in red) to the distribution ``p(z)\propto \exp(-z^2/2)\sigma(20z+4)``, where ``\sigma(a)=1/(1+e^{-a})``. The Laplace approximation is centered on the mode of ``p(z)``.

"""

# ╔═╡ 25f21002-d294-11ef-2198-83495043689d
md"""
## Working out the Laplace Approximation

Assume that we want to approximate a distribution ``f(z)`` by a Gaussian distribution ``q(z)``.

Note that, if ``q(z)`` is a Gaussian distribution, then ``\log q(z)`` is a second-order polynomial in ``z``, so we will find the Gaussian by fitting a parabola to ``\log f(z)``. 

"""

# ╔═╡ 25f21ef8-d294-11ef-2c38-69565004281b
md"""
#### estimation of mean

The mean (``z_0``) of ``q(z)`` is placed on the mode of ``\log f(z)``, i.e., 

```math
z_0 = \arg\max_z \left( \log f(z)\right)  \tag{B-4.126}
```

"""

# ╔═╡ 25f23186-d294-11ef-2260-dfa5c8241116
md"""
#### estimation of precision matrix

Note that since ``\nabla \log f(z) = \frac{1}{f(z)}\nabla f(z)`` and the gradient ``\nabla \left. f(z) \right|_{z=z_0}`` vanishes at the mode ``z=z_0``, we can (Taylor) expand ``\log f(z)`` around ``z=z_0`` as 

```math
\begin{align*}
\log f(z) &\approx \log f(z_0) + \overbrace{\left(\nabla \log f(z_0)\right)^T (z-z_0)}^{=0 \text{ at }z=z_0} + \ldots \\
&\qquad + \frac{1}{2} (z-z_0)^T \left(\nabla \nabla \log f(z_0)\right) (z-z_0) \\
  &= \log f(z_0) - \frac{1}{2} (z-z_0)^T A (z-z_0) \tag{B-4.131}
\end{align*}
```

where the [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) ``A`` is defined by

```math
A = - \nabla \nabla \left. \log f(z) \right|_{z=z_0} \tag{B-4.132}
```

"""

# ╔═╡ 25f24338-d294-11ef-1da0-2d4add34fde9
md"""
#### Laplace approximation construction

After taking exponentials in eq. B-4.131, we obtain

```math
f(z) \approx f(z_0) \exp\left( - \frac{1}{2} (z-z_0)^T A (z-z_0)\right) 
```

"""

# ╔═╡ 25f25742-d294-11ef-281d-a9c52797a6c1
md"""
(end derivation of Laplace Approximation)

We can now identify ``q(z)`` as

```math
q(z) = \mathcal{N}\left( z\,|\,z_0, A^{-1}\right) \tag{B-4.134}
```

with ``z_0`` and ``A`` defined by eqs. B-4.126 and B-4.132. 

All we have done up to now is approximate a function ``f(z)`` by a Gaussian ``q(z)``. This procedure is called the **Laplace Approximation**. Often, the required integrals (for Bayesian marginalization) can be approximately computed if we replace ``f(z)`` by ``q(z)``. 

"""

# ╔═╡ 25f2aa12-d294-11ef-1417-b52f0f67fb5b
md"""
## Bayesian Logistic Regression with the Laplace Approximation

Let's get back to the challenge of computing the predictive class distribution (B-4.145) for Bayesian logistic regression. We first work out the Gaussian Laplace approximation ``q(w)`` to the [posterior weight distribution](#logistic-regression-posterior)

```math
\begin{align*}
\underbrace{p(w | D)}_{\text{posterior}} \propto  \underbrace{\mathcal{N}(w \,|\, m_0, S_0)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)}_{\text{likelihood}}  \tag{B-4.142}
\end{align*}
```

"""

# ╔═╡ 25f2d35c-d294-11ef-1050-ff6188d649b4
md"""
#### A Gausian Laplace approximation to the weights posterior

Since we have a differentiable expression for ``\log p(w | D)``, it is straightforward to compute the gradient and Hessian (for [proof, see optional slide](#gradient-hessian)):

```math
\begin{align*}
\nabla_w \log p(w | D) &= S_0^{-1}\cdot \left(m_0-w\right) + \sum_n (2y_n-1) (1-\sigma_n) x_n \\
\nabla\nabla_w \log p(w | D) &= -S_0^{-1} - \sum_n \sigma_n (1-\sigma_n) x_n x_n^T \tag{B-4.143}
\end{align*}
```

where we used shorthand ``\sigma_n`` for ``\sigma\left( (2y_n-1) w^T x_n\right)``. 

"""

# ╔═╡ 25f2e52c-d294-11ef-357d-053d0c9898e4
md"""
We can now use the gradient ``\nabla_w \log p(w | D)`` to find the **mode** ``w_{N}`` of ``\log p(w|D)`` (eg by some gradient-based optimization procedure) and then use the Hessian ``\nabla\nabla_w \log p(w | D)`` to get the variance of ``q(w)``, leading to a $(HTML("<span id='Laplace-posterior-logistic-regression'>**Gaussian approximate weights posterior**</span>")):

```math
q(w) = \mathcal{N}\left(w\,|\, w_{N}, S_N\right) \tag{B-4.144}
```

with

```math
S_N^{-1} = S_0^{-1} + \sum_n \sigma_n (1-\sigma_n) x_n x_n^T \tag{B-4.143}
```

"""

# ╔═╡ 25f2f898-d294-11ef-3414-c5e9688a37f6
md"""
## Using the Laplace-approximated parameter posterior to evaluate the predictive distribution

In the analytically unsolveable expressions for evidence and the predictive distribution (estimating the class of a new observation), we proceed with using the Laplace approximation to the weights posterior. For a new observation ``x_\bullet``, the class probability is now

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &= \int p(y_\bullet = 1 \,|\, x_\bullet, w) \cdot p(w\,|\, D) \,\mathrm{d}w \\
  &\approx \int p(y_\bullet = 1 \,|\, x_\bullet, w) \cdot \underbrace{q(w)}_{\text{Gaussian}} \,\mathrm{d}w \\
  &= \int \sigma(w^T x_\bullet) \cdot \mathcal{N}\left(w \,|\, w_N, S_N\right) \,\mathrm{d}w \tag{B-4.145}
\end{align*}
```

"""

# ╔═╡ 25f30f34-d294-11ef-2d70-f32b65cc8c69
md"""
This looks better but we need two more clever tricks to evaluate this expression. 

1. First, note that ``w`` appears in ``\sigma(w^T x_\bullet)`` as an inner product, so through substitution of ``a:=w^T x_\bullet``, the expression simplifies to an integral over the scalar ``a`` (see Bishop for derivation):

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &\approx \int \sigma(a) \, \mathcal{N}\left(a\,|\, \mu_a, \sigma_a^2\right) \,\mathrm{d}a \qquad &&\text{(B-4.151)}\\
\mu_a  &= w^T_{N} x_\bullet \qquad &&\text{(B-4.149)}\\
\sigma_a^2 &= x^T_\bullet S_N x_\bullet \qquad &&\text{(B-4.150)}
\end{align*}
```

Secondly, while the integral of the product of a logistic function with a Gaussian is not analytically solvable, the integral of the product of a Gaussian cumulative distribution function (CDF, also known as the [probit function](#scaled-probit)) with a Gaussian *does* have a closed-form solution. Fortunately, 

```math
\Phi(\lambda a) \approx \sigma(a)
```

with the $(HTML("<span id='gaussian-cdf'>Gaussian</span>")) CDF ``\Phi(x)= \frac{1}{\sqrt(2\pi)}\int_{-\infty}^{x}e^{-t^2/2}\mathrm{d}t``, $ \lambda^2= \pi / 8 $ and ``\sigma(a) = 1/(1+e^{-a})``.    Thus, substituting ``\Phi(\lambda a)`` with $ \lambda^2= \pi / 8 $ for ``\sigma(a)`` leads to 

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &=  \int \sigma(w^T x_\bullet) \cdot p(w|D) \,\mathrm{d}w  \\ 
&\approx \int \underbrace{\Phi(\lambda a)}_{\text{probit function}} \cdot \underbrace{\mathcal{N}\left(a\,|\, \mu_a, \sigma_a^2\right)}_{\text{Gaussian}} \,\mathrm{d}a \\ 
&= \Phi\left( \frac{\mu_a}{\sqrt(\lambda^{-2} +\sigma_a^2)}\right) \tag{B-4.152}
\end{align*}
```

"""

# ╔═╡ 25f31dda-d294-11ef-1d57-f16507c1cd6b
md"""
We now have an approximate but **closed-form expression for the predictive class distribution for a new observation** with a Bayesian logistic regression model.  

"""

# ╔═╡ 25f32b7c-d294-11ef-2504-594970435176
md"""
Note that, by [Eq.B-4.143](#Laplace-posterior-logistic-regression), the variance ``S_N`` (and consequently ``\sigma_a^2``) for the weight vector depends on the distribution of the training set. Large uncertainty about the weights (in areas with little training data and uninformative prior variance ``S_0``) increases ``\sigma_a^2`` and takes the posterior class probability eq. B-4.152 closer to ``0.5``. Does that make sense?

"""

# ╔═╡ 25f33920-d294-11ef-3c99-d5233064a5f4
md"""
Apparently, the Laplace approximation leads to a closed-form solutions for Bayesian logistic regression (although admittedly, the derivation is no walk in the park). 

"""

# ╔═╡ 25f34774-d294-11ef-2b4e-995aca972742
md"""
Exam guide: The derivation of closed-form expression eq. B-4.152 for the predictive class distribution requires clever tricks and is therefore not something that you should be able to reproduce at the exam without assistance. You should understand the Laplace Approximation though and be able to work out simpler examples.  

"""

# ╔═╡ 25f356b0-d294-11ef-17b9-8583928f7829
md"""
## ML Estimation for Discriminative Classification

Rather than the computationally involved Laplace approximation for Bayesian inference, in practice, discriminative classification is often executed through maximum likelihood estimation. 

"""

# ╔═╡ 25f365e2-d294-11ef-300e-9914333b1233
md"""
With the usual 1-of-K encoding scheme for classes (``y_{nk}=1`` if ``x_n \in \mathcal{C}_k``, otherwise ``y_{nk}=0``), the log-likelihood for a ``K``-dimensional discriminative classifier is 

```math
\begin{align*}
    \mathrm{L}(\theta) &= \log \prod_n \prod_k {p(\mathcal{C}_k|x_n,\theta)}^{y_{nk}} \\
    &= \log \prod_n \prod_k \Bigg(\underbrace{\frac{e^{\theta_k^T x_n}}{ \sum_j e^{\theta_j^T x_n}}}_{\text{softmax function}}\Bigg)^{y_{nk}} \\
    &= \sum_n \sum_k y_{kn} \log \big( \frac{e^{\theta_k^T x_n}}{ \sum_j e^{\theta_j^T x_n}} \big)
     \end{align*}
```

"""

# ╔═╡ 25f3741a-d294-11ef-1418-f11326406eb6
md"""
Computing the gradient ``\nabla_{\theta_k} \mathrm{L}(\theta)`` leads to (for [proof, see optional slide below](#ML-for-LG)) 

```math
\nabla_{\theta_k} \mathrm{L}(\theta) = \sum_n \underbrace{\big( \underbrace{y_{nk}}_{\text{target}} - \underbrace{\frac{e^{\theta_k^T x_n}}{ \sum_j e^{\theta_j^T x_n}}}_{\text{prediction}} \big)}_{\text{prediction error}}\cdot x_n 
```

"""

# ╔═╡ 25f386e4-d294-11ef-2cec-f56f4a6feb19
md"""
Compare this to the [gradient for *linear* regression](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Regression.ipynb#regression-gradient):

```math
\nabla_\theta \mathrm{L}(\theta) =  \sum_n \left(y_n - \theta^T x_n \right)  x_n
```

"""

# ╔═╡ 25f3965c-d294-11ef-11b8-af605b86f188
md"""
In both cases

```math
\nabla_\theta \mathrm{L} =  \sum_n \left( \text{target}_n - \text{prediction}_n \right) \cdot \text{input}_n 
```

"""

# ╔═╡ 25f3a638-d294-11ef-0cd5-c3a46aa780c6
md"""
The parameter vector ``\theta`` for logistic regression can be estimated through iterative gradient-based adaptation. E.g. (with iteration index ``i``),

```math
\hat{\theta}^{(i+1)} =  \hat{\theta}^{(i)} + \eta \cdot \left. \nabla_\theta   \mathrm{L}(\theta)  \right|_{\theta = \hat{\theta}^{(i)}}
```

Note that, while in the Bayesian approach we get to update ``\theta`` with [**Kalman-gain-weighted** prediction errors](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Gaussian-Distribution.ipynb#precision-weighted-update) (which is optimal), in the maximum likelihood approach, we weigh the prediction errors with **input** values (which is less precise).

"""

# ╔═╡ 25f3bef2-d294-11ef-1438-e9f7e469336f
md"""
## Code Example: ML Estimation for Discriminative Classification

Let us perform ML estimation of ``w`` on the data set from the introduction. To allow an offset in the discrimination boundary, we add a constant 1 to the feature vector ``x``. We only have to specify the (negative) log-likelihood and the gradient w.r.t. ``w``. Then, we use an off-the-shelf optimisation library to minimize the negative log-likelihood.

We plot the resulting maximum likelihood discrimination boundary. For comparison we also plot the ML discrimination boundary obtained from the [code example in the generative Gaussian classifier lesson](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Generative-Classification.ipynb#code-generative-classification-example).

"""

# ╔═╡ 25f3ee5e-d294-11ef-1fb4-e9d84b1e1ec6
md"""
The generative model gives a bad result because the feature distribution of one class is clearly non-Gaussian: the model does not fit the data well. 

The discriminative approach does not suffer from this problem because it makes no assumptions about the feature distribution ``p(x)``. Rather, it just estimates the conditional class distribution ``p(y|x)`` directly.

"""

# ╔═╡ 25f3ff84-d294-11ef-0031-63b23d23324d
md"""
## Why be Bayesian?

Why should you embrace the Bayesian approach to logistic regression? After all, Maximum Likelihood for logistic regression seems simpler.

Still, consider the following:

  * Bayesian logistic regression with the Laplace approximation ultimately leads to very simple analytic rules. Moreover, modern probabilistic programming languages and packages are able to automate the above inference derivations. (We just do them here to gain insight in difficult inference processes.)
  * Bayesian logistic regression offers the option to compute model evidence.
  * Bayesian logistic regression processes uncertainties, e.g., in places where almost no data is observed, the posterior class probability will pull back to the prior class probability rather than predicting some arbitrary probability.

"""

# ╔═╡ 25f41118-d294-11ef-13a8-3fa6587c1bf3
md"""
## Recap Classification

<table> <tr> <td></td><td style="text-align:center"><b>Generative</b></td> <td style="text-align:center"><b>Discriminative (ML)</b></td> </tr> 

<tr> <td>1</td><td>Like <b>density estimation</b>, model joint prob.

```math
p(\mathcal{C}_k) p(x|\mathcal{C}_k) = \pi_k \mathcal{N}(\mu_k,\Sigma)
```

</td> <td>Like (linear) <b>regression</b>, model conditional

```math
p(\mathcal{C}_k|x,\theta)
```

</td> </tr>

<tr> <td>2</td><td>Leads to <b>softmax</b> posterior class probability

```math
 p(\mathcal{C}_k|x,\theta ) = e^{\theta_k^T x}/Z
```

with <b>structured</b> ``\theta``</td> <td> <b>Choose</b> also softmax posterior class probability

```math
 p(\mathcal{C}_k|x,\theta ) = e^{\theta_k^T x}/Z
```

but now with 'free' ``\theta``</td> </tr>

<tr> <td>3</td><td>For Gaussian ``p(x|\mathcal{C}_k)`` and multinomial priors,

```math
\hat \theta_k  = \left[ {\begin{array}{c}
   { - \frac{1}{2} \mu_k^T \sigma^{-1} \mu_k  + \log \pi_k}  \\
   {\sigma^{-1} \mu_k }  \\
\end{array}} \right]
```

<b>in one shot</b>.</td> <td>Find ``\hat\theta_k`` through gradient-based adaptation

```math
\nabla_{\theta_k}\mathrm{L}(\theta) = \sum_n \Big( y_{nk} - \frac{e^{\theta_k^T x_n}}{\sum_{k^\prime} e^{\theta_{k^\prime}^T x_n}} \Big)\, x_n
```

</td> </tr> </table>

"""

# ╔═╡ 25f41df2-d294-11ef-35a1-73e12752c24d
md"""
#  OPTIONAL SLIDES 

"""

# ╔═╡ 25f42e98-d294-11ef-1f51-8b6b81987cc4
md"""
## $(HTML("<span id='gradient-hessian'>Proof of gradient and Hessian for Laplace Approximation of Posterior</span>"))

We will start with the posterior

```math
\begin{align*}
\underbrace{p(w | D)}_{\text{posterior}} \propto  \underbrace{\mathcal{N}(w \,|\, m_0, S_0)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \sigma\big( \underbrace{(2y_n-1) w^T x_n}_{a_n}\big)}_{\text{likelihood}}  \tag{B-4.142}
\end{align*}
```

from which it follows that

```math
\begin{align*}
\log p(w | D) \propto  -\frac{1}{2}\log |S_0| -\frac{1}{2} (w-m_0)^T S_0^{-1} (w-m_0) +\sum_n \log \sigma\left( a_n\right) 
\end{align*}
```

and the gradient

```math
\begin{align*}
\nabla_{w}\log p(w | D) &\propto   \underbrace{S_0^{-1} (m_0-w)}_{\text{SRM-5b}} +\sum_n \underbrace{\frac{1}{\sigma(a_n)}}_{\frac{\partial \log \sigma(a_n)}{\partial \sigma(a_n)}} \cdot \underbrace{\sigma(a_n) \cdot (1-\sigma(a_n))}_{\frac{\partial \sigma(a_n)}{\partial a_n}} \cdot \underbrace{(2y_n-1)x_n}_{\frac{\partial a_n}{\partial w} \text{ (see SRM-5a)}}    \\
&=   S_0^{-1} (m_0-w) + \sum_n (2y_n-1) (1-\sigma(a_n)) x_n \quad \text{(gradient)}
 \end{align*}
```

where we used  ``\sigma^\prime(a) = \sigma(a)\cdot (1-\sigma(a))``.

For the Hessian, we continue to differentiate the transpose of the gradient, leading to

```math
\begin{align*}
\nabla\nabla_{w}\log p(w | D) &=  \nabla_{w} \left(S_0^{-1} (m_0-w)\right)^T - \sum_n (2y_n-1) x_n \nabla_{w}\sigma(a_n)^T \\ &=  -S_0^{-1} - \sum_n (2y_n-1) x_n \cdot \underbrace{\sigma(a_n)\cdot (1-\sigma(a_n))}_{\frac{\partial \sigma(a_n)^T}{\partial a_n^T}}\cdot \underbrace{(2y_n-1) x_n^T}_{\frac{\partial a_n^T}{\partial w}} \\
&= -S_0^{-1} - \sum_n \sigma(a_n)\cdot (1-\sigma(a_n))\cdot x_n x_n^T \quad \text{(Hessian)}
\end{align*}
```

since ``(2y_n-1)^2=1`` for ``y_n \in \{0,1\}``.

"""

# ╔═╡ 25f461c2-d294-11ef-2e85-6f1acc16cf3b
md"""
## $(HTML("<span id='ML-for-LG'>Proof of Derivative of Log-likelihood for Logistic Regression</span>"))

The Log-likelihood is ``\mathrm{L}(\theta) = \log \prod*n \prod*k {\underbrace{p(\mathcal{C}*k|x*n,\theta)}*{p*{nk}}}^{y*{nk}} = \sum*{n,k} y*{nk} \log p*{nk}``

Use the fact that the softmax ``\phi_k \equiv e^{a_k} / {\sum_j e^{a_j}}`` has analytical derivative:

```math
 \begin{align*}
 \frac{\partial \phi_k}{\partial a_j} &= \frac{(\sum_j e^{a_j})e^{a_k}\delta_{kj}-e^{a_j}e^{a_k}}{(\sum_j e^{a_j})^2} = \frac{e^{a_k}}{\sum_j e^{a_j}}\delta_{kj} - \frac{e^{a_j}}{\sum_j e^{a_j}} \frac{e^{a_k}}{\sum_j e^{a_j}}\\
     &= \phi_k \cdot(\delta_{kj}-\phi_j)
 \end{align*}
```



Take the derivative of ``\mathrm{L}(\theta)`` (or: how to spend a hour ...)

```math
\begin{align*} 
\nabla_{\theta_j} \mathrm{L}(\theta) &= \sum_{n,k} \frac{\partial \mathrm{L}_{nk}}{\partial p_{nk}} \cdot\frac{\partial p_{nk}}{\partial a_{nj}}\cdot\frac{\partial a_{nj}}{\partial \theta_j} \\
  &= \sum_{n,k} \frac{y_{nk}}{p_{nk}} \cdot p_{nk} (\delta_{kj}-p_{nj}) \cdot x_n \\
  &= \sum_n \Big( y_{nj} (1-p_{nj}) -\sum_{k\neq j} y_{nk} p_{nj} \Big) \cdot x_n \\
  &= \sum_n \left( y_{nj} - p_{nj} \right)\cdot x_n \\
  &= \sum_n \Big( \underbrace{y_{nj}}_{\text{target}} - \underbrace{\frac{e^{\theta_j^T x_n}}{\sum_{j^\prime} e^{\theta_{j^\prime}^T x_n}}}_{\text{prediction}} \Big)\cdot x_n 
\end{align*}
```

"""

# ╔═╡ 6eee35ee-fd55-498f-9441-f18c2508de19
md"""
# Appendix
"""

# ╔═╡ fcec3c3a-8b0b-4dfd-b010-66abbf330069
function generate_dataset(N::Int64)
	Random.seed!(1234)
    # Generate dataset {(x1,y1),...,(xN,yN)}
    # x is a 2d feature vector [x1;x2]
    # y ∈ {false,true} is a binary class label
    # p(x|y) is multi-modal (mixture of uniform and Gaussian distributions)
	# srand(123)
    X = Matrix{Float64}(undef,2,N); y = Vector{Bool}(undef,N)
    for n=1:N
        if (y[n]=(rand()>0.6)) # p(y=true) = 0.6
            # Sample class 1 conditional distribution
            if rand()<0.5
                X[:,n] = [6.0; 0.5] .* rand(2) .+ [3.0; 6.0]
            else
                X[:,n] = sqrt(0.5) * randn(2) .+ [5.5, 0.0]
            end
        else
            # Sample class 2 conditional distribution
            X[:,n] = randn(2) .+ [1., 4.]
        end
    end

    return (X, y)
end

# ╔═╡ d908270d-81b4-4c97-8298-2ba66bccc45b
X, y = generate_dataset(N) # Generate data set, collect in matrix X and vector y

# ╔═╡ c9dfa502-74ce-4c9d-ad4b-b554fec6ddf2
X_c1 = X[:,findall(.!y)]'; X_c2 = X[:,findall(y)]' # Split X based on class label

# ╔═╡ 476f1aef-fb9b-42a6-a0cb-f0f3da1385da
function plot_dataset()
    result = scatter(X_c1[:,1], X_c1[:,2],markersize=4, label=L"y=0", xlabel=L"x_1", ylabel=L"x_2", xlims=(-1.6, 9), ylims=(-2, 7))
    scatter!(X_c2[:,1], X_c2[:,2],markersize=4, label=L"y=1")
    scatter!([X_test[1]], [X_test[2]], markersize=7, marker=:star, label=L"y=?") 
    return result  
end

# ╔═╡ cc016a47-5c5f-4361-85b9-f6f4141e58d3
plot_dataset()

# ╔═╡ 56598859-2824-4242-a894-684bf1ad1f6e
y_1 = map(y) do val
	val == true ? 1.0 : 0.0
end # class 1 indicator vector

# ╔═╡ 6f483978-29f0-4165-bd8f-650c403e3512
# Extend X with a row of ones to allow an offset in the discrimination boundary
X_ext = vcat(X, ones(1, length(y)))

# ╔═╡ a89af0df-c39b-406e-a30a-4706ad2ea043
function negative_log_likelihood(θ::Vector)
	# P(C1|X,θ)
    p_1 = 1.0 ./ (1.0 .+ exp.(-X_ext' * θ))
	
	# negative log-likelihood
    return -sum(log.( (y_1 .* p_1) + ((1 .- y_1).*(1 .- p_1))) ) 
end

# ╔═╡ a75d69e1-c1e9-45b4-9924-4c2fe59413dc
# Use Optim.jl optimiser to minimize the negative log-likelihood function w.r.t. θ
θ = let
	results = optimize(negative_log_likelihood, zeros(3), LBFGS())
	results.minimizer
end

# ╔═╡ 00488cbb-75c6-4df9-9924-fada8f79a6f1
function build_generative_discrimination_boundary(X::Matrix, y::Vector{Bool})
    # Generate discrimination boundary function x[2] = boundary(x[1]) for a Gaussian generative model:
    # X = [x_1,...,x_N]
    # y = [y_1;...;y_N]
    # x is a 2-d real (feature) vector
    # y ∈ {false,true}
    # x|y ~ 𝓝(x|μ_y, Σ_y)
    # We find the class-conditional Gaussian distributions by MLE
    # See lesson (generative classification) for more details
    (size(X,1)==2) || error("The columns of X should have length 2")

    # MLE of p(y)
    p_1_est = sum(y.==true) / length(y)
    π_hat = [p_1_est; 1 .- p_1_est]

    # MLE of class-conditional multivariate Gaussian densities
    X_cls1 = X[:,y.==true]
    X_cls2 = X[:,y.==false]
    d1 = fit_mle(FullNormal, X_cls1)  # MLE density estimation d1 = N(μ₁, Σ₁)
    d2 = fit_mle(FullNormal, X_cls2)  # MLE density estimation d2 = N(μ₂, Σ₂)
    Σ = π_hat[1] * cov(d1) + π_hat[2] * cov(d2) # Combine Σ₁ and Σ₂ into Σ

    conditionals = [MvNormal(mean(d1), Σ); MvNormal(mean(d2), Σ)] # p(x|C)

    # Discrimination boundary of the posterior (p(apple|x;D) = p(peach|x;D) = 0.5)
    β(k) = inv(Σ)* mean(conditionals[k])
    γ(k) = -0.5 * mean(conditionals[k])' * inv(Σ) * mean(conditionals[k]) + log(π_hat[k])
    function discriminant_x2(x1)
        # Solve discriminant equation for x2
        
        β12 = β(1) .- β(2)
        γ12 = (γ(1) .- γ(2))[1,1]
        return -1 ./ β12[2]*(β12[1]*x1 .+ γ12) 
    end

    return discriminant_x2
end

# ╔═╡ 7ad2f815-9d19-448c-bb7e-044a955f82e0
let
	# Plot the data set and ML discrimination boundary
	plot_dataset()
	p_1(x) = 1.0 / (1.0 + exp(-([x;1.]' * θ)))
	boundary(x1) = -1 / θ[2] * (θ[1]*x1 + θ[3])
	
	generative_boundary = build_generative_discrimination_boundary(X, y)
	
	x_test = [3.75;1.0]
	@debug("P(C1|x•,θ) = $(p_1(x_test))")
	
	plot!([-2., 10.], boundary; label="Discr. boundary", linewidth=2)
	plot!([-2.,10.], generative_boundary; label="Gen. boundary", linewidth=2)
end

# ╔═╡ Cell order:
# ╟─25eefb10-d294-11ef-0734-2daf18636e8e
# ╟─25ef12bc-d294-11ef-1557-d98ba829a804
# ╟─25ef2806-d294-11ef-3cb6-0f3e76b9177e
# ╠═a759653c-0da4-40b7-9e9e-1e3d2e4df4ea
# ╠═ae2a65fa-1322-43b1-80cf-ee3ad1c47312
# ╠═ad6b7f43-ccae-4b85-bd6e-051cd4d771cd
# ╠═d908270d-81b4-4c97-8298-2ba66bccc45b
# ╠═c9dfa502-74ce-4c9d-ad4b-b554fec6ddf2
# ╠═c5777ae3-e499-46a6-998a-05b97693b3e1
# ╠═cc016a47-5c5f-4361-85b9-f6f4141e58d3
# ╠═476f1aef-fb9b-42a6-a0cb-f0f3da1385da
# ╟─25ef6ece-d294-11ef-270a-999c8d457b24
# ╟─25ef7f54-d294-11ef-3f05-0d85fe6e7a17
# ╟─25efa2fe-d294-11ef-172f-9bb09277f59e
# ╟─25efbe42-d294-11ef-3e4e-cfea366757da
# ╟─25efd6b6-d294-11ef-3b21-6363ef531eb5
# ╟─25f02ac6-d294-11ef-26c4-f142b8ac4b5f
# ╟─25f0adde-d294-11ef-353e-4b4773df9ff5
# ╟─25f0f618-d294-11ef-0d94-bf80c8e2957b
# ╟─25f12528-d294-11ef-0c65-97c61935e9c2
# ╟─25f1390c-d294-11ef-364d-17e4c93b9a57
# ╟─25f14226-d294-11ef-369f-e545d5fe2700
# ╟─25f14f82-d294-11ef-02fb-2dc632b8f118
# ╟─25f15e0a-d294-11ef-3737-79a68c9b3c61
# ╟─25f19230-d294-11ef-2dfd-6d4927e86f57
# ╟─25f19ed8-d294-11ef-3298-efa16dda1dde
# ╟─25f1ab08-d294-11ef-32ed-493792e121b7
# ╟─25f1b404-d294-11ef-1c3a-a5a8142bb202
# ╟─25f1c2a0-d294-11ef-009c-69b64e87e5fb
# ╟─25f1d29a-d294-11ef-0ae2-bf73a66952c6
# ╟─25f1e12c-d294-11ef-156a-874d9183f620
# ╟─25f1f202-d294-11ef-3e01-bf5ac8cb92d5
# ╟─25f20242-d294-11ef-1a8c-a52791807d86
# ╟─25f21002-d294-11ef-2198-83495043689d
# ╟─25f21ef8-d294-11ef-2c38-69565004281b
# ╟─25f23186-d294-11ef-2260-dfa5c8241116
# ╟─25f24338-d294-11ef-1da0-2d4add34fde9
# ╟─25f25742-d294-11ef-281d-a9c52797a6c1
# ╟─25f2aa12-d294-11ef-1417-b52f0f67fb5b
# ╟─25f2d35c-d294-11ef-1050-ff6188d649b4
# ╟─25f2e52c-d294-11ef-357d-053d0c9898e4
# ╟─25f2f898-d294-11ef-3414-c5e9688a37f6
# ╟─25f30f34-d294-11ef-2d70-f32b65cc8c69
# ╟─25f31dda-d294-11ef-1d57-f16507c1cd6b
# ╟─25f32b7c-d294-11ef-2504-594970435176
# ╟─25f33920-d294-11ef-3c99-d5233064a5f4
# ╟─25f34774-d294-11ef-2b4e-995aca972742
# ╟─25f356b0-d294-11ef-17b9-8583928f7829
# ╟─25f365e2-d294-11ef-300e-9914333b1233
# ╟─25f3741a-d294-11ef-1418-f11326406eb6
# ╟─25f386e4-d294-11ef-2cec-f56f4a6feb19
# ╟─25f3965c-d294-11ef-11b8-af605b86f188
# ╟─25f3a638-d294-11ef-0cd5-c3a46aa780c6
# ╟─25f3bef2-d294-11ef-1438-e9f7e469336f
# ╠═6a20aa94-e2fa-45ab-9889-62d44cbfc1ba
# ╠═56598859-2824-4242-a894-684bf1ad1f6e
# ╠═6f483978-29f0-4165-bd8f-650c403e3512
# ╠═a89af0df-c39b-406e-a30a-4706ad2ea043
# ╠═a75d69e1-c1e9-45b4-9924-4c2fe59413dc
# ╠═7ad2f815-9d19-448c-bb7e-044a955f82e0
# ╟─25f3ee5e-d294-11ef-1fb4-e9d84b1e1ec6
# ╟─25f3ff84-d294-11ef-0031-63b23d23324d
# ╟─25f41118-d294-11ef-13a8-3fa6587c1bf3
# ╟─25f41df2-d294-11ef-35a1-73e12752c24d
# ╟─25f42e98-d294-11ef-1f51-8b6b81987cc4
# ╟─25f461c2-d294-11ef-2e85-6f1acc16cf3b
# ╟─6eee35ee-fd55-498f-9441-f18c2508de19
# ╠═616e84d7-063d-4d9d-99e4-56aecf3c7ee4
# ╠═fcec3c3a-8b0b-4dfd-b010-66abbf330069
# ╠═00488cbb-75c6-4df9-9924-fada8f79a6f1
