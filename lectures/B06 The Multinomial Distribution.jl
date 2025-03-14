### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ d8422bf2-d294-11ef-0144-098f414c6454
md"""
# Discrete Data and the Multinomial Distribution

"""

# ╔═╡ d8424e52-d294-11ef-0083-fbb77df4d853
md"""
## Preliminaries

Goal 

  * Simple Bayesian and maximum likelihood-based density estimation for discretely valued data sets

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 67-70, 74-76, 93-94

"""

# ╔═╡ d842ad86-d294-11ef-3266-253f80ecf4b7
md"""
## Discrete Data: the 1-of-K Coding Scheme

Consider a coin-tossing experiment with outcomes ``x \in\{0,1\}`` (tail and head) and let ``0\leq \mu \leq 1`` represent the probability of heads. This model can written as a [**Bernoulli distribution**](https://en.wikipedia.org/wiki/Bernoulli_distribution):

```math
 
p(x|\mu) = \mu^{x}(1-\mu)^{1-x}
```

Note that the variable ``x`` acts as a (binary) **selector** for the tail or head probabilities. Think of this as an 'if'-statement in programming.

"""

# ╔═╡ d842d368-d294-11ef-024d-45e58ca994e0
md"""
Now consider a ``K``-sided coin (e.g., a six-faced *die* (pl.: dice)). How should we encode outcomes?

"""

# ╔═╡ d842e5d0-d294-11ef-132f-a3445f00b389
md"""
**Option 1**: ``x \in \{1,2,\ldots,K\}``.

  * E.g., for ``K=6``, if the die lands on the 3rd face ``\,\Rightarrow x=3``.

**Option 2**:  ``x=(x_1,\ldots,x_K)^T`` with **binary selection variables**

```math
x_k = \begin{cases} 1 & \text{if die landed on $k$th face}\\
0 & \text{otherwise} \end{cases}
```

E.g., for ``K=6``, if the die lands on the 3rd face ``\,\Rightarrow x=(0,0,1,0,0,0)^T``.

This coding scheme is called a **1-of-K** or **one-hot** coding scheme.

"""

# ╔═╡ d842f41a-d294-11ef-1888-ddfdb5d236eb
md"""
It turns out that the one-hot coding scheme is mathematically more convenient!

"""

# ╔═╡ d842fe4c-d294-11ef-15a9-a9a6e359f47d
md"""
## The Categorical Distribution

Consider a ``K``-sided die. We use a one-hot coding scheme. Assume the probabilities ``p(x_k=1) = \mu_k`` with  ``\sum_k \mu_k  = 1``. The data generating distribution is then (note the similarity to the Bernoulli distribution)

```math
p(x|\mu) = \mu_1^{x_1} \mu_2^{x_2} \cdots \mu_K^{x_K}=\prod_{k=1}^K \mu_k^{x_k} \tag{B-2.26}
```

"""

# ╔═╡ d84345d2-d294-11ef-2297-39bb0b9d1a3f
md"""
This generalized Bernoulli distribution is called the [**categorical distribution**](https://en.wikipedia.org/wiki/Categorical_distribution) (or sometimes the 'multi-noulli' distribution).



"""

# ╔═╡ d843540a-d294-11ef-3846-2bf27b7e9b30
md"""
## Bayesian Density Estimation for a Loaded Die

Now let's proceed with Bayesian density estimation, i.e., let's learn the parameters for model ``p(x|\theta)`` for an observed data set ``D=\{x_1,\ldots,x_N\}``  of ``N`` independent-and-identically-distributed (IID) rolls of a ``K``-sided die, with 

```math
x_{nk} = \begin{cases} 1 & \text{if the $n$th throw landed on $k$th face}\\
0 & \text{otherwise} \end{cases}
```

"""

# ╔═╡ d84369a4-d294-11ef-38f7-7f393869b705
md"""
#### Model specification

The data generating PDF is

```math
p(D|\mu) = \prod_n \prod_k \mu_k^{x_{nk}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k} \tag{B-2.29}
```

where ``m_k= \sum_n x_{nk}`` is the total number of occurrences that we 'threw' ``k`` eyes. Note that ``\sum_k m_k = N``.

This distribution depends on the observations **only** through the quantities ``\{m_k\}``.

"""

# ╔═╡ d8439866-d294-11ef-230b-dfde21aedfbf
md"""
We need a prior for the parameters ``\mu = (\mu_1,\mu_2,\ldots,\mu_K)``. In the [binary coin toss example](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Bayesian-Machine-Learning.ipynb#beta-prior), 

we used a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) that was conjugate with the binomial and forced us to choose prior pseudo-counts. 

The generalization of the beta prior to the ``K`` parameters ``\{\mu_k\}`` is the [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution):

```math
p(\mu|\alpha) = \mathrm{Dir}(\mu|\alpha) = \frac{\Gamma\left(\sum_k \alpha_k\right)}{\Gamma(\alpha_1)\cdots \Gamma(\alpha_K)} \prod_{k=1}^K \mu_k^{\alpha_k-1} 
```

where ``\Gamma(\cdot)`` is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function). 

The Gamma function can be interpreted as a generalization of the factorial function to the real (``\mathbb{R}``) numbers. If ``n`` is a natural number (``1,2,3, \ldots $), then $\Gamma(n) = (n-1)!``, where ``(n-1)! = (n-1)\cdot (n-2) \cdot 1``.

As before for the Beta distribution in the coin toss experiment, you can interpret ``\alpha_k-1`` as the prior number of (pseudo-)observations that the die landed on the  ``k``-th face.

"""

# ╔═╡ d843a338-d294-11ef-2748-b95f2af1396b
md"""
#### Inference for ``\{\mu_k\}``

The posterior for  ``\{\mu_k\}`` can be obtained through Bayes rule:

```math
\begin{align*}
p(\mu|D,\alpha) &\propto p(D|\mu) \cdot p(\mu|\alpha) \\
  &\propto  \prod_k \mu_k^{m_k} \cdot \prod_k \mu_k^{\alpha_k-1} \\
  &= \prod_k \mu_k^{\alpha_k + m_k -1}\\
  &\propto \mathrm{Dir}\left(\mu\,|\,\alpha + m \right) \tag{B-2.41} \\
  &= \frac{\Gamma\left(\sum_k (\alpha_k + m_k) \right)}{\Gamma(\alpha_1+m_1) \Gamma(\alpha_2+m_2) \cdots \Gamma(\alpha_K + m_K)} \prod_{k=1}^K \mu_k^{\alpha_k + m_k -1}
\end{align*}
```

where ``m = (m_1,m_2,\ldots,m_K)^T``.

"""

# ╔═╡ d843b33c-d294-11ef-195d-2708fbfba49d
md"""
We recognize the ``(\alpha_k-1)``'s as prior pseudo-counts and the Dirichlet distribution shows to be a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) to the categorical/multinomial:

```math
\begin{align*}
\underbrace{\text{Dirichlet}}_{\text{posterior}} &\propto \underbrace{\text{categorical}}_{\text{likelihood}} \cdot \underbrace{\text{Dirichlet}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ d843c228-d294-11ef-0d34-3520dc97859c
md"""
This is actually a generalization of the conjugate relation that we found for the binary coin toss: 

```math
\begin{align*}
\underbrace{\text{Beta}}_{\text{posterior}} &\propto \underbrace{\text{binomial}}_{\text{likelihood}} \cdot \underbrace{\text{Beta}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ d843d0c4-d294-11ef-10b6-cb982615d58a
md"""
#### $(HTML("<span id='prediction-loaded-die'>Prediction of next toss for the loaded die</span>"))

Let's apply what we have learned about the loaded die to compute the probability that we throw the ``k``-th face at the next toss. 

```math
\begin{align*}
p(x_{\bullet,k}=1|D)  &= \int p(x_{\bullet,k}=1|\mu)\,p(\mu|D) \,\mathrm{d}\mu \\
  &= \int_0^1 \mu_k \times  \mathcal{Dir}(\mu|\,\alpha+m) \,\mathrm{d}\mu  \\
  &= \mathrm{E}\left[ \mu_k \right] \\
  &= \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k}
\end{align*}
```

(You can [find the mean of the Dirichlet distribution at its Wikipedia site](https://en.wikipedia.org/wiki/Dirichlet_distribution)). 

This result is simply a generalization of [**Laplace's rule of succession**](https://en.wikipedia.org/wiki/Rule_of_succession).

"""

# ╔═╡ d843defc-d294-11ef-358b-f56f514dcf93
md"""
## Categorical, Multinomial and Related Distributions

In the above derivation, we noticed that the data generating distribution for ``N`` die tosses ``D=\{x_1,\ldots,x_N\}`` only depends on the **data frequencies** ``m_k``:

```math
p(D|\mu) = \prod_n \underbrace{\prod_k \mu_k^{x_{nk}}}_{\text{categorical dist.}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k} \tag{B-2.29}
```

"""

# ╔═╡ d843efdc-d294-11ef-0f3a-630ecdd0acee
md"""
A related distribution is the distribution over data frequency observations ``D_m=\{m_1,\ldots,m_K\}``, which is called the **multinomial distribution**,

```math
p(D_m|\mu) =\frac{N!}{m_1! m_2!\ldots m_K!} \,\prod_k \mu_k^{m_k}\,.
```

"""

# ╔═╡ d84422a6-d294-11ef-148b-c762a90cd620
md"""
When used as a likelihood function for ``\mu``, it makes no difference whether you use ``p(D|\mu)`` or ``p(D_m|\mu)``. Why? 

"""

# ╔═╡ d8443e38-d294-11ef-25db-b16df87850f4
md"""
Verify for yourself that ([Exercise](http://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-The-Multinomial-Distribution.ipynb)): 

  * the categorial distribution is a special case of the multinomial for ``N=1``.
  * the Bernoulli is a special case of the categorial distribution for ``K=2``.
  * the binomial is a special case of the multinomial for ``K=2``.

"""

# ╔═╡ d8449f1a-d294-11ef-3cfa-4fc33a5daa00
md"""
## Maximum Likelihood Estimation for the Multinomial

#### Maximum likelihood as a special case of Bayesian estimation

We can get the maximum likelihood estimate ``\hat{\mu}_k`` for ``\mu_k`` based on ``N`` throws of a ``K``-sided die from the Bayesian framework by using a uniform prior for ``\mu`` and taking the mode of the posterior for ``\mu``:

```math
\begin{align*}
\hat{\mu}_k &= \arg\max_{\mu_k} p(D|\mu) \\
&= \arg\max_{\mu_k} p(D|\mu)\cdot \mathrm{Uniform}(\mu) \\
&= \arg\max_{\mu_k} p(D|\mu) \cdot \left.\mathrm{Dir}(\mu|\alpha)\right|_{\alpha=(1,1,\ldots,1)} \\
&= \arg\max_{\mu_k} \left.p(\mu|D,\alpha)\right|_{\alpha=(1,1,\ldots,1)}  \\
&= \arg\max_{\mu_k} \left.\mathrm{Dir}\left( \mu | m + \alpha \right)\right|_{\alpha=(1,1,\ldots,1)} \\
&= \frac{m_k}{\sum_k m_k} = \frac{m_k}{N}
\end{align*}
```

where we used the fact that the [maximum of the Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Mode) ``\mathrm{Dir}(\{\alpha_1,\ldots,\alpha_K\})`` is obtained at  ``(\alpha_k-1)/(\sum_k\alpha_k - K)``.

"""

# ╔═╡ d844bcfa-d294-11ef-0874-b154f3ed810b
md"""
#### $(HTML("<span id='ML-for-multinomial'>Maximum likelihood estimation by optimizing a constrained log-likelihood</span>"))

Of course, we shouldn't have to go through the full Bayesian framework to get the maximum likelihood estimate. Alternatively, we can find the maximum likelihood (ML) solution directly by optimizing the (constrained) log-likelihood.

The log-likelihood for the multinomial distribution is given by

```math
\begin{align*}
\mathrm{L}(\mu) &\triangleq \log p(D_m|\mu) \propto \log \prod_k \mu_k^{m_k} =  \sum_k m_k \log \mu_k 
\end{align*}
```

"""

# ╔═╡ d844d564-d294-11ef-0454-416352d43524
md"""
When doing ML estimation, we must obey the constraint ``\sum_k \mu_k  = 1``, which can be accomplished by a [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier). The **constrained log-likelihood** with Lagrange multiplier is then

```math
\tilde{\mathrm{L}}(\mu) = \sum_k m_k \log \mu_k  + \lambda \cdot \big(1 - \sum_k \mu_k \big)
```

The method of Lagrange multipliers is a mathematical method for transferring a constrained optimization problem to an unconstrained optimization problem (see Bishop App.E). Unconstrained optimization problems can be solved by setting the derivative to zero. 

"""

# ╔═╡ d844fa76-d294-11ef-172a-85e68842c252
md"""
Setting the derivative of ``\tilde{\mathrm{L}}(\mu)`` to zero yields the **sample proportion** for ``\mu_k`` 

```math
\begin{equation*}
\nabla_{\mu_k}   \tilde{\mathrm{L}}(\mu) = \frac{m_k }
{\hat\mu_k } - \lambda  \overset{!}{=} 0 \; \Rightarrow \; \hat\mu_k = \frac{m_k }{N}
\end{equation*}
```

where we get ``\lambda`` from the constraint 

```math
\begin{equation*}
\sum_k \hat \mu_k = \sum_k \frac{m_k}
{\lambda} = \frac{N}{\lambda} \overset{!}{=}  1
\end{equation*}
```



"""

# ╔═╡ d8453aac-d294-11ef-24c7-71ec0301c913
md"""
## Recap Maximum Likelihood Estimation for Gaussian and Multinomial Distributions

Given ``N`` IID observations ``D=\{x_1,\dotsc,x_N\}``.

For a **multivariate Gaussian** model ``p(x_n|\theta) = \mathcal{N}(x_n|\mu,\Sigma)``, we obtain ML estimates

```math
\begin{align*}
\hat \mu &= \frac{1}{N} \sum_n x_n \qquad &\text{(sample mean)} \\
\hat \Sigma &= \frac{1}{N} \sum_n (x_n-\hat\mu)(x_n - \hat \mu)^T \qquad &\text{(sample variance)}
\end{align*}
```

"""

# ╔═╡ d8455278-d294-11ef-2455-376c205e7edf
md"""
For discrete outcomes modeled by a 1-of-K **categorical distribution** we find

```math
\begin{align*}
\hat\mu_k  = \frac{1}{N} \sum_n x_{nk} \quad \left(= \frac{m_k}{N} \right) \qquad \text{(sample proportion)}
\end{align*}
```

"""

# ╔═╡ d8456524-d294-11ef-0446-891a67740b28
md"""
Note the similarity for the means between discrete and continuous data. 

"""

# ╔═╡ Cell order:
# ╟─d8422bf2-d294-11ef-0144-098f414c6454
# ╟─d8424e52-d294-11ef-0083-fbb77df4d853
# ╟─d842ad86-d294-11ef-3266-253f80ecf4b7
# ╟─d842d368-d294-11ef-024d-45e58ca994e0
# ╟─d842e5d0-d294-11ef-132f-a3445f00b389
# ╟─d842f41a-d294-11ef-1888-ddfdb5d236eb
# ╟─d842fe4c-d294-11ef-15a9-a9a6e359f47d
# ╟─d84345d2-d294-11ef-2297-39bb0b9d1a3f
# ╟─d843540a-d294-11ef-3846-2bf27b7e9b30
# ╟─d84369a4-d294-11ef-38f7-7f393869b705
# ╟─d8439866-d294-11ef-230b-dfde21aedfbf
# ╟─d843a338-d294-11ef-2748-b95f2af1396b
# ╟─d843b33c-d294-11ef-195d-2708fbfba49d
# ╟─d843c228-d294-11ef-0d34-3520dc97859c
# ╟─d843d0c4-d294-11ef-10b6-cb982615d58a
# ╟─d843defc-d294-11ef-358b-f56f514dcf93
# ╟─d843efdc-d294-11ef-0f3a-630ecdd0acee
# ╟─d84422a6-d294-11ef-148b-c762a90cd620
# ╟─d8443e38-d294-11ef-25db-b16df87850f4
# ╟─d8449f1a-d294-11ef-3cfa-4fc33a5daa00
# ╟─d844bcfa-d294-11ef-0874-b154f3ed810b
# ╟─d844d564-d294-11ef-0454-416352d43524
# ╟─d844fa76-d294-11ef-172a-85e68842c252
# ╟─d8453aac-d294-11ef-24c7-71ec0301c913
# ╟─d8455278-d294-11ef-2455-376c205e7edf
# ╟─d8456524-d294-11ef-0446-891a67740b28
