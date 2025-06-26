### A Pluto.jl notebook ###
# v0.20.14

using Markdown
using InteractiveUtils

# ╔═╡ 9021f1a2-276c-43f7-93c0-cdc1f37761d3
md"""
# The Laplace Approximation
When deriving results in Bayesian Machine Learning analytically, you will often end up with integrals that are difficult – or even impossible – to solve. This is unfortunate, because calculating integrals with a computer can be very difficult to do both accurately and efficiently.

The central idea of the Laplace approximation is to approximate a (possibly unnormalized) distribution ``f(z)`` by a Gaussian distribution ``q(z)``. By substituting ``q`` for ``f`` in an integral, we get an approximate result that is easier to solve analytically.
"""

# ╔═╡ a5c709c9-f2cf-4937-9582-669ebe569077


# ╔═╡ f7d34f2f-6515-4d50-9896-8663ab03a101
md"""
We first give the result, followed by the derivation. The **Laplace approximation** of a real-valued distribution ``f(z)`` is given by:

```math
\begin{align*}
f(z) &\approx f(z_0) \exp\left( - \frac{1}{2} (z-z_0)^T A (z-z_0)\right) \\
\text{where: } z_0 &= \arg\max_z \left( \log f(z)\right) \\
A &= - \nabla \nabla \left. \log f(z) \right|_{z=z_0}
\end{align*}
```

"""

# ╔═╡ c3c5f458-2658-460a-9efd-b4fa4c2d9d87
md"""
The Laplace approximation usually serves one or both of the following two purposes: 

1. To approximate a posterior distribution without closed-form expression by a Gaussian distribution.
2. To approximate (part of) the integrand in an integral with purpose to get an analytical solution for the integral.

"""

# ╔═╡ 52ec1121-d6f6-4fb2-8d3e-ae4fec6a0df7
md"""
#### $(HTML("<span id='Laplace-example'>Example</span>"))

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/Figure4.14a.png?raw=true)

(Bishop fig.4.14a). Laplace approximation (in red) to the distribution ``p(z)\propto \exp(-z^2/2)\sigma(20z+4)``, where ``\sigma(a)=1/(1+e^{-a})``. The Laplace approximation is centered on the mode of ``p(z)``.

"""

# ╔═╡ 6d3faf15-5377-4b54-ac2b-410ecf868f63
md"""
## Working out the Laplace Approximation

Assume that we want to approximate a distribution ``f(z)`` by a Gaussian distribution ``q(z)``.

Note that, if ``q(z)`` is a Gaussian distribution, then ``\log q(z)`` is a second-order polynomial in ``z``, so we will find the Gaussian by fitting a parabola to ``\log f(z)``. 

"""

# ╔═╡ 4b7476e4-c67a-4f1d-a7bf-26e49083c93c
md"""
#### estimation of mean

The mean (``z_0``) of ``q(z)`` is placed on the mode of ``\log f(z)``, i.e., 

```math
z_0 = \arg\max_z \left( \log f(z)\right)  \tag{B-4.126}
```

"""

# ╔═╡ 1b8a99c7-ea73-4994-a2f1-19952594d9fc
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

# ╔═╡ eae9f0d4-db52-4370-ad99-bb1fb14ea983
md"""
#### Laplace approximation construction

After taking exponentials in eq. B-4.131, we obtain

```math
f(z) \approx f(z_0) \exp\left( - \frac{1}{2} (z-z_0)^T A (z-z_0)\right) 
```

"""

# ╔═╡ 9026b9ff-5f36-4ced-b5da-47eb26c8f602
md"""
(end derivation of Laplace Approximation)

We can now identify ``q(z)`` as

```math
q(z) = \mathcal{N}\left( z\,|\,z_0, A^{-1}\right) \tag{B-4.134}
```

with ``z_0`` and ``A`` defined by eqs. B-4.126 and B-4.132. 

All we have done up to now is approximate a function ``f(z)`` by a Gaussian ``q(z)``. This procedure is called the **Laplace Approximation**. Often, the required integrals (for Bayesian marginalization) can be approximately computed if we replace ``f(z)`` by ``q(z)``. 

"""

# ╔═╡ 8286b631-e425-4752-af11-cf47ffedaed0
md"""
## Bayesian Logistic Regression with the Laplace Approximation

Let's get back to the challenge of computing the predictive class distribution (B-4.145) for Bayesian logistic regression. We first work out the Gaussian Laplace approximation ``q(w)`` to the [posterior weight distribution](#logistic-regression-posterior)

```math
\begin{align*}
\underbrace{p(w | D)}_{\text{posterior}} \propto  \underbrace{\mathcal{N}(w \,|\, m_0, S_0)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)}_{\text{likelihood}}  \tag{B-4.142}
\end{align*}
```

"""

# ╔═╡ 71c7d88e-d1b2-4edc-85e0-d9eee0501568
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

# ╔═╡ 2096c0e1-c4ff-48ea-92d7-901600a81478
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

# ╔═╡ fff20a37-b6c3-4d4a-8c38-9d219e1398b8
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

# ╔═╡ c32fa1d9-a3e5-4356-952a-027eb75815c9
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

with the $(HTML("<span id='gaussian-cdf'>Gaussian</span>")) CDF ``\Phi(x)= \frac{1}{\sqrt(2\pi)}\int_{-\infty}^{x}e^{-t^2/2}\mathrm{d}t``, ``\lambda^2= \pi / 8`` and ``\sigma(a) = 1/(1+e^{-a})``.    Thus, substituting ``\Phi(\lambda a)`` with ``\lambda^2= \pi / 8`` for ``\sigma(a)`` leads to 

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &=  \int \sigma(w^T x_\bullet) \cdot p(w|D) \,\mathrm{d}w  \\ 
&\approx \int \underbrace{\Phi(\lambda a)}_{\text{probit function}} \cdot \underbrace{\mathcal{N}\left(a\,|\, \mu_a, \sigma_a^2\right)}_{\text{Gaussian}} \,\mathrm{d}a \\ 
&= \Phi\left( \frac{\mu_a}{\sqrt(\lambda^{-2} +\sigma_a^2)}\right) \tag{B-4.152}
\end{align*}
```

"""

# ╔═╡ 6cf91cde-9be7-4ee7-9be2-dd1280aa85b0
md"""
We now have an approximate but **closed-form expression for the predictive class distribution for a new observation** with a Bayesian logistic regression model.  

"""

# ╔═╡ ba35f00c-64e6-442d-9bf3-e6571928d694
md"""
Note that, by [Eq.B-4.143](#Laplace-posterior-logistic-regression), the variance ``S_N`` (and consequently ``\sigma_a^2``) for the weight vector depends on the distribution of the training set. Large uncertainty about the weights (in areas with little training data and uninformative prior variance ``S_0``) increases ``\sigma_a^2`` and takes the posterior class probability eq. B-4.152 closer to ``0.5``. Does that make sense?

"""

# ╔═╡ fba106da-6891-4852-8d9c-ec877b1d0a19
md"""
Apparently, the Laplace approximation leads to a closed-form solutions for Bayesian logistic regression (although admittedly, the derivation is no walk in the park). 

"""

# ╔═╡ 5465e414-7512-4730-97e9-ea6337014d9c
md"""
!!! info
	Exam guide: The derivation of closed-form expression eq. B-4.152 for the predictive class distribution requires clever tricks and is therefore not something that you should be able to reproduce at the exam without assistance. You should understand the Laplace Approximation though and be able to work out simpler examples.  

"""

# ╔═╡ Cell order:
# ╟─9021f1a2-276c-43f7-93c0-cdc1f37761d3
# ╟─a5c709c9-f2cf-4937-9582-669ebe569077
# ╟─f7d34f2f-6515-4d50-9896-8663ab03a101
# ╟─c3c5f458-2658-460a-9efd-b4fa4c2d9d87
# ╟─52ec1121-d6f6-4fb2-8d3e-ae4fec6a0df7
# ╟─6d3faf15-5377-4b54-ac2b-410ecf868f63
# ╟─4b7476e4-c67a-4f1d-a7bf-26e49083c93c
# ╟─1b8a99c7-ea73-4994-a2f1-19952594d9fc
# ╟─eae9f0d4-db52-4370-ad99-bb1fb14ea983
# ╟─9026b9ff-5f36-4ced-b5da-47eb26c8f602
# ╟─8286b631-e425-4752-af11-cf47ffedaed0
# ╟─71c7d88e-d1b2-4edc-85e0-d9eee0501568
# ╟─2096c0e1-c4ff-48ea-92d7-901600a81478
# ╟─fff20a37-b6c3-4d4a-8c38-9d219e1398b8
# ╟─c32fa1d9-a3e5-4356-952a-027eb75815c9
# ╟─6cf91cde-9be7-4ee7-9be2-dd1280aa85b0
# ╟─ba35f00c-64e6-442d-9bf3-e6571928d694
# ╟─fba106da-6891-4852-8d9c-ec877b1d0a19
# ╟─5465e414-7512-4730-97e9-ea6337014d9c
