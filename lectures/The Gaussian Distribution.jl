### A Pluto.jl notebook ###
# v0.20.8

#> [frontmatter]
#> image = "https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/fig-linear-system.png?raw=true"
#> description = "Review of information processing with Gaussian distributions in linear systems."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ c97c495c-f7fe-4552-90df-e2fb16f81d15
using PlutoUI, PlutoTeachingTools

# ╔═╡ 3ec821fd-cf6c-4603-839d-8c59bb931fa9
using Distributions, Plots, LaTeXStrings

# ╔═╡ 69d951b6-58b3-4ce2-af44-4cb799e453ff
using HypertextLiteral

# ╔═╡ 5638c1d0-db95-49e4-bd80-528f79f2947e
using HCubature, LinearAlgebra# Numerical integration package

# ╔═╡ b9abf984-d294-11ef-1eaa-3358379f8b44
begin
  using SpecialFunctions
  let
	X = Normal(0, 1)
	Y = Normal(0, 1)
	pdf_product_std_normals(z::Real) = besselk(0, abs(z))/π
	
	range1 = range(-4,stop=4,length=100)
	plot(range1, t -> pdf(X, t); label=L"p(X)=p(Y)=\mathcal{N}(0,1)", fill=(0, 0.1))
	plot!(range1, t -> pdf(X,t)*pdf(Y,t); label=L"p(X)*p(Y)", fill=(0, 0.1))
	plot!(range1, pdf_product_std_normals; label=L"p(Z=X*Y)", fill=(0, 0.1))
  end
end

# ╔═╡ b9a38e20-d294-11ef-166b-b5597125ed6d
md"""
# Continuous Data and the Gaussian Distribution

"""

# ╔═╡ 5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
PlutoUI.TableOfContents()

# ╔═╡ b9a46c3e-d294-11ef-116f-9b97e0118e5b
md"""
## Preliminaries

Goal 

  * Review of information processing with Gaussian distributions in linear systems

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 85-93
      * [MacKay - 2006 - The Humble Gaussian Distribution](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Mackay-2006-The-humble-Gaussian-distribution.pdf) (highly recommended!)
      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.30-34, section 2.8, the Gaussian distribution
  * References

      * [E.T. Jaynes - 2003 - Probability Theory, The Logic of Science](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf) (best book available on the Bayesian view on probability theory)

"""

# ╔═╡ b9a48c60-d294-11ef-3b90-03053fcd82fb
md"""
## Challenge: Classify a Gaussian Sample Point

Consider a data set as shown in the figure below

"""


# ╔═╡ ba57ecbb-b64e-4dd8-8398-a90af1ac71f3
begin
	N = 100;
	generative_dist = MvNormal([0,1.], [0.8 0.5; 0.5 1.0]);
	D = rand(generative_dist, N);
	x_dot = rand(generative_dist);
	
	let
		scatter(D[1,:], D[2,:], marker=:x, markerstrokewidth=3, label=L"D")
		scatter!([x_dot[1]], [x_dot[2]], label=L"x_\bullet")
		plot!(range(0, 2), [1., 1., 1.], fillrange=2, alpha=0.4, color=:gray,label=L"S")
	end
end

# ╔═╡ 02853a5c-f6aa-4af8-8a25-bfffd4b96afc
md"""

##### Problem 

- Consider a set of observations ``D=\{x_1,…,x_N\}`` in the 2-dimensional plane (see Figure). All observations were generated using the same process. We now draw an extra observation ``x_\bullet = (a,b)`` from the same data-generating process. What is the probability that ``x_\bullet`` lies within the shaded rectangle ``S = \{ (x,y) \in \mathbb{R}^2 | 0 \leq x \leq 2, 1 \leq y \leq 2 \} ``?


##### Solution 

- See later in this lecture. 
"""

# ╔═╡ 71f1c8ee-3b65-4ef8-b36f-3822837de410
md"""
# The Gaussian Distribution
"""

# ╔═╡ b9a4eb62-d294-11ef-06fa-af1f586cbc15
md"""
## The Moment Parameterization 

Consider a random (vector) variable ``x \in \mathbb{R}^M`` that is "normally" (i.e., Gaussian) distributed. The *moment* parameterization of the Gaussian distribution is completely specified by its *mean* ``\mu`` and *variance* ``\Sigma`` parameters, and given by

```math
p(x | \mu, \Sigma) = \mathcal{N}(x|\mu,\Sigma) \triangleq \frac{1}{\sqrt{(2\pi)^M |\Sigma|}} \,\exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)\,,
```

where ``|\Sigma| \triangleq \mathrm{det}(\Sigma)`` is the determinant of ``\Sigma``.  

For a scalar real variable ``x \in \mathbb{R}``, this works out to 

```math
p(x | \mu, \sigma^2) =  \frac{1}{\sqrt{2\pi\sigma^2 }} \,\exp\left(-\frac{(x-\mu)^2}{2 \sigma^2} \right)\,.
```

It is common to write the (scalar) variance parameter as `` \sigma^2 `` to emphasize that the variance is non-negative.

"""

# ╔═╡ b9a50d0c-d294-11ef-0e60-2386cf289478
md"""

## The Canonical (Natural) Parameterization 

Alternatively, the $(HTML("<span id='natural-parameterization'></span>"))*canonical* (a.k.a. *natural*  or *information* ) parameterization of the Gaussian distribution is given by

```math
\begin{equation*}
p(x | \eta, \Lambda) = \mathcal{N}_c(x|\eta,\Lambda)  = \exp\left( a + \eta^T x - \frac{1}{2}x^T \Lambda x \right) \,,
\end{equation*}
```
where
```math
a = -\frac{1}{2} \left( M \log(2 \pi) - \log |\Lambda| + \eta^T \Lambda \eta\right)
```

is the *normalizing* constant that ensures that ``\int p(x)\mathrm{d}x = 1``, and

```math
\Lambda = \Sigma^{-1}
```

is called the *precision* matrix. The parameter

```math
\eta = \Sigma^{-1} \mu
```

is the *natural* mean, or for clarity, often called the *precision-weighted* mean.

The Gaussian distribution can be expressed in both moment and natural parameterizations, which are mathematically equivalent but differ in how the parameters are defined.

"""

# ╔═╡ b9a52b18-d294-11ef-2d42-19c5e3ef3549
md"""
## Why the Gaussian?
"""

# ╔═╡ b9a5589a-d294-11ef-3fc3-0552a69df7b2
md"""

Why is the Gaussian distribution so ubiquitously used in science and engineering? 

1. Operations on probability distributions tend to lead to Gaussian distributions:

    * Any smooth function with a single rounded maximum goes into a Gaussian function, if raised to higher and higher powers. (useful in sequential Bayesian inference).
    * The [Gaussian distribution has higher entropy](https://en.wikipedia.org/wiki/Differential_entropy#Maximization_in_the_normal_distribution) than any other with the same variance. 

        * Therefore, any operation on a probability distribution that discards information but preserves variance gets us closer to a Gaussian.
        * As an example, see [Jaynes, section 7.1.4](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250) for how this leads to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which results from performing convolution operations on distributions.


2. Once the Gaussian has been attained, this form tends to be preserved. e.g.,   

    * The convolution of two Gaussian functions is another Gaussian function (useful in the sum of 2 variables and linear transformations)
    * The product of two Gaussian functions is another Gaussian function (useful in Bayes rule).
    * The Fourier transform of a Gaussian function is another Gaussian function.

See also [Jaynes, section 7.14](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250), and the whole chapter 7 in his book for more details on why the Gaussian distribution is so useful.

"""

# ╔═╡ 9501922f-b928-46e2-8f23-8eb9c64f6198
md"""
# Computing with Gaussians
"""

# ╔═╡ b9a5889c-d294-11ef-266e-d90225222e10
md"""
## Linear Transformations of Gaussian Variables

As shown in the [probability theory lecture](https://bmlip.github.io/colorized/lectures/Probability%20Theory%20Review.html#linear-transformation), under the linear transformation 

```math
z = Ax + b \,,
```
for given ``A`` and ``b``, the mean and covariance of ``z`` are given by ``\mu_z = A\mu_x + b`` and ``\Sigma_z = A\Sigma_x A^\top``, regardless of the distribution of x.

Since a Gaussian distribution is fully specified by its mean and covariance matrix, it follows that a linear transformation ``z=Ax+b`` of a Gaussian variable ``x \sim \mathcal{N}(\mu_x,\Sigma_x)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) \,. \tag{SRG-4a}
```

In case ``x`` is not Gaussian, higher order moments may be needed to specify the distribution for ``z``. 


"""

# ╔═╡ a82378ae-d1be-43f9-b63a-2f897767d1fb
md"""
## Example: The Sum of Gaussian Variables 

A commonly occurring example of a linear transformation is the *sum of two independent Gaussian variables*:


##### Problem

Let ``x \sim \mathcal{N} \left(\mu_x, \sigma_x^2 \right)`` and ``y \sim \mathcal{N} \left(\mu_y, \sigma_y^2 \right)``. What is the PDF for ``z=x+y\, ``?

##### Solution

First, recognize that ``z=x+y`` can be written as a linear transformation ``z = A w``, where

```math
A = \begin{bmatrix} 1 & 1 \end{bmatrix} \text{ ,  and } w = \begin{bmatrix} x \\ y \end{bmatrix} \,.
```

Using the above formula for linear transformations, it follows that
```math
\begin{align*}
p(z) &= \mathcal{N}\big(z\,\big|\,A \mu_w, A \Sigma_w A^T \big) \\
  &= \mathcal{N}\bigg(z\, \bigg|\,\begin{bmatrix} 1 & 1 \end{bmatrix}  \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix} 1 & 1 \end{bmatrix}  \begin{bmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \bigg) \\
  &= \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \sigma_x^2 +\sigma_y^2 \right) \tag{SRG-8}
\end{align*}
```

Consequently, the sum of two independent Gaussian random variables remains Gaussian, with its mean given by the sum of the means and its variance given by the sum of the variances.

Home exercise: Following the example above, can you compute the PDF for ``z`` if ``x`` and ``y`` were *dependent* Gaussian variables?

A common mistake is to confuse the *sum of two Gaussian-distributed variables*, which remains Gaussian-distributed (see above), with the *sum of two Gaussian distributions*, which is typically not a Gaussian distribution.

"""

# ╔═╡ b9a5a82c-d294-11ef-096f-ffee478aca20
md"""
## Example: Gaussian Signals in a Linear System

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/fig-linear-system.png?raw=true)

Given independent variables ``x \sim \mathcal{N}(\mu_x,\sigma_x^2)`` and ``y \sim \mathcal{N}(\mu_y,\sigma_y^2)``, what is the PDF for ``z = A\cdot(x -y) + b`` ? (for answer, see [Exercises](http://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-The-Gaussian-Distribution.ipynb))

"""

# ╔═╡ b9a5b7e0-d294-11ef-213e-4b72b8c88db7
md"""
Think about the role of the Gaussian distribution for stochastic linear systems in relation to what sinusoidals mean for deterministic linear system analysis.

"""

# ╔═╡ b9a5cbc2-d294-11ef-214a-c71fb1272326
md"""
## Bayesian Inference for the Gaussian

##### Problem

Let's estimate a constant ``\theta`` from one ''noisy'' measurement ``x`` about that constant. 

We assume the following measurement equations (the tilde ``\sim`` means: 'is distributed as'):

```math
\begin{align*}
x &= \theta + \epsilon \\
\epsilon &\sim \mathcal{N}(0,\sigma^2)
\end{align*}
```

Also, let's assume a Gaussian prior for ``\theta``

```math
\begin{align*}
\theta &\sim \mathcal{N}(\mu_0,\sigma_0^2) \\
\end{align*}
```

For simplicity, we will assume that ``\sigma^2``, ``\mu_0`` and ``\sigma_0^2`` are given. 

What is the PDF for the posterior ``p(\theta|x)`` ?
"""

# ╔═╡ b9a5dcc0-d294-11ef-2c85-657a460db5cd
md"""
#### Model specification

Note that you can rewrite these specifications in probabilistic notation as follows:

```math
\begin{align*}
    p(x|\theta) &=  \mathcal{N}(x|\theta,\sigma^2) \\
    p(\theta) &=\mathcal{N}(\theta|\mu_0,\sigma_0^2)
\end{align*}
```

"""

# ╔═╡ b9a6557e-d294-11ef-0a90-d74c337ade25
md"""
#### Inference
"""

# ╔═╡ b9a67d06-d294-11ef-297b-eb9039786ea7
md"""
Let's do Bayes rule for the posterior PDF ``p(\theta|x)``. 

```math
\begin{align*}
p(\theta|x)  &= \frac{p(x|\theta) p(\theta)}{p(x)} \propto p(x|\theta) p(\theta)  \\
    &= \mathcal{N}(x|\theta,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2)   \\
    &\propto \exp \left\{   -\frac{(x-\theta)^2}{2\sigma^2} - \frac{(\theta-\mu_0)^2}{2\sigma_0^2} \right\}  \\
    &\propto \exp \left\{ \theta^2 \cdot \left( -\frac{1}{2 \sigma_0^2} - \frac{1}{2\sigma^2}  \right)  + \theta \cdot  \left( \frac{\mu_0}{\sigma_0^2} + \frac{x}{\sigma^2}\right)   \right\} \\
    &= \exp\left\{ -\frac{\sigma_0^2 + \sigma^2}{2 \sigma_0^2 \sigma^2} \left( \theta - \frac{\sigma_0^2 x +  \sigma^2 \mu_0}{\sigma^2 + \sigma_0^2}\right)^2  \right\} 
\end{align*}
```

which we recognize as a Gaussian distribution w.r.t. ``\theta``. 

"""

# ╔═╡ b9a68d3a-d294-11ef-2335-093a39648007
md"""
(Just as an aside,) this computational 'trick' for multiplying two Gaussians is called **completing the square**. The procedure makes use of the equality 

```math
ax^2+bx+c_1 = a\left(x+\frac{b}{2a}\right)^2+c_2
```

"""

# ╔═╡ b9a697fa-d294-11ef-3a57-7b7ba1f4fd70
md"""
In particular, it follows that the posterior for ``\theta`` is

```math
\begin{equation*}
    p(\theta|x) = \mathcal{N} (\theta |\, \mu_1, \sigma_1^2)
\end{equation*}
```

where

```math
\begin{align*}
  \frac{1}{\sigma_1^2}  &= \frac{\sigma_0^2 + \sigma^2}{\sigma^2 \sigma_0^2} = \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}  \\
  \mu_1   &= \frac{\sigma_0^2 x +  \sigma^2 \mu_0}{\sigma^2 + \sigma_0^2} = \sigma_1^2 \, \left(  \frac{1}{\sigma_0^2} \mu_0 + \frac{1}{\sigma^2} x \right) 
\end{align*}
```

So, multiplication of two Gaussian distributions yields another (unnormalized) Gaussian with

  * posterior precision equals **sum of prior precisions**
  * posterior precision-weighted mean equals **sum of prior precision-weighted means**


"""

# ╔═╡ b9a6b7b2-d294-11ef-06dc-4de5ef25c1fd
md"""

## Conjugate Distributions

As we just saw, a Gaussian prior, combined with a Gaussian likelihood, makes Bayesian inference analytically solvable (!), since 

```math
\begin{equation*}
\underbrace{\text{Gaussian}}_{\text{posterior}}
 \propto \underbrace{\text{Gaussian}}_{\text{likelihood}} \times \underbrace{\text{Gaussian}}_{\text{prior}} \,.
\end{equation*}
```


"""

# ╔═╡ 702e7b10-14a4-42da-a192-f7c02a3d470a
md"""
When applying Bayes rule, if the posterior distribution belongs to the same family as the prior (e.g., both are Gaussian distributions), we say that the prior and the likelihood form a conjugate pair.
"""

# ╔═╡ 51d81901-213f-42ce-b77e-10f7ca4a4145

keyconcept("", md"In Bayesian inference, a Gaussian prior distribution is **conjugate** to a Gaussian likelihood (when the variance is known), which ensures that the posterior distribution remains Gaussian. This conjugacy greatly simplifies calculation of Bayes rule.")


# ╔═╡ b9a6c7b6-d294-11ef-0446-c372aa610df8
md"""

## (Multivariate) Gaussian Multiplication


$(HTML("<span id='Gaussian-multiplication'></span>")) In general, the multiplication of two multi-variate Gaussians over ``x`` yields an (unnormalized) Gaussian over ``x``:

```math
\begin{equation*}
\boxed{\mathcal{N}(x|\mu_a,\Sigma_a) \cdot \mathcal{N}(x|\mu_b,\Sigma_b) = \underbrace{\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)}_{\text{normalization constant}} \cdot \mathcal{N}(x|\mu_c,\Sigma_c)} \tag{SRG-6}
\end{equation*}
```

where

```math
\begin{align*}
\Sigma_c^{-1} &= \Sigma_a^{-1} + \Sigma_b^{-1} \\
\Sigma_c^{-1} \mu_c &= \Sigma_a^{-1}\mu_a + \Sigma_b^{-1}\mu_b
\end{align*}
```

"""

# ╔═╡ b9a6ecd2-d294-11ef-02af-37c977f2814b
md"""
Check out that normalization constant ``\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)``. Amazingly, this constant can also be expressed by a Gaussian!

"""

# ╔═╡ b9a6f916-d294-11ef-38cb-b78c0c448550
md"""

Also note that Bayesian inference is trivial in the [*canonical* parameterization of the Gaussian](#natural-parameterization), where we would get

```math
\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}
```

This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 

"""

# ╔═╡ d2bedf5f-a0ea-4604-b5da-adf9f11e80be
md"""
It is important to distinguish between two concepts: the *product of Gaussian distributions*, which results in a (possibly unnormalized) Gaussian distribution, and the *product of Gaussian-distributed variables*, which generally does not yield a Gaussian-distributed variable. See the [optional slides below](#product-of-gaussians) for further discussion.
"""

# ╔═╡ 1e587633-06ab-442f-b6ed-3a994e19a613
TODO("link to optional slide does not seem to work.")

# ╔═╡ 45c2fb37-a078-4284-9e04-176156cffb1e
d1 = Normal(0, 1); # μ=0, σ^2=1

# ╔═╡ d81483db-3826-4ff4-9d52-e23418da07d0
d2 = Normal(3, 2); # μ=3, σ^2=4

# ╔═╡ e6a2d2ed-0100-4570-85c1-fc9d8f84e32e
TODO("Can we play with these parameter values so that the plot below moves.")

# ╔═╡ 14fd14db-26da-4f0b-81d0-59ee4ab1a35c
md"""
We can calculate the parameters of the product `d1*d2`.
"""

# ╔═╡ f9cf453a-6369-4d38-9dad-fb3412497635
s2_prod = (d1.σ^-2 + d2.σ^-2)^-1

# ╔═╡ 9f939dd4-18e8-464c-a12e-eb320d5fd88b
m_prod = s2_prod * ((d1.σ^-2)*d1.μ + (d2.σ^-2)*d2.μ)

# ╔═╡ fa03d8c8-bead-456f-833e-cc0690e1b528
TODO("Fons: I don't like 16 decimals for m_prod =
0.600 000 000 000 0001. Can you just round to 2 decimals?")

# ╔═╡ 6cbf7a96-9e73-4289-9970-88e30cea28a5
d_prod = Normal(m_prod, sqrt(s2_prod)) # (Note that we neglect the normalization constant.)

# ╔═╡ df8867ed-0eff-4a52-8f5e-2472467e1aa2
let
	x = range(-4, stop=8, length=100)
	fill = (0, 0.1)
	
	# Plot the first Gaussian
	plot(x, pdf.(d1,x); label=L"\mathcal{N}(0,1)", fill)
	
	# Plot the second Gaussian
	plot!(x, pdf.(d2,x); label=L"\mathcal{N}(3,4)", fill)
	
	#  Plot the exact product
	plot!(x, pdf.(d1,x) .* pdf.(d2,x); label=L"\mathcal{N}(0,1) \mathcal{N}(3,4)", fill)
	
	# Plot the normalized Gaussian product
	plot!(x, pdf.(d_prod,x); label=L"Z^{-1} \mathcal{N}(0,1) \mathcal{N}(3,4)", fill)
end

# ╔═╡ b9a885a8-d294-11ef-079e-411d3f1cda03
md"""
## Conditioning and Marginalization of a Gaussian

Let ``z = \begin{bmatrix} x \\ y \end{bmatrix}`` be jointly normal distributed as

```math
\begin{align*}
p(z) &= \mathcal{N}(z | \mu, \Sigma) 
  =\mathcal{N} \left( \begin{bmatrix} x \\ y \end{bmatrix} \left| \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, 
  \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \right. \right)
\end{align*}
```

Since covariance matrices are by definition symmetric, it follows that ``\Sigma_x`` and ``\Sigma_y`` are symmetric and ``\Sigma_{xy} = \Sigma_{yx}^T``.

Let's factorize ``p(z) = p(x,y)`` as ``p(x,y) = p(y|x) p(x)`` through conditioning and marginalization.

##### conditioning
```math
\begin{equation*}
\boxed{ p(y|x) = \mathcal{N}\left(y\,|\,\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x),\, \Sigma_y - \Sigma_{yx}\Sigma_x^{-1}\Sigma_{xy} \right)}
\end{equation*}
```

##### marginalization
```math
\begin{equation*}
 \boxed{ p(x) = \mathcal{N}\left( x|\mu_x, \Sigma_x \right)}
\end{equation*}
```

**proof**: in [Bishop](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) pp.87-89

Hence, conditioning and marginalization in Gaussians lead to Gaussians again. This is very useful for applications in Bayesian inference in jointly Gaussian systems.

With a natural parameterization of the Gaussian ``p(z) = \mathcal{N}_c(z|\eta,\Lambda)`` with precision matrix ``\Lambda = \Sigma^{-1} = \begin{bmatrix} \Lambda_x & \Lambda_{xy} \\ \Lambda_{yx} & \Lambda_y \end{bmatrix}``,  the conditioning operation results in a simpler result, see Bishop pg.90, eqs. 2.96 and 2.97. 

As an exercise, interpret the formula for the conditional mean (``\mathbb{E}[y|x]=\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x)``) as a prediction-correction operation.

"""

# ╔═╡ b9a99fcc-d294-11ef-3de4-5369d9796de7
let
	# Define the joint distribution p(x,y)
	μ = [1.0; 2.0]
	Σ = [0.3 0.7;
	     0.7 2.0]
	joint = MvNormal(μ,Σ)
	
	# Define the marginal distribution p(x)
	marginal_x = Normal(μ[1], sqrt(Σ[1,1]))
	
	# Plot p(x,y)
	x_range = y_range = range(-2,stop=5,length=1000)
	joint_pdf = [ pdf(joint, [x_range[i];y_range[j]]) for  j=1:length(y_range), i=1:length(x_range)]
	plot_1 = heatmap(x_range, y_range, joint_pdf, title = L"p(x, y)")
	
	# Plot p(x)
	plot_2 = plot(range(-2,stop=5,length=1000), pdf.(marginal_x, range(-2,stop=5,length=1000)), title = L"p(x)", label="", fill=(0, 0.1))
	
	# Plot p(y|x = 0.1)
	x = 0.1
	conditional_y_m = μ[2]+Σ[2,1]*inv(Σ[1,1])*(x-μ[1])
	conditional_y_s2 = Σ[2,2] - Σ[2,1]*inv(Σ[1,1])*Σ[1,2]
	conditional_y = Normal(conditional_y_m, sqrt.(conditional_y_s2))
	plot_3 = plot(range(-2,stop=5,length=1000), pdf.(conditional_y, range(-2,stop=5,length=1000)), title = L"p(y|x = %$x)", label="", fill=(0, 0.1))
	plot(plot_1, plot_2, plot_3, layout=(1,3), size=(1200,300))
end

# ╔═╡ f4ce24c7-7d03-4574-81e0-d4cbb818a897
TODO("Fons can you let the student play with some parameter values?")

# ╔═╡ b9a9b8e0-d294-11ef-348d-c197c4ce2b8c
md"""
As is clear from the plots, the conditional distribution is a renormalized slice from the joint distribution.

"""

# ╔═╡ b9a9dca8-d294-11ef-04ec-a9202c319f89
md"""
## Gaussian Conditioning Revisited

Consider (again) the system 

```math
\begin{align*}
p(x\,|\,\theta) &= \mathcal{N}(x\,|\,\theta,\sigma^2) \\
p(\theta) &= \mathcal{N}(\theta\,|\,\mu_0,\sigma_0^2)
\end{align*}
```

"""

# ╔═╡ b9a9f98e-d294-11ef-193a-0dbdbfffa86f
md"""
Let ``z = \begin{bmatrix} x \\ \theta \end{bmatrix}``. The distribution for ``z`` is then given by (see [Gaussian distribution Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-The-Gaussian-Distribution.ipynb))

```math
p(z) = p\left(\begin{bmatrix} x \\ \theta \end{bmatrix}\right) = \mathcal{N} \left( \begin{bmatrix} x\\ 
  \theta  \end{bmatrix} 
  \,\left|\, \begin{bmatrix} \mu_0\\ 
  \mu_0\end{bmatrix}, 
         \begin{bmatrix} \sigma_0^2+\sigma^2  & \sigma_0^2\\ 
         \sigma_0^2 &\sigma_0^2 
  \end{bmatrix} 
  \right. \right)
```

"""

# ╔═╡ b9aa27da-d294-11ef-0780-af9d89f9f599
md"""
Direct substitution of the rule for Gaussian conditioning leads to the $(HTML("<span id='precision-weighted-update'>posterior</span>")) (derivation as an Exercise):

```math
\begin{align*}
p(\theta|x) &= \mathcal{N} \left( \theta\,|\,\mu_1, \sigma_1^2 \right)\,,
\end{align*}
```

with

```math
\begin{align*}
K &= \frac{\sigma_0^2}{\sigma_0^2+\sigma^2} \qquad \text{($K$ is called: Kalman gain)}\\
\mu_1 &= \mu_0 + K \cdot (x-\mu_0)\\
\sigma_1^2 &= \left( 1-K \right) \sigma_0^2  
\end{align*}
```

"""

# ╔═╡ b9aa3950-d294-11ef-373f-d5d330694bfd

keyconcept("", md"For jointly Gaussian systems, inference can be performed in a single step using closed-form expressions for conditioning and marginalization of (multivariate) Gaussian distributions.")


# ╔═╡ 0072e73e-1569-4ce4-bffb-280823499f0d
md"""
# Advanced Bayesian Inference
"""

# ╔═╡ b9a80522-d294-11ef-39d8-53a536d66bf9
md"""
## Bayesian Inference with Multiple Observations


#### model specification

Now consider that we measure a data set ``D = \{x_1, x_2, \ldots, x_N\}``, with measurements

```math
\begin{aligned}
x_n &= \theta + \epsilon_n \\
\epsilon_n &\sim \mathcal{N}(0,\sigma^2) \,,
\end{aligned}
```

and the same prior for ``\theta``:

```math
\theta \sim \mathcal{N}(\mu_0,\sigma_0^2) \\
```

Let's derive the predictive distribution ``p(x_{N+1}|D)`` for the next sample. 


#### inference

First, we derive the posterior for ``\theta``:

```math
\begin{align*}
p(\theta|D) \propto  \underbrace{\mathcal{N}(\theta|\mu_0,\sigma_0^2)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \mathcal{N}(x_n|\theta,\sigma^2)}_{\text{likelihood}} \,.
\end{align*}
```

Since the posterior is formed by multiplying ``N+1`` Gaussian distributions in ``\theta``, the result is also Gaussian in ``\theta``, due to the closure of the Gaussian family under multiplication (up to a normalization constant).

Using the property that precisions and precision-weighted means add when Gaussians are multiplied, we can immediately write the posterior as

```math
p(\theta|D) = \mathcal{N} (\theta |\, \mu_N, \sigma_N^2)
```

where 

```math
\begin{align*}
  \frac{1}{\sigma_N^2}  &= \frac{1}{\sigma_0^2} + \sum_n \frac{1}{\sigma^2}  \tag{B-2.142} \\
  \mu_N   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \tag{B-2.141}
\end{align*}
```

#### application: prediction of future sample

With the posterior over the model parameters in hand, we can now evaluate the posterior predictive distribution for the next sample ``x_{N+1}`` as

```math
\begin{align*}
 p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &= \int \mathcal{N}(x_{N+1}|\theta,\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{1}{=} \int \mathcal{N}(\theta|x_{N+1},\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{2}{=} \int  \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta \tag{use SRG-6} \\
  &= \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \underbrace{\int \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta}_{=1} \\
  &=\mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 )
\end{align*}
```

Note that uncertainty about ``x_{N+1}`` involves both uncertainty about the parameter (``\sigma_N^2``) and observation noise ``\sigma^2``.

To follow the above derivation of ``p(x_{N+1}|D)``, note that transition ``1`` relies on the identity
```math
\mathcal{N}(x|\mu,\Sigma) = \mathcal{N}(\mu|x,\Sigma)
```
and transition ``2`` derives from using the multiplication rule for Gaussians.
"""

# ╔═╡ b9a85716-d294-11ef-10e0-a7b08b800a98
md"""
## Maximum Likelihood Estimation for the Gaussian

In order to determine the *maximum likelihood* estimate of ``\theta``, we let ``\sigma_0^2 \rightarrow \infty`` (leads to uniform prior for ``\theta``), yielding ``\frac{1}{\sigma_N^2} = \frac{N}{\sigma^2}`` and consequently

```math
\begin{align*}
  \mu_{\text{ML}}  = \left.\mu_N\right\vert_{\sigma_0^2 \rightarrow \infty} = \sigma_N^2 \, \left(   \frac{1}{\sigma^2}\sum_n  x_n  \right) = \frac{1}{N} \sum_{n=1}^N x_n 
  \end{align*}
```

As expected, having an expression for the maximum likelihood estimate, it is now possible to rewrite the (Bayesian) posterior mean for ``\theta`` as the combination of a prior-based prediction and likelihood-based correction:

```math
\begin{align*}
  \underbrace{\mu_N}_{\text{posterior}}   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \\
  &= \frac{\sigma_0^2 \sigma^2}{N\sigma_0^2 + \sigma^2} \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \\
  &= \frac{ \sigma^2}{N\sigma_0^2 + \sigma^2}   \mu_0 + \frac{N \sigma_0^2}{N\sigma_0^2 + \sigma^2} \mu_{\text{ML}}   \\
  &= \underbrace{\mu_0}_{\text{prior}} + \underbrace{\underbrace{\frac{N \sigma_0^2}{N \sigma_0^2 + \sigma^2}}_{\text{gain}}\cdot \underbrace{\left(\mu_{\text{ML}} - \mu_0 \right)}_{\text{prediction error}}}_{\text{correction}}\tag{B-2.141}
\end{align*}
```
Hence, the posterior mean always lies somewhere between the prior mean ``\mu_0`` and the maximum likelihood estimate (the "data" mean) ``\mu_{\text{ML}}``.

(Of course, in practical applications, the maximum likelihood estimate is not obtained by first computing the full Bayesian posterior and then applying simplifications. This derivation is included solely to illuminate the connection between Bayesian inference and maximum likelihood estimation.)
"""

# ╔═╡ b9aa930a-d294-11ef-37ec-8d17be226c74
md"""
## Recursive Bayesian Estimation for Adaptive Signal Processing

##### Problem

Consider a signal 

```math
x_t=\theta+\epsilon_t \, \text{,    with    } \epsilon_t \sim \mathcal{N}(0,\sigma^2)\,,
```
where ``D_t= \left\{x_1,\ldots,x_t\right\}`` is observed *sequentially* (over time). Derive a **recursive** algorithm for 
```math
p(\theta|D_t) \,,
```
i.e., an update rule for (posterior) ``p(\theta|D_t)``, based on (prior) ``p(\theta|D_{t-1})`` and (a new observation) ``x_t``.

"""

# ╔═╡ b9aabe9a-d294-11ef-2489-e9fc0dbb760a
md"""
#### Model specification

The data-generating distribution is given as
```math
p(x_t|\theta) = \mathcal{N}(x_t\,|\, \theta,\sigma^2)\,.
```

For a given new measurement ``x_t`` and given ``\sigma^2``, this equation can also be read as a likelihood function for $\theta$. 

We now need a prior for $\theta$. Let's define the estimate for $\theta$ after ``t`` observations (i.e., our *solution* ) as ``p(\theta|D_t) = \mathcal{N}(\theta\,|\,\mu_t,\sigma_t^2)``. The prior is then given by

```math
p(\theta|D_{t-1}) = \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2)\,.
```

"""

# ╔═╡ b9aad50e-d294-11ef-23d2-8d2bb3b47574
md"""
#### Inference

Use Bayes rule,

```math
\begin{align*}
p(\theta|D_t) &= p(\theta|x_t,D_{t-1}) \\
  &\propto p(x_t,\theta | D_{t-1}) \\
  &= p(x_t|\theta) \, p(\theta|D_{t-1}) \\
  &= \mathcal{N}(x_t|\theta,\sigma^2) \, \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2) \\
  &= \mathcal{N}(\theta|x_t,\sigma^2) \, \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2) \;\;\text{(note this trick)}\\
  &= \mathcal{N}(\theta|\mu_t,\sigma_t^2) \;\;\text{(use Gaussian multiplication formula SRG-6)}
\end{align*}
```

with

```math
\begin{align*}
K_t &= \frac{\sigma_{t-1}^2}{\sigma_{t-1}^2+\sigma^2} \qquad \text{(Kalman gain)}\\
\mu_t &= \mu_{t-1} + K_t \cdot (x_t-\mu_{t-1})\\
\sigma_t^2 &= \left( 1-K_t \right) \sigma_{t-1}^2 
\end{align*}
```

"""

# ╔═╡ b9aaee4a-d294-11ef-2ed7-0dcb360d8bb7
md"""
This *online* (recursive) estimator of mean and variance in Gaussian observations is called a **Kalman Filter**.

 

"""

# ╔═╡ b9aafc6e-d294-11ef-1b1a-df718c1f1a58
md"""
Note that the so-called Kalman gain ``K_t`` serves as a "learning rate" (step size) in the update equation for the posterior mean ``\mu_t``.

"""

# ╔═╡ e2fc4945-4f88-4520-b56c-c7208b62c29d
keyconcept("", md"Bayesian inference does not require manual tuning of a learning rate; instead, it adapts its own effective learning rate via balancing prior beliefs with incoming evidence.")
 

# ╔═╡ b9ab0b46-d294-11ef-13c5-8314655f7867
md"""
Note that the uncertainty about ``\theta`` decreases over time (since ``0<(1-K_t)<1``). If we assume that the statistics of the system do not change (stationarity), each new sample provides new information about the process, so the uncertainty decreases. 

"""

# ╔═╡ b9ab1dd4-d294-11ef-2e86-31c4a4389475
md"""
Recursive Bayesian estimation as discussed here is the basis for **adaptive signal processing** algorithms such as the [Least Mean Squares](https://en.wikipedia.org/wiki/Least_mean_squares_filter) (LMS) filter and the [Recursive Least Squares](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) (RLS) filter. Both RLS and LMS are special cases of Recursive Bayesian estimation.

"""

# ╔═╡ 85b15f0a-650f-44be-97ab-55d52cb817ed
n = 100         # specify number of observations

# ╔═╡ c940a43a-0980-4c15-a7ea-9dfe95ccc4f2
θ = 2.0         # true value of the parameter we would like to estimate

# ╔═╡ 25db91d0-69b3-4909-ab74-1414817575e9
noise_σ2 = 0.3  # variance of observation noise

# ╔═╡ 929a01f6-9eaf-4c9d-ae0c-c23eee2a5205
observations = noise_σ2 * randn(n) .+ θ

# ╔═╡ 115eabf2-c476-40f8-8d7b-868a7359c1b6
function perform_kalman_step(prior :: Normal, x :: Float64, noise_σ2 :: Float64)
    K = prior.σ / (noise_σ2 + prior.σ)          # compute the Kalman gain
    posterior_μ = prior.μ + K*(x - prior.μ)     # update the posterior mean
    posterior_σ = prior.σ * (1.0 - K)           # update the posterior standard deviation
    return Normal(posterior_μ, posterior_σ)     # return the posterior distribution
end

# ╔═╡ d37f14bb-8f88-4635-90d6-c6ca17669b33
begin
	post_μ = fill!(Vector{Float64}(undef,n + 1), NaN)     # means of p(θ|D) over time
	post_σ2 = fill!(Vector{Float64}(undef,n + 1), NaN)    # variances of p(θ|D) over time

	# specify the prior distribution (you can play with the parameterization of this to get a feeling of how the Kalman filter converges)
	prior = Normal(0, 1)

	# save prior mean and variance to show these in plot
	post_μ[1] = prior.μ
	post_σ2[1] = prior.σ
	
	
	# note that this loop demonstrates Bayesian learning on streaming data; we update the prior distribution using observation(s), after which this posterior becomes the new prior for future observations
	for (i, x) in enumerate(observations)
		# compute the posterior distribution given the observation
	    posterior = perform_kalman_step(prior, x, noise_σ2)
		# save the mean of the posterior distribution
	    post_μ[i + 1] = posterior.μ
		# save the variance of the posterior distribution
	    post_σ2[i + 1] = posterior.σ
		# the posterior becomes the prior for future observations
	    prior = posterior
	end
end

# ╔═╡ 661082eb-f0c9-49a9-b046-8705f4342b37
let
	obs_scale = collect(2:n+1)
	# scatter the observations
	scatter(obs_scale, observations, label=L"D", )  
	post_scale = collect(1:n+1)
	# lineplot our estimated means of intermediate posterior distributions
	plot!(post_scale, post_μ, ribbon=sqrt.(post_σ2), linewidth=3, label=L"p(θ | D_t)")
	# plot the true value of θ
	plot!(post_scale, θ*ones(n + 1), linewidth=2, label=L"θ")
end

# ╔═╡ 7cd105c4-362b-4fa2-904a-d0bdc26b323b
TODO("Fons, can you do something with the number of points and a slider?")

# ╔═╡ b9ab9e28-d294-11ef-3a73-1f5cefdab3d8
md"""
The shaded area represents 2 standard deviations of posterior ``p(\theta|D)``. The variance of the posterior is guaranteed to decrease monotonically for the standard Kalman filter.

"""

# ╔═╡ b9ac5190-d294-11ef-0a99-a9d369b34045
let
	# Maximum likelihood estimation of 2D Gaussian
	N = length(sum(D,dims=1))
	μ = 1/N * sum(D,dims=2)[:,1]
	D_min_μ = D - repeat(μ, 1, N)
	Σ = Hermitian(1/N * D_min_μ*D_min_μ')
	m = MvNormal(μ, convert(Matrix, Σ));
	
	contour(range(-3, 4, length=100), range(-3, 4, length=100), (x, y) -> pdf(m, [x, y]))
	
	# Numerical integration of p(x|m) over S:
	(val,err) = hcubature((x)->pdf(m,x), [0., 1.], [2., 2.])
	@debug("p(x⋅∈S|m) ≈ $(val)")
	
	scatter!(D[1,:], D[2,:]; marker=:x, markerstrokewidth=3, label=L"D")
	scatter!([x_dot[1]], [x_dot[2]]; label=L"x_\bullet")
	plot!(range(0, 2), [1., 1., 1.]; fillrange=2, alpha=0.4, color=:gray, label=L"S")
end

# ╔═╡ 7a57afce-f325-4e14-815d-ec74eeee7d08
TODO("in general, I dont like 16 decimals, in particular when the answer is approximate. Let's make it 2 decimals.")

# ╔═╡ b9ac7486-d294-11ef-13e5-29b7ffb440bc
md"""
# Summary

A **linear transformation** ``z=Ax+b`` of a Gaussian variable ``x \sim \mathcal{N}(\mu_x,\Sigma_x)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) 
```

Bayesian inference with a Gaussian prior and Gaussian likelihood leads to an analytically computable Gaussian posterior, because of the **multiplication rule for Gaussians**:

```math
\begin{equation*}
\mathcal{N}(x|\mu_a,\Sigma_a) \cdot \mathcal{N}(x|\mu_b,\Sigma_b) = \underbrace{\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)}_{\text{normalization constant}} \cdot \mathcal{N}(x|\mu_c,\Sigma_c)
\end{equation*}
```

where

```math
\begin{align*}
\Sigma_c^{-1} &= \Sigma_a^{-1} + \Sigma_b^{-1} \\
\Sigma_c^{-1} \mu_c &= \Sigma_a^{-1}\mu_a + \Sigma_b^{-1}\mu_b
\end{align*}
```

**Conditioning and marginalization** of a multivariate Gaussian distribution yields Gaussian distributions. In particular, the joint distribution

```math
\mathcal{N} \left( \begin{bmatrix} x \\ y \end{bmatrix} \left| \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, 
  \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \right. \right)
```

can be decomposed as

```math
\begin{align*}
 p(y|x) &= \mathcal{N}\left(y\,|\,\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x),\, \Sigma_y - \Sigma_{yx}\Sigma_x^{-1}\Sigma_{xy} \right) \\
p(x) &= \mathcal{N}\left( x|\mu_x, \Sigma_x \right)
\end{align*}
```

Here's a nice [summary of Gaussian calculations](https://github.com/bertdv/AIP-5SSB0/raw/master/lessons/notebooks/files/RoweisS-gaussian_formulas.pdf) by Sam Roweis. 

"""

# ╔═╡ b9aca5b6-d294-11ef-2178-456126c0a874
md"""
#  OPTIONAL SLIDES

"""

# ╔═╡ b9acd5d4-d294-11ef-1ae5-ed4e13d238ef
md"""
## $(HTML("<span id='inference-for-precision'>Inference for the Precision Parameter of the Gaussian</span>"))



"""



# ╔═╡ 72e9420c-80da-48ef-8849-71988a4f8dda
TODO("Bert to update this")

# ╔═╡ b9acf7a8-d294-11ef-13d9-81758355cb1e
md"""

##### Problem



Consider again a Gaussian data-generating (measurement) model

```math
\mathcal{N}\left(x_n \,|\, \mu, \lambda^{-1} \right) \,.
```

(We express here the variance as the inverse of a precision parameter ``\lambda``, rather than using ``\sigma^2``, since this simplifies the subsequent Bayesian computations.)

Earlier in this lecture, we discussed Bayesian inference from a data set for the mean ``\mu``, when the variance ``\lambda^{-1}`` was given. 

We now derive the posterior distribution over the precision parameter ``\lambda``, assuming that the mean ``\mu`` is known. We omit the more general case in which both ``\mu`` and ``\lambda`` are treated as unknowns, since the resulting calculations are considerably more involved (but still result in a closed-form solution).


"""

# ╔═╡ b9ad0842-d294-11ef-2035-31bceab4ace1
md"""
#### model specification

The likelihood for the precision parameter is 

```math
\begin{align*}
p(D|\lambda) &= \prod_{n=1}^N \mathcal{N}\left(x_n \,|\, \mu, \lambda^{-1} \right) \\
  &\propto \lambda^{N/2} \exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^N \left(x_n - \mu \right)^2\right\} \tag{B-2.145}
\end{align*}
```

"""

# ╔═╡ b9ad1b70-d294-11ef-3931-d1dcd2343ac9
md"""
The conjugate distribution for this function of ``\lambda`` is the [*Gamma* distribution](https://en.wikipedia.org/wiki/Gamma_distribution), given by

```math
p(\lambda\,|\,a,b) = \mathrm{Gam}\left( \lambda\,|\,a,b \right) \triangleq \frac{1}{\Gamma(a)} b^{a} \lambda^{a-1} \exp\left\{ -b \lambda\right\}\,, \tag{B-2.146}
```

where ``a>0`` and ``b>0`` are known as the *shape* and *rate* parameters, respectively. 

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/B-fig-2.13.png?raw=true)

(Bishop fig.2.13). Plots of the Gamma distribution ``\mathrm{Gam}\left( \lambda\,|\,a,b \right)`` for different values of ``a`` and ``b``.

"""

# ╔═╡ b9ad299e-d294-11ef-36d7-2f73d3cd1fa7
md"""
The mean and variance of the Gamma distribution evaluate to ``\mathrm{E}\left( \lambda\right) = \frac{a}{b}`` and ``\mathrm{var}\left[\lambda\right] = \frac{a}{b^2}``. 

For this example, we consider a prior 
```math
p(\lambda) = \mathrm{Gam}\left( \lambda\,|\,a_0, b_0\right) \,. 
```

"""

# ╔═╡ b9ad5100-d294-11ef-0e8b-3f67ddb2d86d
md"""
#### inference

The posterior is given by Bayes rule, 

```math
\begin{align*}
p(\lambda\,|\,D) &\propto \underbrace{\lambda^{N/2} \exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^N \left(x_n - \mu \right)^2\right\} }_{\text{likelihood}} \cdot \underbrace{\frac{1}{\Gamma(a_0)} b_0^{a_0} \lambda^{a_0-1} \exp\left\{ -b_0 \lambda\right\}}_{\text{prior}} \\
  &\propto \mathrm{Gam}\left( \lambda\,|\,a_N,b_N \right) 
\end{align*}
```

with

```math
\begin{align*}
a_N &= a_0 + \frac{N}{2} \qquad &&\text{(B-2.150)} \\
b_N &= b_0 + \frac{1}{2}\sum_n \left( x_n-\mu\right)^2 \qquad &&\text{(B-2.151)}
\end{align*}
```

"""

# ╔═╡ b9ad6238-d294-11ef-3fed-bbcc7d7443ee
md"""
Hence the **posterior is again a Gamma distribution**. By inspection of B-2.150 and B-2.151, we deduce that we can interpret ``2a_0`` as the number of a priori (pseudo-)observations. 

"""

# ╔═╡ b9ad71a6-d294-11ef-185f-f1f6e6ac4464
md"""
Since the most uninformative prior is given by ``a_0=b_0 \rightarrow 0``, we can derive the **maximum likelihood estimate** for the precision as

```math
\lambda_{\text{ML}} = \left.\mathrm{E}\left[ \lambda\right]\right\vert_{a_0=b_0\rightarrow 0} = \left. \frac{a_N}{b_N}\right\vert_{a_0=b_0\rightarrow 0} = \frac{N}{\sum_{n=1}^N \left(x_n-\mu \right)^2}
```

"""

# ╔═╡ b9ad85a4-d294-11ef-2af2-953ac0ab8927
md"""
In short, if we do density estimation with a Gaussian distribution ``\mathcal{N}\left(x_n\,|\,\mu,\sigma^2 \right)`` for an observed data set ``D = \{x_1, x_2, \ldots, x_N\}``, the $(HTML("<span id='ML-for-Gaussian'>maximum likelihood estimates</span>")) for ``\mu`` and ``\sigma^2`` are given by

```math
\begin{align*}
\mu_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N x_n \qquad &&\text{(B-2.121)} \\
\sigma^2_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N \left(x_n - \mu_{\text{ML}} \right)^2 \qquad &&\text{(B-2.122)}
\end{align*}
```

These estimates are also known as the *sample mean* and *sample variance* respectively. 

"""

# ╔═╡ b9abadce-d294-11ef-14a6-9131c5b1b802
md"""
## $(HTML("<span id='product-of-gaussians'>Product of Normally Distributed Variables</span>"))

(We've seen that) the sum of two Gausssian-distributed variables is also Gaussian distributed.

Has the *product* of two Gaussian distributed variables also a Gaussian distribution?

**No**! In general, this is a difficult computation. As an example, let's compute ``p(z)`` for ``Z=XY`` for the special case that ``X\sim \mathcal{N}(0,1)`` and ``Y\sim \mathcal{N}(0,1)``.

```math
\begin{align*}
p(z) &= \int_{X,Y} p(z|x,y)\,p(x,y)\,\mathrm{d}x\mathrm{d}y \\
  &= \frac{1}{2 \pi}\int  \delta(z-xy) \, e^{-(x^2+y^2)/2} \, \mathrm{d}x\mathrm{d}y \\
  &=  \frac{1}{\pi} \int_0^\infty \frac{1}{x} e^{-(x^2+z^2/x^2)/2} \, \mathrm{d}x \\
  &= \frac{1}{\pi} \mathrm{K}_0( \lvert z\rvert )\,.
\end{align*}
```

where  ``\mathrm{K}_n(z)`` is a [modified Bessel function of the second kind](http://mathworld.wolfram.com/ModifiedBesselFunctionoftheSecondKind.html).

"""

# ╔═╡ b9ac09c4-d294-11ef-2cb8-270289d01f25
md"""
In short, Gaussian-distributed variables remain Gaussian in linear systems, but this is not the case in non-linear systems. 

"""

# ╔═╡ d5f74bd7-cc31-4efc-bb9b-a289f26b0121
challenge_header(
	title; 
	color="green",
	big::Bool=false,
	header_level::Int=2,
	challenge_text="Challenge:",
) = HypertextLiteral.@htl """
	
<$("h$header_level") class="ptt-section $(big ? "big" : "")" style="--ptt-accent: $(color);"><span>$(challenge_text)</span> $(title)</$("h$header_level")>
	
<style>
.ptt-section::before {
	content: "";
	display: block;
	position: absolute;
	left: -25px;
	right: -6px;
	top: -4px;
	height: 200px;
	border: 4px solid salmon;
	border-bottom: none;
	border-image-source: linear-gradient(to bottom, var(--ptt-accent), transparent);
	border-image-slice: 1;
	opacity: .7;
	pointer-events: none;
}

.big.ptt-section::before {
	height: 500px;
}
	

.ptt-section > span {
	color: color-mix(in hwb, var(--ptt-accent) 60%, black);
	@media (prefers-color-scheme: dark) {
		color: color-mix(in hwb, var(--ptt-accent) 30%, white);
	}
	font-style: italic;
}
	
</style>
"""

# ╔═╡ b9a7073a-d294-11ef-2330-49ffa7faff21
md"""

$(challenge_header("Product of Two Gaussian PDFs"; challenge_text="Code Example:"))


Let's plot the exact product of two Gaussian PDFs as well as the normalized product according to the above derivation.

"""

# ╔═╡ b9a9565c-d294-11ef-1b67-83d1ab18035b
md"""
##  

$(challenge_header("Joint, Marginal, and Conditional Gaussian Distributions"; challenge_text="Code Example:"))

Let's plot the joint, marginal, and conditional distributions for some Gaussians.

"""

# ╔═╡ b9ab2e32-d294-11ef-2ccc-9760ead59972
md"""
$(challenge_header("Kalman Filtering"; challenge_text="Code Example:"))

Let's implement the Kalman filter described above. We'll use it to recursively estimate the value of ``\theta`` based on noisy observations.

"""

# ╔═╡ b9ac2d3c-d294-11ef-0d37-65a65525ad28
md"""
$(challenge_header("Classify a Gaussian Sample"; challenge_text="Challenge Revisited:", header_level=1))

Let's solve the challenge from the beginning of the lecture. We apply maximum likelihood estimation to fit a 2-dimensional Gaussian model (``m``) to data set ``D``. Next, we evaluate ``p(x_\bullet \in S | m)`` by (numerical) integration of the Gaussian pdf over ``S``: ``p(x_\bullet \in S | m) = \int_S p(x|m) \mathrm{d}x``.

"""

# ╔═╡ b9abdc7e-d294-11ef-394a-a708c96c86fc
md"""
$(challenge_header("Product of Gaussian Distributions"; challenge_text="Code Example:"))


We plot ``p(Z=XY)`` and ``p(X)p(Y)`` for ``X\sim\mathcal{N}(0,1)`` and ``Y \sim \mathcal{N}(0,1)`` to give an idea of how these distributions differ.

"""


# ╔═╡ Cell order:
# ╟─b9a38e20-d294-11ef-166b-b5597125ed6d
# ╠═c97c495c-f7fe-4552-90df-e2fb16f81d15
# ╠═3ec821fd-cf6c-4603-839d-8c59bb931fa9
# ╠═69d951b6-58b3-4ce2-af44-4cb799e453ff
# ╠═5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
# ╟─b9a46c3e-d294-11ef-116f-9b97e0118e5b
# ╟─b9a48c60-d294-11ef-3b90-03053fcd82fb
# ╠═ba57ecbb-b64e-4dd8-8398-a90af1ac71f3
# ╟─02853a5c-f6aa-4af8-8a25-bfffd4b96afc
# ╟─71f1c8ee-3b65-4ef8-b36f-3822837de410
# ╟─b9a4eb62-d294-11ef-06fa-af1f586cbc15
# ╟─b9a50d0c-d294-11ef-0e60-2386cf289478
# ╟─b9a52b18-d294-11ef-2d42-19c5e3ef3549
# ╟─b9a5589a-d294-11ef-3fc3-0552a69df7b2
# ╟─9501922f-b928-46e2-8f23-8eb9c64f6198
# ╟─b9a5889c-d294-11ef-266e-d90225222e10
# ╟─a82378ae-d1be-43f9-b63a-2f897767d1fb
# ╟─b9a5a82c-d294-11ef-096f-ffee478aca20
# ╟─b9a5b7e0-d294-11ef-213e-4b72b8c88db7
# ╟─b9a5cbc2-d294-11ef-214a-c71fb1272326
# ╟─b9a5dcc0-d294-11ef-2c85-657a460db5cd
# ╟─b9a6557e-d294-11ef-0a90-d74c337ade25
# ╟─b9a67d06-d294-11ef-297b-eb9039786ea7
# ╟─b9a68d3a-d294-11ef-2335-093a39648007
# ╟─b9a697fa-d294-11ef-3a57-7b7ba1f4fd70
# ╟─b9a6b7b2-d294-11ef-06dc-4de5ef25c1fd
# ╟─702e7b10-14a4-42da-a192-f7c02a3d470a
# ╟─51d81901-213f-42ce-b77e-10f7ca4a4145
# ╟─b9a6c7b6-d294-11ef-0446-c372aa610df8
# ╟─b9a6ecd2-d294-11ef-02af-37c977f2814b
# ╟─b9a6f916-d294-11ef-38cb-b78c0c448550
# ╟─d2bedf5f-a0ea-4604-b5da-adf9f11e80be
# ╠═1e587633-06ab-442f-b6ed-3a994e19a613
# ╟─b9a7073a-d294-11ef-2330-49ffa7faff21
# ╠═45c2fb37-a078-4284-9e04-176156cffb1e
# ╠═d81483db-3826-4ff4-9d52-e23418da07d0
# ╠═e6a2d2ed-0100-4570-85c1-fc9d8f84e32e
# ╟─14fd14db-26da-4f0b-81d0-59ee4ab1a35c
# ╠═f9cf453a-6369-4d38-9dad-fb3412497635
# ╠═9f939dd4-18e8-464c-a12e-eb320d5fd88b
# ╠═fa03d8c8-bead-456f-833e-cc0690e1b528
# ╠═6cbf7a96-9e73-4289-9970-88e30cea28a5
# ╟─df8867ed-0eff-4a52-8f5e-2472467e1aa2
# ╟─b9a885a8-d294-11ef-079e-411d3f1cda03
# ╟─b9a9565c-d294-11ef-1b67-83d1ab18035b
# ╠═b9a99fcc-d294-11ef-3de4-5369d9796de7
# ╟─f4ce24c7-7d03-4574-81e0-d4cbb818a897
# ╟─b9a9b8e0-d294-11ef-348d-c197c4ce2b8c
# ╟─b9a9dca8-d294-11ef-04ec-a9202c319f89
# ╟─b9a9f98e-d294-11ef-193a-0dbdbfffa86f
# ╟─b9aa27da-d294-11ef-0780-af9d89f9f599
# ╟─b9aa3950-d294-11ef-373f-d5d330694bfd
# ╟─0072e73e-1569-4ce4-bffb-280823499f0d
# ╟─b9a80522-d294-11ef-39d8-53a536d66bf9
# ╟─b9a85716-d294-11ef-10e0-a7b08b800a98
# ╟─b9aa930a-d294-11ef-37ec-8d17be226c74
# ╟─b9aabe9a-d294-11ef-2489-e9fc0dbb760a
# ╟─b9aad50e-d294-11ef-23d2-8d2bb3b47574
# ╟─b9aaee4a-d294-11ef-2ed7-0dcb360d8bb7
# ╟─b9aafc6e-d294-11ef-1b1a-df718c1f1a58
# ╟─e2fc4945-4f88-4520-b56c-c7208b62c29d
# ╟─b9ab0b46-d294-11ef-13c5-8314655f7867
# ╟─b9ab1dd4-d294-11ef-2e86-31c4a4389475
# ╟─b9ab2e32-d294-11ef-2ccc-9760ead59972
# ╠═85b15f0a-650f-44be-97ab-55d52cb817ed
# ╠═c940a43a-0980-4c15-a7ea-9dfe95ccc4f2
# ╠═25db91d0-69b3-4909-ab74-1414817575e9
# ╠═929a01f6-9eaf-4c9d-ae0c-c23eee2a5205
# ╠═115eabf2-c476-40f8-8d7b-868a7359c1b6
# ╠═d37f14bb-8f88-4635-90d6-c6ca17669b33
# ╠═661082eb-f0c9-49a9-b046-8705f4342b37
# ╠═7cd105c4-362b-4fa2-904a-d0bdc26b323b
# ╟─b9ab9e28-d294-11ef-3a73-1f5cefdab3d8
# ╟─b9ac2d3c-d294-11ef-0d37-65a65525ad28
# ╠═5638c1d0-db95-49e4-bd80-528f79f2947e
# ╠═b9ac5190-d294-11ef-0a99-a9d369b34045
# ╠═7a57afce-f325-4e14-815d-ec74eeee7d08
# ╟─b9ac7486-d294-11ef-13e5-29b7ffb440bc
# ╟─b9aca5b6-d294-11ef-2178-456126c0a874
# ╟─b9acd5d4-d294-11ef-1ae5-ed4e13d238ef
# ╠═72e9420c-80da-48ef-8849-71988a4f8dda
# ╟─b9acf7a8-d294-11ef-13d9-81758355cb1e
# ╟─b9ad0842-d294-11ef-2035-31bceab4ace1
# ╟─b9ad1b70-d294-11ef-3931-d1dcd2343ac9
# ╟─b9ad299e-d294-11ef-36d7-2f73d3cd1fa7
# ╟─b9ad5100-d294-11ef-0e8b-3f67ddb2d86d
# ╟─b9ad6238-d294-11ef-3fed-bbcc7d7443ee
# ╟─b9ad71a6-d294-11ef-185f-f1f6e6ac4464
# ╟─b9ad85a4-d294-11ef-2af2-953ac0ab8927
# ╟─b9abadce-d294-11ef-14a6-9131c5b1b802
# ╟─b9abdc7e-d294-11ef-394a-a708c96c86fc
# ╟─b9abf984-d294-11ef-1eaa-3358379f8b44
# ╟─b9ac09c4-d294-11ef-2cb8-270289d01f25
# ╠═d5f74bd7-cc31-4efc-bb9b-a289f26b0121
