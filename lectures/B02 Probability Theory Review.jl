### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3e17df5e-d294-11ef-38c7-f573724871d8
md"""
# Probability Theory Review

"""

# ╔═╡ 3e1803d0-d294-11ef-0304-df2b9b698cd1
md"""
## Preliminaries

Goal 

  * Review of Probability Theory as a theory for rational/logical reasoning with uncertainties (i.e., a Bayesian interpretation)

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 12-24
      * [Ariel Caticha, Entropic Inference and the Foundations of Physics (2012)](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-56 (ch.2: probability)

          * Great introduction to probability theory, in particular w.r.t. its correct interpretation as a state-of-knowledge.
          * Absolutely worth your time to read the whole chapter, even if you skip section 2.2.4 (pp.15-18) on Cox's proof.
      * [Edwin Jaynes, Probability Theory–The Logic of Science (2003)](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf). 

          * Brilliant book on Bayesian view on probability theory. Just for fun, scan the annotated bibliography and references.
      * [Aubrey Clayton, Bernoulli's Fallacy–Statistical Illogic and the Crisis of Modern Science (2021)](https://aubreyclayton.com/bernoulli)

          * A very readable account on the history of statistics and probability theory. Discusses why most popular statistics recipes are very poor scientific analysis tools. Use probability theory instead!
      * [Joram Soch et al ., The Book of Statistical Proofs (2023 - )](https://statproofbook.github.io/)

          * On-line resource for proofs in probability theory and statistical inference.

"""

# ╔═╡ 3e1823b0-d294-11ef-3dba-9997a7230cdf
md"""
## [Data Analysis: A Bayesian Tutorial](https://www.amazon.com/Data-Analysis-Bayesian-Devinderjit-Sivia/dp/0198568320)

The following is an excerpt from the book [Data Analysis: A Bayesian Tutorial](https://www.amazon.com/Data-Analysis-Bayesian-Devinderjit-Sivia/dp/0198568320) (2006), by D.S. Sivia with J.S. Skilling:

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/preface-data-analysis-a-Bayesian-tutorial.png?raw=true)

Does this fragment resonate with your own experience? 

In this lesson we introduce *Probability Theory* (PT) again. As we will see in the next lessons, PT is all you need to make sense of machine learning, artificial intelligence, statistics, etc. 

"""

# ╔═╡ 3e185ab0-d294-11ef-3f7d-9bd465518274
md"""
## Challenge: Disease Diagnosis

**Problem**: Given a disease with prevalence of  1%  and a test procedure  with sensitivity ('true positive' rate) of  95%  and specificity ('true negative' rate) of  85% , what is the chance that somebody who tests positive actually has the disease?

"""

# ╔═╡ 3e1876f8-d294-11ef-22bf-7904df3c1182
md"""
**Solution**: Use probabilistic inference, to be discussed in this lecture. 

"""

# ╔═╡ 3e1889b8-d294-11ef-17bb-496655fbd618
md"""
## The Design of Probability Theory

Define an **event** (or "proposition") ``A`` as a statement that can be considered for its truth by a person. For instance, 

```math
𝐴= \texttt{'there is life on Mars'}
```

"""

# ╔═╡ 3e189fa2-d294-11ef-1f2b-2151b6c128f8
md"""
If we assume the fact 

```math
I = \texttt{'All known life forms require water'}
```

as background information, and a new piece of information 

```math
x = \texttt{'There is water on Mars'}
```

becomes available, how *should* our degree of belief in event ``A`` be affected *if we were rational*? 

"""

# ╔═╡ 3e18b2fa-d294-11ef-1255-df048f0dcec2
md"""
[Richard T. Cox (1946)](https://aapt.scitation.org/doi/10.1119/1.1990764) developed a **calculus for rational reasoning** about how to represent and update the degree of *beliefs* about the truth value of events when faced with new information.  

"""

# ╔═╡ 3e18c25c-d294-11ef-11bc-a93c2572b107
md"""
In developing this calculus, only some very agreeable assumptions were made, including:

  * (Representation). Degrees of rational belief (or, as we shall later call them, probabilities) about the truth value of propositions are represented by real numbers.
  * (Transitivity). If the belief in ``A`` is greater than the belief in ``B``, and the belief in ``B`` is greater than the belief in ``C``, then the belief in ``A`` must be greater than the belief in ``C``.
  * (Consistency). If the belief in an event can be inferred in two different ways, then the two ways must agree on the resulting belief.

"""

# ╔═╡ 3e18d2ea-d294-11ef-35e9-2332dd31dbf0
md"""
This effort resulted in confirming that the **sum and product rules of Probability Theory** [(to be discussed below)](#PT-calculus) are the **only** proper rational way to process belief intensities. 

"""

# ╔═╡ 3e18e4bc-d294-11ef-38bc-cb97cb4e0963
md"""
```math
\Rightarrow
```

Probability theory (PT) provides the **theory of optimal processing of incomplete information** (see [Cox theorem](https://en.wikipedia.org/wiki/Cox%27s_theorem), and [Caticha](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-24).

"""

# ╔═╡ 3e18f18c-d294-11ef-33e4-b7f9495e0508
md"""
## Why Probability Theory for Machine Learning?

Machine learning concerns updating our beliefs about appropriate settings for model parameters from new information (namely a data set), and therefore PT provides the *optimal calculus for machine learning*. 

"""

# ╔═╡ 3e1906ea-d294-11ef-236e-c966a9474170
md"""
In general, nearly all interesting questions in machine learning (and information processing in general) can be stated in the following form (a conditional probability):

```math
p(\texttt{whatever-we-want-to-know}\, | \,\texttt{whatever-we-do-know})
```

where ``p(a|b)`` means the probability that ``a`` is true, given that ``b`` is true.

"""

# ╔═╡ 3e191b6c-d294-11ef-3174-d1b4b36e252b
md"""
Examples

  * Predictions

```math
p(\,\texttt{future-observations}\,|\,\texttt{past-observations}\,)
```

Classify a received data point ``x`` 

```math
p(\,x\texttt{-belongs-to-class-}k \,|\,x\,)
```

Update a model based on a new observation

```math
p(\,\texttt{model-parameters} \,|\,\texttt{new-observation},\,\texttt{past-observations}\,)
```

"""

# ╔═╡ 3e192ef4-d294-11ef-1fc4-87175eeec5eb
md"""
## Frequentist vs. Bayesian Interpretation of Probabilities

The interpretation of a probability as a **degree-of-belief** about the truth value of an event is also called the **Bayesian** interpretation.  

"""

# ╔═╡ 3e19436c-d294-11ef-11c5-f9914f7a3a57
md"""
In the **Bayesian** interpretation, the probability is associated with a **state-of-knowledge** (usually held by a person, but formally by a rational agent). 

  * For instance, in a coin tossing experiment, ``p(\texttt{tail}) = 0.4`` should be interpreted as the belief that there is a 40% chance that ``\texttt{tail}`` comes up if the coin were tossed.
  * Under the Bayesian interpretation, PT calculus (sum and product rules) **extends boolean logic to rational reasoning with uncertainty**.

"""

# ╔═╡ 3e194ef2-d294-11ef-3b38-1ddc3063ff35
md"""
The Bayesian interpretation contrasts with the **frequentist** interpretation of a probability as the relative frequency that an event would occur under repeated execution of an experiment.

  * For instance, if the experiment is tossing a coin, then ``p(\texttt{tail}) = 0.4`` means that in the limit of a large number of coin tosses, 40% of outcomes turn up as ``\texttt{tail}``.

"""

# ╔═╡ 3e1964b4-d294-11ef-373d-712257fc130f
md"""
The Bayesian viewpoint is more generally applicable than the frequentist viewpoint, e.g., it is hard to apply the frequentist viewpoint to events like '``\texttt{it will rain tomorrow}``'. 

"""

# ╔═╡ 3e196d6a-d294-11ef-0795-41c045079251
md"""
The Bayesian viewpoint is clearly favored in the machine learning community. (In this class, we also strongly favor the Bayesian interpretation). 

"""

# ╔═╡ 3e198336-d294-11ef-26fd-03cd15876486
md"""
Aubrey Clayton, in his wonderful book [Bernoulli's fallacy](https://aubreyclayton.com/bernoulli) (2021), writes about this issue: 

> “Compared with Bayesian methods, standard [frequentist] statistical techniques use only a small fraction of the available information about a research hypothesis (how well it predicts some observation), so naturally they will struggle when that limited information proves inadequate. Using standard statistical methods is like driving a car at night on a poorly lit highway: to keep from going in a ditch, we could build an elaborate system of bumpers and guardrails and equip the car with lane departure warnings and sophisticated navigation systems, and even then we could at best only drive to a few destinations. Or we could turn on the headlights.”


"""

# ╔═╡ 3e198ba6-d294-11ef-3fe7-d70bf4833fa6
md"""
In this class, we aim to turn on the headlights and illuminate the elegance and power of the Bayesian approach to information processing. 

"""

# ╔═╡ 3e19a2a0-d294-11ef-18b4-7987534916d2
md"""
## Events

Technically, a probability expresses a degree-of-belief in the truth value of an event. Let's first define an event. 

We define an **event** ``A`` as a statement, whose truth can be contemplated by a person, e.g.,

```math
A = \text{`it will rain tomorrow'}
```

"""

# ╔═╡ 3e19ac30-d294-11ef-10b7-fbba9ae2a2c3
md"""
We write the denial of ``A``, i.e. the event **not**-A, as ``\bar{A}``. 

"""

# ╔═╡ 3e19c06c-d294-11ef-197a-f549e8107a57
md"""
Events can be logically combined to create new events. Given two events ``A`` and ``B``, we will shortly write the **conjunction** (logical-and) ``A \wedge B`` as ``A,B`` or ``AB``. The conjunction ``AB`` is true only if both ``A`` and ``B`` are true. 

"""

# ╔═╡ 3e19d39a-d294-11ef-1a50-7fe8a24777dc
md"""
We will write the **disjunction** (logical-or) ``A \lor B`` also as ``A + B``, which is true if either ``A`` or ``B`` is true or both ``A`` and ``B`` are true. (Note that the plus-sign is not an arithmetic here but rather a logical operator to process truth values). 

"""

# ╔═╡ 3e19e95a-d294-11ef-3da4-6d23922a5150
md"""
## Probability

For any event ``A``, with background knowledge ``I``, the **conditional probability of ``A`` given ``I``**, is written as 

```math
p(A|I)\,.
```

"""

# ╔═╡ 3e19fd2a-d294-11ef-2a52-b9245f6d02ba
md"""
```math
p(A|I)
```

indicates the degree-of-belief in event ``A``, given that ``I`` is true. 

"""

# ╔═╡ 3e1a36b4-d294-11ef-2242-f36061b0b754
md"""
In principle, all probabilities are conditional probabilities of the type ``p(A|I)``, since there is always some background knowledge. However, we often write ``p(A)`` rather than ``p(A|I)`` if the background knowledge ``I`` is assumed to be obviously present. E.g., we usually write ``p(A)`` rather than ``p(\,A\,|\,\text{the-sun-comes-up-tomorrow}\,)``.

"""

# ╔═╡ 3e1a522a-d294-11ef-1a7a-bdcfbd5dae09
md"""
The expression ``p(A,B)`` is called the **joint probability** of events ``A`` and ``B``. Note that ``p(A,B) = p(B,A)``, since ``AB=BA``. Therefore the order of arguments in a joint probability distribution does not matter: ``p(A,B,C,D) = p(C,A,D,B)``, etc.

"""

# ╔═╡ 3e1a69f4-d294-11ef-103e-efc47025fb8f
md"""
Note that, if ``X`` is a variable, then an *assignment* ``X=x`` (where ``x`` is a value, e.g., ``X=5``) can be interpreted as an event. Hence, the expression ``p(X=5)`` should be interpreted as the degree-of-belief that variable ``X`` takes on the value ``5``. 

"""

# ╔═╡ 3e1a7c8e-d294-11ef-1f97-55e608d49141
md"""
If ``X`` is a *discretely* valued variable, then ``p(X=x)`` is a probability *mass* function (PMF) with ``0\le p(X=x)\le 1`` and normalization ``\sum_x p(x) =1``. 

"""

# ╔═╡ 3e1a8eca-d294-11ef-1ef0-c15b24d05990
md"""
If ``X`` is *continuously* valued, then ``p(X=x)`` is a probability *density* function (PDF) with ``p(X=x)\ge 0``  and normalization ``\int_x p(x)\mathrm{d}x=1``. 

  * Note that if ``X`` is continuously valued, then the value of ``p(x)`` is not necessarily ``\le 1``. E.g., a uniform distribution on the continuous domain ``[0,.5]`` has value ``p(x) = 2``.

"""

# ╔═╡ 3e1a9fdc-d294-11ef-288f-37fd1b7ee281
md"""
The notational conventions in PT are unfortunately a bit sloppy:( For instance, in the context of a variable ``X``, we often write ``p(x)`` rather than ``p(X=x)``, assuming that the reader understands the context.  

"""

# ╔═╡ 3e1ab104-d294-11ef-1a98-412946949fba
md"""
## $(HTML("<span id='PT-calculus'>Probability Theory Calculus</span>"))

"""

# ╔═╡ 3e1ac32c-d294-11ef-13be-558397a6cc2a
md"""
The following product and sum rules are also known as the **axioms of probability theory**, but as discussed above, under some mild assumptions, they can be derived as the unique rules for *rational reasoning under uncertainty* ([Cox theorem, 1946](https://en.wikipedia.org/wiki/Cox%27s_theorem), and [Caticha, 2012](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-26).

"""

# ╔═╡ 3e1ad2ea-d294-11ef-02c4-3f06e14ea4d8
md"""
**Sum rule**. The disjunction of two events ``A`` and ``B`` with given background ``I`` is 

```math
 \boxed{p(A+B|I) = p(A|I) + p(B|I) - p(A,B|I)}
```

"""

# ╔═╡ 3e1ae30a-d294-11ef-3a5d-d91b7d7723d3
md"""
**Product rule**. The conjuction of two events ``A`` and ``B`` with given background ``I`` is

```math
 \boxed{p(A,B|I) = p(A|B,I)\,p(B|I)}
```

"""

# ╔═╡ 3e1af354-d294-11ef-1971-dfb39016cfcd
md"""
PT extends (propositional) logic in reasoning about the truth value of events. Logic reasons about the truth value of events on the binary domain {0,1} (FALSE and TRUE), whereas PT extends the range to degrees-of-belief, represented by real numbers in [0,1].

"""

# ╔═╡ 3e1b05ee-d294-11ef-33de-efed64d01c0d
md"""
**All legitimate probabilistic relations can be derived from the sum and product rules!**

"""

# ╔═╡ 3e1b4b1c-d294-11ef-0423-9152887cc403
md"""
## Independent, Exclusive and Exhaustive Events

It will be helpful to introduce some terms concerning special relationships between events.  

Two events ``A`` and ``B`` are said to be **independent** if the probability of one event is not altered by information about the truth of the other event, i.e., 

```math
p(A|B) = p(A)\,.
```

```math
\Rightarrow
```

If ``A`` and ``B`` are independent, then the product rule simplifies to 

```math
p(A,B) = p(A) p(B)\,.
```

```math
A
```

and ``B`` with given background ``I`` are said to be **conditionally independent** if ``p(A|B,I) = p(A|I)``. In that case, the product rule simplifies to ``p(A,B|I) = p(A|I) p(B|I)``.

Two events ``A_1`` and ``A_2`` are said to be **mutually exclusive** ('disjoint') if they cannot be true simultanously, i.e., if ``p(A_1,A_2)=0``.

```math
\Rightarrow
```

For mutually exclusive events, probability adds (this follows from the sum rule): 

```math
p(A_1+A_2) = p(A_1) + p(A_2)
```

A set of events ``A_1, A_2, \ldots, A_N`` is said to be **collectively exhaustive** if one of the statements is necessarily true, i.e., ``A_1+A_2+\cdots +A_N=\mathrm{TRUE}``, or equivalently 

```math
p(A_1+A_2+\cdots +A_N)=1
```

If a set of events ``A_1, A_2, \ldots, A_n`` are both **mutually exclusive** and **collectively exhausitive** events, then we say that they **partition the universe**. Technically, this means that 

```math
\sum_{n=1}^N p(A_n) = p(A_1 + \ldots + A_N) = 1
```



"""

# ╔═╡ 3e1b5c9c-d294-11ef-137f-d75b3731eae4
md"""
We mentioned before that every inference problem in PT can be evaluated through the sum and product rules. Next, we present two useful corollaries: (1) *Marginalization* and (2) *Bayes rule*. 

"""

# ╔═╡ 3e1b6bce-d294-11ef-2bd9-29c0634b1856
md"""
## Marginalization

"""

# ╔═╡ 3e1b7d14-d294-11ef-0d10-1148a928dd57
md"""
Let ``A`` and ``B_1,B_2,\ldots,B_n`` be events, where ``B_1,B_2,\ldots,B_n`` partitions the universe. Then

```math
\sum_{i=1}^n p(A,B_i) = p(A) \,.
```

This rule is called the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability). Proof:

```math
\begin{align*}
  \sum_i p(A,B_i) &= p(\sum_i AB_i)  &&\quad \text{(since all $AB_i$ are disjoint)}\\
  &= p(A,\sum_i B_i) \\
  &= p(A,\text{TRUE}) &&\quad \text{(since $B_i$ are exhaustive)} \\
  &= p(A)
  \end{align*}
```

"""

# ╔═╡ 3e1b8bf4-d294-11ef-04cc-6364e46fdd64
md"""
A very practical application of this law is to get rid of a variable that we are not interested in. For instance, if ``X`` and ``Y \in \{y_1,y_2,\ldots,y_n\}`` are discrete variables, then

```math
p(X) = \sum_{i=1}^n p(X,Y=y_i)\,.
```

"""

# ╔═╡ 3e1b9ba8-d294-11ef-18f2-db8eed3d87d0
md"""
Summing ``Y`` out of a joint distribution ``p(X,Y)`` is called **marginalization** and the result ``p(X)`` is sometimes referred to as the **marginal probability** of ``X``. 

"""

# ╔═╡ 3e1babca-d294-11ef-37c1-cd821a6488b2
md"""
Note that marginalization can be understood as applying a "generalized" sum rule. Bishop (p.14) and some other authors also refer to this as the sum rule, but we do not follow that terminology.

"""

# ╔═╡ 3e1bba8e-d294-11ef-1f61-295af16078ce
md"""
Of course, in the continuous domain, marginalization becomes

```math
p(X)=\int_Y p(X,Y) \,\mathrm{d}Y
```

"""

# ╔═╡ 3e1bcb00-d294-11ef-2795-bd225bd00496
md"""
## $(HTML("<span id='Bayes-rule'>Bayes Rule</span>"))

Consider two variables ``D`` and ``\theta``. It follows from symmetry arguments that 

```math
p(D,\theta)=p(\theta,D)
```

, and hence that

```math
p(D|\theta)p(\theta)=p(\theta|D)p(D)
```

or, equivalently,

```math
 p(\theta|D) = \frac{p(D|\theta) }{p(D)}p(\theta)\,.\qquad \text{(Bayes rule)}
```

"""

# ╔═╡ 3e1bdd02-d294-11ef-19e8-2f44eccf58af
md"""
This last formula is called **Bayes rule**, named after its inventor [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes) (1701-1761). While Bayes rule is always true, a particularly useful application occurs when ``D`` refers to an observed data set and ``\theta`` is set of unobserved model parameters. In that case,

  * the **prior** probability ``p(\theta)`` represents our **state-of-knowledge** about proper values for ``\theta``, before seeing the data ``D``.
  * the **posterior** probability ``p(\theta|D)`` represents our state-of-knowledge about ``\theta`` after we have seen the data.

"""

# ╔═╡ 3e1bf116-d294-11ef-148b-f7a1ca3f3bad
md"""
```math
\Rightarrow
```

Bayes rule tells us how to update our knowledge about model parameters when facing new data. Hence, 

 <span style="font-size:large; color:red"> Bayes rule is the fundamental rule for learning from data! </span> 

"""

# ╔═╡ 3e1bffec-d294-11ef-2a49-9ff0f6331add
md"""
## Bayes Rule Nomenclature

Some nomenclature associated with Bayes rule:

```math
\underbrace{p(\theta | D)}_{\text{posterior}} = \frac{\overbrace{p(D|\theta)}^{\text{likelihood}} \times \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(D)}_{\text{evidence}}}
```

"""

# ╔═╡ 3e1c0e80-d294-11ef-0d19-375e01988f16
md"""
Note that the evidence (a.k.a. *marginal likelihood* ) can be computed from the numerator through marginalization since

```math
 p(D) = \int p(D,\theta) \,\mathrm{d}\theta = \int p(D|\theta)\,p(\theta) \,\mathrm{d}\theta
```

"""

# ╔═╡ 3e1c1e3e-d294-11ef-0955-bdf9d0ba3c53
md"""
Hence, having access to likelihood and prior is in principle sufficient to compute both the evidence and the posterior. To emphasize that point, Bayes rule is sometimes written as a transformation:

```math
 \underbrace{\underbrace{p(\theta|D)}_{\text{posterior}}\cdot \underbrace{p(D)}_{\text{evidence}}}_{\text{this is what we want to compute}} = \underbrace{\underbrace{p(D|\theta)}_{\text{likelihood}}\cdot \underbrace{p(\theta)}_{\text{prior}}}_{\text{this is available}}
```

"""

# ╔═╡ 3e1c4224-d294-11ef-2707-49470aaae6eb
md"""
For a given data set ``D``, the posterior probabilities of the parameters scale relatively against each other as

```math
p(\theta|D) \propto p(D|\theta) p(\theta)
```

```math
\Rightarrow
```

All that we can learn from the observed data is contained in the likelihood function ``p(D|\theta)``. This is called the **likelihood principle**.

"""

# ╔═╡ 3e1c51e2-d294-11ef-2c6d-d32a98308c6f
md"""
## The Likelihood Function vs the Sampling Distribution

Consider a distribution ``p(D|\theta)``, where ``D`` relates to variables that are observed (i.e., a "data set") and ``\theta`` are model parameters.

"""

# ╔═╡ 3e1c60ba-d294-11ef-3a01-cf9e97512857
md"""
In general, ``p(D|\theta)`` is just a function of the two variables ``D`` and ``\theta``. We distinguish two interpretations of this function, depending on which variable is observed (or given by other means). 

"""

# ╔═╡ 3e1c70be-d294-11ef-14ed-0d46515541c5
md"""
The **sampling distribution** (a.k.a. the **data-generating** distribution) 

```math
p(D|\theta=\theta_0)
```

(which is a function of ``D`` only) describes a probability distribution for data ``D``, assuming that it is generated by the given model with parameters fixed at ``\theta = \theta_0``.

"""

# ╔═╡ 3e1c806a-d294-11ef-1fad-17e5625279f7
md"""
In a machine learning context, often the data is observed, and ``\theta`` is the free variable. In that case, for given observations ``D=D_0``, the **likelihood function** (which is a function only of the model parameters ``\theta``) is defined as 

```math
\mathrm{L}(\theta) \triangleq p(D=D_0|\theta)
```

"""

# ╔═╡ 3e1c9184-d294-11ef-3e35-5393d97fbc44
md"""
Note that ``\mathrm{L}(\theta)`` is not a probability distribution for ``\theta`` since in general ``\sum_\theta \mathrm{L}(\theta) \neq 1``.

"""

# ╔═╡ 3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
md"""
## Code Example: Sampling Distribution and Likelihood Function for the Coin Toss

Consider the following simple model for the outcome (head or tail) ``y \in \{0,1\}`` of a biased coin toss with parameter ``\theta \in [0,1]``:

```math
\begin{align*}
p(y|\theta) = \theta^y (1-\theta)^{1-y}\\
\end{align*}
```

Next, we use Julia to plot both the sampling distribution 

```math
p(y|\theta=0.5) = \begin{cases} 0.5 & \text{if }y=0 \\ 0.5 & \text{if } y=1 \end{cases}
```

and the likelihood function 

```math
L(\theta) = p(y=1|\theta) = \theta \,.
```

"""

# ╔═╡ 3e1d0c88-d294-11ef-0f69-1b12eb875b23
using Plots
using LaTeXStrings

f(y,θ) = θ.^y .* (1 .- θ).^(1 .- y) # p(y|θ)

θ = 0.5
p1 = plot([0,1], f([0,1], θ), 
  line=:stem, marker=:circle, xrange=(-0.5, 1.5), yrange=(0,1), title="Sampling Distribution", xlabel="y", ylabel=L"p(y|θ=%$θ)", label="")

_θ = 0:0.01:1
y=1
p2 = plot(_θ, f(y, _θ), 
  ylabel=L"p(y=%$y | θ)", xlabel=L"θ", title="Likelihood Function", label="")

plot(p1, p2)

# ╔═╡ 3e1d20e0-d294-11ef-2044-e1fe6590a600
md"""
The (discrete) sampling distribution is a valid probability distribution. 

However, the likelihood function ``L(\theta)`` clearly isn't, since ``\int_0^1 L(\theta) \mathrm{d}\theta \neq 1``. 

"""

# ╔═╡ 3e1d33c8-d294-11ef-0a08-bdc419949925
md"""
## Probabilistic Inference

**Probabilistic inference** refers to computing

```math
p(\,\text{whatever-we-want-to-know}\, | \,\text{whatever-we-already-know}\,)
```

For example: 

```math
\begin{align*}
 p(\,\text{Mr.S.-killed-Mrs.S.} \;&|\; \text{he-has-her-blood-on-his-shirt}\,) \\
 p(\,\text{transmitted-codeword} \;&|\;\text{received-codeword}\,) 
  \end{align*}
```

"""

# ╔═╡ 3e1d4160-d294-11ef-0dc6-d7aa9bce56a1
md"""
This can be accomplished by repeated application of sum and product rules.

"""

# ╔═╡ 3e1d51a0-d294-11ef-228e-294b503d2e3d
md"""
In particular, consider a joint distribution ``p(X,Y,Z)``. Assume we are interested in ``p(X|Z)``:

```math
\begin{align*}
p(X|Z) \stackrel{p}{=} \frac{p(X,Z)}{p(Z)} \stackrel{s}{=} \frac{\sum_Y p(X,Y,Z)}{\sum_{X,Y} p(X,Y,Z)} \,,
\end{align*}
```

where the 's' and 'p' above the equality sign indicate whether the sum or product rule was used. 

"""

# ╔═╡ 3e1d5efc-d294-11ef-0627-cfa86f1447ca
md"""
In the rest of this course, we'll encounter many long probabilistic derivations. For each manipulation, you should be able to associate an 's' (for sum rule), a 'p' (for product or Bayes rule) or an 'm' (for a simplifying model assumption) above any equality sign.

"""

# ╔═╡ 3e1d6d00-d294-11ef-1081-e11b8397eb91
md"""
## Revisiting the Challenge: Disease Diagnosis

**Problem**: Given a disease ``D`` with prevalence of ``1\%`` and a test procedure ``T`` with sensitivity ('true positive' rate) of ``95\%`` and specificity ('true negative' rate) of ``85\%``, what is the chance that somebody who tests positive actually has the disease?

"""

# ╔═╡ 3e1d9cd2-d294-11ef-2eb0-f99ae4e66ec7
md"""
**Solution**: The given information is ``p(D=1)=0.01``, ``p(T=1|D=1)=0.95`` and ``p(T=0|D=0)=0.85``. We are asked to derive ``p( D=1 | T=1)``. We just follow the sum and product rules to derive the requested probability:

```math
\begin{align*}
p( D=1 &| T=1) \\
&\stackrel{p}{=} \frac{p(T=1,D=1)}{p(T=1)} \\
&\stackrel{p}{=} \frac{p(T=1|D=1)p(D=1)}{p(T=1)} \\
&\stackrel{s}{=} \frac{p(T=1|D=1)p(D=1)}{p(T=1|D=1)p(D=1)+p(T=1|D=0)p(D=0)} \\
&= \frac{0.95\times0.01}{0.95\times0.01 + 0.15\times0.99} = 0.0601
\end{align*}
```

"""

# ╔═╡ 3e1dc900-d294-11ef-331a-9b17133817d2
md"""
Note that ``p(\text{sick}|\text{positive test}) = 0.06`` while ``p(\text{positive test} | \text{sick}) = 0.95``. This is a huge difference that is sometimes called the "medical test paradox" or the [base rate fallacy](https://en.wikipedia.org/wiki/Base_rate_fallacy). 

Many people have trouble distinguishing ``p(A|B)`` from ``p(B|A)`` in their heads. This has led to major negative consequences. For instance, unfounded convictions in the legal arena and even lots of unfounded conclusions in the pursuit of scientific results. See [Ioannidis (2005)](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124) and [Clayton (2021)](https://aubreyclayton.com/bernoulli).

"""

# ╔═╡ 3e1de32c-d294-11ef-1f63-f190c8361404
md"""
## Inference Exercise: Bag Counter

**Problem**:  A bag contains one ball, known to be either white or black. A white ball is put in, the bag is shaken,

and a ball is drawn out, which proves to be white. What is now the  chance of drawing a white ball?

"""

# ╔═╡ 3e1e134c-d294-11ef-18c0-21742fe74fa6
md"""
**Solution**: Again, use Bayes and marginalization to arrive at ``p(\text{white}|\text{data})=2/3``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.

```math
\Rightarrow
```

Note that probabilities describe **a person's state of knowledge** rather than a 'property of nature'.

"""

# ╔═╡ 3e1e2b96-d294-11ef-3a68-fdc78232142e
md"""
## Inference Exercise: Causality?

**Problem**: A dark bag contains five red balls and seven green ones. (a) What is the probability of drawing a red ball on the first draw? Balls are not returned to the bag after each draw. (b) If you know that on the second draw the ball was a green one, what is now the probability of drawing a red ball on the first draw?

"""

# ╔═╡ 3e1e3a34-d294-11ef-0053-393751b94f2c
md"""
**Solution**: (a) ``5/12``. (b) ``5/11``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.

```math
\Rightarrow
```

Again, we conclude that conditional probabilities reflect **implications for a state of knowledge** rather than temporal causality.

"""

# ╔═╡ 3e1e4dda-d294-11ef-33b7-4bbe3300ca22
md"""
## Moments of the PDF

Distributions can often usefully be summarized by a set of values known as moments of the distribution.  

Consider a distribution ``p(x)``. The first moment, also known as **expected value** or **mean** of ``p(x)`` is defined as 

```math
\mu_x = \mathbb{E}[x] \triangleq  \int x \,p(x) \,\mathrm{d}{x}
```

"""

# ╔═╡ 3e1e5a5a-d294-11ef-2fdf-efee4eb1a0f2
md"""
The second central moment, also known as **variance** of ``x`` is defined as 

```math
\Sigma_x \triangleq \mathbb{E} \left[(x-\mu_x)(x-\mu_x)^T \right]
```

"""

# ╔═╡ 3e1e7742-d294-11ef-1204-f9be24da07ab
md"""
The **covariance** matrix between *vectors* ``x`` and ``y`` is a mixed central moment, defined as

```math
\begin{align*}
    \Sigma_{xy} &\triangleq \mathbb{E}\left[ (x-\mu_x) (y-\mu_y)^T \right]\\
    &= \mathbb{E}\left[ (x-\mu_x) (y^T-\mu_y^T) \right]\\
    &= \mathbb{E}[x y^T] - \mu_x \mu_y^T
\end{align*}
```

Clearly, if ``x`` and ``y`` are independent, then ``\Sigma_{xy} = 0``, since in that case ``\mathbb{E}[x y^T] = \mathbb{E}[x] \mathbb{E}[y^T] = \mu_x \mu_y^T``.

Exercise: Proof that ``\Sigma_{xy} = \Sigma_{yx}^{T}`` (making use of ``(AB)^T = B^TA^T``).

"""

# ╔═╡ 3e1e9224-d294-11ef-38b3-137c2be22400
md"""
## $(HTML("<span id='linear-transformation'>Linear Transformations</span>"))

Consider an arbitrary distribution ``p(X)`` with mean ``\mu_x`` and variance ``\Sigma_x`` and the linear transformation 

```math
Z = A X + b \,.
```

No matter the specification of ``p(X)``, we can derive that (see [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook)

```math
\begin{align*}
\mu_z &= A\mu_x + b &\qquad \text{(SRG-3a)}\\
\Sigma_z &= A\,\Sigma_x\,A^T &\qquad \text{(SRG-3b)}
\end{align*}
```

(The tag (SRG-3a) refers to the corresponding eqn number in Sam Roweis' [Gaussian identities](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Roweis-1999-gaussian-identities.pdf) notes.)

"""

# ╔═╡ 3e1ea442-d294-11ef-1364-8dd9986325f7
md"""
## PDF for the Sum of Two Variables

Given eqs SRG-3a and SRG-3b (previous cell), you should now be able to derive the following: for any distribution of variable ``X`` and ``Y`` and sum ``Z = X+Y`` (proof by [Exercise](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb))

```math
\begin{align*}
    \mu_z &= \mu_x + \mu_y \\
    \Sigma_z &= \Sigma_x + \Sigma_y + 2\Sigma_{xy} 
\end{align*}
```

"""

# ╔═╡ 3e1eba72-d294-11ef-2f53-b56f1862fcbb
md"""
Clearly, it follows that if ``X`` and ``Y`` are **independent**, then

```math
\Sigma_z = \Sigma_x + \Sigma_y 
```

"""

# ╔═╡ 3e1ed1a4-d294-11ef-2de4-d7cc540e06a1
md"""
More generally, assume two jointly continuous variables ``X`` and ``Y``, with joint PDF ``p_{xy}(x,y)``. Let ``Z=X+Y``, then

```math
\begin{align*}
\text{Prob}(Z\leq z) &= \text{Prob}(X+Y\leq z)\\
&= \int_{-\infty}^\infty \biggl( \int_{-\infty}^{z-x} p_{xy}(x,y) \mathrm{d}y \biggr) \mathrm{d}x \\
&= \int_{-\infty}^\infty \biggl( \int_{-\infty}^{z} p_{xy}(x,t-x) \mathrm{d}t \biggr) \mathrm{d}x \\
&= \int_{-\infty}^z \biggl( \underbrace{\int_{-\infty}^{\infty} p_{xy}(x,t-x) \mathrm{d}x}_{p_z(t)} \biggr) \mathrm{d}t
\end{align*}
```

Hence, the PDF for the sum ``Z`` is given by ``p_z(z) = \int_{-\infty}^{\infty} p_{xy}(x,z-x) \mathrm{d}x``.

In particular, if ``X`` and ``Y`` are **independent** variables, then

```math
p_z (z) = \int_{-\infty}^{\infty}  p_x(x) p_y(z - x)\,\mathrm{d}{x} = p_x(z) * p_y(z)\,,
```

which is the **convolution** of the two marginal PDFs. 

"""

# ╔═╡ 3e1eeb14-d294-11ef-1702-f5d2cf6fe60a
md"""
[https://en.wikipedia.org/wiki/List*of*convolutions*of*probability_distributions](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions) shows how these convolutions work out for a few common probability distributions. 

"""

# ╔═╡ 3e1f130a-d294-11ef-292f-37578d61ea52
md"""


"""

# ╔═╡ 3e1f225a-d294-11ef-04c6-f3ca018ab286
md"""
## Code Example: Sum of Two Gaussian Distributed Variables

Consider two independent Gaussian-distributed variables ``X`` and ``Y`` (see [wiki:normal-distribution](https://en.wikipedia.org/wiki/Normal_distribution) for definition of a Gaussian (=Normal) distribution):

```math
\begin{align*}
p_X(x) &= \mathcal{N}(\,x\,|\,\mu_X,\sigma_X^2\,) \\ 
p_Y(y) &= \mathcal{N}(\,y\,|\,\mu_Y,\sigma_Y^2\,) 
\end{align*}
```

Let ``Z = X + Y``. Performing the convolution (nice exercise) yields a Gaussian PDF for ``Z``: 

```math
p_Z(z) = \mathcal{N}(\,z\,|\,\mu_X+\mu_Y,\sigma_X^2+\sigma_Y^2\,).
```

We illustrate the distributions for ``X``, ``Y`` and ``Z`` using Julia:

"""

# ╔═╡ 3e1f3d8a-d294-11ef-307d-452ce35c16a4
using Plots, Distributions, LaTeXStrings

μx = 2.
σx = 1.
μy = 2.
σy = 0.5
μz = μx+μy; σz = sqrt(σx^2 + σy^2)
x = Normal(μx, σx)
y = Normal(μy, σy)
z = Normal(μz, σz)
range_min = minimum([μx-2*σx, μy-2*σy, μz-2*σz])
range_max = maximum([μx+2*σx, μy+2*σy, μz+2*σz])
range_grid = range(range_min, stop=range_max, length=100)
plot(range_grid, pdf.(x,range_grid), label=L"p_x", fill=(0, 0.1))
plot!(range_grid, pdf.(y,range_grid), label=L"p_y", fill=(0, 0.1))
plot!(range_grid, pdf.(z,range_grid), label=L"p_z", fill=(0, 0.1))

# ╔═╡ 3e1f4f46-d294-11ef-29b8-69e546763781
md"""
## PDF for the Product of Two Variables

For two continuous **independent** variables

```math
X
```

and ``Y``, with PDF's ``p_x(x)`` and ``p_y(y)``, the PDF of  :Z = X Y $ is given by 

```math
p_z(z) = \int_{-\infty}^{\infty} p_x(x) \,p_y(z/x)\, \frac{1}{|x|}\,\mathrm{d}x\,.
```

For proof, see [https://en.wikipedia.org/wiki/Product_distribution](https://en.wikipedia.org/wiki/Product_distribution).

"""

# ╔═╡ 3e1f68fa-d294-11ef-31b2-e7670da8c08c
md"""
Generally, this integral does not lead to an analytical expression for ``p_z(z)``. 

For example, [the product of two independent variables that are both Gaussian-distributed does not lead to a Gaussian distribution](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Gaussian-Distribution.ipynb#product-of-gaussians).

  * Exception: the distribution of the product of two variables that both have [log-normal distributions](https://en.wikipedia.org/wiki/Log-normal_distribution) is again a lognormal distribution. (If ``X`` has a normal distribution, then ``Y=\exp(X)`` has a log-normal distribution.)

"""

# ╔═╡ 3e1f7d5e-d294-11ef-2878-05744036f32c
md"""
## Variable Transformations

Suppose ``x`` is a **discrete** variable with probability **mass** function ``P_x(x)``, and ``y = h(x)`` is a one-to-one function with ``x = g(y) = h^{-1}(y)``. Then

```math
P_y(y) = P_x(g(y))\,.
```

"""

# ╔═╡ 3e1f8e48-d294-11ef-0f8a-b58294a8543d
md"""
**Proof**: ``P_y(\hat{y}) = P(y=\hat{y}) = P(h(x)=\hat{y}) = P(x=g(\hat{y})) = P_x(g(\hat{y})). \,\square``

"""

# ╔═╡ 3e1fa04a-d294-11ef-00c3-a51d1aaa5553
md"""
If ``x`` is defined on a **continuous** domain, and ``p_x(x)`` is a probability **density** function, then probability mass is represented by the area under a (density) curve. Let ``a=g(c)`` and ``b=g(d)``. Then

```math
\begin{align*}
P(a ≤ x ≤ b) &= \int_a^b p_x(x)\mathrm{d}x \\
  &= \int_{g(c)}^{g(d)} p_x(x)\mathrm{d}x \\
  &= \int_c^d p_x(g(y))\mathrm{d}g(y) \\
  &= \int_c^d \underbrace{p_x(g(y)) g^\prime(y)}_{p_y(y)}\mathrm{d}y \\  
  &= P(c ≤ y ≤ d)
\end{align*}
```

Equating the two probability masses leads to identification of the relation 

```math
p_y(y) = p_x(g(y)) g^\prime(y)\,,
```

which is also known as the [Change-of-Variable theorem](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function). 

If the tranformation ``y = h(x)`` is not invertible, then ``x=g(y)`` does not exist. In that case, you can still work out the transformation by equating equivalent probability masses in the two domains. 

"""

# ╔═╡ 3e1fb370-d294-11ef-1fb6-63a41a024691
md"""
## Example: Transformation of a Gaussian Variable

Let ``p_x(x) = \mathcal{N}(x|\mu,\sigma^2)`` and ``y = \frac{x-\mu}{\sigma}``. 

**Problem**: What is ``p_y(y)``? 

**Solution**: Note that ``h(x)`` is invertible with ``x = g(y) = \sigma y + \mu``. The change-of-variable formula leads to

```math
\begin{align*}
p_y(y) &= p_x(g(y)) \cdot g^\prime(y) \\
  &= p_x(\sigma y + \mu) \cdot \sigma \\
  &= \frac{1}{\sigma\sqrt(2 \pi)} \exp\left( - \frac{(\sigma y + \mu - \mu)^2}{2\sigma^2}\right) \cdot \sigma \\
  &=  \frac{1}{\sqrt(2 \pi)} \exp\left( - \frac{y^2 }{2}\right)\\
  &= \mathcal{N}(y|0,1) 
\end{align*}
```

"""

# ╔═╡ 3e1fc4da-d294-11ef-12f5-d51f9728fcc0
md"""
## A Notational Convention

Finally, here is a notational convention that you should be precise about (but many authors are not).

If you want to write that a variable ``x`` is distributed as a Gaussian with mean ``\mu`` and covariance matrix ``\Sigma``, you can write this properly in either of two ways:

```math
\begin{align*} 
p(x) &= \mathcal{N}(x|\mu,\Sigma) \\
x &\sim \mathcal{N}(\mu,\Sigma)
\end{align*}
```

In the second version, the symbol ``\sim`` can be interpreted as "is distributed as" (a Gaussian with parameters ``\mu`` and ``\Sigma``).

Don't write ``p(x) = \mathcal{N}(\mu,\Sigma)`` because ``p(x)`` is a function of ``x`` but ``\mathcal{N}(\mu,\Sigma)`` is not. 

Also, ``x \sim \mathcal{N}(x|\mu,\Sigma)`` is not proper because you already named the argument at the right-hand-site. On the other hand, ``x \sim \mathcal{N}(\cdot|\mu,\Sigma)`` is fine, as is the shorter ``x \sim \mathcal{N}(\mu,\Sigma)``.

"""

# ╔═╡ 3e1fd38a-d294-11ef-05d3-ad467328be96
md"""
## Summary

Probabilities should be interpretated as degrees of belief, i.e., a state-of-knowledge, rather than a property of nature.

"""

# ╔═╡ 3e1fe0de-d294-11ef-0d8c-35187e394292
md"""
We can do everything with only the **sum rule** and the **product rule**. In practice, **Bayes rule** and **marginalization** are often very useful for inference, i.e., for computing

```math
p(\,\text{what-we-want-to-know}\,|\,\text{what-we-already-know}\,)\,.
```

"""

# ╔═╡ 3e1fedfc-d294-11ef-30ee-a396bb877037
md"""
Bayes rule 

```math
 p(\theta|D) = \frac{p(D|\theta)p(\theta)} {p(D)} 
```

is the fundamental rule for learning from data!

"""

# ╔═╡ 3e1ffc5c-d294-11ef-27b1-4f6ccb64c5d6
md"""
For a variable ``X`` with distribution ``p(X)`` with mean ``\mu_x`` and variance ``\Sigma_x``, the mean and variance of the **Linear Transformation** ``Z = AX +b`` is given by 

```math
\begin{align}
\mu_z &= A\mu_x + b \tag{SRG-3a}\\
\Sigma_z &= A\,\Sigma_x\,A^T \tag{SRG-3b}
\end{align}
```

"""

# ╔═╡ 3e2009e2-d294-11ef-255d-8d4a44865663
md"""
That's really about all you need to know about probability theory, but you need to *really* know it, so do the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb)!

"""

# ╔═╡ 3e2023c6-d294-11ef-3ac0-75118015870e
open("../../styles/aipstyle.html") do f
    display("text/html", read(f,String))
end

# ╔═╡ Cell order:
# ╟─3e17df5e-d294-11ef-38c7-f573724871d8
# ╟─3e1803d0-d294-11ef-0304-df2b9b698cd1
# ╟─3e1823b0-d294-11ef-3dba-9997a7230cdf
# ╟─3e185ab0-d294-11ef-3f7d-9bd465518274
# ╟─3e1876f8-d294-11ef-22bf-7904df3c1182
# ╟─3e1889b8-d294-11ef-17bb-496655fbd618
# ╟─3e189fa2-d294-11ef-1f2b-2151b6c128f8
# ╟─3e18b2fa-d294-11ef-1255-df048f0dcec2
# ╟─3e18c25c-d294-11ef-11bc-a93c2572b107
# ╟─3e18d2ea-d294-11ef-35e9-2332dd31dbf0
# ╟─3e18e4bc-d294-11ef-38bc-cb97cb4e0963
# ╟─3e18f18c-d294-11ef-33e4-b7f9495e0508
# ╟─3e1906ea-d294-11ef-236e-c966a9474170
# ╟─3e191b6c-d294-11ef-3174-d1b4b36e252b
# ╟─3e192ef4-d294-11ef-1fc4-87175eeec5eb
# ╟─3e19436c-d294-11ef-11c5-f9914f7a3a57
# ╟─3e194ef2-d294-11ef-3b38-1ddc3063ff35
# ╟─3e1964b4-d294-11ef-373d-712257fc130f
# ╟─3e196d6a-d294-11ef-0795-41c045079251
# ╟─3e198336-d294-11ef-26fd-03cd15876486
# ╟─3e198ba6-d294-11ef-3fe7-d70bf4833fa6
# ╟─3e19a2a0-d294-11ef-18b4-7987534916d2
# ╟─3e19ac30-d294-11ef-10b7-fbba9ae2a2c3
# ╟─3e19c06c-d294-11ef-197a-f549e8107a57
# ╟─3e19d39a-d294-11ef-1a50-7fe8a24777dc
# ╟─3e19e95a-d294-11ef-3da4-6d23922a5150
# ╟─3e19fd2a-d294-11ef-2a52-b9245f6d02ba
# ╟─3e1a36b4-d294-11ef-2242-f36061b0b754
# ╟─3e1a522a-d294-11ef-1a7a-bdcfbd5dae09
# ╟─3e1a69f4-d294-11ef-103e-efc47025fb8f
# ╟─3e1a7c8e-d294-11ef-1f97-55e608d49141
# ╟─3e1a8eca-d294-11ef-1ef0-c15b24d05990
# ╟─3e1a9fdc-d294-11ef-288f-37fd1b7ee281
# ╟─3e1ab104-d294-11ef-1a98-412946949fba
# ╟─3e1ac32c-d294-11ef-13be-558397a6cc2a
# ╟─3e1ad2ea-d294-11ef-02c4-3f06e14ea4d8
# ╟─3e1ae30a-d294-11ef-3a5d-d91b7d7723d3
# ╟─3e1af354-d294-11ef-1971-dfb39016cfcd
# ╟─3e1b05ee-d294-11ef-33de-efed64d01c0d
# ╟─3e1b4b1c-d294-11ef-0423-9152887cc403
# ╟─3e1b5c9c-d294-11ef-137f-d75b3731eae4
# ╟─3e1b6bce-d294-11ef-2bd9-29c0634b1856
# ╟─3e1b7d14-d294-11ef-0d10-1148a928dd57
# ╟─3e1b8bf4-d294-11ef-04cc-6364e46fdd64
# ╟─3e1b9ba8-d294-11ef-18f2-db8eed3d87d0
# ╟─3e1babca-d294-11ef-37c1-cd821a6488b2
# ╟─3e1bba8e-d294-11ef-1f61-295af16078ce
# ╟─3e1bcb00-d294-11ef-2795-bd225bd00496
# ╟─3e1bdd02-d294-11ef-19e8-2f44eccf58af
# ╟─3e1bf116-d294-11ef-148b-f7a1ca3f3bad
# ╟─3e1bffec-d294-11ef-2a49-9ff0f6331add
# ╟─3e1c0e80-d294-11ef-0d19-375e01988f16
# ╟─3e1c1e3e-d294-11ef-0955-bdf9d0ba3c53
# ╟─3e1c4224-d294-11ef-2707-49470aaae6eb
# ╟─3e1c51e2-d294-11ef-2c6d-d32a98308c6f
# ╟─3e1c60ba-d294-11ef-3a01-cf9e97512857
# ╟─3e1c70be-d294-11ef-14ed-0d46515541c5
# ╟─3e1c806a-d294-11ef-1fad-17e5625279f7
# ╟─3e1c9184-d294-11ef-3e35-5393d97fbc44
# ╟─3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
# ╠═3e1d0c88-d294-11ef-0f69-1b12eb875b23
# ╟─3e1d20e0-d294-11ef-2044-e1fe6590a600
# ╟─3e1d33c8-d294-11ef-0a08-bdc419949925
# ╟─3e1d4160-d294-11ef-0dc6-d7aa9bce56a1
# ╟─3e1d51a0-d294-11ef-228e-294b503d2e3d
# ╟─3e1d5efc-d294-11ef-0627-cfa86f1447ca
# ╟─3e1d6d00-d294-11ef-1081-e11b8397eb91
# ╟─3e1d9cd2-d294-11ef-2eb0-f99ae4e66ec7
# ╟─3e1dc900-d294-11ef-331a-9b17133817d2
# ╟─3e1de32c-d294-11ef-1f63-f190c8361404
# ╟─3e1e134c-d294-11ef-18c0-21742fe74fa6
# ╟─3e1e2b96-d294-11ef-3a68-fdc78232142e
# ╟─3e1e3a34-d294-11ef-0053-393751b94f2c
# ╟─3e1e4dda-d294-11ef-33b7-4bbe3300ca22
# ╟─3e1e5a5a-d294-11ef-2fdf-efee4eb1a0f2
# ╟─3e1e7742-d294-11ef-1204-f9be24da07ab
# ╟─3e1e9224-d294-11ef-38b3-137c2be22400
# ╟─3e1ea442-d294-11ef-1364-8dd9986325f7
# ╟─3e1eba72-d294-11ef-2f53-b56f1862fcbb
# ╟─3e1ed1a4-d294-11ef-2de4-d7cc540e06a1
# ╟─3e1eeb14-d294-11ef-1702-f5d2cf6fe60a
# ╟─3e1f130a-d294-11ef-292f-37578d61ea52
# ╟─3e1f225a-d294-11ef-04c6-f3ca018ab286
# ╠═3e1f3d8a-d294-11ef-307d-452ce35c16a4
# ╟─3e1f4f46-d294-11ef-29b8-69e546763781
# ╟─3e1f68fa-d294-11ef-31b2-e7670da8c08c
# ╟─3e1f7d5e-d294-11ef-2878-05744036f32c
# ╟─3e1f8e48-d294-11ef-0f8a-b58294a8543d
# ╟─3e1fa04a-d294-11ef-00c3-a51d1aaa5553
# ╟─3e1fb370-d294-11ef-1fb6-63a41a024691
# ╟─3e1fc4da-d294-11ef-12f5-d51f9728fcc0
# ╟─3e1fd38a-d294-11ef-05d3-ad467328be96
# ╟─3e1fe0de-d294-11ef-0d8c-35187e394292
# ╟─3e1fedfc-d294-11ef-30ee-a396bb877037
# ╟─3e1ffc5c-d294-11ef-27b1-4f6ccb64c5d6
# ╟─3e2009e2-d294-11ef-255d-8d4a44865663
# ╠═3e2023c6-d294-11ef-3ac0-75118015870e
