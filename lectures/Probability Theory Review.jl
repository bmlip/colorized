### A Pluto.jl notebook ###
# v0.20.8

#> [frontmatter]
#> description = "Review of probability theory as a foundation for rational reasoning and Bayesian inference."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ eeb9a1f5-b857-4843-920b-2e4a9656f66b
using Plots, LaTeXStrings

# ╔═╡ 5394e37c-ae00-4042-8ada-3bbf32fbca9e
using Distributions

# ╔═╡ b305a905-06c2-4a15-8042-72ef6375720f
using PlutoUI, PlutoTeachingTools

# ╔═╡ 7910a84c-18b3-4081-9f01-e59258a01adb
using HypertextLiteral

# ╔═╡ 3e17df5e-d294-11ef-38c7-f573724871d8
md"""
# Probability Theory Review

"""

# ╔═╡ bcb4be20-0439-4809-a166-8c50b6b9206b
PlutoUI.TableOfContents()

# ╔═╡ 3e1803d0-d294-11ef-0304-df2b9b698cd1
md"""
## Preliminaries

##### Goal 

- Review of Probability Theory as a theory for rational/logical reasoning with uncertainties (i.e., a Bayesian interpretation)

##### Materials        

- Mandatory

  - These lecture notes

- Optional

  - Bishop pp. 12-24
      
  - [Edwin Jaynes, Probability Theory–The Logic of Science (2003)](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf). 
    - Brilliant book on the Bayesian view of probability theory. Just for fun, scan the annotated bibliography and references.

  - [Aubrey Clayton, Bernoulli's Fallacy–Statistical Illogic and the Crisis of Modern Science (2021)](https://aubreyclayton.com/bernoulli)
    - A very readable account of the history of statistics and probability theory. Discusses why most popular statistics recipes are very poor scientific analysis tools. Use probability theory instead!

  - [Ariel Caticha, Entropic Inference and the Foundations of Physics (2012)](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-56 (ch.2: probability)
    - Great introduction to probability theory, in particular w.r.t. its correct interpretation as a state-of-knowledge.
    - Absolutely worth your time to read the whole chapter, even if you skip section 2.2.4 (pp.15-18) on Cox's proof.

  - [Joram Soch et al ., The Book of Statistical Proofs (2023 - )](https://statproofbook.github.io/)
    - Online resource for proofs in probability theory and statistical inference.

"""

# ╔═╡ 3e1823b0-d294-11ef-3dba-9997a7230cdf
md"""
## 📕 Data Analysis: A Bayesian Tutorial

The following is an excerpt from the book [Data Analysis: A Bayesian Tutorial](https://global.oup.com/academic/product/data-analysis-9780198568322) (2006), by D.S. Sivia with J.S. Skilling:

> #### Preface
> "As an undergraduate, I always found the subject of statistics to be rather mysterious. This topic wasn’t entirely new to me, as we had been taught a little bit about probability earlier at high school; for example, I was already familiar with the binomial, Poisson and normal distributions. Most of this made sense, but only seemed to relate to things like rolling dice, flipping coins, shuffling cards and so on. However, having aspirations of becoming a scientist, what I really wanted to know was how to analyse experimental data. Thus, I eagerly looked forward to the lectures on statistics. Sadly, they were a great disappointment. Although many of the tests and procedures expounded were intuitively reasonable, there was something deeply unsatisfactory about the whole affair: there didn’t seem to be any underlying basic principles! Hence, the course on ‘probability and statistics’ had led to an unfortunate dichotomy: probability made sense, but was just a game; statistics was important, but it was a bewildering collection of tests with little obvious rhyme or reason. While not happy with this situation, I decided to put aside the subject and concentrate on real science. After all, the predicament was just a reflection of my own inadequacies and I’d just have to work at it when the time came to really analyse my data.
> 
> The story above is not just my own, but is the all too common experience of many scientists. Fortunately, it doesn’t have to be like this. What we were not told in our undergraduate lectures is that there is an alternative approach to the whole subject of data analysis which uses only probability theory. In one sense, it makes the topic of statistics entirely superfluous. In another, it provides the logical justification for many of the prevalent statistical tests and procedures, making explicit the conditions and approximations implicitly assumed in their use."


Does this fragment resonate with your own experience? 

In this lesson we introduce *Probability Theory* (PT) again. As we will see in the next lessons, PT is all you need to make sense of machine learning, artificial intelligence, statistics, etc. 

"""

# ╔═╡ 840ab4dc-0d2e-4bf8-acc7-5f1ee2b0dcaf
md"""
# Probability Theory as Rational Reasoning
"""

# ╔═╡ 41bee964-a0a9-4a7f-8505-54a9ee12ef0d
md"""
## Propositional (Boolean) Logic 

Define an **event** (or "proposition") ``A`` as a statement that can be considered for its truth by a person. For instance, 

```math
𝐴= \texttt{``there is life on Mars''}
```

Boolean logic (or propositional logic) is a formal system of logic based on binary truth values: every proposition is either true (with assigned value ``1``) or false (with assigned value ``0``). It is named after George Boole, who developed the algebraic formulation of logic in the mid-19th century.

With Boolean operators (``\lor``, ``\land``, ``\implies``, etc.), we can create and evaluate compound propositions, e.g.,

- Given two events ``A`` and ``B``, the **conjunction** (logical-and) ``A \land B`` is true only if both ``A`` and ``B`` are true. We write ``A \land B`` also shortly as ``A,B`` or ``AB``. 

- The **disjunction** (logical-or) ``A \lor B``, is true if either ``A`` or ``B`` is true or both ``A`` and ``B`` are true. We write ``A \lor B`` also as ``A + B`` (Note that the plus-sign is here not an arithmetic operator, but rather a logical operator to process truth values.)

- The denial of ``A``, i.e., the event **not**-A, is written as ``\bar{A}``. 

Boolean logic provides the rules of inference for **deductive reasoning** and underpins all formal reasoning systems in mathematics and philosophy. 
"""

# ╔═╡ 3e1889b8-d294-11ef-17bb-496655fbd618
md"""
## The Design of Probability Theory

Consider the truth value of the proposition 
```math
𝐴= \texttt{``there is life on Mars''}
```

with 

```math
I = \texttt{``All known life forms require water''}
```

as background information. Now assume that a new piece of information 

```math
x = \texttt{``There is water on Mars''}
```

becomes available, how **should** our degree of belief in event ``A`` be affected *if we were rational*? 

"""

# ╔═╡ 3e18b2fa-d294-11ef-1255-df048f0dcec2
md"""
[Richard T. Cox (1946)](https://aapt.scitation.org/doi/10.1119/1.1990764) developed a **calculus for rational reasoning** about how to represent and update the **degree-of-belief** about the truth value of an event when faced with new information.  

"""

# ╔═╡ 3e18c25c-d294-11ef-11bc-a93c2572b107
md"""
In developing this calculus, only some very agreeable assumptions were made, including:

- Degrees of belief are represented by real numbers.

- Plausibility assessments are consistent, e.g.,
  - if ``A`` becomes more plausible under new information ``B``, the assigned degree-of-belief should increase accordingly.
  - If the belief in ``A`` is greater than the belief in ``B``, and the belief in ``B`` is greater than the belief in ``C``, then the belief in ``A`` must be greater than the belief in ``C``.

- Logical equivalences are preserved, e.g., 
  - If the belief in an event can be inferred in two different ways, then the two ways must agree on the resulting belief.

"""

# ╔═╡ 3e18d2ea-d294-11ef-35e9-2332dd31dbf0
md"""
Under these assumptions, Cox showed that any consistent system of reasoning about uncertainty must obey the **rules of probability theory** (see [Cox theorem, 1946](https://en.wikipedia.org/wiki/Cox%27s_theorem), and [Caticha, 2012](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-26). These rules are the sum and product rules:

##### The sum rule

- The degree of belief in the disjunction of two events ``A`` and ``B``, with given background information ``I``, is evaluated as

```math
 p(A+B|I) = p(A|I) + p(B|I) - p(A,B|I)
```

##### The product rule: 

- The degree of belief in the conjunction of two events ``A`` and ``B``, with given background information ``I``, is evaluated as

```math
 p(A,B|I) = p(A|B,I)\,p(B|I)
```

Cox’s Theorem derives the rules of probability theory from first principles, not as arbitrary postulates but as consequences of rational reasoning. 
In other words: **Probability = extended logic**.

"""

# ╔═╡ dd11e93a-3dad-4e97-8642-fb70edfa6aae
md"""
In the above sum and product rules, the **conditional probability** of ``A`` given ``I``, denoted by ``p(A|I)``, indicates the degree of belief in event ``A``, given that ``I`` is true. 
"""

# ╔═╡ 3e18e4bc-d294-11ef-38bc-cb97cb4e0963
keyconcept(" ", 
	md"""
	
	If you want to assign real numbers to **degrees of belief**, and you want those assignments to be logically consistent, then you are **forced to follow the sum and product rules of probability theory** (PT). PT is therefore the **optimal calculus for information processing under uncertainty**.  
	
	"""
)

# ╔═╡ 3e1b05ee-d294-11ef-33de-efed64d01c0d
keyconcept(
	"",
	md"""
	All legitimate probabilistic relations can be **derived from the sum and product rules**!
	
	"""
)

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
##### Examples

  * Predictions

```math
p(\,\texttt{future-observations}\,|\,\texttt{past-observations}\,)
```

  * Classify a received data point ``x`` 

```math
p(\,x\texttt{-belongs-to-class-}k \,|\,x\,)
```

  * Update a model based on a new observation

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

  * For instance, in a coin tossing experiment, ``p(\texttt{outcome} = \texttt{tail}) = 0.4`` should be interpreted as the belief that there is a 40% chance that ``\texttt{tail}`` comes up if the coin were tossed.
  * Under the Bayesian interpretation, PT calculus (sum and product rules) **extends boolean logic to rational reasoning with uncertainty**.

"""

# ╔═╡ 4edf38ab-a940-4ab0-be22-fa95cf571146
md"""
In the Bayesian interpretation, all probabilities are, in principle, conditional probabilities of the type ``p(A|I)``, since there is always some background knowledge. However, we often write ``p(A)`` rather than ``p(A|I)`` if the background knowledge ``I`` is assumed to be obviously present. E.g., we usually write ``p(A)`` rather than ``p(\,A\,|\,\text{the-sun-comes-up-tomorrow}\,)``.

"""

# ╔═╡ 3e194ef2-d294-11ef-3b38-1ddc3063ff35
md"""
The Bayesian interpretation contrasts with the **frequentist** interpretation of a probability as the relative frequency that an event would occur under repeated execution of an experiment.

  * For instance, if the experiment is tossing a coin, then ``p(\texttt{outcome} = \texttt{tail}) = 0.4`` means that in the limit of a large number of coin tosses, 40% of outcomes turn up as ``\texttt{tail}``.

"""

# ╔═╡ 3e1964b4-d294-11ef-373d-712257fc130f
md"""
The Bayesian viewpoint is more generally applicable than the frequentist viewpoint, e.g., it is hard to apply the frequentist viewpoint to events like ``\texttt{`it will rain tomorrow'}``. 

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

# ╔═╡ 3e19e95a-d294-11ef-3da4-6d23922a5150
md"""
## Variable Assignments as Propositions 


"""

# ╔═╡ 3e1a69f4-d294-11ef-103e-efc47025fb8f
md"""
If ``X`` is a variable, then an *assignment* ``X=x`` (where ``x`` is a value, e.g., ``X=5``) can be interpreted as an event. Hence, the expression ``p(X=5)`` should be interpreted as the *degree-of-belief of the event* that variable ``X`` takes on the value ``5``. 

"""

# ╔═╡ 3e1a7c8e-d294-11ef-1f97-55e608d49141
md"""
If ``X`` is a *discretely* valued variable, then ``p(X=x)`` is a probability *mass* function (PMF) with ``0\le p(X=x)\le 1`` and normalization ``\sum_x p(x) =1``. 

"""

# ╔═╡ 3e1a8eca-d294-11ef-1ef0-c15b24d05990
md"""
If ``X`` is *continuously* valued, then ``p(X=x)`` is a probability *density* function (PDF) with ``p(X=x)\ge 0``  and normalization ``\int_x p(x)\mathrm{d}x=1``. 

  * Note that if ``X`` is continuously valued, then the value of ``p(x)`` is not necessarily ``\le 1``. E.g., a uniform distribution on the continuous domain ``[0,.5]`` has value ``p(x) = 2`` over its domain.

"""

# ╔═╡ 3e1fc4da-d294-11ef-12f5-d51f9728fcc0
md"""
## Notational Conventions

Here is a notational convention that you should be precise about (but many authors are not).

If you want to write that a variable ``x`` is distributed as a Gaussian with mean ``\mu`` and covariance matrix ``\Sigma``, you can write this in either of two ways:

```math
\begin{align*} 
p(x) &= \mathcal{N}(x|\mu,\Sigma) \\
x &\sim \mathcal{N}(\mu,\Sigma)
\end{align*}
```

In the second version, the symbol ``\sim`` can be interpreted as "is distributed as" (a Gaussian with parameters ``\mu`` and ``\Sigma``).

Don't write ``p(x) = \mathcal{N}(\mu,\Sigma)`` because ``p(x)`` is a function of ``x`` but ``\mathcal{N}(\mu,\Sigma)`` is not. 

Also, ``x \sim \mathcal{N}(x|\mu,\Sigma)`` is not entirely proper because you already named the argument on the right-hand-site. On the other hand, ``x \sim \mathcal{N}(\cdot|\mu,\Sigma)`` is fine, as is the shorter ``x \sim \mathcal{N}(\mu,\Sigma)``.

"""

# ╔═╡ 3e1ab104-d294-11ef-1a98-412946949fba
md"""
# $(HTML("<span id='PT-calculus'>Probability Theory Calculus</span>"))

"""

# ╔═╡ fea8ae4c-8ef9-4b74-ad13-1314afef97de
md"""
## True and False Events

In probability theory, events that are certainly true or certainly false are treated as special cases of general events, and they correspond to probabilities of ``1`` and ``0``, respectively.

Let ``\Omega`` be the sample space, i.e., the set of all possible outcomes of an experiment. Then:
  - The true event corresponds to the entire sample space ``\Omega``.
  - It always happens, regardless of the outcome.
  - Its probability is
```math
p(\Omega) = 1\,.
```

The false event corresponds to the empty set ``\emptyset``.
  - It never happens (no possible outcomes).
  - Its probability is
```math
p(\emptyset) = 0 \,.
```

"""

# ╔═╡ 3e1b4b1c-d294-11ef-0423-9152887cc403
md"""
## Independent, Exclusive, and Exhaustive Events

It will be helpful to introduce some terms concerning special relationships between events.  

##### Joint events

The expression ``p(A,B)`` for the probability of the conjuction ``A \land B`` is also called the **joint probability** of events ``A`` and ``B``. Note that 

```math
p(A,B) = p(B,A)\,,
```

since ``A\land B = B \land A``. Therefore, the order of arguments in a joint probability distribution does not matter: ``p(A,B,C,D) = p(C,A,D,B)``, etc.

##### Independent events

Two events ``A`` and ``B`` are said to be **independent** if the probability of one event is not altered by information about the truth of the other event, i.e., 

```math
p(A|B) = p(A)\,.
```

It follows that, if ``A`` and ``B`` are independent, then the product rule simplifies to 

```math
p(A,B) = p(A) p(B)\,.
```

``A`` and ``B`` with given background ``I`` are said to be **conditionally independent** for given ``I``, if 

```math
p(A|B,I) = p(A|I)\,.
```

In that case, the product rule simplifies to ``p(A,B|I) = p(A|I) p(B|I)``.

##### Mutually exclusive events

Two events ``A_1`` and ``A_2`` are said to be **mutually exclusive** ('disjoint') if they cannot be true simultaneously, i.e., if

```math
p(A_1,A_2)=0 \,.
```

For mutually exclusive events, probabilities add (this follows from the sum rule), hence 

```math
p(A_1 + A_2) = p(A_1) + p(A_2)
```

##### Collectively exhaustive events

A set of events ``A_1, A_2, \ldots, A_N`` is said to be **collectively exhaustive** if one of the statements is necessarily true, i.e., ``A_1+A_2+\cdots +A_N=\mathrm{TRUE}``, or equivalently 

```math
p(A_1+A_2+\cdots +A_N) = 1 \,.
```

##### Partitioning the universe

If a set of events ``A_1, A_2, \ldots, A_n`` is both **mutually exclusive** and **collectively exhaustive**, then we say that they **partition the universe**. Technically, this means that 

```math
\sum_{n=1}^N p(A_n) = p(A_1 + \ldots + A_N) = 1
```



"""

# ╔═╡ 3e1b5c9c-d294-11ef-137f-d75b3731eae4
md"""
We mentioned before that every inference problem in PT can be evaluated through the sum and product rules. Next, we present two useful corollaries: (1) *Marginalization* and (2) *Bayes rule*. 

"""

# ╔═╡ 3e1b7d14-d294-11ef-0d10-1148a928dd57
md"""
## Marginalization

Let ``A`` and ``B_1,B_2,\ldots,B_n`` be events, where ``B_1,B_2,\ldots,B_n`` partitions the universe. Then

```math
\sum_{i=1}^n p(A,B_i) = p(A) \,.
```

This rule is called the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability). Proof:

```math
\begin{align*}
  \sum_i p(A,B_i) &= p(\sum_i AB_i)  &&\quad \text{(since all $AB_i$ are disjoint)}\\
  &= p(A,\sum_i B_i) \\
  &= p(A,\Omega) &&\quad \text{($\Omega$ is true event, since $B_i$ are exhaustive)} \\
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
p(D,\theta)=p(\theta,D)\,,
```

and hence that

```math
p(D|\theta)p(\theta)=p(\theta|D)p(D)\,,
```

or equivalently,

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

Bayes rule tells us how to update our knowledge about model parameters when facing new data. Hence, 
"""

# ╔═╡ 16c2eb59-16b8-4347-9aab-6e4b99016c79
keyconcept("", md"Bayes rule is the fundamental rule for learning from data!")

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

Hence, all that we can learn from the observed data is contained in the likelihood function ``p(D|\theta)``. This is called the **likelihood principle**.

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
L(\theta) \triangleq p(D=D_0|\theta)
```

"""

# ╔═╡ 3e1c9184-d294-11ef-3e35-5393d97fbc44
md"""
Note that ``L(\theta)`` is not a probability distribution for ``\theta`` since in general ``\sum_\theta L(\theta) \neq 1``.

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

This can be accomplished by repeatedly applying the sum and product rules.

In particular, consider a joint distribution ``p(X,Y,Z)``. Assume we are interested in ``p(X|Z)``:

```math
\begin{align*}
p(X|Z) \stackrel{p}{=} \frac{p(X,Z)}{p(Z)} \stackrel{s}{=} \frac{\sum_Y p(X,Y,Z)}{\sum_{X,Y} p(X,Y,Z)} \,,
\end{align*}
```

where the ``s`` and ``p`` above the equality sign indicate whether the sum or product rule was used. 

In the rest of this course, we'll encounter many lengthy probabilistic derivations. For each manipulation, you should be able to associate an 's' (for sum rule), a 'p' (for product or Bayes rule) or an 'm' (for a simplifying model assumption like conditional independency) above any equality sign.
"""

# ╔═╡ b176ceae-884e-4460-9f66-020c1ac447f1
md"""
# Examples
"""

# ╔═╡ fc733d61-fd0f-4a13-9afc-4505ac0253df
f(y,θ) = θ.^y .* (1 .- θ).^(1 .- y) # p(y|θ)

# ╔═╡ 8a7dd8b7-5faf-4091-8451-9769f842accb
let
	θ = 0.5
	p1 = plot(
			[0,1], f([0,1], θ);
			line=:stem, 
			marker=:circle, 
			xrange=(-0.5, 1.5), yrange=(0,1), 
			title="Sampling Distribution", 
			xlabel="y", ylabel=L"p(y|θ=%$θ)", label=""
		 )
	
	_θ = 0:0.01:1
	y=1
	p2 = plot(
			_θ, f(y, _θ);
			ylabel=L"p(y=%$y | θ)", xlabel=L"θ", 
			title="Likelihood Function", label=""
		 )
	
	plot(p1, p2)
end

# ╔═╡ 3e1d20e0-d294-11ef-2044-e1fe6590a600
md"""
The (discrete) sampling distribution is a valid probability distribution. 

However, the likelihood function ``L(\theta)`` clearly isn't, since ``\int_0^1 L(\theta) \mathrm{d}\theta \neq 1``. 

"""

# ╔═╡ 178721d2-624c-4ac4-8fa1-ded23da7feef
keyconcept("", md"Probabilities describe beliefs (a ''state of knowledge''), rather than actual properties of nature.")

# ╔═╡ 2156f96e-eebe-4190-8ce9-c76825c6da71
md"""

##### Solution 

- The given information is ``p(D=1)=0.01``, ``p(T=1|D=1)=0.95`` and ``p(T=0|D=0)=0.85``. We are asked to derive ``p( D=1 | T=1)``. We just follow the sum and product rules to derive the requested probability:

```math
\begin{align*}
p( D=1 &| T=1) \\
&\stackrel{p}{=} \frac{p(T=1,D=1)}{p(T=1)} \\
&\stackrel{p}{=} \frac{p(T=1|D=1)p(D=1)}{p(T=1)} \\
&\stackrel{s}{=} \frac{p(T=1|D=1)p(D=1)}{p(T=1|D=1)p(D=1)+p(T=1|D=0)p(D=0)} \\
&= \frac{0.95\times$(specificity)}{0.95\times0.01 + 0.15\times0.99} = 0.0601
\end{align*}
```


Note that ``p(\text{sick}|\text{positive test}) = 0.06`` while ``p(\text{positive test} | \text{sick}) = 0.95``. This is a huge difference that is sometimes called the "medical test paradox" or the [base rate fallacy](https://en.wikipedia.org/wiki/Base_rate_fallacy). 

Many people have trouble distinguishing ``p(A|B)`` from ``p(B|A)`` in their heads. This has led to major negative consequences. For instance, unfounded convictions in the legal arena and numerous unfounded conclusions in the pursuit of scientific results. See [Ioannidis (2005)](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124) and [Clayton (2021)](https://aubreyclayton.com/bernoulli).

"""

# ╔═╡ c593cb52-0538-4180-b9d5-479bdaeb3fc1
TODO("tell students that they can scorll. And make this interactive")

# ╔═╡ ef264651-854e-4374-8ea8-5476c85150c4
md"# Moments and Transformations"

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

Home exercise: Proof that ``\Sigma_{xy} = \Sigma_{yx}^{T}`` (making use of ``(AB)^T = B^TA^T``).

"""

# ╔═╡ 3e1e9224-d294-11ef-38b3-137c2be22400
md"""
## $(HTML("<span id='linear-transformation'>Linear Transformations</span>"))

If we have two variables ``X`` and ``Y``, what can we say about arithmetic transformations of these variables? Such as ``X + Y``, ``5X + 7`` or ``\sqrt{X}``?

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

Given eqs SRG-3a and SRG-3b (previous section), you should now be able to derive the following: for any distribution of variable ``X`` and ``Y`` and sum ``Z = X+Y`` (proof by [Exercise](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb))

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
[Wikipedia's List of convolutions of probability distributions](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions) shows how these convolutions work out for a few common probability distributions. 

"""

# ╔═╡ a83f164b-c66a-4e56-9151-5969302179d5
TODO("Use Code Example box for next cell")

# ╔═╡ 98fa17a6-7c8b-46e4-b32d-52db183d88f8
md"""
Set the parameters for the distributions of ``X`` and ``Y``:
"""

# ╔═╡ 27ec154a-a4c3-4d71-b2a0-45f2b456a8e4
μx = 2.0; σx = 1.0;

# ╔═╡ de4dbfc9-9340-4ae2-b323-49abfd77f198
μy = 2.0; σy = 0.5;

# ╔═╡ 1cb8b2c4-e1ae-4973-ba53-fc6c7fe1f37a
md"""
Compute the parameters for the distribution of ``Z = X + Y``:
"""

# ╔═╡ 91a91472-ee6d-416b-b18e-acbedc03a7fe
μz = μx + μy

# ╔═╡ 6485575d-c5a5-4891-8210-f50d6f75476f
σz = sqrt(σx^2 + σy^2)

# ╔═╡ 0abaed25-decc-4dcd-aa04-b68ec0d5c73e


# ╔═╡ 218d3b6e-50b6-4b98-a00c-a19dd33d2c03
md"""
Let's plot the distributions for ``X``, ``Y``, and ``Z``
"""

# ╔═╡ e836f877-5ed6-4865-ba3a-1ca5a86b2349
begin
	x = Normal(μx, σx)
	y = Normal(μy, σy)
	z = Normal(μz, σz)
end;

# ╔═╡ 842fd4e6-7873-45d4-aa29-e4aa9eb94fe4
begin
	# Calculate the x-range for plotting
	range_min = min(μx-2*σx, μy-2*σy, μz-2*σz)
	range_max = max(μx+2*σx, μy+2*σy, μz+2*σz)
	range_grid = range(range_min, stop=range_max, length=100)
end;

# ╔═╡ c0ea3253-a06b-426c-91a3-a6dd33e42779
let
	plot(range_grid, t -> pdf(x,t), label=L"p_x", fill=(0, 0.1))
	plot!(range_grid, t -> pdf(y,t), label=L"p_y", fill=(0, 0.1))
	plot!(range_grid, t -> pdf(z,t), label=L"p_z", fill=(0, 0.1))
end

# ╔═╡ 3e1f4f46-d294-11ef-29b8-69e546763781
md"""
## PDF for the Product of Two Variables

For two continuous **independent** variables ``X`` and ``Y``, with PDF's ``p_x(x)`` and ``p_y(y)``, the PDF of  ``Z = X Y`` is given by 

```math
p_z(z) = \int_{-\infty}^{\infty} p_x(x) \,p_y(z/x)\, \frac{1}{|x|}\,\mathrm{d}x\,.
```

For proof, see [https://en.wikipedia.org/wiki/Product_distribution](https://en.wikipedia.org/wiki/Product_distribution).

"""

# ╔═╡ 3e1f68fa-d294-11ef-31b2-e7670da8c08c
md"""
Generally, this integral does not lead to an analytical expression for ``p_z(z)``. 

As a crucial example, [the product of two independent variables that are both Gaussian-distributed does **not** lead to a Gaussian distribution](https://bmlip.github.io/colorized/lectures/The%20Gaussian%20Distribution.html#product-of-gaussians).

  * Exception: the distribution of the product of two variables that both have [log-normal distributions](https://en.wikipedia.org/wiki/Log-normal_distribution) is again a lognormal distribution. (If ``X`` has a normal distribution, then ``Y=\exp(X)`` has a log-normal distribution.)

"""

# ╔═╡ 3e1f7d5e-d294-11ef-2878-05744036f32c
md"""
## General Variable Transformations

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

Equating the two probability masses leads to the identification of the relation 

```math
p_y(y) = p_x(g(y)) g^\prime(y)\,,
```

which is also known as the [Change-of-Variable theorem](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function). 

If the transformation ``y = h(x)`` is not invertible, then ``x=g(y)`` does not exist. In that case, you can still work out the transformation by equating equivalent probability masses in the two domains. 

"""

# ╔═╡ 3e1fd38a-d294-11ef-05d3-ad467328be96
md"""
# Summary

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

# ╔═╡ dd31ec7c-708d-4fd7-958d-f9887798a5bc
md"""
# Appendix
"""

# ╔═╡ 9ba42512-0a9b-4a37-95ee-510aae5ec0be
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

# ╔═╡ 3e185ab0-d294-11ef-3f7d-9bd465518274
md"""
$(challenge_header("Disease Diagnosis"))

##### Problem
  - Given is a disease with a prevalence of 1%  and a test procedure with sensitivity ('true positive' rate) of 95%, and specificity ('true negative' rate) of 85%. What is the chance that somebody who tests positive actually has the disease?

##### Solution
  - Use probabilistic inference, to be discussed in this lecture. 
"""

# ╔═╡ 3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
md"""

$(challenge_header("Sampling Distribution and Likelihood Function for the Coin Toss", challenge_text="Code Example:"))


Consider the following simple model for the outcome ``y \in \{0,1\}`` (tail = ``0``, head = ``1``) of a biased coin toss with a real parameter ``\theta \in [0,1]``:

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
L(\theta) \triangleq p(y=1|\theta) = \theta \,.
```

"""

# ╔═╡ 3e1de32c-d294-11ef-1f63-f190c8361404
md"""
$(challenge_header("Bag Counter"; challenge_text="Inference Exercise:"))

##### Problem  

- A bag contains one ball, known to be either white or black. A white ball is put in, the bag is shaken. Next, a ball is drawn out, which proves to be white. What is now the  chance of drawing a white ball?

##### Solution 

- Again, use Bayes and marginalization to arrive at ``p(\text{white}|\text{data})=2/3``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.

``\Rightarrow`` Note that probabilities describe **a person's state of knowledge** rather than a 'property of nature'.

"""

# ╔═╡ 3e1e2b96-d294-11ef-3a68-fdc78232142e
md"""
$(challenge_header("Causality?"; challenge_text="Inference Exercise:"))

##### Problem 

- A dark bag contains five red balls and seven green ones. 
  - (a) What is the probability of drawing a red ball on the first draw? 
- Balls are not returned to the bag after each draw. 
  - (b) If you know that on the second draw the ball was a green one, what is now the probability of drawing a red ball on the first draw?

##### Solution 

- (a) ``5/12``. (b) ``5/11``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.


``\Rightarrow`` Again, we conclude that conditional probabilities reflect **implications for a state of knowledge** rather than temporal causality.


"""

# ╔═╡ 3e1d6d00-d294-11ef-1081-e11b8397eb91
## Revisiting the Challenge: Disease Diagnosis
md"""
$(challenge_header("Disease Diagnosis"; big=true, challenge_text="Revisiting the Challenge:", header_level=2))

##### Problem 

- Given a disease ``D \in \{0, 1\}`` with prevalence (overall occurence percentage) of $(@bind prevalence Scrubbable(0:0.01:0.1; format=".0%", default=0.01)) and a test procedure ``T  \in \{0, 1\}`` with sensitivity (true positive rate) of $(@bind sensitivity Scrubbable(0:0.01:1; format=".0%", default=0.95)) and specificity (true negative' rate) of $(@bind specificity Scrubbable(0:0.01:1; format=".0%", default=0.85)), what is the chance that somebody who tests positive actually has the disease?

"""

# ╔═╡ 444f1e2b-6a2a-47d3-b101-53994087b3a9
prevalence, sensitivity, specificity

# ╔═╡ 3e1f225a-d294-11ef-04c6-f3ca018ab286
md"""
$(challenge_header("Sum of Two Gaussian-distributed Variables"; big=true, challenge_text="Code Example:", header_level=2))  

Consider two independent Gaussian-distributed variables ``X`` and ``Y`` (see [wikipedia normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) for definition of a Gaussian (=Normal) distribution):

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

# ╔═╡ 3e1fb370-d294-11ef-1fb6-63a41a024691
md"""
$(challenge_header("Transformation of a Gaussian Variable"; big=true, challenge_text="Example:", header_level=2))  

Let ``p_x(x) = \mathcal{N}(x|\mu,\sigma^2)`` and ``y = \frac{x-\mu}{\sigma}``. 

##### Problem

- What is ``p_y(y)``? 

##### Solution

- Note that ``h(x)`` is invertible with ``x = g(y) = \sigma y + \mu``. The change-of-variable formula leads to

```math
\begin{align*}
p_y(y) &= p_x(g(y)) \cdot g^\prime(y) \\
  &= p_x(\sigma y + \mu) \cdot \sigma \\
  &= \frac{1}{\sigma\sqrt(2 \pi)} \exp\left( - \frac{(\sigma y + \mu - \mu)^2}{2\sigma^2}\right) \cdot \sigma \\
  &=  \frac{1}{\sqrt(2 \pi)} \exp\left( - \frac{y^2 }{2}\right)\\
  &= \mathcal{N}(y|0,1) 
\end{align*}
```

In the statistics literature, ``y = \frac{x-\mu}{\sigma}`` is called the **standardized** variable since it transforms a general normal variable into a standard normal one.

"""

# ╔═╡ 0fb6912c-d705-4a59-a922-ecd68bf650f1
@htl """
$(challenge_header("test test tesT"; big=false, ))
<p>
Consider two variables 
 and 
. It follows from symmetry arguments that


and hence that


or equivalently,



<p>
This last formula is called Bayes rule, named after its inventor Thomas Bayes (1701-1761). While Bayes rule is always true, a particularly useful application
"""

# ╔═╡ Cell order:
# ╟─3e17df5e-d294-11ef-38c7-f573724871d8
# ╟─bcb4be20-0439-4809-a166-8c50b6b9206b
# ╟─3e1803d0-d294-11ef-0304-df2b9b698cd1
# ╟─3e1823b0-d294-11ef-3dba-9997a7230cdf
# ╟─3e185ab0-d294-11ef-3f7d-9bd465518274
# ╟─840ab4dc-0d2e-4bf8-acc7-5f1ee2b0dcaf
# ╟─41bee964-a0a9-4a7f-8505-54a9ee12ef0d
# ╟─3e1889b8-d294-11ef-17bb-496655fbd618
# ╟─3e18b2fa-d294-11ef-1255-df048f0dcec2
# ╟─3e18c25c-d294-11ef-11bc-a93c2572b107
# ╟─3e18d2ea-d294-11ef-35e9-2332dd31dbf0
# ╟─dd11e93a-3dad-4e97-8642-fb70edfa6aae
# ╟─3e18e4bc-d294-11ef-38bc-cb97cb4e0963
# ╟─3e1b05ee-d294-11ef-33de-efed64d01c0d
# ╟─3e18f18c-d294-11ef-33e4-b7f9495e0508
# ╟─3e1906ea-d294-11ef-236e-c966a9474170
# ╟─3e191b6c-d294-11ef-3174-d1b4b36e252b
# ╟─3e192ef4-d294-11ef-1fc4-87175eeec5eb
# ╟─3e19436c-d294-11ef-11c5-f9914f7a3a57
# ╟─4edf38ab-a940-4ab0-be22-fa95cf571146
# ╟─3e194ef2-d294-11ef-3b38-1ddc3063ff35
# ╟─3e1964b4-d294-11ef-373d-712257fc130f
# ╟─3e196d6a-d294-11ef-0795-41c045079251
# ╟─3e198336-d294-11ef-26fd-03cd15876486
# ╟─3e198ba6-d294-11ef-3fe7-d70bf4833fa6
# ╟─3e19e95a-d294-11ef-3da4-6d23922a5150
# ╟─3e1a69f4-d294-11ef-103e-efc47025fb8f
# ╟─3e1a7c8e-d294-11ef-1f97-55e608d49141
# ╟─3e1a8eca-d294-11ef-1ef0-c15b24d05990
# ╟─3e1fc4da-d294-11ef-12f5-d51f9728fcc0
# ╟─3e1ab104-d294-11ef-1a98-412946949fba
# ╟─fea8ae4c-8ef9-4b74-ad13-1314afef97de
# ╟─3e1b4b1c-d294-11ef-0423-9152887cc403
# ╟─3e1b5c9c-d294-11ef-137f-d75b3731eae4
# ╟─3e1b7d14-d294-11ef-0d10-1148a928dd57
# ╟─3e1b8bf4-d294-11ef-04cc-6364e46fdd64
# ╟─3e1b9ba8-d294-11ef-18f2-db8eed3d87d0
# ╟─3e1babca-d294-11ef-37c1-cd821a6488b2
# ╟─3e1bba8e-d294-11ef-1f61-295af16078ce
# ╟─3e1bcb00-d294-11ef-2795-bd225bd00496
# ╟─3e1bdd02-d294-11ef-19e8-2f44eccf58af
# ╟─3e1bf116-d294-11ef-148b-f7a1ca3f3bad
# ╟─16c2eb59-16b8-4347-9aab-6e4b99016c79
# ╟─3e1bffec-d294-11ef-2a49-9ff0f6331add
# ╟─3e1c0e80-d294-11ef-0d19-375e01988f16
# ╟─3e1c1e3e-d294-11ef-0955-bdf9d0ba3c53
# ╟─3e1c4224-d294-11ef-2707-49470aaae6eb
# ╟─3e1c51e2-d294-11ef-2c6d-d32a98308c6f
# ╟─3e1c60ba-d294-11ef-3a01-cf9e97512857
# ╟─3e1c70be-d294-11ef-14ed-0d46515541c5
# ╟─3e1c806a-d294-11ef-1fad-17e5625279f7
# ╟─3e1c9184-d294-11ef-3e35-5393d97fbc44
# ╟─3e1d33c8-d294-11ef-0a08-bdc419949925
# ╟─b176ceae-884e-4460-9f66-020c1ac447f1
# ╟─3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
# ╠═eeb9a1f5-b857-4843-920b-2e4a9656f66b
# ╠═fc733d61-fd0f-4a13-9afc-4505ac0253df
# ╟─8a7dd8b7-5faf-4091-8451-9769f842accb
# ╟─3e1d20e0-d294-11ef-2044-e1fe6590a600
# ╟─3e1de32c-d294-11ef-1f63-f190c8361404
# ╟─3e1e2b96-d294-11ef-3a68-fdc78232142e
# ╟─178721d2-624c-4ac4-8fa1-ded23da7feef
# ╠═3e1d6d00-d294-11ef-1081-e11b8397eb91
# ╟─2156f96e-eebe-4190-8ce9-c76825c6da71
# ╠═444f1e2b-6a2a-47d3-b101-53994087b3a9
# ╠═c593cb52-0538-4180-b9d5-479bdaeb3fc1
# ╟─ef264651-854e-4374-8ea8-5476c85150c4
# ╟─3e1e4dda-d294-11ef-33b7-4bbe3300ca22
# ╟─3e1e5a5a-d294-11ef-2fdf-efee4eb1a0f2
# ╟─3e1e7742-d294-11ef-1204-f9be24da07ab
# ╟─3e1e9224-d294-11ef-38b3-137c2be22400
# ╟─3e1ea442-d294-11ef-1364-8dd9986325f7
# ╟─3e1eba72-d294-11ef-2f53-b56f1862fcbb
# ╟─3e1ed1a4-d294-11ef-2de4-d7cc540e06a1
# ╟─3e1eeb14-d294-11ef-1702-f5d2cf6fe60a
# ╠═a83f164b-c66a-4e56-9151-5969302179d5
# ╟─3e1f225a-d294-11ef-04c6-f3ca018ab286
# ╟─98fa17a6-7c8b-46e4-b32d-52db183d88f8
# ╠═27ec154a-a4c3-4d71-b2a0-45f2b456a8e4
# ╠═de4dbfc9-9340-4ae2-b323-49abfd77f198
# ╟─1cb8b2c4-e1ae-4973-ba53-fc6c7fe1f37a
# ╠═91a91472-ee6d-416b-b18e-acbedc03a7fe
# ╠═6485575d-c5a5-4891-8210-f50d6f75476f
# ╟─0abaed25-decc-4dcd-aa04-b68ec0d5c73e
# ╟─218d3b6e-50b6-4b98-a00c-a19dd33d2c03
# ╠═5394e37c-ae00-4042-8ada-3bbf32fbca9e
# ╠═e836f877-5ed6-4865-ba3a-1ca5a86b2349
# ╠═c0ea3253-a06b-426c-91a3-a6dd33e42779
# ╠═842fd4e6-7873-45d4-aa29-e4aa9eb94fe4
# ╟─3e1f4f46-d294-11ef-29b8-69e546763781
# ╟─3e1f68fa-d294-11ef-31b2-e7670da8c08c
# ╟─3e1f7d5e-d294-11ef-2878-05744036f32c
# ╟─3e1f8e48-d294-11ef-0f8a-b58294a8543d
# ╟─3e1fa04a-d294-11ef-00c3-a51d1aaa5553
# ╟─3e1fb370-d294-11ef-1fb6-63a41a024691
# ╟─3e1fd38a-d294-11ef-05d3-ad467328be96
# ╟─3e1fe0de-d294-11ef-0d8c-35187e394292
# ╟─3e1fedfc-d294-11ef-30ee-a396bb877037
# ╟─3e1ffc5c-d294-11ef-27b1-4f6ccb64c5d6
# ╟─3e2009e2-d294-11ef-255d-8d4a44865663
# ╟─dd31ec7c-708d-4fd7-958d-f9887798a5bc
# ╠═b305a905-06c2-4a15-8042-72ef6375720f
# ╟─0fb6912c-d705-4a59-a922-ecd68bf650f1
# ╟─9ba42512-0a9b-4a37-95ee-510aae5ec0be
# ╠═7910a84c-18b3-4081-9f01-e59258a01adb
