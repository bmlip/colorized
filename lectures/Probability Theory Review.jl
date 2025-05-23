### A Pluto.jl notebook ###
# v0.20.8

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
## 📕 Data Analysis: A Bayesian Tutorial

The following is an excerpt from the book [Data Analysis: A Bayesian Tutorial](https://global.oup.com/academic/product/data-analysis-9780198568322) (2006), by D.S. Sivia with J.S. Skilling:

> #### Preface
> "As an undergraduate, I always found the subject of statistics to be rather mysterious. This topic wasn’t entirely new to me, as we had been taught a little bit about probability earlier at high school; for example, I was already familiar with the binomial, Poisson and normal distributions. Most of this made sense, but only seemed to relate to things like rolling dice, flipping coins, shuffling cards and so on. However, having aspirations of becoming a scientist, what I really wanted to know was how to analyse experimental data. Thus, I eagerly looked forward to the lectures on statistics. Sadly, they were a great disappointment. Although many of the tests and procedures expounded were intuitively reasonable, there was something deeply unsatisfactory about the whole affair: there didn’t seem to be any underlying basic principles! Hence, the course on ‘probability and statistics’ had led to an unfortunate dichotomy: probability made sense, but was just a game; statistics was important, but it was a bewildering collection of tests with little obvious rhyme or reason. While not happy with this situation, I decided to put aside the subject and concentrate on real science. After all, the predicament was just a reflection of my own inadequacies and I’d just have to work at it when the time came to really analyse my data.
> 
> The story above is not just my own, but is the all too common experience of many scientists. Fortunately, it doesn’t have to be like this. What we were not told in our undergraduate lectures is that there is an alternative approach to the whole subject of data analysis which uses only probability theory. In one sense, it makes the topic of statistics entirely superfluous. In another, it provides the logical justification for many of the prevalent statistical tests and procedures, making explicit the conditions and approximations implicitly assumed in their use."


Does this fragment resonate with your own experience? 

In this lesson we introduce *Probability Theory* (PT) again. As we will see in the next lessons, PT is all you need to make sense of machine learning, artificial intelligence, statistics, etc. 

"""

# ╔═╡ 3e1876f8-d294-11ef-22bf-7904df3c1182
md"""
**Solution**: Use probabilistic inference, to be discussed in this lecture. 

"""

# ╔═╡ 3e1889b8-d294-11ef-17bb-496655fbd618
md"""
# The Design of Probability Theory

Define an **event** (or "proposition") ``A`` as a statement that can be considered for its truth by a person. For instance, 

```math
𝐴= \texttt{``there is life on Mars''}
```

If we assume the fact 

```math
I = \texttt{``All known life forms require water''}
```

as background information, and a new piece of information 

```math
x = \texttt{``There is water on Mars''}
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
keyconcept(" ", 
	md"""
	
	Probability theory (PT) provides the **theory of optimal processing of incomplete information** (see [Cox theorem](https://en.wikipedia.org/wiki/Cox%27s_theorem), and [Caticha](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-24).
	
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
#### Examples

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
We write the denial of ``A``, i.e., the event **not**-A, as ``\bar{A}``. 

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
p(A|I)\,,
```

which indicates the degree-of-belief in event ``A``, given that ``I`` is true. 
"""

# ╔═╡ 3e1a36b4-d294-11ef-2242-f36061b0b754
md"""
In principle, all probabilities are conditional probabilities of the type ``p(A|I)``, since there is always some background knowledge. However, we often write ``p(A)`` rather than ``p(A|I)`` if the background knowledge ``I`` is assumed to be obviously present. E.g., we usually write ``p(A)`` rather than ``p(\,A\,|\,\text{the-sun-comes-up-tomorrow}\,)``.

"""

# ╔═╡ 3e1a522a-d294-11ef-1a7a-bdcfbd5dae09
md"""
The expression ``p(A,B)`` is called the **joint probability** of events ``A`` and ``B``. Note that 

```math
p(A,B) = p(B,A)\,,
```

since ``AB = BA``. Therefore, the order of arguments in a joint probability distribution does not matter: ``p(A,B,C,D) = p(C,A,D,B)``, etc.

"""

# ╔═╡ 3e1a69f4-d294-11ef-103e-efc47025fb8f
md"""
Note that, if ``X`` is a variable, then an *assignment* ``X=x`` (where ``x`` is a value, e.g., ``X=5``) can be interpreted as an event. Hence, the expression ``p(X=5)`` should be interpreted as the *degree-of-belief of the event* that variable ``X`` takes on the value ``5``. 

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

# ╔═╡ 3e1a9fdc-d294-11ef-288f-37fd1b7ee281
md"""
The notational conventions in PT are unfortunately a bit sloppy:( For instance, in the context of a variable ``X``, we often write ``p(x)`` rather than ``p(X=x)``, assuming that the reader understands the context.  

"""

# ╔═╡ 3e1ab104-d294-11ef-1a98-412946949fba
md"""
# $(HTML("<span id='PT-calculus'>Probability Theory Calculus</span>"))

"""

# ╔═╡ 3e1ac32c-d294-11ef-13be-558397a6cc2a
md"""

## The Sum and Product Rules

The following product and sum rules are also known as the **axioms of probability theory**, but as discussed above, under some mild assumptions, they can be derived as the unique rules for *rational reasoning under uncertainty* ([Cox theorem, 1946](https://en.wikipedia.org/wiki/Cox%27s_theorem), and [Caticha, 2012](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.7-26).

"""

# ╔═╡ 3e1ad2ea-d294-11ef-02c4-3f06e14ea4d8
md"""
**Sum rule**. The disjunction of two events ``A`` and ``B`` with given background ``I`` is 

```math
 p(A+B|I) = p(A|I) + p(B|I) - p(A,B|I)
```

"""

# ╔═╡ 3e1ae30a-d294-11ef-3a5d-d91b7d7723d3
md"""
**Product rule**. The conjuction of two events ``A`` and ``B`` with given background ``I`` is

```math
 p(A,B|I) = p(A|B,I)\,p(B|I)
```

"""

# ╔═╡ 3e1af354-d294-11ef-1971-dfb39016cfcd
md"""
PT extends (propositional) logic in reasoning about the truth value of events. Logic reasons about the truth value of events on the binary domain {0,1} (FALSE and TRUE), whereas PT extends the range to degrees-of-belief, represented by real numbers in [0,1].

"""

# ╔═╡ 3e1b05ee-d294-11ef-33de-efed64d01c0d
keyconcept(
	"",
	md"""
	All legitimate probabilistic relations can be **derived from the sum and product rules**!
	
	"""
)

# ╔═╡ 3e1b4b1c-d294-11ef-0423-9152887cc403
md"""
## Independent, Exclusive, and Exhaustive Events

It will be helpful to introduce some terms concerning special relationships between events.  

#### Independent events

Two events ``A`` and ``B`` are said to be **independent** if the probability of one event is not altered by information about the truth of the other event, i.e., 

```math
p(A|B) = p(A)\,.
```

It follows that, if ``A`` and ``B`` are independent, then the product rule simplifies to 

```math
p(A,B) = p(A) p(B)\,.
```

``A`` and ``B`` with given background ``I`` are said to be **conditionally independent** if 

```math
p(A|B,I) = p(A|I)\,.
```

In that case, the product rule simplifies to ``p(A,B|I) = p(A|I) p(B|I)``.

#### Mutually exclusive events

Two events ``A_1`` and ``A_2`` are said to be **mutually exclusive** ('disjoint') if they cannot be true simultaneously, i.e., if

```math
p(A_1,A_2)=0 \,.
```

For mutually exclusive events, probabilities add (this follows from the sum rule), hence 

```math
p(A_1 + A_2) = p(A_1) + p(A_2)
```

#### Collectively exhaustive events

A set of events ``A_1, A_2, \ldots, A_N`` is said to be **collectively exhaustive** if one of the statements is necessarily true, i.e., ``A_1+A_2+\cdots +A_N=\mathrm{TRUE}``, or equivalently 

```math
p(A_1+A_2+\cdots +A_N) = 1 \,.
```

#### Partitioning the universe

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

# ╔═╡ dc4665b8-f803-4c18-845c-4901553eaa99
TODO("Can you make a box for ''Code Example''?")

# ╔═╡ 3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
md"""
### Code Example: Sampling Distribution and Likelihood Function for the Coin Toss

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

where the ``s`` and ``p`` above the equality sign indicate whether the sum or product rule was used. 

"""

# ╔═╡ 3e1d5efc-d294-11ef-0627-cfa86f1447ca
md"""
In the rest of this course, we'll encounter many long probabilistic derivations. For each manipulation, you should be able to associate an 's' (for sum rule), a 'p' (for product or Bayes rule) or an 'm' (for a simplifying model assumption) above any equality sign.

"""

# ╔═╡ 3e1e134c-d294-11ef-18c0-21742fe74fa6
md"""
**Solution**: Again, use Bayes and marginalization to arrive at ``p(\text{white}|\text{data})=2/3``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.

``\Rightarrow`` Note that probabilities describe **a person's state of knowledge** rather than a 'property of nature'.

"""

# ╔═╡ 3e1e3a34-d294-11ef-0053-393751b94f2c
md"""
**Solution**: (a) ``5/12``. (b) ``5/11``, see the [Exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-Probability-Theory-Review.ipynb) notebook.


``\Rightarrow`` Again, we conclude that conditional probabilities reflect **implications for a state of knowledge** rather than temporal causality.

"""

# ╔═╡ 178721d2-624c-4ac4-8fa1-ded23da7feef
keyconcept("", md"Probabilities describe beliefs (a ''state of knowledge''), rather than actual properties of nature.")

# ╔═╡ 2156f96e-eebe-4190-8ce9-c76825c6da71
md"""


**Solution**: The given information is ``p(D=1)=0.01``, ``p(T=1|D=1)=0.95`` and ``p(T=0|D=0)=0.85``. We are asked to derive ``p( D=1 | T=1)``. We just follow the sum and product rules to derive the requested probability:

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

Many people have trouble distinguishing ``p(A|B)`` from ``p(B|A)`` in their heads. This has led to major negative consequences. For instance, unfounded convictions in the legal arena and even lots of unfounded conclusions in the pursuit of scientific results. See [Ioannidis (2005)](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124) and [Clayton (2021)](https://aubreyclayton.com/bernoulli).

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

Exercise: Proof that ``\Sigma_{xy} = \Sigma_{yx}^{T}`` (making use of ``(AB)^T = B^TA^T``).

"""

# ╔═╡ 3e1e9224-d294-11ef-38b3-137c2be22400
md"""
## $(HTML("<span id='linear-transformation'>Linear Transformations</span>"))

If we have two variables ``X`` and ``Y``, what can we say about transformations of these variables? Such as ``X + Y``, ``5X + 7`` or ``\sqrt{X}``?

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

# ╔═╡ 3e1f225a-d294-11ef-04c6-f3ca018ab286
md"""
### Code Example: Sum of Two Gaussian-distributed Variables

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

# ╔═╡ 3e1fb370-d294-11ef-1fb6-63a41a024691
md"""
### Example: Transformation of a Gaussian Variable

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

In the statistics literature, ``y = \frac{x-\mu}{\sigma}`` is called the **standardized** variable since it transforms a general normal variable into a standard normal one.

"""

# ╔═╡ 3e1fc4da-d294-11ef-12f5-d51f9728fcc0
md"""
# Notational Conventions

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

**Problem**: Given a disease with prevalence of 1%  and a test procedure  with sensitivity ('true positive' rate) of  95%  and specificity ('true negative' rate) of  85%, what is the chance that somebody who tests positive actually has the disease?

"""

# ╔═╡ 3e1de32c-d294-11ef-1f63-f190c8361404
md"""
$(challenge_header("Bag Counter"; challenge_text="Inference Exercise:"))

**Problem**:  A bag contains one ball, known to be either white or black. A white ball is put in, the bag is shaken,

and a ball is drawn out, which proves to be white. What is now the  chance of drawing a white ball?

"""

# ╔═╡ 3e1e2b96-d294-11ef-3a68-fdc78232142e
md"""
$(challenge_header("Causality?"; challenge_text="Inference Exercise:"))

**Problem**: A dark bag contains five red balls and seven green ones. (a) What is the probability of drawing a red ball on the first draw? Balls are not returned to the bag after each draw. (b) If you know that on the second draw the ball was a green one, what is now the probability of drawing a red ball on the first draw?

"""

# ╔═╡ 3e1d6d00-d294-11ef-1081-e11b8397eb91
## Revisiting the Challenge: Disease Diagnosis
md"""
$(challenge_header("Disease Diagnosis"; big=true, challenge_text="Revisiting the Challenge:", header_level=2))

**Problem**: Given a disease ``D \in \{0, 1\}`` with prevalence (overall occurence percentage) of $(@bind prevalence Scrubbable(0:0.01:0.1; format=".0%", default=0.01)) and a test procedure ``T  \in \{0, 1\}`` with sensitivity (true positive rate) of $(@bind sensitivity Scrubbable(0:0.01:1; format=".0%", default=0.95)) and specificity (true negative' rate) of $(@bind specificity Scrubbable(0:0.01:1; format=".0%", default=0.85)), what is the chance that somebody who tests positive actually has the disease?


"""

# ╔═╡ 444f1e2b-6a2a-47d3-b101-53994087b3a9
prevalence, sensitivity, specificity

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.117"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "3afc239179c519e4ba30516488624731ba1e1c5f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "7ffa4049937aeba2e5e1242274dc052b0362157a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.14"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "98fc192b4e4b938775ecd276ce88f539bcec358e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.14+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6ac9e4acc417a5b534ace12690bc6973c25b862f"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "4ef1c538614e3ec30cb6383b9eb0326a5c3a9763"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+4"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "809ba625a00c605f8d00cd2a9ae19ce34fc24d68"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.13"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "cedc9f9013f7beabd8a9c6d2e22c0ca7c5c2a8ed"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.6"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d62610ec45e4efeabf7032d67de2ffdea8344bed"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cad2cf2c8d80f1d17320652b3ea7778b30f473f"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.3.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "002748401f7b520273e2b506f61cab95d4701ccf"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.48+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "c950ae0a3577aec97bfccf3381f66666bc416729"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.8.1+0"
"""

# ╔═╡ Cell order:
# ╟─3e17df5e-d294-11ef-38c7-f573724871d8
# ╟─bcb4be20-0439-4809-a166-8c50b6b9206b
# ╟─3e1803d0-d294-11ef-0304-df2b9b698cd1
# ╟─3e1823b0-d294-11ef-3dba-9997a7230cdf
# ╟─3e185ab0-d294-11ef-3f7d-9bd465518274
# ╟─3e1876f8-d294-11ef-22bf-7904df3c1182
# ╟─3e1889b8-d294-11ef-17bb-496655fbd618
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
# ╠═dc4665b8-f803-4c18-845c-4901553eaa99
# ╠═3e1ca4a8-d294-11ef-1a4f-a3443b74fe63
# ╠═eeb9a1f5-b857-4843-920b-2e4a9656f66b
# ╠═fc733d61-fd0f-4a13-9afc-4505ac0253df
# ╠═8a7dd8b7-5faf-4091-8451-9769f842accb
# ╟─3e1d20e0-d294-11ef-2044-e1fe6590a600
# ╠═3e1d33c8-d294-11ef-0a08-bdc419949925
# ╟─3e1d4160-d294-11ef-0dc6-d7aa9bce56a1
# ╟─3e1d51a0-d294-11ef-228e-294b503d2e3d
# ╟─3e1d5efc-d294-11ef-0627-cfa86f1447ca
# ╟─3e1de32c-d294-11ef-1f63-f190c8361404
# ╟─3e1e134c-d294-11ef-18c0-21742fe74fa6
# ╟─3e1e2b96-d294-11ef-3a68-fdc78232142e
# ╟─3e1e3a34-d294-11ef-0053-393751b94f2c
# ╟─178721d2-624c-4ac4-8fa1-ded23da7feef
# ╟─3e1d6d00-d294-11ef-1081-e11b8397eb91
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
# ╠═3e1fb370-d294-11ef-1fb6-63a41a024691
# ╟─3e1fc4da-d294-11ef-12f5-d51f9728fcc0
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
