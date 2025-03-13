### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 96547560-d294-11ef-0fa7-6b6489f7baba
md"""
# Factor Graphs

"""

# ╔═╡ 9654ea3e-d294-11ef-335c-657af1ceaf19
md"""
## Preliminaries

Goal 

  * Introduction to Forney-style factor graphs and message passing-based inference

Materials        

  * Mandatory

      * These lecture notes
      * Loeliger (2007), [The factor graph approach to model based signal processing](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf), pp. 1295-1302 (until section V)
  * Optional

      * Frederico Wadehn (2015), [Probabilistic graphical models: Factor graphs and more](https://www.youtube.com/watch?v=Fv2YbVg9Frc&t=31) video lecture (**recommended**)
  * References

      * Forney (2001), [Codes on graphs: normal realizations](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Forney-2001-Codes-on-graphs-normal-realizations.pdf)

"""

# ╔═╡ 96552348-d294-11ef-16d8-b53563054687
md"""
## Why Factor Graphs?

A probabilistic inference task gets its computational load mainly through the need for marginalization (i.e., computing integrals). E.g., for a model ``p(x_1,x_2,x_3,x_4,x_5)``, the inference task ``p(x_2|x_3)`` is given by 

```math
p(x_2|x_3) = \frac{p(x_2,x_3)}{p(x_3)} = \frac{\int \cdots \int p(x_1,x_2,x_3,x_4,x_5) \, \mathrm{d}x_1  \mathrm{d}x_4 \mathrm{d}x_5}{\int \cdots \int p(x_1,x_2,x_3,x_4,x_5) \, \mathrm{d}x_1  \mathrm{d}x_2 \mathrm{d}x_4 \mathrm{d}x_5}
```

"""

# ╔═╡ 965531da-d294-11ef-1639-db0dd32c16d1
md"""
Since these computations (integrals or sums) suffer from the "curse of dimensionality", we often need to solve a simpler problem in order to get an answer. 

"""

# ╔═╡ 96555a72-d294-11ef-1270-f14e47749893
md"""
Factor graphs provide a computationally efficient approach to solving inference problems **if the probabilistic model can be factorized**. 

"""

# ╔═╡ 9655959e-d294-11ef-0ca6-5f20aa579e91
md"""
$(HTML("<span id='factorization-helps'>Factorization helps</span>")). For instance, if ``p(x_1,x_2,x_3,x_4,x_5) = p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4)``, then

```math
p(x_2|x_3) = \frac{\int \cdots \int p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4) \, \mathrm{d}x_1  \mathrm{d}x_4 \mathrm{d}x_5}{\int \cdots \int p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4) \, \mathrm{d}x_1  \mathrm{d}x_2 \mathrm{d}x_4 \mathrm{d}x_5} 
  = \frac{p(x_2,x_3)}{\int p(x_2,x_3) \mathrm{d}x_2}
```

which is computationally much cheaper than the general case above.

"""

# ╔═╡ 9655a94e-d294-11ef-00af-8f49c8821a19
md"""
In this lesson, we discuss how computationally efficient inference in *factorized* probability distributions can be automated by message passing-based inference in factor graphs.

"""

# ╔═╡ 9655b2c2-d294-11ef-057f-9b3984064411
md"""
## Factor Graph Construction Rules

Consider a function 

```math
f(x_1,x_2,x_3,x_4,x_5) = f_a(x_1,x_2,x_3) \cdot f_b(x_3,x_4,x_5) \cdot f_c(x_4)
```

"""

# ╔═╡ 9655c1ae-d294-11ef-061a-991947cee620
md"""
The factorization of this function can be graphically represented by a **Forney-style Factor Graph** (FFG):

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-example-1.png?raw=true)

"""

# ╔═╡ 9655d360-d294-11ef-0f06-ab58e2ad0e5f
md"""
An FFG is an **undirected** graph subject to the following construction rules ([Forney, 2001](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Forney-2001-Codes-on-graphs-normal-realizations.pdf))

1. A **node** for every factor;
2. An **edge** (or **half-edge**) for every variable;
3. Node ``f_\bullet`` is connected to edge ``x`` **iff** variable ``x`` appears in factor ``f_\bullet``.

"""

# ╔═╡ 9655e06c-d294-11ef-0393-9355d6e20afb
md"""
A **configuration** is an assigment of values to all variables.

A configuration ``\omega=(x_1,x_2,x_3,x_4,x_5)`` is said to be **valid** iff ``f(\omega) \neq 0``

"""

# ╔═╡ 9655ed6e-d294-11ef-370f-937b590036f3
md"""
## Equality Nodes for Branching Points

Note that a variable can appear in maximally two factors in an FFG (since an edge has only two end points).

"""

# ╔═╡ 9655fb88-d294-11ef-1ceb-91585012d142
md"""
Consider the factorization (where ``x_2`` appears in three factors) 

```math
 f(x_1,x_2,x_3,x_4) = f_a(x_1,x_2)\cdot f_b(x_2,x_3) \cdot f_c(x_2,x_4)
```

"""

# ╔═╡ 965606f2-d294-11ef-305b-870427879e50
md"""
For the factor graph representation, we will instead consider the function ``g``, defined as

```math
\begin{align*}
 g(x_1,x_2&,x_2^\prime,x_2^{\prime\prime},x_3,x_4) 
  = f_a(x_1,x_2)\cdot f_b(x_2^\prime,x_3) \cdot f_c(x_2^{\prime\prime},x_4) \cdot f_=(x_2,x_2^\prime,x_2^{\prime\prime})
\end{align*}
```

where 

```math
f_=(x_2,x_2^\prime,x_2^{\prime\prime}) \triangleq \delta(x_2-x_2^\prime)\, \delta(x_2-x_2^{\prime\prime})
```

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-wEquality-node.png?raw=true)

"""

# ╔═╡ 96561594-d294-11ef-1590-198382927808
md"""
Note that through introduction of auxiliary variables ``X_2^{\prime}`` and ``X_2^{\prime\prime}`` and a factor ``f_=(x_2,x_2^\prime,x_2^{\prime\prime})``, each variable in ``g`` appears in maximally two factors.

"""

# ╔═╡ 96563a44-d294-11ef-3ba4-23fd588b99a5
md"""
The constraint ``f_=(x,x^\prime,x^{\prime\prime})`` enforces that ``X=X^\prime=X^{\prime\prime}`` **for every valid configuration**.

"""

# ╔═╡ 9656566e-d294-11ef-37c9-a536fc435e36
md"""
Since ``f`` is a marginal of ``g``, i.e., 

```math
f(x_1,x_2,x_3,x_4) = \iint g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4)\, \mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime}
```

it follows that any inference problem on ``f`` can be executed by a corresponding inference problem on ``g``, e.g.,

```math
\begin{align*}
f(x_1 \mid x_2) &\triangleq \frac{\iint f(x_1,x_2,x_3,x_4) \,\mathrm{d}x_3 \mathrm{d}x_4 }{ \int\cdots\int f(x_1,x_2,x_3,x_4) \,\mathrm{d}x_1 \mathrm{d}x_3 \mathrm{d}x_4} \\
  &= \frac{\int\cdots\int g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4) \,\mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime} \mathrm{d}x_3 \mathrm{d}x_4 }{ \int\cdots\int g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4) \,\mathrm{d}x_1 \mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime} \mathrm{d}x_3 \mathrm{d}x_4} \\
  &= g(x_1 \mid x_2)
\end{align*}
```

"""

# ╔═╡ 965679f0-d294-11ef-13e0-bf28c9a9a505
md"""
```math
\Rightarrow
```

**Any factorization of a global function ``f`` can be represented by a Forney-style Factor Graph**.

"""

# ╔═╡ 9656b67c-d294-11ef-1541-3d3607375fd2
md"""
## Probabilistic Models as Factor Graphs

FFGs can be used to express conditional independence (factorization) in probabilistic models. 

"""

# ╔═╡ 9656cf72-d294-11ef-03aa-b715dd686c09
md"""
For example, the (previously shown) graph for 

```math
f_a(x_1,x_2,x_3) \cdot f_b(x_3,x_4,x_5) \cdot f_c(x_4)
```

could represent the probabilistic model

```math
p(x_1,x_2,x_3,x_4,x_5) = p(x_1,x_2|x_3) \cdot p(x_3,x_5|x_4) \cdot p(x_4)
```

where we identify 

```math
\begin{align*}
f_a(x_1,x_2,x_3) &= p(x_1,x_2|x_3) \\
f_b(x_3,x_4,x_5) &= p(x_3,x_5|x_4) \\
f_c(x_4) &= p(x_4)
\end{align*}
```

"""

# ╔═╡ 9656d850-d294-11ef-21a1-474b07ea7729
md"""
This is the graph

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-example-prob-model.png?raw=true)

"""

# ╔═╡ 9656e606-d294-11ef-1daa-312623552a5b
md"""
## Inference by Closing Boxes

Factorizations provide opportunities to cut on the amount of needed computations when doing inference. In what follows, we will use FFGs to process these opportunities in an automatic way by message passing between the nodes of the graph. 

"""

# ╔═╡ 9656ee62-d294-11ef-38f4-7bc8031df7ee
md"""
Assume we wish to compute the marginal

```math
\bar{f}(x_3) \triangleq \sum\limits_{x_1,x_2,x_4,x_5,x_6,x_7}f(x_1,x_2,\ldots,x_7) 
```

for a model ``f`` with given factorization 

```math
f(x_1,x_2,\ldots,x_7) = f_a(x_1) f_b(x_2) f_c(x_1,x_2,x_3) f_d(x_4) f_e(x_3,x_4,x_5) f_f(x_5,x_6,x_7) f_g(x_7)
```

"""

# ╔═╡ 9656fae2-d294-11ef-10d8-ff921d5956bd
md"""
Note that, if each variable ``x_i`` can take on ``10`` values, then the computing the marginal ``\bar{f}(x_3)`` takes about ``10^6`` (1 million) additions. 

"""

# ╔═╡ 96570d3e-d294-11ef-0178-c34dda717495
md"""
Due to the factorization and the [Generalized Distributive Law](https://en.wikipedia.org/wiki/Generalized_distributive_law), we can decompose this sum-of-products to the following product-of-sums:

```math
\begin{align*}\bar{f}&(x_3) = \\
  &\underbrace{ \Bigg( \sum_{x_1,x_2} \underbrace{f_a(x_1)}_{\overrightarrow{\mu}_{X_1}(x_1)}\, \underbrace{f_b(x_2)}_{\overrightarrow{\mu}_{X_2}(x_2)}\,f_c(x_1,x_2,x_3)\Bigg) }_{\overrightarrow{\mu}_{X_3}(x_3)} 
  \underbrace{ \cdot\Bigg( \sum_{x_4,x_5} \underbrace{f_d(x_4)}_{\overrightarrow{\mu}_{X_4}(x_4)}\,f_e(x_3,x_4,x_5) \cdot \underbrace{ \big( \sum_{x_6,x_7} f_f(x_5,x_6,x_7)\,\underbrace{f_g(x_7)}_{\overleftarrow{\mu}_{X_7}(x_7)}\big) }_{\overleftarrow{\mu}_{X_5}(x_5)} \Bigg) }_{\overleftarrow{\mu}_{X_3}(x_3)}
\end{align*}
```

which, in case ``x_i`` has ``10`` values, requires a few hundred additions and is therefore computationally (much!) lighter than executing the full sum ``\sum_{x_1,\ldots,x_7}f(x_1,x_2,\ldots,x_7)``

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-message-passing.png?raw=true)

"""

# ╔═╡ 96571c34-d294-11ef-11ef-29beeb1f96c2
md"""
Note that the auxiliary factor ``\overrightarrow{\mu}_{X_3}(x_3)`` is obtained by multiplying all enclosed factors (``f_a``, ``f_b, f_c``) by the red dashed box, followed by marginalization (summing) over all enclosed variables (``x_1``, ``x_2``).

"""

# ╔═╡ 96572be8-d294-11ef-2cd1-256972de7b23
md"""
This is the **Closing the Box**-rule, which is a general recipe for marginalization of latent variables (inside the box) and leads to a new factor that has the variables (edges) that cross the box as arguments. For instance, the argument of the remaining factor ``\overrightarrow{\mu}_{X_3}(x_3)`` is the variable on the edge that crosses the red box (``x_3``).

"""

# ╔═╡ 96573a0c-d294-11ef-2e99-67fdf2ee2eab
md"""
Hence, ``\overrightarrow{\mu}_{X_3}(x_3)`` can be interpreted as a **message from the red box toward variable** ``x_3``.

"""

# ╔═╡ 96574a88-d294-11ef-31a1-e949e6875a3d
md"""
We drew *directed edges* in the FFG in order to distinguish forward messages ``\overrightarrow{\mu}_\bullet(\cdot)`` (in the same direction as the arrow of the edge) from backward messages ``\overleftarrow{\mu}_\bullet(\cdot)`` (in opposite direction). This is just a notational convenience since an FFG is computationally an undirected graph. 

"""

# ╔═╡ 96575dd4-d294-11ef-31d6-b39b4c4bdea1
md"""
## Sum-Product Algorithm

Closing-the-box can also be interpreted as a **message update rule** for an outgoing message from a node. For a node ``f(y,x_1,\ldots,x_n)`` with incoming messages ``\overrightarrow{\mu}_{X_1}(x_1), \overrightarrow{\mu}_{X_1}(x_1), \ldots,\overrightarrow{\mu}_{X_n}(x_n)``, the outgoing message is given by ([Loeliger (2007), pg.1299](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf)): 

```math
 \boxed{
\underbrace{\overrightarrow{\mu}_{Y}(y)}_{\substack{ \text{outgoing}\\ \text{message}}} = \sum_{x_1,\ldots,x_n} \underbrace{\overrightarrow{\mu}_{X_1}(x_1)\cdots \overrightarrow{\mu}_{X_n}(x_n)}_{\substack{\text{incoming} \\ \text{messages}}} \cdot \underbrace{f(y,x_1,\ldots,x_n)}_{\substack{\text{node}\\ \text{function}}} }
```

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-sum-product.png?raw=true)

This is called the **Sum-Product Message** (SPM) update rule. (Look at the formula to understand why it's called the SPM update rule).

"""

# ╔═╡ 96576b24-d294-11ef-027d-9d71159afa34
md"""
Note that all SPM update rules can be computed from information that is **locally available** at each node.

"""

# ╔═╡ 96577adc-d294-11ef-0157-37c636011697
md"""
If the factor graph for a function ``f`` has **no cycles** (i.e., the graph is a tree), then the marginal ``\bar{f}(x_3) = \sum_{x_1,x_2,x_4,x_5,x_6,x_7}f(x_1,x_2,\ldots,x_7)`` is given by multiplying the forward and backward messages on that edge:

```math
 \boxed{
\bar{f}(x_3) = \overrightarrow{\mu}_{X_3}(x_3)\cdot \overleftarrow{\mu}_{X_3}(x_3)}
```

"""

# ╔═╡ 965798e4-d294-11ef-291e-89c674ec5689
md"""
It follows that the marginal ``\bar{f}(x_3) = \sum_{x_1,x_2,x_4,x_5,x_6,x_7}f(x_1,x_2,\ldots,x_7)`` can be efficiently computed through sum-product messages. Executing inference through SP message passing is called the **Sum-Product Algorithm** (or alternatively, the **belief propagation** algorithm).

"""

# ╔═╡ 9657b088-d294-11ef-3017-e95c4c69b62b
md"""
Just as a final note, inference by sum-product message passing is much like replacing the sum-of-products

```math
ac + ad + bc + bd
```

by the following product-of-sums:

```math
(a + b)(c + d) \,.
```

Which of these two computations is cheaper to execute?

"""

# ╔═╡ 9657f32a-d294-11ef-2d6b-330969a7e395
md"""
## $(HTML("<span id='sp-for-equality-node'>Sum-Product Messages for the Equality Node</span>"))

As an example, let´s evaluate the SP messages for the **equality node** ``f_=(x,y,z) = \delta(z-x)\delta(z-y)``: 

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-equality-node.png?raw=true)

```math
\begin{align*}
\overrightarrow{\mu}_{Z}(z) &= \iint  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(y) \,\delta(z-x)\delta(z-y) \,\mathrm{d}x \mathrm{d}y \\
   &=  \overrightarrow{\mu}_{X}(z)  \int  \overrightarrow{\mu}_{Y}(y) \,\delta(z-y) \,\mathrm{d}y \\
   &=  \overrightarrow{\mu}_{X}(z) \overrightarrow{\mu}_{Y}(z) 
\end{align*}
```

By symmetry, this also implies (for the same equality node) that

```math
\begin{align*}
\overleftarrow{\mu}_{X}(x) &= \overrightarrow{\mu}_{Y}(x) \overleftarrow{\mu}_{Z}(x) \quad \text{and} \\
\overleftarrow{\mu}_{Y}(y) &= \overrightarrow{\mu}_{X}(y) \overleftarrow{\mu}_{Z}(y)\,.
\end{align*}
```

Let us now consider the case of Gaussian messages ``\overrightarrow{\mu}_{X}(x) = \mathcal{N}(x|\overrightarrow{m}_X,\overrightarrow{V}_X)``, ``\overrightarrow{\mu}_{Y}(y) = \mathcal{N}(y| \overrightarrow{m}_Y,\overrightarrow{V}_Y)`` and ``\overrightarrow{\mu}_{Z}(z) = \mathcal{N}(z|\overrightarrow{m}_Z,\overrightarrow{V}_Z)``. Let´s also define the precision matrices ``\overrightarrow{W}_X \triangleq \overrightarrow{V}_X^{-1}`` and similarly for ``Y`` and ``Z``. Then applying the SP update rule leads to multiplication of two Gaussian distributions (see [Roweis notes](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Roweis-1999-gaussian-identities.pdf)), resulting in 

```math
\begin{align*}
\overrightarrow{W}_Z &= \overrightarrow{W}_X + \overrightarrow{W}_Y \qquad &\text{(precisions add)}\\ 
\overrightarrow{W}_Z \overrightarrow{m}_z &= \overrightarrow{W}_X \overrightarrow{m}_X + \overrightarrow{W}_Y \overrightarrow{m}_Y \qquad &\text{(natural means add)}
\end{align*}
```

It follows that **message passing through an equality node is similar to applying Bayes rule**, i.e., fusion of two information sources. Does this make sense?

"""

# ╔═╡ 9658041e-d294-11ef-228d-09e94ca50366
md"""
## Message Passing Schedules

In a non-cyclic (ie, tree) graph, start with messages from the terminals and keep passing messages through the internal nodes towards the "target" variable (``x_3`` in above problem) until you have both the forward and backward message for the target variable. 

"""

# ╔═╡ 965812b0-d294-11ef-24d0-29e7897375db
md"""
In a tree graph, if you continue to pass messages throughout the graph, the Sum-Product Algorithm computes **exact** marginals for all hidden variables.

"""

# ╔═╡ 96582192-d294-11ef-31b5-aba2da3170c5
md"""
If the graph contains cycles, we have in principle an infinite tree by "unrolling" the graph. In this case, the SP Algorithm is not guaranteed to find exact marginals. In practice, if we apply the SP algorithm for just a few iterations ("unrolls"), then we often find satisfying approximate marginals.   

"""

# ╔═╡ 9658329c-d294-11ef-0d03-45e6872c4985
md"""
## Terminal Nodes and Processing Observations

We can use terminal nodes to represent observations, e.g., add a factor ``f(y)=\delta(y−3)`` to terminate the half-edge for variable ``Y``  if  ``y=3``  is observed.

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-observation-y-3.png?raw=true)

Terminal nodes that carry observations are denoted by small black boxes.

"""

# ╔═╡ 965842e6-d294-11ef-2810-bbd070da18ba
md"""
The message out of a **terminal node** (attached to only 1 edge) is the factor itself. For instance, closing a box around terminal node ``f_a(x_1)`` would lead to 

```math
\overrightarrow{\mu}_{X_1}(x_1) \triangleq \sum_{ \stackrel{ \textrm{enclosed} }{ \textrm{variables} } } \;\prod_{\stackrel{ \textrm{enclosed} }{ \textrm{factors} }} f_a(x_1) = f_a(x_1)\,
```

since there are no enclosed variables. 

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-terminal-node-message.png?raw=true)

"""

# ╔═╡ 965856b2-d294-11ef-2c5a-d91b9c678730
md"""
The message from a half-edge is ``1`` (one). You can verify this by imagining that a half-edge ``x`` can be terminated by a node function ``f(x)=1`` without affecting any inference issue.

"""

# ╔═╡ 96587a66-d294-11ef-2c7a-9fd7bea76582
md"""
## Automating Bayesian Inference by Message Passing

The foregoing message update rules can be worked out in closed-form and put into tables (e.g., see Tables 1 through 6 in [Loeliger (2007)](./files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf) for many standard factors such as essential probability distributions and operations such as additions, fixed-gain multiplications and branching (equality nodes).

In the optional slides below, we have worked out a few more update rules for the [addition node](#sp-for-addition-node) and the [multiplication node](#sp-for-multiplication-node).

If the update rules for all node types in a graph have been tabulated, then inference by message passing comes down to executing a set of table-lookup operations, thus creating a completely **automatable Bayesian inference framework**. 

In our research lab [BIASlab](http://biaslab.org) (FLUX 7.060), we are developing [RxInfer](http://rxinfer.ml), which is a (Julia) toolbox for automating Bayesian inference by message passing in a factor graph.

"""

# ╔═╡ 96589eb0-d294-11ef-239a-2513a805cdcf
md"""
## Example: Bayesian Linear Regression by Message Passing

"""

# ╔═╡ 9658c106-d294-11ef-01db-cfcff611ed81
md"""
Assume we want to estimate some function ``f: \mathbb{R}^D \rightarrow \mathbb{R}`` from a given data set ``D = \{(x_1,y_1), \ldots, (x_N,y_N)\}``, with model assumption ``y_i = f(x_i) + \epsilon_i``.

"""

# ╔═╡ 96594d44-d294-11ef-22b8-95165fb08ce4
md"""
### model specification

We will assume a linear model with white Gaussian noise and a Gaussian prior on the coefficients ``w``:

```math
\begin{align*}
  y_i &= w^T x_i  + \epsilon_i \\
  \epsilon_i &\sim \mathcal{N}(0, \sigma^2) \\ 
  w &\sim \mathcal{N}(0,\Sigma)
\end{align*}
```

or equivalently

```math
\begin{align*}
p(w,\epsilon,D) &= \overbrace{p(w)}^{\text{weight prior}} \prod_{i=1}^N  \overbrace{p(y_i\,|\,x_i,w,\epsilon_i)}^{\text{regression model}} \overbrace{p(\epsilon_i)}^{\text{noise model}} \\
  &= \mathcal{N}(w\,|\,0,\Sigma) \prod_{i=1}^N \delta(y_i - w^T x_i - \epsilon_i) \mathcal{N}(\epsilon_i\,|\,0,\sigma^2) 
\end{align*}
```

"""

# ╔═╡ 96597ce0-d294-11ef-3478-25c6bbef601e
md"""
### Inference (parameter estimation)

We are interested in inferring the posterior ``p(w|D)``. We will execute inference by message passing on the FFG for the model.

"""

# ╔═╡ 965998a8-d294-11ef-1d18-85876e3656c5
md"""
The left figure shows the factor graph for this model. 

The right figure shows the message passing scheme. 

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-bayesian-linear-regression.png?raw=true)

"""

# ╔═╡ 9659ab66-d294-11ef-027a-d3f7206050af
md"""
### CODE EXAMPLE

Let's solve this problem by message passing-based inference with Julia's FFG toolbox [RxInfer](https://biaslab.github.io/rxinfer-website/).

"""

# ╔═╡ 965a08f4-d294-11ef-0604-1586ff37c0d4
using Plots, LinearAlgebra, LaTeXStrings

# Parameters
Σ = 1e5 * Diagonal(I,3) # Covariance matrix of prior on w
σ2 = 2.0                # Noise variance

# Generate data set
w = [1.0; 2.0; 0.25]
N = 30
z = 10.0*rand(N)
x_train = [[1.0; z; z^2] for z in z] # Feature vector x = [1.0; z; z^2]
f(x) = (w'*x)[1]
y_train = map(f, x_train) + sqrt(σ2)*randn(N) # y[i] = w' * x[i] + ϵ
scatter(z, y_train, label="data", xlabel=L"z", ylabel=L"f([1.0, z, z^2]) + \epsilon")

# ╔═╡ 965a1df0-d294-11ef-323c-3da765f1104a
md"""
Now build the factor graph in RxInfer, perform sum-product message passing and plot results (mean of posterior).

"""

# ╔═╡ 965a37e8-d294-11ef-340f-0930b229dd32
using RxInfer, Random
# Build model
@model function linear_regression(y,x, N, Σ, σ2)

    w ~ MvNormalMeanCovariance(zeros(3),Σ)
    
    for i in 1:N
        y[i] ~ NormalMeanVariance(dot(w , x[i]), σ2)
    end
end
# Run message passing algorithm 
results = infer(
    model      = linear_regression(N=length(x_train), Σ=Σ, σ2=σ2),
    data       = (y = y_train, x = x_train),
    returnvars = (w = KeepLast(),),
    iterations = 20,
);
# Plot result
w = results.posteriors[:w]
println("Posterior distribution of w: $(w)")
plt = scatter(z, y_train, label="data", xlabel=L"z", ylabel=L"f([1.0, z, z^2]) + \epsilon")
z_test = collect(0:0.2:12)
x_test = [[1.0; z; z^2] for z in z_test]
for i=1:10
    w_sample = rand(results.posteriors[:w])
    f_est(x) = (w_sample'*x)[1]
    plt = plot!(z_test, map(f_est, x_test), alpha=0.3, label="");
end
display(plt)

# ╔═╡ 965a6c20-d294-11ef-1c91-4bd237afbd20
md"""
## Final thoughts: Modularity and Abstraction

The great Michael Jordan (no, not [this one](https://youtu.be/cuLprHh_BRg), but [this one](https://people.eecs.berkeley.edu/~jordan/)), wrote:   

> "I basically know of two principles for treating complicated systems in simple ways: the first is the principle of **modularity** and the second is the principle of **abstraction**. I am an apologist for computational probability in machine learning because I believe that probability theory implements these two principles in deep and intriguing ways — namely through factorization and through averaging. Exploiting these two mechanisms as fully as possible seems to me to be the way forward in machine learning." — Michael Jordan, 1997 (quoted in [Fre98](https://mitpress.mit.edu/9780262062022/)).


Factor graphs realize these ideas nicely, both visually and computationally.

Visually, the modularity of conditional independencies in the model are displayed by the graph structure. Each node hides internal complexity and by closing-the-box, we can hierarchically move on to higher levels of abstraction. 

Computationally, message passing-based inference uses the Distributive Law to avoid any unnecessary computations.  

What is the relevance of this lesson? RxInfer is not yet a finished project. Still, my prediction is that in 5-10 years, this lesson on Factor Graphs will be the final lecture of part-A of this class, aimed at engineers who need to develop machine learning applications. In principle you have all the tools now to work out the 4-step machine learning recipe (1. model specification, 2. parameter learning, 3. model evaluation, 4. application) that was proposed in the [Bayesian machine learning lesson](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Bayesian-Machine-Learning.ipynb#Bayesian-design). You can propose any model and execute the (learning, evaluation, and application) stages by executing the corresponding inference task automatically in RxInfer. 

Part-B of this class would be about on advanced methods on how to improve automated inference by RxInfer or a similar probabilistic programming package. The Bayesian approach fully supports separating model specification from the inference task. 

"""

# ╔═╡ 965a8a1a-d294-11ef-1d2f-65abf76665e8
md"""
# OPTIONAL SLIDES

"""

# ╔═╡ 965aa14c-d294-11ef-226f-65d587fefa64
md"""
## $(HTML("<span id='sp-for-multiplication-node'>Sum-Product Messages for Multiplication Nodes</span>"))

Next, let us consider a **multiplication** by a fixed (invertible matrix) gain ``f_A(x,y) = \delta(y-Ax)``

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-gain-node.png?raw=true)

"""

# ╔═╡ 965ab77c-d294-11ef-2510-95b1a998589f
md"""
```math
\begin{align*}
\overrightarrow{\mu}_{Y}(y) &= \int  \overrightarrow{\mu}_{X}(x) \,\delta(y-Ax) \,\mathrm{d}x \\
&= \int  \overrightarrow{\mu}_{X}(x) \,|A|^{-1}\delta(x-A^{-1}y) \,\mathrm{d}x \\
&= |A|^{-1}\overrightarrow{\mu}_{X}(A^{-1}y) \,.
\end{align*}
```

"""

# ╔═╡ 965af708-d294-11ef-112c-f5470031dbbe
md"""
For a Gaussian message input message ``\overrightarrow{\mu}_{X}(x) = \mathcal{N}(x|\overrightarrow{m}_{X},\overrightarrow{V}_{X})``, the output message is also Gaussian with 

```math
\begin{align*}
\overrightarrow{m}_{Y} = A\overrightarrow{m}_{X} \,,\,\text{and}\,\,
\overrightarrow{V}_{Y} = A\overrightarrow{V}_{X}A^T
\end{align*}
```

since 

```math
\begin{align*}
\overrightarrow{\mu}_{Y}(y) &= |A|^{-1}\overrightarrow{\mu}_{X}(A^{-1}y) \\
  &\propto \exp \left( -\frac{1}{2} \left( A^{-1}y - \overrightarrow{m}_{X}\right)^T \overrightarrow{V}_{X}^{-1} \left(  A^{-1}y - \overrightarrow{m}_{X}\right)\right) \\
   &= \exp \big( -\frac{1}{2} \left( y - A\overrightarrow{m}_{X}\right)^T \underbrace{A^{-T}\overrightarrow{V}_{X}^{-1} A^{-1}}_{(A \overrightarrow{V}_{X} A^T)^{-1}} \left( y - A\overrightarrow{m}_{X}\right)\big) \\
  &\propto  \mathcal{N}(y| A\overrightarrow{m}_{X},A\overrightarrow{V}_{X}A^T) \,.
\end{align*}
```

"""

# ╔═╡ 965b11a4-d294-11ef-1d04-dbdf39ce91a3
md"""
<b>Exercise</b>: Proof that, for the same factor ``\delta(y-Ax)`` and Gaussian messages, the (backward) sum-product message ``\overleftarrow{\mu}_{X}`` is given by 

```math
\begin{align*}
\overleftarrow{\xi}_{X} &= A^T\overleftarrow{\xi}_{Y} \\
\overleftarrow{W}_{X} &= A^T\overleftarrow{W}_{Y}A
\end{align*}
```

where ``\overleftarrow{\xi}_X \triangleq \overleftarrow{W}_X \overleftarrow{m}_X`` and ``\overleftarrow{W}_{X} \triangleq \overleftarrow{V}_{X}^{-1}`` (and similarly for ``Y``).

"""

# ╔═╡ 965b25ac-d294-11ef-0b9a-9d5a50a76069
md"""
## $(HTML("<span id='sp-for-addition-node'>Code example: Gaussian forward and backward messages for the Addition node</span>"))

Let's calculate the Gaussian forward and backward messages for the addition node in RxInfer.  ![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-addition-node.png?raw=true)

"""

# ╔═╡ 965b4732-d294-11ef-3a9e-81c66860177a
println("Forward message on Z:")
@call_rule typeof(+)(:out, Marginalisation) (m_in1 = NormalMeanVariance(1.0, 1.0), m_in2 = NormalMeanVariance(2.0, 1.0))

# ╔═╡ 965b6cd8-d294-11ef-1267-bd980921b98c
println("Backward message on X:")
@call_rule typeof(+)(:in1, Marginalisation) (m_out = NormalMeanVariance(3.0, 1.0), m_in2 = NormalMeanVariance(2.0, 1.0))

# ╔═╡ 965b886e-d294-11ef-1b10-0319896874cf
md"""
## Code Example: forward and backward messages for the Matrix Multiplication node

In the same way we can also investigate the forward and backward messages for the matrix multiplication ("gain") node  ![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-gain-node.png?raw=true)

"""

# ╔═╡ 965bcdec-d294-11ef-3c5f-a5a7675e5665
println("Forward message on Y:")
@call_rule typeof(*)(:out, Marginalisation) (m_A = PointMass(4.0), m_in = NormalMeanVariance(1.0, 1.0))

# ╔═╡ 965c0442-d294-11ef-22a3-518443336d03
println("Backward message on X:")
@call_rule typeof(*)(:in, Marginalisation) (m_out = NormalMeanVariance(2.0, 1.0), m_A = PointMass(4.0))

# ╔═╡ 965c18f8-d294-11ef-2456-b945a46241f4
md"""
## Example: Sum-Product Algorithm to infer a posterior

Consider a generative model 

```math
p(x,y_1,y_2) = p(x)\,p(y_1|x)\,p(y_2|x) .
```

This model expresses the assumption that ``Y_1`` and ``Y_2`` are independent measurements of ``X``.

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-observations.png?raw=true)

"""

# ╔═╡ 965c2a4e-d294-11ef-1aab-73725568c64e
md"""
Assume that we are interested in the posterior for ``X`` after observing ``Y_1= \hat y_1`` and ``Y_2= \hat y_2``. The posterior for ``X`` can be inferred by applying the sum-product algorithm to the following graph:

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ffg-observations-2.png?raw=true)

"""

# ╔═╡ 965c39a8-d294-11ef-1d83-bde85e3ca790
md"""
(Note that) we usually draw terminal nodes for observed variables in the graph by smaller solid-black squares. This is just to help the visualization of the graph, since the computational rules are no different than for other nodes. 

"""

# ╔═╡ 965c5f28-d294-11ef-324e-4df3e38b5045
md"""
## Code for Sum-Product Algorithm to infer  a posterior

We'll use RxInfer to build the above graph, and perform sum-product message passing to infer the posterior ``p(x|y_1,y_2)``. We assume ``p(y_1|x)`` and ``p(y_2|x)`` to be Gaussian likelihoods with known variances:

```math
\begin{align*}
    p(y_1\,|\,x) &= \mathcal{N}(y_1\,|\,x, v_{y1}) \\
    p(y_2\,|\,x) &= \mathcal{N}(y_2\,|\,x, v_{y2})
\end{align*}
```

Under this model, the posterior is given by:

```math
\begin{align*}
    p(x\,|\,y_1,y_2) &\propto \overbrace{p(y_1\,|\,x)\,p(y_2\,|\,x)}^{\text{likelihood}}\,\overbrace{p(x)}^{\text{prior}} \\
    &=\mathcal{N}(x\,|\,\hat{y}_1, v_{y1})\, \mathcal{N}(x\,|\,\hat{y}_2, v_{y2}) \, \mathcal{N}(x\,|\,m_x, v_x) 
\end{align*}
```

so we can validate the answer by solving the Gaussian multiplication manually.

"""

# ╔═╡ 965c8018-d294-11ef-1274-bd71086649c6
# Data
y1_hat = 1.0
y2_hat = 2.0

# Construct the factor graph
@model function my_model(y1,y2)

    # `x` is the hidden states
    x ~ NormalMeanVariance(0.0, 4.0)

    # `y1` and `y2` are "clamped" observations
    y1 ~ NormalMeanVariance(x, 1.0)
    y2 ~ NormalMeanVariance(x, 2.0)
    
    return x
end

result = infer(model=my_model(), data=(y1=y1_hat, y2 = y2_hat,))
println("Sum-product message passing result: p(x|y1,y2) = 𝒩($(mean(result.posteriors[:x])),$(var(result.posteriors[:x])))")

# Calculate mean and variance of p(x|y1,y2) manually by multiplying 3 Gaussians (see lesson 4 for details)
v = 1 / (1/4 + 1/1 + 1/2)
m = v * (0/4 + y1_hat/1.0 + y2_hat/2.0)
println("Manual result: p(x|y1,y2) = 𝒩($(m), $(v))")

# ╔═╡ 965c97ea-d294-11ef-36e3-2b436feb90f6
open("../../styles/aipstyle.html") do f display("text/html", read(f, String)) end

# ╔═╡ Cell order:
# ╟─96547560-d294-11ef-0fa7-6b6489f7baba
# ╟─9654ea3e-d294-11ef-335c-657af1ceaf19
# ╟─96552348-d294-11ef-16d8-b53563054687
# ╟─965531da-d294-11ef-1639-db0dd32c16d1
# ╟─96555a72-d294-11ef-1270-f14e47749893
# ╟─9655959e-d294-11ef-0ca6-5f20aa579e91
# ╟─9655a94e-d294-11ef-00af-8f49c8821a19
# ╟─9655b2c2-d294-11ef-057f-9b3984064411
# ╟─9655c1ae-d294-11ef-061a-991947cee620
# ╟─9655d360-d294-11ef-0f06-ab58e2ad0e5f
# ╟─9655e06c-d294-11ef-0393-9355d6e20afb
# ╟─9655ed6e-d294-11ef-370f-937b590036f3
# ╟─9655fb88-d294-11ef-1ceb-91585012d142
# ╟─965606f2-d294-11ef-305b-870427879e50
# ╟─96561594-d294-11ef-1590-198382927808
# ╟─96563a44-d294-11ef-3ba4-23fd588b99a5
# ╟─9656566e-d294-11ef-37c9-a536fc435e36
# ╟─965679f0-d294-11ef-13e0-bf28c9a9a505
# ╟─9656b67c-d294-11ef-1541-3d3607375fd2
# ╟─9656cf72-d294-11ef-03aa-b715dd686c09
# ╟─9656d850-d294-11ef-21a1-474b07ea7729
# ╟─9656e606-d294-11ef-1daa-312623552a5b
# ╟─9656ee62-d294-11ef-38f4-7bc8031df7ee
# ╟─9656fae2-d294-11ef-10d8-ff921d5956bd
# ╟─96570d3e-d294-11ef-0178-c34dda717495
# ╟─96571c34-d294-11ef-11ef-29beeb1f96c2
# ╟─96572be8-d294-11ef-2cd1-256972de7b23
# ╟─96573a0c-d294-11ef-2e99-67fdf2ee2eab
# ╟─96574a88-d294-11ef-31a1-e949e6875a3d
# ╟─96575dd4-d294-11ef-31d6-b39b4c4bdea1
# ╟─96576b24-d294-11ef-027d-9d71159afa34
# ╟─96577adc-d294-11ef-0157-37c636011697
# ╟─965798e4-d294-11ef-291e-89c674ec5689
# ╟─9657b088-d294-11ef-3017-e95c4c69b62b
# ╟─9657f32a-d294-11ef-2d6b-330969a7e395
# ╟─9658041e-d294-11ef-228d-09e94ca50366
# ╟─965812b0-d294-11ef-24d0-29e7897375db
# ╟─96582192-d294-11ef-31b5-aba2da3170c5
# ╟─9658329c-d294-11ef-0d03-45e6872c4985
# ╟─965842e6-d294-11ef-2810-bbd070da18ba
# ╟─965856b2-d294-11ef-2c5a-d91b9c678730
# ╟─96587a66-d294-11ef-2c7a-9fd7bea76582
# ╟─96589eb0-d294-11ef-239a-2513a805cdcf
# ╟─9658c106-d294-11ef-01db-cfcff611ed81
# ╟─96594d44-d294-11ef-22b8-95165fb08ce4
# ╟─96597ce0-d294-11ef-3478-25c6bbef601e
# ╟─965998a8-d294-11ef-1d18-85876e3656c5
# ╟─9659ab66-d294-11ef-027a-d3f7206050af
# ╠═965a08f4-d294-11ef-0604-1586ff37c0d4
# ╟─965a1df0-d294-11ef-323c-3da765f1104a
# ╠═965a37e8-d294-11ef-340f-0930b229dd32
# ╟─965a6c20-d294-11ef-1c91-4bd237afbd20
# ╟─965a8a1a-d294-11ef-1d2f-65abf76665e8
# ╟─965aa14c-d294-11ef-226f-65d587fefa64
# ╟─965ab77c-d294-11ef-2510-95b1a998589f
# ╟─965af708-d294-11ef-112c-f5470031dbbe
# ╟─965b11a4-d294-11ef-1d04-dbdf39ce91a3
# ╟─965b25ac-d294-11ef-0b9a-9d5a50a76069
# ╠═965b4732-d294-11ef-3a9e-81c66860177a
# ╠═965b6cd8-d294-11ef-1267-bd980921b98c
# ╟─965b886e-d294-11ef-1b10-0319896874cf
# ╠═965bcdec-d294-11ef-3c5f-a5a7675e5665
# ╠═965c0442-d294-11ef-22a3-518443336d03
# ╟─965c18f8-d294-11ef-2456-b945a46241f4
# ╟─965c2a4e-d294-11ef-1aab-73725568c64e
# ╟─965c39a8-d294-11ef-1d83-bde85e3ca790
# ╟─965c5f28-d294-11ef-324e-4df3e38b5045
# ╠═965c8018-d294-11ef-1274-bd71086649c6
# ╠═965c97ea-d294-11ef-36e3-2b436feb90f6
