### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3ceb490e-d294-11ef-1883-a50aadd2d519
md"""
# Machine Learning Overview

"""

# ╔═╡ 3cebc804-d294-11ef-32bd-29507524ddb2
md"""
## Preliminaries

Goal

  * Top-level overview of machine learning

Materials

  * Mandatory  

      * this notebook
  * Optional

      * Study Bishop pp. 1-4

"""

# ╔═╡ 3cebf2d4-d294-11ef-1fde-bf03ecfb9b99
md"""
## What is Machine Learning?

Machine Learning relates to **building models from data and using these models in applications**.

"""

# ╔═╡ 3cec06e6-d294-11ef-3359-5740f25965da
md"""
**Problem**: Suppose we want to develop an algorithm for a complex process about which we have little knowledge (so hand-programming is not possible).

"""

# ╔═╡ 3cec1032-d294-11ef-1b9d-237c491b2eb2
md"""
**Solution**: Get the computer to develop the algorithm by itself by showing it examples of the behavior that we want.

"""

# ╔═╡ 3cec1832-d294-11ef-1317-07fe5c4e69c2
md"""
Practically, we choose a library of models, and write a program that picks a model and tunes it to fit the data.

"""

# ╔═╡ 3cec20f4-d294-11ef-1012-c19579a786e4
md"""
This field is known in various scientific communities with slight variations under different names such as machine learning, statistical inference, system identification, data mining, source coding, data compression, data science, etc.

"""

# ╔═╡ 3cec3062-d294-11ef-3dd6-bfc5588bdf1f
md"""
## Machine Learning and the Scientific Inquiry Loop

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/scientific-inquiry-loop.png?raw=true)

Machine learning technology uses the scientific inquiry loop to develop models and use these models in applications.

"""

# ╔═╡ 3cec43d4-d294-11ef-0a9f-43eb506527a6
md"""
## Machine Learning is Difficult

Modeling (Learning) Problems

  * Is there any regularity in the data anyway?
  * What is our prior knowledge and how to express it mathematically?
  * How to pick the model library?
  * How to tune the models to the data?
  * How to measure the generalization performance?

"""

# ╔═╡ 3cec5b96-d294-11ef-39e0-15e93768d2b1
md"""
Quality of Observed Data

  * Not enough data
  * Too much data?
  * Available data may be messy (measurement noise, missing data points, outliers)

"""

# ╔═╡ 3cec86cc-d294-11ef-267d-7743fd241c64
md"""
## A Machine Learning Taxonomy

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/ml-taxonomy.png?raw=true)

**Supervised Learning**: Given examples of inputs and corresponding

desired outputs, predict outputs on future inputs. Essentially, supervised learning relates to learning functions. [Functions describe the world, at least, that's what this guy says.](https://youtu.be/BWZTlfrneD8?si=8Oa4XLlkgaQZpW1H)

Examples: classification, regression, time series prediction

"""

# ╔═╡ 3cec9250-d294-11ef-01ac-9d94676a65a3
md"""
**Unsupervised Learning**: (a.k.a. **density estimation**). Given only inputs, automatically discover representations, features, structure, etc.

  * Examples: clustering, outlier detection, compression

"""

# ╔═╡ 3cecbc46-d294-11ef-24cb-2d9e41fb35d9
md"""
**Trial Design**: (a.k.a. experimental design, active learning). Learn to make actions that optimize some performance criterion about the expected future. 

  * Examples: playing games like chess, self-driving cars, robotics.
  * Two major approaches include **reinforcement learning** and **active inference**

      * **Reinforcement Learning**: Given an observed sequence of input signals and (occasionally observed) rewards for those inputs, *learn* to select actions that maximize *expected* future rewards.
      * **Active inference**: Given an observed sequence of input signals and a prior probability distribution about future observations, *learn* to select actions

that minimize *expected* prediction errors (i.e., minimize actual minus predicted sensation).    

"""

# ╔═╡ 3cecdb48-d294-11ef-20a1-1df2731ac57c
md"""
Other stuff, like **preference learning**, **learning to rank**, etc., can often be (re-)formulated as special cases of either a supervised, unsupervised or trial design problem.

"""

# ╔═╡ 3ced0d0c-d294-11ef-3000-7b63362a2351
md"""
## Supervised Learning

Given observations of desired input-output behavior ``D=\{(x_1,y_1),\dots,(x_N,y_N)\}`` (with ``x_i`` inputs and ``y_i`` outputs), the goal is to estimate the conditional distribution ``p(y|x)`` (i.e., how does ``y`` depend on ``x``?).

#### Classification

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/Bishop-Figure4.5b.png?raw=true)

The target variable ``y`` is a *discrete-valued* vector representing class labels 

The special case ``y \in \{\text{true},\text{false}\}`` is called **detection**. 

"""

# ╔═╡ 3ced29ae-d294-11ef-158b-09fcdaa47d1c
md"""
#### Regression

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/Bishop-Figure1.2.png?raw=true)

Same problem statement as classification but now the target variable is a *real-valued* vector.

Regression is sometimes called **curve fitting**.

"""

# ╔═╡ 3ced3fc2-d294-11ef-3fac-d5e80eacc488
md"""
## Unsupervised Learning

Given data ``D=\{x_1,\ldots,x_N\}``, model the (unconditional) probability distribution ``p(x)`` (a.k.a. **density estimation**). The two primary applications are **clustering** and **compression** (a.k.a. dimensionality reduction).  

#### Clustering

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/fig-Zoubin-clustering-example.png?raw=true)

Group data into clusters such that all data points in a cluster have similar properties.

Clustering can be interpreted as ''unsupervised classification''.

"""

# ╔═╡ 3ced567c-d294-11ef-2657-df20e23a00fa
md"""
#### Compression / dimensionality reduction

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/fig-compression-example.png?raw=true)

Output from coder is much smaller in size than original, but if coded signal if further processed by a decoder, then the result is very close (or exactly equal) to the original.

Usually, the compressed image comprises continuously valued variables. In that case, compression can be interpreted as ''unsupervised regression''.

"""

# ╔═╡ 3ced6df4-d294-11ef-1091-474e512d605c
md"""
## Trial Design and Decision-making

Given the state of the world (obtained from sensory data), the agent must *learn* to produce actions (like making a movement or making a decision) that optimize some performance criterion about the expected future.

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/RL-example.png?raw=true)

In contrast to supervised and unsupervised learning, an agent is able to affect its data set by making actions, e.g., a robot can change its input video data stream by turning the head of its camera. 

In this course, we focus on the active inference approach to trial design, see the [Intelligent Agent lesson](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Intelligent-Agents-and-Active-Inference.ipynb) for details. 

"""

# ╔═╡ 3ced839a-d294-11ef-3dd0-1f8c5ef11b75
md"""
## $(HTML("<span id='some-ml-apps'>Some Machine Learning Applications</span>"))

computer speech recognition, speaker recognition

face recognition, iris identification

printed and handwritten text parsing

financial prediction, outlier detection (credit-card fraud)

user preference modeling (amazon); modeling of human perception

modeling of the web (google)

machine translation

medical expert systems for disease diagnosis (e.g., mammogram)

strategic games (chess, go, backgammon), self-driving cars

In summary, **any 'knowledge-poor' but 'data-rich' problem**

"""

# ╔═╡ 3ced947a-d294-11ef-0403-512f2407a2d2
md"""

"""

# ╔═╡ 3cedcc6a-d294-11ef-345a-6d2af13d4e9c
open("../../styles/aipstyle.html") do f 
    display("text/html", read(f,String)) 
end

# ╔═╡ Cell order:
# ╟─3ceb490e-d294-11ef-1883-a50aadd2d519
# ╟─3cebc804-d294-11ef-32bd-29507524ddb2
# ╟─3cebf2d4-d294-11ef-1fde-bf03ecfb9b99
# ╟─3cec06e6-d294-11ef-3359-5740f25965da
# ╟─3cec1032-d294-11ef-1b9d-237c491b2eb2
# ╟─3cec1832-d294-11ef-1317-07fe5c4e69c2
# ╟─3cec20f4-d294-11ef-1012-c19579a786e4
# ╟─3cec3062-d294-11ef-3dd6-bfc5588bdf1f
# ╟─3cec43d4-d294-11ef-0a9f-43eb506527a6
# ╟─3cec5b96-d294-11ef-39e0-15e93768d2b1
# ╟─3cec86cc-d294-11ef-267d-7743fd241c64
# ╟─3cec9250-d294-11ef-01ac-9d94676a65a3
# ╟─3cecbc46-d294-11ef-24cb-2d9e41fb35d9
# ╟─3cecdb48-d294-11ef-20a1-1df2731ac57c
# ╟─3ced0d0c-d294-11ef-3000-7b63362a2351
# ╟─3ced29ae-d294-11ef-158b-09fcdaa47d1c
# ╟─3ced3fc2-d294-11ef-3fac-d5e80eacc488
# ╟─3ced567c-d294-11ef-2657-df20e23a00fa
# ╟─3ced6df4-d294-11ef-1091-474e512d605c
# ╟─3ced839a-d294-11ef-3dd0-1f8c5ef11b75
# ╟─3ced947a-d294-11ef-0403-512f2407a2d2
# ╠═3cedcc6a-d294-11ef-345a-6d2af13d4e9c
