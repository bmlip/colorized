### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6a23b828-d294-11ef-371a-05d061144a43
md"""
# Bayesian Machine Learning

"""

# ╔═╡ 6a23df9e-d294-11ef-3ddf-a51d4cea00fc
md"""
## Preliminaries

Goals

  * Introduction to Bayesian (i.e., probabilistic) modeling

Materials

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 68-74 (on the coin toss example)
      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.35-44 (section 2.9, on deriving Bayes rule for updating probabilities)

    

"""

# ╔═╡ 6a24376c-d294-11ef-348a-e9027bd0ec29
md"""
## Challenge: Predicting a Coin Toss

**Problem**: We observe the following sequence of heads (outcome ``=1``) and tails (outcome ``=0``) when tossing the same coin repeatedly 

```math
D=\{1011001\}\,.
```

What is the probability that heads comes up next?

"""

# ╔═╡ 6a24ae24-d294-11ef-3825-b7d13df50212
md"""
**Solution**: later in this lecture. 

"""

# ╔═╡ 6a24b9e4-d294-11ef-3ead-9d272fbf89be
md"""
## The Bayesian Machine Learning Framework

Suppose that your application is to predict a future observation ``x``, based on ``N`` past observations ``D=\{x_1,\dotsc,x_N\}``.

"""

# ╔═╡ 6a24c3e6-d294-11ef-3581-2755a9ba15ba
md"""
The $(HTML("<span id='Bayesian-design'>Bayesian design</span>")) approach to solving this task involves four stages: 

<div style="background-color:rgba(0, 0, 0, 0.0470588); padding:10px 0;font-family:monospace;"> <font color = "red">&nbsp;&nbsp;REPEAT</font></br> &nbsp;&nbsp;&nbsp; 1- Model specification</br> &nbsp;&nbsp;&nbsp; 2- Parameter estimation</br> &nbsp;&nbsp;&nbsp; 3- Model evaluation</br> <font color = "red">&nbsp;&nbsp;UNTIL model performance is satisfactory</font></br> &nbsp;&nbsp;&nbsp; 4- Apply model </div>

"""

# ╔═╡ 6a24c9f4-d294-11ef-20cc-172ea50da901
md"""
In principle, based on the model evaluation results, you may want to re-specify your model and *repeat* the design process (a few times), until model performance is acceptable. 

"""

# ╔═╡ 6a24cee0-d294-11ef-35cb-71ab9ef935e5
md"""
Next, we discuss these four stages in a bit more detail.

"""

# ╔═╡ 6a24d478-d294-11ef-2a75-9d03a5ba7ff8
md"""
## (1) Model specification

Your first task is to propose a probabilistic model for generating the observations ``x``.

"""

# ╔═╡ 6a24fde8-d294-11ef-29bf-ad3e20a53c29
md"""
A probabilistic model ``m`` consists of a joint distribution ``p(x,\theta|m)`` that relates observations ``x`` to model parameters ``\theta``. Usually, the model is proposed in the form of a data generating  distribution ``p(x|\theta,m)`` and a prior ``p(\theta|m)``. 

"""

# ╔═╡ 6a251a08-d294-11ef-171a-27b9d0f818bc
md"""
*You* are responsible to choose the data generating distribution ``p(x|\theta)`` based on your physical understanding of the data generating process. (For brevity, if we are working on one given model ``m`` with no alternative models, we usually drop the given dependency on ``m`` from the notation).

"""

# ╔═╡ 6a252250-d294-11ef-33cd-89b18066817d
md"""
*You* must also choose the prior ``p(\theta)`` to reflect what you know about the parameter values before you see the data ``D``.

"""

# ╔═╡ 6a25307e-d294-11ef-0662-3db678b32e99
md"""
## (2) Parameter estimation

Note that, for a given data set ``D=\{x_1,x_2,\dots,x_N\}`` with *independent* observations ``x_n``, the likelihood factorizes as 

```math
 p(D|\theta) = \prod_{n=1}^N p(x_n|\theta)\,,
```

so usually you select a model for generating one observation ``x_n`` and then use (in-)dependence assumptions to combine these models into a likelihood function for the model parameters.

"""

# ╔═╡ 6a25379a-d294-11ef-3e07-87819f6d75cb
md"""
The likelihood and prior both contain information about the model parameters. Next, you use Bayes rule to fuse these two information sources into a posterior distribution for the parameters:

```math
\begin{align*}
\underbrace{p(\theta|D) }_{\text{posterior}} &= \frac{p(D|\theta) p(\theta)}{p(D)} \\
&= \frac{p(D|\theta) p(\theta)}{\int p(D|\theta) p(\theta) \mathrm{d}\theta}
\end{align*}
```

"""

# ╔═╡ 6a254460-d294-11ef-1890-230b75b6b9ee
md"""
Note that there's **no need for you to design some clever parameter estimation algorithm**. Bayes rule *is* the parameter estimation algorithm, which can be entirely expressed in terms of the likelihood and prior. The only complexity lies in the computational issues! 

"""

# ╔═╡ 6a2552ac-d294-11ef-08d6-179e068bc297
md"""
This parameter estimation "recipe" works if the right-hand side (RHS) factors can be evaluated; the computational details can be quite challenging and this is what machine learning is about.     

```math
\Rightarrow
```

**Machine learning is EASY, apart from computational details :)**

"""

# ╔═╡ 6a2561c0-d294-11ef-124d-373846e3120c
md"""
## (3) Model Evaluation

In the framework above, parameter estimation was executed by "perfect" Bayesian reasoning. So is everything settled now? 

"""

# ╔═╡ 6a257020-d294-11ef-0490-e151934b2f42
md"""
No, there appears to be one remaining problem: how good really were our model assumptions ``p(x|\theta)`` and ``p(\theta)``? We want to "score" the model performance.

"""

# ╔═╡ 6a257f34-d294-11ef-2928-fbb800e81124
md"""
Note that this question is only interesting in practice if we have alternative models to choose from. After all, if you don't have an alternative model, any value for the model evidence would still not lead you to switch to another model.  

"""

# ╔═╡ 6a25a11e-d294-11ef-1c51-09482dad86f2
md"""
Let's assume that we have more candidate models, say ``\mathcal{M} = \{m_1,\ldots,m_K\}`` where each model relates to specific prior ``p(\theta|m_k)`` and likelihood ``p(D|\theta,m_k)``? Can we evaluate the relative performance of a model against another model from the set?

"""

# ╔═╡ 6a25edfc-d294-11ef-3411-6f74c376461e
md"""
Start again with **model specification**. *You* must now specify a prior ``p(m_k)`` (next to the likelihood ``p(D|\theta,m_k)`` and prior ``p(\theta|m_k)``) for each of the models and then solve the desired inference problem:      

```math
\begin{align*} 
\underbrace{p(m_k|D)}_{\substack{\text{model}\\\text{posterior}}} &= \frac{p(D|m_k) p(m_k)}{p(D)} \\
  &\propto p(m_k) \cdot p(D|m_k) \\
  &= p(m_k)\cdot \int_\theta p(D,\theta|m_k) \,\mathrm{d}\theta\\
  &= \underbrace{p(m_k)}_{\substack{\text{model}\\\text{prior}}}\cdot \underbrace{\int_\theta \underbrace{p(D|\theta,m_k)}_{\text{likelihood}} \,\underbrace{p(\theta|m_k)}_{\text{prior}}\, \mathrm{d}\theta }_{\substack{\text{evidence }p(D|m_k)\\\text{= model likelihood}}}\\
\end{align*}
```

"""

# ╔═╡ 6a261278-d294-11ef-25a0-5572de58ad06
md"""
You *can* evaluate the RHS of this equation since you selected the model priors ``p(m_k)``, the parameter priors ``p(\theta|m_k)`` and the likelihoods ``p(D|\theta,m_k)``.

"""

# ╔═╡ 6a262182-d294-11ef-23e9-ed45e1da9f46
md"""
You can now compare posterior distributions ``p(m_k|D)`` for a set of models ``\{m_k\}`` and decide on the merits of each model relative to alternative models. This procedure is called **Bayesian model comparison**.

"""

# ╔═╡ 6a26549a-d294-11ef-1f10-15c4d14ae41f
md"""
Note that, to evaluate the model posterior, you must calculate the "model evidence" ``p(D|m_k)``, which can be interpreted as a likelihood function for model ``m_k``. 

"""

# ╔═╡ 6a2664c6-d294-11ef-0a49-5192e17fb9ea
md"""
```math
\Rightarrow
```

In a Bayesian framework, **model estimation** follows the same recipe as parameter estimation; it just works at one higher hierarchical level. Compare the required calulations:

```math
\begin{align*}
p(\theta|D) &\propto p(D|\theta) p(\theta) \; &&\text{(parameter estimation)} \\
p(m_k|D) &\propto p(D|m_k) p(m_k) \; &&\text{(model comparison)}
\end{align*}
```

"""

# ╔═╡ 6a2672d6-d294-11ef-1886-3195c9c7cfa9
md"""
Again, **no need to invent a special algorithm for estimating the performance of your model**. Straightforward application of probability theory takes care of all that. 

"""

# ╔═╡ 6a269568-d294-11ef-02e3-13402d296391
md"""
In principle, you could proceed with asking how good your choice for the candidate model set ``\mathcal{M}`` was. You would have to provide a set of alternative model sets ``\{\mathcal{M}_1,\mathcal{M}_2,\ldots,\mathcal{M}_M\}`` with priors ``p(\mathcal{M}_m)`` for each set and compute posteriors ``p(\mathcal{M}_m|D)``. And so forth ...  

"""

# ╔═╡ 6a26a31e-d294-11ef-2c2f-b349d0859a27
md"""
With the (relative) performance evaluation scores of your model in hand, you could now re-specify your model (hopefully an improved model) and *repeat* the design process until the model performance score is acceptable. 

"""

# ╔═╡ 6a26b7bc-d294-11ef-03e7-2715b6f8dcc7
md"""
#### Bayes Factors

"""

# ╔═╡ 6a26f244-d294-11ef-0488-c1e4ec6e739d
md"""
As an aside, in the (statistics and machine learning) literature, performance comparison between two models is often reported by the [Bayes Factor](https://en.wikipedia.org/wiki/Bayes_factor), which is defined as the ratio of model evidences: 

```math
\begin{align*}
\underbrace{\frac{p(D|m_1)}{p(D|m_2)}}_{\text{Bayes Factor}} &= \frac{\frac{p(D,m_1)}{p(m_1)}}{\frac{p(D,m_2)}{p(m_2)}}  \\
&= \frac{p(D,m_1)}{p(m_1)} \cdot \frac{p(m_2)}{p(D,m_2)} \\
&= \frac{p(m_1|D) p(D)}{p(m_1)} \cdot \frac{p(m_2)}{p(m_2|D) p(D)} \\
&= \underbrace{\frac{p(m_1|D)}{p(m_2|D)}}_{\substack{\text{posterior} \\ \text{ratio}}} \cdot \underbrace{\frac{p(m_2)}{p(m_1)}}_{\substack{\text{prior} \\ \text{ratio}}}
\end{align*}
```

Hence, for equal model priors (``p(m_1)=p(m_2)=0.5``), the Bayes Factor reports the posterior probability ratio for the two models. 

In principle, any hard decision on which is the better model has to accept some *ad hoc* arguments, but [Jeffreys (1961)](https://www.amazon.com/Theory-Probability-Classic-Physical-Sciences/dp/0198503687/ref=sr_1_1?qid=1663516628&refinements=p_27%3Athe+late+Harold+Jeffreys&s=books&sr=1-1&text=the+late+Harold+Jeffreys) advises the following interpretation of the log-Bayes factor 

```math
\log_{10} B_{12} =\log_{10}\frac{p(D|m_1)}{p(D|m_2)}
```

<table width="60%" align="center" text-align="center"> <tr><td> ``\log_{10} B_{12}``</td><td > <b>Evidence for ``m_1`` </b></td></tr> <tr><td>0 to 0.5</td><td>not worth mentioning</td></tr> <tr><td>0.5 to 1</td><td>substantial</td></tr> <tr><td>1 to 2</td><td>strong</td></tr> <tr><td> >2</td><td>decisive</td></tr> </table> 

"""

# ╔═╡ 6a2707e6-d294-11ef-02ad-31bf84662c70
md"""
## (4) Prediction

Once we are satisfied with the evidence for a (trained) model, we can apply the model to our prediction/classification/etc task.

"""

# ╔═╡ 6a271a56-d294-11ef-0046-add807cc0b4f
md"""
Given the data ``D``, our knowledge about the yet unobserved datum ``x`` is captured by (everything is conditioned on the selected model)

```math
\begin{align*}
p(x|D) &\stackrel{s}{=} \int p(x,\theta|D) \,\mathrm{d}\theta\\
 &\stackrel{p}{=} \int p(x|\theta,D) p(\theta|D) \,\mathrm{d}\theta\\
 &\stackrel{m}{=} \int \underbrace{p(x|\theta)}_{\text{data generation dist.}} \cdot \underbrace{p(\theta|D)}_{\text{posterior}} \,\mathrm{d}\theta\\
\end{align*}
```

"""

# ╔═╡ 6a272cc6-d294-11ef-2844-0fa9091f97de
md"""
In the last equation, the simplification ``p(x|\theta,D) = p(x|\theta)`` follows from our model specification. We assumed a *parametric* data generating distribution ``p(x|\theta)`` with no explicit dependency on the data set ``D``. The information from the data set ``D`` has been absorded in the posterior ``p(\theta|D)``, so all information from ``D`` is passed to ``x`` through the (posterior distribution over the) parameters ``\theta``. Technically, ``x`` is conditionally independent from ``D``, given the parameters ``\theta``.

"""

# ╔═╡ 6a273ae0-d294-11ef-2c00-9b3eaed93f6d
md"""
Again, **no need to invent a special prediction algorithm**. Probability theory takes care of all that. The complexity of prediction is just computational: how to carry out the marginalization over ``\theta``.

"""

# ╔═╡ 6a274948-d294-11ef-0563-1796b8883306
md"""
Note that the application of the learned posterior ``p(\theta|D)`` not necessarily has to be a prediction task. We use it here as an example, but other applications (e.g., classification, regression etc.) are of course also possible. 

"""

# ╔═╡ 6a275a52-d294-11ef-1323-9d83972f611a
md"""
#### Prediction with multiple models

When you have a posterior ``p(m_k|D)`` for the models, you don't *need* to choose one model for the prediction task. You can do prediction by **Bayesian model averaging**, which combines the predictive power from all models:

```math
\begin{align*}
p(x|D) &= \sum_k \int p(x,\theta,m_k|D)\,\mathrm{d}\theta \\
 &= \sum_k \int  p(x|\theta,m_k) \,p(\theta|m_k,D)\, p(m_k|D) \,\mathrm{d}\theta \\
  &= \sum_k \underbrace{p(m_k|D)}_{\substack{\text{model}\\\text{posterior}}} \cdot \int \underbrace{p(\theta|m_k,D)}_{\substack{\text{parameter}\\\text{posterior}}} \, \underbrace{p(x|\theta,m_k)}_{\substack{\text{data generating}\\\text{distribution}}} \,\mathrm{d}\theta
\end{align*}
```

"""

# ╔═╡ 6a27684e-d294-11ef-040e-c302cdad714a
md"""
Alternatively, if you do need to work with one model (e.g. due to computational resource constraints), you can for instance select the model with largest posterior ``p(m_k|D)`` and use that model for prediction. This is called **Bayesian model selection**.

"""

# ╔═╡ 6a2777d0-d294-11ef-1ac3-add102c097d6
md"""
Bayesian model averaging is the principal way to apply PT to machine learning. You don't throw away information by discarding lesser performant models, but rather use PT (marginalization of models) to compute 

```math
p(\text{what-I-am-interested-in} \,|\, \text{all available information})\,.
```

"""

# ╔═╡ 6a278784-d294-11ef-11ae-65bd398910d5
md"""
## We're Done!

In principle, you now have the recipe in your hands now to solve all your prediction/classification/regression etc problems by the same method:

1. specify a model
2. train the model (by PT)
3. evaluate the model (by PT); if not satisfied, goto 1
4. apply the model (by PT)

"""

# ╔═╡ 6a27951c-d294-11ef-2e1a-b5a4ce84aceb
md"""
Crucially, there is no need to invent clever machine learning algorithms, and there is no need to invent a clever prediction algorithm nor a need to invent a model performance criterion. Instead, you propose a model and, from there on, you let PT reason about everything that you care about. 

"""

# ╔═╡ 6a27a28a-d294-11ef-1f33-41b444761429
md"""
Your problems are only of computational nature. Perhaps the integral to compute the evidence may not be analytically tractable, etc.

"""

# ╔═╡ 6a27b114-d294-11ef-099d-1d55968934a6
md"""
## Bayesian Evidence as a Model Performance Criterion

I'd like to convince you that $(HTML("<span id='Bayesian-model-evidence'>Bayesian model evidence</span>")) is an excellent criterion for assessing your model's performance. To do so, let us consider a decomposition that relates model evidence to other highly-valued criteria such as **accuracy** and **model complexity**.

"""

# ╔═╡ 6a27beca-d294-11ef-1895-d57b11b827c1
md"""
Consider a model ``p(x,\theta|m)`` and a data set ``D = \{x_1,x_2, \ldots,x_N\}``.

"""

# ╔═╡ 6a27cc80-d294-11ef-244a-01307ec86188
md"""
Given the data set ``D``, the log-evidence for model ``m`` decomposes as follows (please check the derivation):

```math
\begin{align*}
\underbrace{\log p(D|m)}_{\text{log-evidence}} &= \log p(D|m) \cdot   \underbrace{\int p(\theta|D,m)\mathrm{d}\theta}_{\text{evaluates to }1} \\
 &= \int p(\theta|D,m) \log p(D|m) \mathrm{d}\theta  \qquad \text{(move $\log p(D|m)$ into the integral)} \\
 &= \int p(\theta|D,m) \log \underbrace{\frac{p(D|\theta,m) p(\theta|m)}{p(\theta|D,m)}}_{\text{by Bayes rule}} \mathrm{d}\theta \\
  &= \underbrace{\int p(\theta|D,m) \log p(D|\theta,m) \mathrm{d}\theta}_{\text{accuracy (a.k.a. data fit)}} - \underbrace{\int p(\theta|D,m) \log  \frac{p(\theta|D,m)}{p(\theta|m)} \mathrm{d}\theta}_{\text{complexity}}
\end{align*}
```

"""

# ╔═╡ 6a27efc6-d294-11ef-2dc2-3b2ef95e72f5
md"""
The "accuracy" term (also known as data fit) measures how well the model predicts the data set ``D``. We want this term to be high because good models should predict the data ``D`` well. Indeed, higher accuracy leads to higher model evidence. To achieve high accuracy, applying Bayes' rule will shift the posterior ``p(\theta|D)`` away from the prior towards the likelihood function ``p(D|\theta)``.

"""

# ╔═╡ 6a280132-d294-11ef-10ac-f3890cb3f78b
md"""
The second term (complexity) is technically a [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (KLD) between the posterior and prior distributions, see [OPTIONAL SLIDE](#KLD). The KLD is an information-theoretic quantity that can be interpreted as a "distance" measure between two distributions. In other words, the complexity term measures how much the beliefs about ``\theta`` changed, due to learning from the data ``D``. Generally, we like the complexity term to be low, because moving away means forgetting previously acquired information represented by the prior. Indeed, lower complexity leads to higher model evidence.

"""

# ╔═╡ 6a2814b0-d294-11ef-3a76-9b93c1fcd4d5
md"""
Models with high evidence ``p(D|m)`` prefer both high accuracy and low complexity. Therefore, models with high evidence tend to predict the training data ``D`` well (high accuracy), yet also try to preserve the information encoded by the prior (low complexity). These types of models are said to *generalize* well, since they can be applied to different data sets without specific adaptations for each data set.  

"""

# ╔═╡ 6a282892-d294-11ef-2c12-4b1c7374617c
md"""
Focussing only on accuracy maximization could lead to *overfitting*. Focussing only on complexity minimization could lead to *underfitting* of the model. Bayesian ML attends to both terms and avoids both underfitting and overfitting.  

"""

# ╔═╡ 6a286b04-d294-11ef-1b34-8b7a85c0048c
md"""
```math
\Rightarrow
```

Bayesian learning automatically leads to models that generalize well. There is **no need for early stopping or validation data sets**. There is also **no need for tuning parameters** in the learning process. Just learn on the full data set and all behaves well.  

"""

# ╔═╡ 6a2879e6-d294-11ef-37db-df7babe24d25
md"""
This latter point accentuates that the common practice in machine learning to divide a data set into a training, test and validation set is just an ad hoc mechanism that compensates for failing to frame the learning task as a Bayesian inference task. 

"""

# ╔═╡ 6a2889ae-d294-11ef-2439-e1a541a5ccd7
md"""
## Bayesian Machine Learning and the Scientific Method Revisited

The Bayesian design process provides a unified framework for the Scientific Inquiry method. We can now add equations to the design loop. (Trial design to be discussed in [Intelligent Agent lesson](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Intelligent-Agents-and-Active-Inference.ipynb).) 

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/scientific-inquiry-loop-w-BML-eqs.png?raw=true)

"""

# ╔═╡ 6a2898ea-d294-11ef-39ec-31e4bac1e048
md"""
## Revisiting the Challenge: Predicting a Coin Toss

At the beginning of this lesson, we posed the following challenge:

We observe a the following sequence of heads (outcome = ``1``) and tails (outcome = ``0``) when tossing the same coin repeatedly 

```math
D=\{1011001\}\,.
```

What is the probability that heads comes up next? We solve this in the next slides ...

"""

# ╔═╡ 6a28a704-d294-11ef-1bf2-efbdb0cb4cbc
md"""
## Coin toss example (1): Model Specification

We observe a sequence of ``N`` coin tosses ``D=\{x_1,\ldots,x_N\}`` with ``n`` heads. 

"""

# ╔═╡ 6a28b44c-d294-11ef-15da-81be8753d311
md"""
Let us denote outcomes by 

```math
x_k = \begin{cases} 1 & \text{if heads comes up} \\
  0 & \text{otherwise (tails)} \end{cases}
  
```

"""

# ╔═╡ 6a28c9b4-d294-11ef-222b-97bf0912efe7
md"""
#### Likelihood

Assume a [**Bernoulli** distributed](https://en.wikipedia.org/wiki/Bernoulli_distribution) variable ``p(x_k=1|\mu)=\mu`` for a single coin toss, leading to 

```math
p(x_k|\mu)=\mu^{x_k} (1-\mu)^{1-x_k} \,.
```

Assume ``n`` times heads were thrown out of a total of ``N`` throws. The likelihood function then follows a [**binomial** distribution](https://en.wikipedia.org/wiki/Binomial_distribution) :

```math
   
p(D|\mu) = \prod_{k=1}^N p(x_k|\mu) = \mu^n (1-\mu)^{N-n}
```

"""

# ╔═╡ 6a28d81e-d294-11ef-2a9f-d32daa5556ae
md"""
#### $(HTML("<span id='beta-prior'>Prior</span>"))

Assume the prior beliefs for ``\mu`` are governed by a [**beta distribution**](https://en.wikipedia.org/wiki/Beta_distribution)

```math
p(\mu) = \mathrm{Beta}(\mu|\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1}(1-\mu)^{\beta-1}
```

where the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) is sort-of a generalized factorial function. In particular, if ``\alpha,\beta`` are integers, then 

```math
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} = \frac{(\alpha+\beta-1)!}{(\alpha-1)!\,(\beta-1)!}
```

"""

# ╔═╡ 6a28e674-d294-11ef-391b-0d33fd609fb8
md"""
A *what* distribution? Yes, the **beta distribution** is a [**conjugate prior**](https://en.wikipedia.org/wiki/Conjugate_prior) for the binomial distribution, which means that 

```math
\underbrace{\text{beta}}_{\text{posterior}} \propto \underbrace{\text{binomial}}_{\text{likelihood}} \times \underbrace{\text{beta}}_{\text{prior}}
```

so we get a closed-form posterior.

"""

# ╔═╡ 6a28f466-d294-11ef-3af9-e34de9736c71
md"""
```math
\alpha
```

and ``\beta`` are called **hyperparameters**, since they parameterize the distribution for another parameter (``\mu``). E.g., ``\alpha=\beta=1`` leads to a uniform prior for ``\mu``. We use Julia below to visualize some priors ``\mathrm{Beta}(\mu|\alpha,\beta)`` for different values for ``\alpha, \beta``.

"""

# ╔═╡ 6a291b12-d294-11ef-2554-eb7c2b28929d
using Distributions, StatsPlots, SpecialFunctions
using Plots, LaTeXStrings, Plots.PlotMeasures

# ╔═╡ 6a293250-d294-11ef-3702-fbee25b7a456
# maintain a vector of log evidences to plot later
params = [
    (α=0.1, β=0.1),
    (α=1.0, β=1.0),
    (α=2.0, β=3.0),
    (α=8.0, β=4.0)
]

x = 0:0.01:1

plots = []
for (i, (α, β)) in enumerate(params)
    beta_dist = Beta(α, β)
    y = pdf.(beta_dist, x)
    
    xlabel = i in [3, 4] ? "μ" : ""
    ylabel = i in [1, 3] ? "Density" : ""
    
    push!(plots, plot(x, y, label="α=$α, β=$β", xlabel=xlabel, ylabel=ylabel))
end

plot(plots..., layout=(2, 2), suptitle="PDFs of Beta Distributions", legend=:topleft, link=:both, padding=10)

# ╔═╡ 6a294790-d294-11ef-270b-5b2152431426
md"""
Before observing any data, you can express your state-of-knowledge about the coin by choosing values for ``\alpha`` and ``\beta`` that reflect your beliefs. Stronger yet, you *must* choose values for ``\alpha`` and ``\beta``, because the Bayesian framework does not allow you to walk away from your responsibility to explicitly state your beliefs before the experiment.  

"""

# ╔═╡ 6a29bfcc-d294-11ef-30d9-59b2f7c49f0b
md"""
## Coin toss example (2): Parameter estimation

Infer posterior PDF over ``\mu`` (and evidence) through Bayes rule

```math
\begin{align*}
p(\mu&|D) \cdot p(D) = p(D|\mu)\cdot p(\mu)  \\
  &=  \underbrace{\biggl( \mu^n (1-\mu)^{N-n}\biggr)}_{\text{likelihood}} \cdot \underbrace{\biggl( \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1}(1-\mu)^{\beta-1} \biggr)}_{\text{prior}} \\
  &= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1} \\
        &= \underbrace{\biggl(\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \frac{\Gamma(n+\alpha) \Gamma(N-n+\beta)}{\Gamma(N+\alpha+\beta)}\biggr)}_{\text{evidence }p(D)} \cdot \underbrace{\biggl( \frac{\Gamma(N+\alpha+\beta)}{\Gamma(n+\alpha)\Gamma(N-n+\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1}\biggr)}_{\text{posterior }p(\mu|D)=\mathrm{Beta}(\mu|n+\alpha, N-n+\beta)}
\end{align*}
```

hence the posterior is also beta-distributed as

```math
p(\mu|D) = \mathrm{Beta}(\mu|\,n+\alpha, N-n+\beta)
```

"""

# ╔═╡ 6a29d548-d294-11ef-1361-ad2230cad02b
md"""
## Coin toss example (3): Model Evaluation

It follow from the above calculation that the evidence for model ``m`` can be analytically expressed as

```math
p(D|m) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \frac{\Gamma(n+\alpha) \Gamma(N-n+\beta)}{\Gamma(N+\alpha+\beta)}
```

The model evidence is a scalar. The absolute value is not important. However, you may want to compare the model evidence of this model to the evidence for another model on the same data set.  

"""

# ╔═╡ 6a29e25e-d294-11ef-15ce-5bf3d8cdb64c
md"""
## Coin Toss Example (4): Prediction

Once we have accepted a model, let's apply it to the application, in this case, predicting future observations. 

"""

# ╔═╡ 6a29f1c2-d294-11ef-147f-877f99e5b57c
md"""
Marginalize over the parameter posterior to get the predictive PDF for a new coin toss ``x_\bullet``, given the data ``D``,

```math
\begin{align*}
p(x_\bullet=1|D)  &= \int_0^1 p(x_\bullet=1|\mu)\,p(\mu|D) \,\mathrm{d}\mu \\
  &= \int_0^1 \mu \times  \mathrm{Beta}(\mu|\,n+\alpha, N-n+\beta) \,\mathrm{d}\mu  \\
  &= \frac{n+\alpha}{N+\alpha+\beta}
\end{align*}
```

This result is known as [**Laplace's rule of succession**](https://en.wikipedia.org/wiki/Rule_of_succession).

"""

# ╔═╡ 6a2a000e-d294-11ef-17d6-bdcddeedc65d
md"""
The above integral computes the mean of a beta distribution, which is given by ``\mathbb{E}[x] = \frac{a}{a+b}`` for ``x \sim \mathrm{Beta}(a,b)``, see [wikipedia](https://en.wikipedia.org/wiki/Beta_distribution).

"""

# ╔═╡ 6a2a0f18-d294-11ef-02c2-ef117377ca66
md"""
Finally, we're ready to solve our challenge: for ``D=\{1011001\}`` and uniform prior (``\alpha=\beta=1``), we get

```math
 p(x_\bullet=1|D)=\frac{n+1}{N+2} = \frac{4+1}{7+2} = \frac{5}{9}
```

In other words, given the model assumptions (the Bernoulli data-generating distribution and Beta prior as specified above), and the observations ``D=\{1011001\}``, the probability for observing heads (outcome=``1``) on the next toss is ``\frac{5}{9}``.

"""

# ╔═╡ 6a2a1daa-d294-11ef-2a67-9f2ac60a14c5
md"""
Be aware that there is no such thing as an "objective" or "correct" prediction. Every prediction is conditional on the selected model and the used data set. 

"""

# ╔═╡ 6a2a2af2-d294-11ef-0072-bdc3c6f95bb3
md"""
## Coin Toss Example: What did we learn from the data?

What did we learn from the data? Before seeing any data, we think that the probability for throwing heads is 

```math
\left. p(x_\bullet=1|D) \right|_{n=N=0} = \left.\frac{n+\alpha}{N+\alpha+\beta}\right|_{n=N=0} = \frac{\alpha}{\alpha + \beta}\,.
```

"""

# ╔═╡ 6a2a389e-d294-11ef-1b8c-b55de794b65c
md"""
Hence, ``\alpha`` and ``\beta`` can be interpreted as prior pseudo-counts for heads and tails, respectively. 

"""

# ╔═╡ 6a2a465e-d294-11ef-2aa0-43c954a6439e
md"""
If we were to assume zero pseudo-counts, i.e. ``\alpha=\beta \rightarrow 0``, then our prediction for throwing heads after ``N`` coin tosses is completely based on the data, given by

```math
\left. p(x_\bullet=1|D) \right|_{\alpha=\beta \rightarrow 0} = \left.\frac{n+\alpha}{N+\alpha+\beta}\right|_{\alpha=\beta \rightarrow 0} = \frac{n}{N}\,.
```

"""

# ╔═╡ 6a2a5496-d294-11ef-0f1a-e9a70c44288a
md"""
Note the following decomposition

```math
\begin{align*}
    p(x_\bullet=1|\,D) &= \frac{n+\alpha}{N+\alpha+\beta} \\
    &= \frac{\alpha}{N+\alpha+\beta} + \frac{n}{N+\alpha+\beta}  \\
    &= \frac{\alpha}{N+\alpha+\beta}\cdot \frac{\alpha+\beta}{\alpha+\beta} + \frac{n}{N+\alpha+\beta}\cdot \frac{N}{N}  \\
    &= \frac{\alpha}{\alpha+\beta}\cdot \frac{\alpha+\beta}{N+\alpha+\beta} + \frac{N}{N+\alpha+\beta}\cdot \frac{n}{N}  \\
    &= \frac{\alpha}{\alpha+\beta}\cdot \biggl(1-\frac{N}{N+\alpha+\beta} \biggr) + \frac{N}{N+\alpha+\beta}\cdot \frac{n}{N}  \\
        &= \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} + \underbrace{\underbrace{\frac{N}{N+\alpha+\beta}}_{\text{gain}}\cdot \underbrace{\biggl( \underbrace{\frac{n}{N}}_{\substack{\text{data-based}\\\text{prediction}}} - \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} \biggr)}_{\text{prediction error}}}_{\text{correction}}
\end{align*}
```

"""

# ╔═╡ 6a2a9faa-d294-11ef-1284-cfccb1da444e
md"""
Let's interpret this decomposition of the posterior prediction. Before the data ``D`` was observed, our model generated a *prior prediction* $ p(x*\bullet=1) = \frac{\alpha}{\alpha+\beta}$. Next, the degree to which the actually observed data matches this prediction is represented by the _prediction error* $ \frac{n}{N} - \frac{\alpha}{\alpha-\beta}``. The prior prediction is then updated to a _posterior prediction_ $p(x_\bullet=1|D)`` by adding a fraction of the prediction error to the prior prediction. Hence, the data plays the role of "correcting" the prior prediction. 

Note that, since ``0\leq \underbrace{\frac{N}{N+\alpha+\beta}}_{\text{gain}} \lt 1``, the Bayesian prediction lies between (fuses) the prior and data-based predictions.

"""

# ╔═╡ 6a2aad42-d294-11ef-3129-3be5be8c82d6
md"""
For large ``N``, the gain goes to ``1`` and ``\left. p(x_\bullet=1|D)\right|_{N\rightarrow \infty} \rightarrow \frac{n}{N}`` goes to the data-based prediction (the observed relative frequency).

"""

# ╔═╡ 6a2abb16-d294-11ef-0243-d376e8a39bb0
md"""
## Code Example: Bayesian evolution for the coin toss

"""

# ╔═╡ 6a2acb7e-d294-11ef-185c-9d49ce79c31b
md"""
Next, we code an example for a sequence of coin tosses, where we assume that the true coin generates data ``x_n \in \{0,1\}`` by a Bernoulli distribution:

```math
p(x_n|\mu=0.4)=0.4^{x_n} \cdot 0.6^{1-x_n}
```

So, this coin is biased!

In order predict the outcomes of future coin tosses, we'll compare two models.

All models have the same data generating distribution (also Bernoulli)

```math
p(x_n|\mu,m_k) = \mu^{x_n} (1-\mu)^{1-x_n} \quad \text{for }k=1,2
```

but they have different priors:

```math
\begin{aligned}
p(\mu|m_1) &= \mathrm{Beta}(\mu|\alpha=100,\beta=500) \\
p(\mu|m_2) &= \mathrm{Beta}(\mu|\alpha=8,\beta=13) \\
\end{aligned}
```

You can verify that model ``m_2`` has the best prior, since

```math
\begin{align*}
p(x_n=1|m_1) &= \left.\frac{\alpha}{\alpha+\beta}\right|_{m_1} = 100/600 \approx 0.17 \\
p(x_n=1|m_2) &= \left.\frac{\alpha}{\alpha+\beta}\right|_{m_2} = 8/21 \approx 0.38 \,,
\end{align*}
```

( but you are not supposed to know that the real coin has probability for heads ``p(x_n=1|\mu) = 0.4`` ). 

Let's run ``500`` tosses:

"""

# ╔═╡ 6a2aeb0e-d294-11ef-31b5-cbe1a7179f16
# computes log10 of Gamma function
function log10gamma(num)
    num = convert(BigInt, num)
    return log10(gamma(num))
end


μ  = 0.4;                        # specify model parameter
n_tosses = 500                   # specify number of coin tosses
samples = rand(n_tosses) .<= μ   # Flip 200 coins

function handle_coin_toss(prior :: Beta, observation :: Bool)
    posterior = Beta(prior.α + observation, prior.β + (1 - observation))
    return posterior
end

function log_evidence_prior(prior :: Beta, N :: Int64, n :: Int64)
    log_evidence = log10gamma(prior.α + prior.β) - log10gamma(prior.α) - log10gamma(prior.β) + log10gamma(n+prior.α) + log10gamma((N-n)+prior.β) - log10gamma(N+prior.α+prior.β)
    return log_evidence
end

priors = [Beta(100., 500.), Beta(8., 13.)]  # specify prior distributions 
n_models = length(priors)

# save a sequence of posterior distributions for every prior, starting with the prior itself
posterior_distributions = [[d] for d in priors] 
log_evidences = [[] for _ in priors] 

for (N, sample) in enumerate(samples)                       # for every sample we want to update our posterior
    for (i, prior) in enumerate(priors)                     # at every sample we want to update all distributions

        posterior = handle_coin_toss(prior, sample)         # do bayesian updating
        push!(posterior_distributions[i], posterior)        # add posterior to vector of posterior distributions
        
        # compute log evidence and add to vector
        log_evidence = log_evidence_prior(posterior_distributions[i][N], N, sum(samples[1:N]))    
        push!(log_evidences[i], log_evidence)

        # the prior for the next sample is the posterior from the current sample
        priors[i] = posterior                               
    end
end

# ╔═╡ 6a2af90a-d294-11ef-07bd-018326577791
md"""
For each model, as a function of the number of coin tosses, we plot the evolution of the parameter posteriors 

```math
p(\mu|D_n,m_\bullet)
```

"""

# ╔═╡ 6a2b1106-d294-11ef-0d64-dbc26ba3eb44
#animate posterior distributions over time in a gif

anim = @animate for i in 1:n_tosses
    p = plot(title=string("n = ", i))
    for j in 1:n_models
        plot!(posterior_distributions[j][i+1], xlims = (0, 1), fill=(0, .2,), label=string("Posterior m", j), linewidth=2, ylims=(0,28), xlabel="μ")
    end
end

gif(anim, "figures/anim_bay_ct.gif", show_msg = false)

# ╔═╡ 6a2b1f5a-d294-11ef-25d0-e996c07958b9
md"""
(If the GIF animation is not rendered, you can try to [view it here](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/Bayesian-Machine-Learning.ipynb)).

"""

# ╔═╡ 6a2b2d44-d294-11ef-33ba-15db357708b1
md"""
Note that both posteriors move toward the "correct" value (``\mu=0.4``). However, the posterior for ``m_1`` (blue) moves much slower because we assumed far more pseudo-observations for ``m_1`` than for ``m_2``. 

As we get more observations, the influence of the prior diminishes. 

"""

# ╔═╡ 6a2b3ba4-d294-11ef-3c28-176be260cb15
md"""
We have an intuition that ``m_2`` is superior over ``m_1``. Let's check this by plotting over time the relative Bayesian evidences for each model:

```math
\frac{p(D_n|m_i)}{\sum_{i=1}^2 p(D_n|m_i)}
```

"""

# ╔═╡ 6a2b533e-d294-11ef-3f74-adae9d9c63f0
using LaTeXStrings

evidences = map(model -> exp.(model), log_evidences)

anim = @animate for i in 1:n_tosses
    p = plot(title=string(L"\frac{p_i(\mathbf{x}_{1:n})}{\sum_i p_i(\mathbf{x}_{1:n})}","   n = ", i), ylims=(0, 1), legend=:topleft)
    total = sum([evidences[j][i] for j in 1:n_models])
    bar!([(evidences[j][i] / total) for j in 1:n_models], group=["Model $i" for i in 1:n_models])
end

gif(anim, "figures/anim_bay_ct_log_evidence.gif", show_msg = false)

# ╔═╡ 6a2b9676-d294-11ef-241a-89ff7aa676f9
md"""
(If the GIF animation is not rendered, you can try to [view it here](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/Bayesian-Machine-Learning.ipynb)).

Over time, the relative evidence of model ``m_1`` converges to 0. Can you explain this behavior?

"""

# ╔═╡ 6a2bb18a-d294-11ef-23bb-99082caf6e01
md"""
## From Posterior to Point-Estimate

In the example above, Bayesian parameter estimation and prediction were tractable in closed-form. This is often not the case. We will need to approximate some of the computations. 

"""

# ╔═╡ 6a2bd3ac-d294-11ef-0543-6fe202ca35b6
md"""
Recall Bayesian prediction

```math
p(x|D) = \int p(x|\theta)p(\theta|D)\,\mathrm{d}{\theta}
```

"""

# ╔═╡ 6a2bf332-d294-11ef-1ff1-cdbfb7732cf1
md"""
If we approximate posterior ``p(\theta|D)`` by a delta function for one 'best' value ``\hat\theta``, then the predictive distribution collapses to

```math
p(x|D)= \int p(x|\theta)\,\delta(\theta-\hat\theta)\,\mathrm{d}{\theta} = p(x|\hat\theta)
```

"""

# ╔═╡ 6a2c008e-d294-11ef-2f07-11cdfb2bddca
md"""
This is just the data generating distribution ``p(x|\theta)`` evaluated at ``\theta=\hat\theta``, which is easy to evaluate.

"""

# ╔═╡ 6a2c11e6-d294-11ef-173b-23fc6dbfefca
md"""
The next question is how to get the parameter estimate ``\hat{\theta}``? (See next slide).

"""

# ╔═╡ 6a2c229e-d294-11ef-2f24-ebe43cbfbfa4
md"""
## Some Well-known Point-Estimates

**Bayes estimate** (the mean of the posterior)

```math
\hat \theta_{bayes}  = \int \theta \, p\left( \theta |D \right)
\,\mathrm{d}{\theta}
```

"""

# ╔═╡ 6a2c3036-d294-11ef-23cb-c3b36c475e8f
md"""
**Maximum A Posteriori** (MAP) estimate 

```math
\hat \theta_{\text{map}}=  \arg\max _{\theta} p\left( \theta |D \right) =
\arg \max_{\theta}  p\left(D |\theta \right) \, p\left(\theta \right)
```

"""

# ╔═╡ 6a2c4058-d294-11ef-2312-d9c672d49701
md"""
**Maximum Likelihood** (ML) estimate

```math
\hat \theta_{ml}  = \arg \max_{\theta}  p\left(D |\theta\right)
```

Note that Maximum Likelihood is MAP with uniform prior

ML is the most common approximation to the full Bayesian posterior.

"""

# ╔═╡ 6a2c505c-d294-11ef-1c92-c1b0e9d50da5
md"""
## Bayesian vs Maximum Likelihood Learning

Consider the task: predict a datum ``x`` from an observed data set ``D``.

<table> <tr><td></td><td style="text-align:center"> <b>Bayesian</b></td><td style="text-align:center"> <b>Maximum Likelihood </b></td></tr> <tr><td>1. <b>Model Specification</b></td><td>Choose a model ``m`` with data generating distribution ``p(x|\theta,m)`` and parameter prior ``p(\theta|m)``</td><td>Choose a model ``m`` with same data generating distribution ``p(x|\theta,m)``. No need for priors.</td></tr> <tr><td>2. <b>Learning</b></td><td>use Bayes rule to find the parameter posterior,

```math
p(\theta|D) \propto p(D|\theta) p(\theta)
```

</td><td>By Maximum Likelihood (ML) optimization,

```math
 
    \hat \theta  = \arg \max_{\theta}  p(D |\theta)
```

</td></tr> <tr><td>3. <b>Prediction</b></td><td>

```math
p(x|D) = \int p(x|\theta) p(\theta|D) \,\mathrm{d}\theta
```

</td><td>

```math
 
    p(x|D) =  p(x|\hat\theta)
```

</td></tr> </table>

"""

# ╔═╡ 6a2c5e08-d294-11ef-213d-97bcfa16eb5a
md"""
## Report Card on Maximum Likelihood Estimation

Maximum Likelihood (ML) is MAP with uniform prior. MAP is sometimes called a 'penalized' ML procedure:

```math
\hat \theta_{map}  = \arg \max _\theta  \{ \underbrace{\log
p\left( D|\theta  \right)}_{\text{log-likelihood}} + \underbrace{\log
p\left( \theta \right)}_{\text{penalty}} \}
```

"""

# ╔═╡ 6a2c7230-d294-11ef-05a2-3ff2f65d10e0
md"""
(good!). ML works rather well if we have a lot of data because the influence of the prior diminishes with more data.

"""

# ╔═╡ 6a2c7f5a-d294-11ef-2e17-9108a39df280
md"""
(good!). Computationally often do-able. Useful fact that makes the optimization easier (since ``\log`` is monotonously increasing):

```math
\arg\max_\theta \log p(D|\theta) =  \arg\max_\theta p(D|\theta)
```

"""

# ╔═╡ 6a2c8f4a-d294-11ef-213c-dfa929a403bc
md"""
(bad). Cannot be used for model comparison! When doing ML estimation, the Bayesian model evidence always evaluates to zero because the prior probability mass under the likelihood function goes to zero. Therefore, when doing ML estimation, Bayesian model evidence cannot be used to evaluate model performance: 

```math
\begin{align*}
\underbrace{p(D|m)}_{\substack{\text{Bayesian}\\ \text{evidence}}} &= \int p(D|\theta) \cdot p(\theta|m)\,\mathrm{d}\theta \\
  &= \lim_{(b-a)\rightarrow \infty} \int p(D|\theta)\cdot \text{Uniform}(\theta|a,b)\,\mathrm{d}\theta \\
  &= \lim_{(b-a)\rightarrow \infty} \frac{1}{b-a}\underbrace{\int_a^b p(D|\theta)\,\mathrm{d}\theta}_{<\infty}  \\
    &= 0
\end{align*}
```

In fact, this is a serious problem because Bayesian evidence is fundamentally the correct performance assessment criterion that follows from straighforward PT. In practice, when estimating parameters by maximum likelihood, we often evaluate model performance by an *ad hoc* performance measure such as mean-squared-error on a testing data set.

"""

# ╔═╡ 6a2ca496-d294-11ef-0043-1f350b36773e
md"""
```math
\Rightarrow
```

**Maximum Likelihood estimation is at best an approximation to Bayesian learning**, but for good reason a very popular learning method when faced with lots of available data.

"""

# ╔═╡ 6a2cb25e-d294-11ef-1d88-1fc784b33df0
md"""
# OPTIONAL SLIDES

"""

# ╔═╡ 6a2ccd16-d294-11ef-22ee-a5cff62ccd9c
md"""
## The Kullback-Leibler Divergence

The $(HTML("<span id='KLD'>Kullback-Leibler Divergence</span>")) (a.k.a. relative entropy) between two distributions ``q`` and ``p`` is defined as

```math
D_{\text{KL}}[q,p] \equiv \sum_z q(z) \log \frac{q(z)}{p(z)}
```

The following [Gibbs Inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality) holds (see [wikipedia](https://en.wikipedia.org/wiki/Gibbs%27_inequality) for proof): 

```math
D_{\text{KL}}[q,p] \geq 0 \quad \text{with equality only if } p=q 
```

The KL divergence can be interpreted as a distance between two probability distributions.

As an aside, note that ``D_{\text{KL}}[q,p] \neq D_{\text{KL}}[p,q]``. Both divergences are relevant. 

"""

# ╔═╡ 6a2cd9be-d294-11ef-33cf-4b23b92e1cbf
md"""
Here is an animation that shows the KL divergence between two Gaussian distributions:

"""

# ╔═╡ 6a2cfda6-d294-11ef-02ee-052eb727fcd3
using Distributions, StatsPlots, Plots.PlotMeasures, LaTeXStrings

function kullback_leibler(q :: Normal, p :: Normal)                                 
    # Calculates the KL Divergence between two gaussians 
    # (see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence for calculations)
    return log((p.σ / q.σ)) + ((q.σ)^2 + (p.μ - q.μ)^2) / (2*p.σ^2) - (1. / 2.)
end

# statistics of distributions we'll keep constant (we'll vary the mean of q)
# feel free to change these and see what happens
μ_p = 0             
σ_p = 1
σ_q = 1

p = Normal(μ_p, σ_p)

anim = @animate for i in 1:100
    μ_seq = [(j / 10.) - 5. + μ_p for j in 1:i]                                                                                                         #compute the sequence of means tested so far (to compute sequence of KL divergences)
    kl = [kullback_leibler(Normal(μ, σ_q), p) for μ in μ_seq]                                                                                           #compute KL divergence data
    viz = plot(right_margin=8mm, title=string(L"D_{KL}(Q || P) = ", round(100 * kl[i]) / 100.), legend=:topleft)                                        #build plot and format title and margins
    μ_q = μ_seq[i]                                                                                                                                      #extract mean of current frame from mean sequence
    q = Normal(μ_q, σ_q)
    plot!(p, xlims = (μ_p - 8, μ_p + 8), fill=(0, .2,), label=string("P"), linewidth=2, ylims=(0,0.5))                                                  #plot p
    plot!(q, fill=(0, .2,), label=string("Q"), linewidth=2, ylims=(0,0.5))                                                                              #plot q
    plot!(twinx(), μ_seq, kl, xticks=:none, ylims=(0, maximum(kl) + 3), linewidth = 3,                                                                  #plot KL divergence data with different y-axis scale and different legend
        legend=:topright,xlims = (μ_p - 8, μ_p + 8), color="green", label=L"D_{KL}(Q || P)")
end
gif(anim, "figures/anim_lat_kl.gif", show_msg = false)

# ╔═╡ 6a2d0b84-d294-11ef-1988-0171c783a412
md"""
If the GIF animation is not rendered, you can try to [view it here](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/Bayesian-Machine-Learning.ipynb).

"""

# ╔═╡ 6a2d20cc-d294-11ef-1830-e5880043a967
open("../../styles/aipstyle.html") do f display("text/html", read(f, String)) end

# ╔═╡ Cell order:
# ╟─6a23b828-d294-11ef-371a-05d061144a43
# ╟─6a23df9e-d294-11ef-3ddf-a51d4cea00fc
# ╟─6a24376c-d294-11ef-348a-e9027bd0ec29
# ╟─6a24ae24-d294-11ef-3825-b7d13df50212
# ╟─6a24b9e4-d294-11ef-3ead-9d272fbf89be
# ╟─6a24c3e6-d294-11ef-3581-2755a9ba15ba
# ╟─6a24c9f4-d294-11ef-20cc-172ea50da901
# ╟─6a24cee0-d294-11ef-35cb-71ab9ef935e5
# ╟─6a24d478-d294-11ef-2a75-9d03a5ba7ff8
# ╟─6a24fde8-d294-11ef-29bf-ad3e20a53c29
# ╟─6a251a08-d294-11ef-171a-27b9d0f818bc
# ╟─6a252250-d294-11ef-33cd-89b18066817d
# ╟─6a25307e-d294-11ef-0662-3db678b32e99
# ╟─6a25379a-d294-11ef-3e07-87819f6d75cb
# ╟─6a254460-d294-11ef-1890-230b75b6b9ee
# ╟─6a2552ac-d294-11ef-08d6-179e068bc297
# ╟─6a2561c0-d294-11ef-124d-373846e3120c
# ╟─6a257020-d294-11ef-0490-e151934b2f42
# ╟─6a257f34-d294-11ef-2928-fbb800e81124
# ╟─6a25a11e-d294-11ef-1c51-09482dad86f2
# ╟─6a25edfc-d294-11ef-3411-6f74c376461e
# ╟─6a261278-d294-11ef-25a0-5572de58ad06
# ╟─6a262182-d294-11ef-23e9-ed45e1da9f46
# ╟─6a26549a-d294-11ef-1f10-15c4d14ae41f
# ╟─6a2664c6-d294-11ef-0a49-5192e17fb9ea
# ╟─6a2672d6-d294-11ef-1886-3195c9c7cfa9
# ╟─6a269568-d294-11ef-02e3-13402d296391
# ╟─6a26a31e-d294-11ef-2c2f-b349d0859a27
# ╟─6a26b7bc-d294-11ef-03e7-2715b6f8dcc7
# ╟─6a26f244-d294-11ef-0488-c1e4ec6e739d
# ╟─6a2707e6-d294-11ef-02ad-31bf84662c70
# ╟─6a271a56-d294-11ef-0046-add807cc0b4f
# ╟─6a272cc6-d294-11ef-2844-0fa9091f97de
# ╟─6a273ae0-d294-11ef-2c00-9b3eaed93f6d
# ╟─6a274948-d294-11ef-0563-1796b8883306
# ╟─6a275a52-d294-11ef-1323-9d83972f611a
# ╟─6a27684e-d294-11ef-040e-c302cdad714a
# ╟─6a2777d0-d294-11ef-1ac3-add102c097d6
# ╟─6a278784-d294-11ef-11ae-65bd398910d5
# ╟─6a27951c-d294-11ef-2e1a-b5a4ce84aceb
# ╟─6a27a28a-d294-11ef-1f33-41b444761429
# ╟─6a27b114-d294-11ef-099d-1d55968934a6
# ╟─6a27beca-d294-11ef-1895-d57b11b827c1
# ╟─6a27cc80-d294-11ef-244a-01307ec86188
# ╟─6a27efc6-d294-11ef-2dc2-3b2ef95e72f5
# ╟─6a280132-d294-11ef-10ac-f3890cb3f78b
# ╟─6a2814b0-d294-11ef-3a76-9b93c1fcd4d5
# ╟─6a282892-d294-11ef-2c12-4b1c7374617c
# ╟─6a286b04-d294-11ef-1b34-8b7a85c0048c
# ╟─6a2879e6-d294-11ef-37db-df7babe24d25
# ╟─6a2889ae-d294-11ef-2439-e1a541a5ccd7
# ╟─6a2898ea-d294-11ef-39ec-31e4bac1e048
# ╟─6a28a704-d294-11ef-1bf2-efbdb0cb4cbc
# ╟─6a28b44c-d294-11ef-15da-81be8753d311
# ╟─6a28c9b4-d294-11ef-222b-97bf0912efe7
# ╟─6a28d81e-d294-11ef-2a9f-d32daa5556ae
# ╟─6a28e674-d294-11ef-391b-0d33fd609fb8
# ╟─6a28f466-d294-11ef-3af9-e34de9736c71
# ╠═6a291b12-d294-11ef-2554-eb7c2b28929d
# ╠═6a293250-d294-11ef-3702-fbee25b7a456
# ╟─6a294790-d294-11ef-270b-5b2152431426
# ╟─6a29bfcc-d294-11ef-30d9-59b2f7c49f0b
# ╟─6a29d548-d294-11ef-1361-ad2230cad02b
# ╟─6a29e25e-d294-11ef-15ce-5bf3d8cdb64c
# ╟─6a29f1c2-d294-11ef-147f-877f99e5b57c
# ╟─6a2a000e-d294-11ef-17d6-bdcddeedc65d
# ╟─6a2a0f18-d294-11ef-02c2-ef117377ca66
# ╟─6a2a1daa-d294-11ef-2a67-9f2ac60a14c5
# ╟─6a2a2af2-d294-11ef-0072-bdc3c6f95bb3
# ╟─6a2a389e-d294-11ef-1b8c-b55de794b65c
# ╟─6a2a465e-d294-11ef-2aa0-43c954a6439e
# ╟─6a2a5496-d294-11ef-0f1a-e9a70c44288a
# ╟─6a2a9faa-d294-11ef-1284-cfccb1da444e
# ╟─6a2aad42-d294-11ef-3129-3be5be8c82d6
# ╟─6a2abb16-d294-11ef-0243-d376e8a39bb0
# ╟─6a2acb7e-d294-11ef-185c-9d49ce79c31b
# ╠═6a2aeb0e-d294-11ef-31b5-cbe1a7179f16
# ╟─6a2af90a-d294-11ef-07bd-018326577791
# ╠═6a2b1106-d294-11ef-0d64-dbc26ba3eb44
# ╟─6a2b1f5a-d294-11ef-25d0-e996c07958b9
# ╟─6a2b2d44-d294-11ef-33ba-15db357708b1
# ╟─6a2b3ba4-d294-11ef-3c28-176be260cb15
# ╠═6a2b533e-d294-11ef-3f74-adae9d9c63f0
# ╟─6a2b9676-d294-11ef-241a-89ff7aa676f9
# ╟─6a2bb18a-d294-11ef-23bb-99082caf6e01
# ╟─6a2bd3ac-d294-11ef-0543-6fe202ca35b6
# ╟─6a2bf332-d294-11ef-1ff1-cdbfb7732cf1
# ╟─6a2c008e-d294-11ef-2f07-11cdfb2bddca
# ╟─6a2c11e6-d294-11ef-173b-23fc6dbfefca
# ╟─6a2c229e-d294-11ef-2f24-ebe43cbfbfa4
# ╟─6a2c3036-d294-11ef-23cb-c3b36c475e8f
# ╟─6a2c4058-d294-11ef-2312-d9c672d49701
# ╟─6a2c505c-d294-11ef-1c92-c1b0e9d50da5
# ╟─6a2c5e08-d294-11ef-213d-97bcfa16eb5a
# ╟─6a2c7230-d294-11ef-05a2-3ff2f65d10e0
# ╟─6a2c7f5a-d294-11ef-2e17-9108a39df280
# ╟─6a2c8f4a-d294-11ef-213c-dfa929a403bc
# ╟─6a2ca496-d294-11ef-0043-1f350b36773e
# ╟─6a2cb25e-d294-11ef-1d88-1fc784b33df0
# ╟─6a2ccd16-d294-11ef-22ee-a5cff62ccd9c
# ╟─6a2cd9be-d294-11ef-33cf-4b23b92e1cbf
# ╠═6a2cfda6-d294-11ef-02ee-052eb727fcd3
# ╟─6a2d0b84-d294-11ef-1988-0171c783a412
# ╠═6a2d20cc-d294-11ef-1830-e5880043a967
