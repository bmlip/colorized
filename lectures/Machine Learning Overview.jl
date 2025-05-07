### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ a5d43e01-8f73-4c48-b565-f10eb807a9ab
using PlutoTeachingTools

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

# ╔═╡ fa1d5123-db02-4fda-93d2-3e5e2efed515
html"""
<style>
pluto-output img {
	background: white;
	border-radius: 3px;
}
</style>
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

# ╔═╡ 40fddaa0-cb9d-4873-b2f6-3fd2a742ecd2
md"""
IDEE: herhalen in andere lectures
""" |> TODO

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

# ╔═╡ 69555c82-a593-4fe5-97e2-8c6898253991
md"""
IDEE: genereren in code (Bayesian multinomial regression) (mss een mini)

IDEE: klikken om data toe te voegen, en dan runt de classificatie live (2 klasses mag ook)

IDEE: idem voor gressie
""" |> TODO

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

# ╔═╡ 61741b02-f407-4a66-be00-28f92896490e
md"""
IDEE: hier ook live
""" |> TODO

# ╔═╡ 81c85536-2ac9-4c65-9c34-f4661a2a796b
TODO("ook references van toekomstige lectures")

# ╔═╡ 3ced567c-d294-11ef-2657-df20e23a00fa
md"""
#### Compression / dimensionality reduction

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/fig-compression-example.png?raw=true)

Output from coder is much smaller in size than original, but if coded signal if further processed by a decoder, then the result is very close (or exactly equal) to the original.

Usually, the compressed image comprises continuously valued variables. In that case, compression can be interpreted as ''unsupervised regression''.

"""

# ╔═╡ d5a490a9-be04-41bf-b59b-fdeaba999073
md"""
IDEE: (deze lecture is eruit gegaan) 
""" |> TODO

# ╔═╡ 3ced6df4-d294-11ef-1091-474e512d605c
md"""
## Trial Design and Decision-making

Given the state of the world (obtained from sensory data), the agent must *learn* to produce actions (like making a movement or making a decision) that optimize some performance criterion about the expected future.

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./figures/RL-example.png?raw=true)

In contrast to supervised and unsupervised learning, an agent is able to affect its data set by making actions, e.g., a robot can change its input video data stream by turning the head of its camera. 

In this course, we focus on the active inference approach to trial design, see the [Intelligent Agent lesson](https://bmlip.github.io/colorized/lectures/Intelligent%20Agents%20and%20Active%20Inference.html) for details. 

"""

# ╔═╡ 0bdc66e2-7e66-44b9-9685-0174dce20b94
md"""
IDEE: demo van active inference agent uit laatste lecture (mountain car)
""" |> TODO

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

# ╔═╡ f3cff8d8-1b49-4479-be78-ee632233be5c
md"""
IDEE: twee keer hetzelfde cointoss experiment, maar eentje met ML estimator (je krijgt alleen een verticale lijn) en een met bayesian (posterior wordt smaller)

je realiseert niet dat je met ML informatie bent kwijtgeraakt
""" |> TODO

# ╔═╡ 442f9293-319e-4d43-8890-64b1c5a2a118
md"""
IDEE: alle voorbeelden twee plots, een tweede met onzekerheidsmarges

(gradient!! voor classificatie)


""" |> TODO

# ╔═╡ 3ced947a-d294-11ef-0403-512f2407a2d2
md"""

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"

[compat]
PlutoTeachingTools = "~0.2.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "c146a51490f1c1be20bdd2498a3737ddbddc95f4"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

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

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a434e811d10e7cbf4f0674285542e697dca605d0"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.42"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "5cf59106f9b47014c58c5053a1ce09c0a2e0333c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.3"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─3ceb490e-d294-11ef-1883-a50aadd2d519
# ╟─3cebc804-d294-11ef-32bd-29507524ddb2
# ╟─3cebf2d4-d294-11ef-1fde-bf03ecfb9b99
# ╟─3cec06e6-d294-11ef-3359-5740f25965da
# ╟─3cec1032-d294-11ef-1b9d-237c491b2eb2
# ╟─3cec1832-d294-11ef-1317-07fe5c4e69c2
# ╟─3cec20f4-d294-11ef-1012-c19579a786e4
# ╟─3cec3062-d294-11ef-3dd6-bfc5588bdf1f
# ╟─fa1d5123-db02-4fda-93d2-3e5e2efed515
# ╟─3cec43d4-d294-11ef-0a9f-43eb506527a6
# ╟─3cec5b96-d294-11ef-39e0-15e93768d2b1
# ╟─3cec86cc-d294-11ef-267d-7743fd241c64
# ╠═40fddaa0-cb9d-4873-b2f6-3fd2a742ecd2
# ╟─3cec9250-d294-11ef-01ac-9d94676a65a3
# ╟─3cecbc46-d294-11ef-24cb-2d9e41fb35d9
# ╟─3cecdb48-d294-11ef-20a1-1df2731ac57c
# ╟─3ced0d0c-d294-11ef-3000-7b63362a2351
# ╠═69555c82-a593-4fe5-97e2-8c6898253991
# ╟─3ced29ae-d294-11ef-158b-09fcdaa47d1c
# ╟─3ced3fc2-d294-11ef-3fac-d5e80eacc488
# ╠═61741b02-f407-4a66-be00-28f92896490e
# ╠═81c85536-2ac9-4c65-9c34-f4661a2a796b
# ╟─3ced567c-d294-11ef-2657-df20e23a00fa
# ╠═d5a490a9-be04-41bf-b59b-fdeaba999073
# ╟─3ced6df4-d294-11ef-1091-474e512d605c
# ╠═0bdc66e2-7e66-44b9-9685-0174dce20b94
# ╟─3ced839a-d294-11ef-3dd0-1f8c5ef11b75
# ╠═f3cff8d8-1b49-4479-be78-ee632233be5c
# ╠═442f9293-319e-4d43-8890-64b1c5a2a118
# ╠═a5d43e01-8f73-4c48-b565-f10eb807a9ab
# ╟─3ced947a-d294-11ef-0403-512f2407a2d2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
