### A Pluto.jl notebook ###
# v0.20.15

#> [frontmatter]
#> description = "Course Syllabus"
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ f96d047f-9efa-4889-8b4e-a8d96677d072
using PlutoUI, PlutoTeachingTools

# ╔═╡ 0cfd4bc0-d294-11ef-3537-630954a9dd27
md"""
# 5SSD0 Course Syllabus

"""

# ╔═╡ 467c8189-b5d3-4eaf-8886-6ae53136dd8f
PlutoUI.TableOfContents()

# ╔═╡ 0cffef7e-d294-11ef-3dd5-1fd862260b70
md"""
## Learning Goals

This course provides an introduction to Bayesian machine learning and information processing systems. The Bayesian approach affords a unified and consistent treatment of many useful information processing systems. 

Upon successful completion of the course, students should be able to:

  * understand the essence of the Bayesian approach to information processing.
  * specify a solution to an information processing problem as a Bayesian inference task on a probabilistic model.
  * design a probabilistic model by a specifying a likelihood function and prior distribution;
  * Code the solution in a probabilistic programming package.
  * execute the Bayesian inference task either analytically or approximately.
  * evaluate the resulting solution by examination of Bayesian evidence.
  * be aware of the properties of commonly used probability distribitions such as the Gaussian, Gamma and multinomial distribution; models such as hidden Markov models and Gaussian mixture models; and inference methods such as the Laplace approximation, variational Bayes and message passing in a factor graph.

"""

# ╔═╡ 0d013750-d294-11ef-333c-d9eb7578fab2
md"""
## Entrance Requirements (pre-knowledge)

Undergraduate courses in Linear Algebra and Probability Theory (or Statistics). 

Some scientific programming experience, eg in MATLAB or Python. In this class, we use the [Julia](https://julialang.org/) programming language, which has a similar syntax to MATLAB, but is (close to) as fast as C. 

"""

# ╔═╡ 0d0142b6-d294-11ef-0297-e5bb923ad942
md"""
## Important Links

Please bookmark the following three websites:

1. The course homepage [http://bmlip.nl](https://biaslab.github.io/teaching/bmlip/) (or try [https://biaslab.github.io/teaching/bmlip](https://biaslab.github.io/teaching/bmlip/) ) contains links to all materials such as lecture notes and video lectures.
2. The [Piazza course site](https://piazza.com/tue.nl/winter2025/5ssd0/home) will be used for Q&A and communication.
3. The [Canvas course site](https://canvas.tue.nl/courses/30024) will be sparingly used for communication (mostly by ESA staff)

"""

# ╔═╡ 0d015ab4-d294-11ef-2e53-5339062c435c
md"""
## Materials

All materials can be accessed from the [course homepage](https://biaslab.github.io/teaching/bmlip).

Materials consist of the following resources:

  * Mandatory

      * Lecture notes
      * Probabilistic Programming (PP) notes
      * The lecture notes and probabilistic programming notes contain the mandatory materials. Some lecture notes are extended by a reading assignment, see the first cell in the lecture notes. These reading assignment are also part of the mandatory materials.
  * Optional materials to help understand the lecture and PP notes

      * video recordings of the Q2-2023 lecture series
      * exercises
      * Q&A at Piazza
      * practice exams



Source materials are available at github repo at [https://github.com/bmlip/course](https://github.com/bmlip/course). You do not need to bother with this site. If you spot an error in the materials, please raise the issue at Piazza.  

"""

# ╔═╡ 0d016cf8-d294-11ef-0c84-336979a02dd7
md"""
## Study Guide

Slides that are not required for the exam are moved to the end of the notes and preceded by an [OPTIONAL SLIDES](#optional) header.

<p style="color:red">Please study the lecture notes before you come to class!!</p> 

Optionally, you can view the video recordings of the Q2-2023 lecture series for addional explanations. 

Then come to the class!

  * During the scheduled classroom meetings, I will not teach all materials in the lecture notes.
  * Rather, I will first discuss a summary of the lecture notes and then be available for any additional questions that you may still have.

Still got any sticky issues regarding the lecture notes?

  * Pose you question at the **Piazza site**!
  * Your questions will be answered at the Piazza site by fellow students and accorded (or corrected) by the teaching staff.

Each class also comes with a set of exercises. They are often a bit challenging and test more of your quantitative skills than you will need for the exam. When doing exercises, feel free to make use of Sam Roweis' cheat sheets for [Matrix identities](https://github.com/bmlip/course/blob/main/assets/files/Roweis-1999-matrix-identities.pdf) and [Gaussian identities](https://github.com/bmlip/course/blob/main/assets/files/Roweis-1999-gaussian-identities.pdf). Also accessible from the course homepage.   

"""

# ╔═╡ 0d017b82-d294-11ef-2d11-df36557202c9
md"""
## Piazza (Q&A)

We will be using Piazza for Q&A and news. The system is highly catered to getting you help fast and efficiently from both classmates and the teaching staff. 

[Sign up for Piazza](http://piazza.com/tue.nl/winter2025/5ssd0) today if you have not done so. And install the Piazza app on your phone! 

The quicker you begin asking questions on Piazza (rather than via emails), the quicker you'll benefit from the collective knowledge of your classmates and instructors. We encourage you to ask questions when you're struggling to understand a concept—you can even do so anonymously.

We will also disseminate news and announcements via Piazza.

Unless it is a personal issue, pose your course-related questions at Piazza (in the right folder). 

Please contribute to the class by answering questions at Piazza. 

  * If so desired, you can contribute anonymously.
  * Answering technical questions at Piazza is a great way to learn. If you really want to understand a topic, you should try to explain it to others.
  * Every question has just a single students' answer that students can edit collectively (and a single instructors’ answer for instructors).

You can use LaTeX in Piazza for math (and please do so!). 

Piazza has a great ``search`` feature. Use search before putting in new questions.

"""

# ╔═╡ 0d018ee2-d294-11ef-3b3d-e34d0532a953
md"""
## Exam Guide

The course will be scored by two programming assignments and a final written exam. See the [course homepage](https://biaslab.github.io/teaching/bmlip/) for how the final score is computed.

The written exam in multiple-choice format. 

You are not allowed to use books nor bring printed or handwritten formula sheets to the exam. Difficult-to-remember formulas are supplied at the exam sheet.

No smartphones at the exam.

The tested material consists of the mandatory lecture + PP notes (+ mandatory reading assignments as assigned in the first cell/slide of each lecture notebook).

The class homepage contains two representative practice exams from previous terms. 



"""

# ╔═╡ 0d019cde-d294-11ef-0563-6b41bc2ca80f
md"""
## Preview

Check out [a recording from last year](https://youtu.be/k9DO26O6dIg?si=b8EiK12O_s76btPn) to understand what this class will be like. 

"""

# ╔═╡ 0d01a404-d294-11ef-3fe4-df9726debd05
md"""
#  $(HTML("<span id='optional'>OPTIONAL SLIDES</span>")) 

"""

# ╔═╡ 0d01b03e-d294-11ef-3b2f-53f22689075c
md"""
## Title

The slides below the `OPTIONAL SLIDES` marker are optional for the exam.  

"""

# ╔═╡ f3b97e01-f8d2-4865-ae81-2df412f7515a
md"""
# Appendix
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoTeachingTools = "~0.4.5"
PlutoUI = "~0.7.70"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "488f17c0ecac3e1a18f0c355ec86750587e5f38a"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

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

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

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
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "85778cdf2bed372008e6646c64340460764a5b85"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fcfec547342405c7a8529ea896f98c0ffcc4931d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.70"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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
# ╟─0cfd4bc0-d294-11ef-3537-630954a9dd27
# ╟─467c8189-b5d3-4eaf-8886-6ae53136dd8f
# ╟─0cffef7e-d294-11ef-3dd5-1fd862260b70
# ╟─0d013750-d294-11ef-333c-d9eb7578fab2
# ╟─0d0142b6-d294-11ef-0297-e5bb923ad942
# ╟─0d015ab4-d294-11ef-2e53-5339062c435c
# ╟─0d016cf8-d294-11ef-0c84-336979a02dd7
# ╟─0d017b82-d294-11ef-2d11-df36557202c9
# ╟─0d018ee2-d294-11ef-3b3d-e34d0532a953
# ╟─0d019cde-d294-11ef-0563-6b41bc2ca80f
# ╟─0d01a404-d294-11ef-3fe4-df9726debd05
# ╟─0d01b03e-d294-11ef-3b2f-53f22689075c
# ╟─f3b97e01-f8d2-4865-ae81-2df412f7515a
# ╠═f96d047f-9efa-4889-8b4e-a8d96677d072
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
