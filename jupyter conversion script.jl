### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 77cabc8b-29b8-4635-a5fd-d0c48240249e
using JupyterPlutoConverter

# ╔═╡ 78a608ef-5546-4ee2-a74a-f77e3f6b9f0f
using Markdown: LaTeX

# ╔═╡ 259172cb-f835-4408-9404-c9dda6bcae14
file = "/Users/fons/Documents/BMLIP/lessons/notebooks/The-Gaussian-Distribution.ipynb"

# ╔═╡ 5bdace51-e1a7-4eb8-a2fd-b044fe8cce9d


# ╔═╡ 9359d000-6ff7-4161-b164-dc9b619f4947
s = raw"""### The Gaussian Distribution 

- Consider a random (vector) variable $x \in \mathbb{R}^M$ that is "normally" (i.e., Gaussian) distributed. The _moment_ parameterization of the Gaussian distribution is completely specified by its _mean_ $\mu$ and _variance_ $\Sigma$ and given by
$$
p(x | \mu, \Sigma) = \mathcal{N}(x|\mu,\Sigma) \triangleq \frac{1}{\sqrt{(2\pi)^M |\Sigma|}} \,\exp\left\{-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right\}\,.
$$
where $|\Sigma| \triangleq \mathrm{det}(\Sigma)$ is the determinant of $\Sigma$.  
  - For the scalar real variable $x \in \mathbb{R}$, this works out to 
$$
p(x | \mu, \sigma^2) =  \frac{1}{\sqrt{2\pi\sigma^2 }} \,\exp\left\{-\frac{(x-\mu)^2}{2 \sigma^2} \right\}\,.
$$"""

# ╔═╡ a7477c4d-34ef-4387-88b5-0fbca9761503
multiline_latex_regex = r"\$\$\n*(.*?)\n*\$\$"ms

# ╔═╡ 2522faf4-b084-4b97-b5a1-3142cdcc7400
multiline_latex_regex_replacer = s"\n\n```math\n\1\n```\n"

# ╔═╡ 5b546a5d-f090-469a-89e0-ee77f2e356e0
match(multiline_latex_regex, s)

# ╔═╡ f5087e8d-aa4d-4c8a-9aa8-379f620c78bd
replace(s, multiline_latex_regex => multiline_latex_regex_replacer)

# ╔═╡ 9a2cf635-c10c-4491-97ac-05eda9f5e380
basename(file)

# ╔═╡ 9d2c5770-b7a7-11ef-151b-c93f6b093d21
m = md"""
# asdf _s_

- Asdf ``adsf`` $\sqrt{a}$ yooo

```math
xoxoxo
```

```asdf
ff
```
"""

# ╔═╡ 0ff5c40f-96c0-47fd-bb2e-de410d04cc05

function Markdown.plain(io::IO, l::LaTeX)
    println(io, "```math")
    println(io, l.formula)
    println(io, "```")
end


# ╔═╡ 3cd1d102-e7d5-44fc-b47a-a955f18e85bc

function Markdown.plaininline(io::IO, l::LaTeX)
    print(io, "``", l.formula, "``")
end


# ╔═╡ 4847286f-c50d-463d-8929-ef6a86d4aed3
Text(m)

# ╔═╡ 13a0a367-862d-4311-aad8-f71536a30a0f
dump(m)

# ╔═╡ 3ae3d6a6-c6c0-4f9c-8302-98d226243ee9
md"""
# Removing the loplevel lists
"""

# ╔═╡ e0272eb8-86dc-458d-80e6-56f9940ead04
function remove_toplevel_lists(md::Markdown.MD)
	Iterators.flatmap(md.content) do c
		if c isa Markdown.List
			c.items
		else
			[c]
		end


		
	end |> collect |> Markdown.MD

end

# ╔═╡ 72c060ae-714b-44ff-be14-c7ff8faa1d85
function transform_md(str::String)
	# removing the md""" prefix and suffix
	md_str = replace(str, r"(^\s*md\"\"?\"?)|(\"\"?\"?\s*$)" => "")

	# fix the $$ multiline latex (which is not fully supported by Julia Markdown)
	md_str = replace(md_str, multiline_latex_regex => multiline_latex_regex_replacer)

	md = Markdown.parse(md_str)

	md = remove_toplevel_lists(md)
	s = string(md)
		
	@info "yay" md Text(md_str) Text(str) Text(s)

	"md\"\"\"\n$(s)\n\"\"\""
end

# ╔═╡ 453bfe18-088c-4683-a0aa-c8cc04236f09
jupyter2pluto(file; overwrite=true, transform_md)

# ╔═╡ bbbff55c-eec8-4ff1-939f-08e22971c6de
m1 = md"""
- ``\Rightarrow`` Note that Bayesian inference is trivial in the [_canonical_ parameterization of the Gaussian](#natural-parameterization), where we would get
$$\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}$$
  - This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 
""" 

# ╔═╡ de754bcf-062a-4522-ae08-52086f603ed0
remove_toplevel_lists(m1)

# ╔═╡ 02c0eb08-4b65-4435-8b6a-f81e67e8e6d1
dump(m1d)

# ╔═╡ 66500acb-00ae-4538-9931-147ae28f03d4


# ╔═╡ 1e16e207-14ab-48e2-8da6-007ea00afa4c
m2 = md"""
- $\Rightarrow$ Note that Bayesian inference is trivial in the [_canonical_ parameterization of the Gaussian](#natural-parameterization), where we would get
$$\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}$$
  - This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 
""" 

# ╔═╡ 1de8add3-9a6b-4ea0-b514-ab6bf6f48fdc
m3 = md"""
- a
- b
  - c
  - d
e
- f
""" 

# ╔═╡ 41900791-87b5-4bc6-8e76-1c82a4fe1a6e
dump(m3)

# ╔═╡ Cell order:
# ╠═77cabc8b-29b8-4635-a5fd-d0c48240249e
# ╠═259172cb-f835-4408-9404-c9dda6bcae14
# ╠═5bdace51-e1a7-4eb8-a2fd-b044fe8cce9d
# ╠═72c060ae-714b-44ff-be14-c7ff8faa1d85
# ╠═9359d000-6ff7-4161-b164-dc9b619f4947
# ╠═a7477c4d-34ef-4387-88b5-0fbca9761503
# ╠═2522faf4-b084-4b97-b5a1-3142cdcc7400
# ╠═5b546a5d-f090-469a-89e0-ee77f2e356e0
# ╠═f5087e8d-aa4d-4c8a-9aa8-379f620c78bd
# ╠═453bfe18-088c-4683-a0aa-c8cc04236f09
# ╠═9a2cf635-c10c-4491-97ac-05eda9f5e380
# ╠═9d2c5770-b7a7-11ef-151b-c93f6b093d21
# ╠═78a608ef-5546-4ee2-a74a-f77e3f6b9f0f
# ╠═0ff5c40f-96c0-47fd-bb2e-de410d04cc05
# ╠═3cd1d102-e7d5-44fc-b47a-a955f18e85bc
# ╠═4847286f-c50d-463d-8929-ef6a86d4aed3
# ╠═13a0a367-862d-4311-aad8-f71536a30a0f
# ╟─3ae3d6a6-c6c0-4f9c-8302-98d226243ee9
# ╠═e0272eb8-86dc-458d-80e6-56f9940ead04
# ╠═de754bcf-062a-4522-ae08-52086f603ed0
# ╠═bbbff55c-eec8-4ff1-939f-08e22971c6de
# ╠═02c0eb08-4b65-4435-8b6a-f81e67e8e6d1
# ╠═66500acb-00ae-4538-9931-147ae28f03d4
# ╠═1e16e207-14ab-48e2-8da6-007ea00afa4c
# ╠═41900791-87b5-4bc6-8e76-1c82a4fe1a6e
# ╠═1de8add3-9a6b-4ea0-b514-ab6bf6f48fdc
