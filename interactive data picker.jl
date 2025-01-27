### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 62abfc12-dcab-11ef-3ee6-1991c0729d7e
using HypertextLiteral

# ╔═╡ edbcaefa-84ab-4658-98d0-3e2bed6c5488
function data_picker_1D(initial_data::Vector{<:Real}=Float64[]; xlim=(0,100))
@htl """
    <script id="yolo">

	

    const Plot = this?.plotlib ?? await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");
	const div = this ?? document.createElement("div")

	const initial_data = $(initial_data)

	let data = [...initial_data]
	const reset_button = document.createElement("input")
	reset_button.type = "button"
	reset_button.value = "Reset"

	let plot = undefined


	const radius = 7.0

	const render = () => {
		
		plot = Plot.plot({
		  height: 160,
		  marks: [
		    Plot.dotX(data, Plot.dodgeY({x: (el=>el), r: radius, title: "name", fill: "currentColor", padding:0}))
		  ],
			x: {
				domain: $(xlim),
			}
		})

	    div.innerHTML = ""
		div.value = data
	    div.append(plot);
		div.append(reset_button)
	}

render()


div.addEventListener("click", (e) => {
	// if(e.target !== plot) return
	if(!plot.contains(e.target)) return

	const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(plot.getScreenCTM().inverse());

	const domain = plot.scale("x").domain

	const randn = (Math.random() + Math.random() + Math.random() - Math.random() - Math.random() - Math.random()) / 1.7
	
	const noise = 0 * randn * radius

	const clicked = plot.scale("x").invert(p.x + noise)

	data.push(clicked + noise)
	render()
	div.dispatchEvent(new CustomEvent("input"))
})


reset_button.addEventListener("click", (e) => {
console.log(123)
	data = [...initial_data]

	render()
	div.dispatchEvent(new CustomEvent("input"))
})


    div.plotlib = Plot
    return div

    </script>
"""
end

# ╔═╡ fcc155ed-6176-4223-843a-9dc905d339bc
@bind yollo data_picker_1D()

# ╔═╡ 1d766b00-22c2-4a12-a663-d37b3c8bdbb2
yollo

# ╔═╡ 5840dc91-ea47-4410-a8a6-19796c797167
function data_picker_2D(initial_data::Vector{<:Real}=Float64[]; xlim=(0,100), ylim=(0,100))
@htl """
    <script id="yolo">

	

    const Plot = this?.plotlib ?? await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");
	const div = this ?? document.createElement("div")

	const initial_data = $(initial_data)

	let data = [...initial_data]
	const reset_button = document.createElement("input")
	reset_button.type = "button"
	reset_button.value = "Reset"

	let plot = undefined


	const radius = 7.0

	const render = () => {
		
		plot = Plot.plot({
		  marks: [
		    Plot.dot(data, { r: radius, title: "name", fill: "currentColor", padding:0})
		  ],
			x: {
				domain: $(xlim),
			},
			y: {
				domain: $(ylim),
			},
		})

	    div.innerHTML = ""
		div.value = data
	    div.append(plot);
		div.append(reset_button)
	}

render()


div.addEventListener("click", (e) => {
	// if(e.target !== plot) return
	if(!plot.contains(e.target)) return

	const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(plot.getScreenCTM().inverse());


	data.push([plot.scale("x").invert(p.x), plot.scale("y").invert(p.y)])
	render()
	div.dispatchEvent(new CustomEvent("input"))
})


reset_button.addEventListener("click", (e) => {
console.log(123)
	data = [...initial_data]

	render()
	div.dispatchEvent(new CustomEvent("input"))
})


    div.plotlib = Plot
    return div

    </script>
"""
end

# ╔═╡ 541901aa-9ab5-4035-a817-ba82a04ea15e
@bind aaa data_picker_2D()

# ╔═╡ 1692b522-296f-4591-9985-357a8c6c893e
aaa

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"

[compat]
HypertextLiteral = "~0.9.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "60bb17116de90af65672383384a5610ccd921125"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"
"""

# ╔═╡ Cell order:
# ╠═62abfc12-dcab-11ef-3ee6-1991c0729d7e
# ╠═1d766b00-22c2-4a12-a663-d37b3c8bdbb2
# ╠═fcc155ed-6176-4223-843a-9dc905d339bc
# ╟─edbcaefa-84ab-4658-98d0-3e2bed6c5488
# ╠═1692b522-296f-4591-9985-357a8c6c893e
# ╠═541901aa-9ab5-4035-a817-ba82a04ea15e
# ╠═5840dc91-ea47-4410-a8a6-19796c797167
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
