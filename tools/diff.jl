
source = joinpath(@__DIR__, "..")
cd(source)

using Pkg
Pkg.activate(joinpath(source, "pluto-slider-server-environment"))
Pkg.instantiate()

using PlutoNotebookComparison








# you can test me by setting 
function drama_broken_link(di::DramaContext)
	for (cell_id, cell) in di.new_state["cell_results"]
		b = string(cell["output"]["body"])

		href_pattern = r"<a\s+[^>]*?href=['\"](.*?)['\"]"
	    for m in eachmatch(href_pattern, b)
	        href = m.captures[1]
			@info "Found href" href
		end
	end
end








sources_old = [
# PSSCache("pluto_state_cache")
# WebsiteDir("gh_pages_dir")
# WebsiteAddress("https://bmlip.github.io/colorized/")
SafePreview()
]

sources_new = [
# PSSCache("pluto_state_cache")
# RunWithPlutoSliderServer()
SafePreview()
]

drama_checkers = Function[
    drama_broken_link,
]

PlutoNotebookComparison.compare_PR(source;
sources_old,
sources_new,
drama_checkers,
)