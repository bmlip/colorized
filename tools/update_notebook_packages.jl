if !isdir("pluto-slider-server-environment") || length(ARGS) != 1
    error("""
    Run me from the root of the repository directory, using:

    julia tools/update_notebook_packages.jl <level>
    
    Where <level> is one of: PATCH, MINOR, MAJOR
    """)
end

if !(v"1.11.0-aaa" < VERSION < v"1.12.0")
    error("Our notebook package environments need to be updated with Julia 1.11. Go to julialang.org/downloads to install it.")
end

import Pkg
Pkg.activate("./pluto-slider-server-environment")
Pkg.instantiate()

import Pluto

flatmap(args...) = vcat(map(args...)...)


getfrom(dir) = flatmap(walkdir("lectures")) do (root, _dirs, files)
    joinpath.((root,), files)
end

all_files_recursive = [getfrom("lectures")..., getfrom("minis")...]

all_notebooks = filter(Pluto.is_pluto_notebook, all_files_recursive)

level = getfield(Pkg, Symbol("UPLEVEL_$(ARGS[1])"))

for n in all_notebooks
    @info "Updating" n
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
    Pluto.update_notebook_environment(n; backup=false, level)
end

@info "All notebooks done!"
