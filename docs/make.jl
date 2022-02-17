using Documenter
using MLUtils

makedocs(
    sitename = "MLUtils",
    format = Documenter.HTML(),
    modules = [MLUtils]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
