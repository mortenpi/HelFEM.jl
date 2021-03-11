using Documenter
using HelFEM
using Plots

makedocs(
    sitename="HelFEM.jl",
    pages = [
        "Overview" => "index.md",
        "polynomial.md",
        "radialbasis.md",
        "FEM basis" => "fembasis.md",
        "utilities.md",
    ],
)

deploydocs(repo = "github.com/mortenpi/HelFEM.jl")
