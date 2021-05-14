using Documenter, Literate
using HelFEM
using DataFrames, DataFramesMeta, Plots, StatsPlots

Literate.markdown(
    joinpath(@__DIR__, "..", "examples", "particleinabox.jl"),
    joinpath(@__DIR__, "src", "examples"),
    documenter = true,
)

makedocs(
    sitename="HelFEM.jl",
    pages = [
        "Overview" => "index.md",
        "polynomial.md",
        "radialbasis.md",
        "FEM basis" => "fembasis.md",
        "utilities.md",
        "Examples" => [
            "examples/particleinabox.md",
        ],
    ],
)

deploydocs(repo = "github.com/mortenpi/HelFEM.jl", push_preview=true)
