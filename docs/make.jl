using Documenter, HelFEM

makedocs(
    sitename="HelFEM.jl",
    pages = [
        "Overview" => "index.md",
    ]
)

deploydocs(repo = "github.com/mortenpi/HelFEM.jl")
