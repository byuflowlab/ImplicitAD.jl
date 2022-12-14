using ImplicitAD
using Documenter

DocMeta.setdocmeta!(ImplicitAD, :DocTestSetup, :(using ImplicitAD); recursive=true)

makedocs(;
    modules=[ImplicitAD],
    authors="Andrew Ning <aning@byu.edu> and contributors",
    repo="https://github.com/byuflowlab/ImplicitAD.jl/blob/{commit}{path}#{line}",
    sitename="ImplicitAD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://byuflowlab.github.io/ImplicitAD.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "reference.md",
        "Theory" => "theory.md",
    ],
)

deploydocs(;
    repo="github.com/byuflowlab/ImplicitAD.jl",
    devbranch="main",
)
