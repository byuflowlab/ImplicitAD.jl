using ImplicitAD
using Documenter

DocMeta.setdocmeta!(ImplicitAD, :DocTestSetup, :(using ImplicitAD); recursive=true)

makedocs(;
    modules=[ImplicitAD],
    authors="Andrew Ning <aning@byu.edu> and contributors",
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
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/byuflowlab/ImplicitAD.jl",
    devbranch="main",
)
