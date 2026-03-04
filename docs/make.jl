using Documenter: Documenter, DocMeta, deploydocs, makedocs
using FusionTensors: FusionTensors
using ITensorFormatter: ITensorFormatter

DocMeta.setdocmeta!(FusionTensors, :DocTestSetup, :(using FusionTensors); recursive = true)

ITensorFormatter.make_index!(pkgdir(FusionTensors))

makedocs(;
    modules = [FusionTensors],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "FusionTensors.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/FusionTensors.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/FusionTensors.jl", devbranch = "main", push_preview = true
)
