## Initialisation script.
using Pkg
Pkg.activate(@__DIR__, io = Base.DevNull());

## Some constants.
const AVec = AbstractVector;
const AMat = AbstractMatrix;
const AArr = AbstractArray;
const AFloat = AbstractFloat;
const AString = AbstractString;

const PROJECT_ROOT = @__DIR__;
const SRC_PATH = joinpath(PROJECT_ROOT, "src");
const LIB_PATH = joinpath(SRC_PATH, "julia", "lib");
const MODELS_PATH = joinpath(SRC_PATH, "julia", "models");
const SCRIPTS_PATH = joinpath(SRC_PATH, "julia", "scripts");
const OUTPUT_PATH = joinpath(PROJECT_ROOT, "output");
const INPUT_PATH = joinpath(PROJECT_ROOT, "input");
const SIMEXP_PATH = joinpath(OUTPUT_PATH, "simulation-experiments");

push!(LOAD_PATH, LIB_PATH); # Add possibility to load local modules.
