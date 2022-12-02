include("../../../config.jl");
import ArgParse

s = ArgParse.ArgParseSettings();
ArgParse.@add_arg_table! s begin
    "--inputfolder"
        help = string("input file")
        arg_type = String
        default = joinpath(OUTPUT_PATH, "simulation-experiments", "block-metrics-sys-lgcpr")
    "--outfolder"
        help = "folder where to place results"
        default = joinpath(INPUT_PATH, "lgcpr-data")
        arg_type = String
    "--verbose"
        help = "display messages?"
        action = :store_true
end;
args = ArgParse.parse_args(ARGS, s);
inputfile = let
    folder = args["inputfolder"]; 
    file = only(readdir(folder));
    joinpath(folder, file);
end
outfolder = args["outfolder"];
verbose = args["verbose"];
if !isfile(inputfile)
    msg = "Invalid `inputfile` $inputfile: not found";
    throw(ArgumentError(msg));
else
    verbose && println("Found file $inputfile");
end
include(joinpath(LIB_PATH, "bbcpf-blocking.jl"));
using JLD2, DataFrames
using StatsBase: mean

jldopen(inputfile, "r") do file
    global DATASEED = file["args"]["seed"];
    global opt_blocks = file["opt-blocks"];
    global opt_blocksizes = file["opt-blocksizes"];
end
datadts = jldopen(joinpath(INPUT_PATH, "lgcpr-data", string(DATASEED) * ".jld2"), "r") do file
    file["data"].dt
end

mkpath(outfolder);
filename = "blocking-" * string(DATASEED) * "-" * "cpf-metrics" * ".jld2";
outfilepath = joinpath(outfolder, filename);
verbose && println("Saving file $outfilepath");
jldopen(outfilepath, "w") do file
    file["seed"] = DATASEED;
    file["dts"] = datadts;
    file["blocks"] = opt_blocks;
    file["blocksizes"] = opt_blocksizes;
end
