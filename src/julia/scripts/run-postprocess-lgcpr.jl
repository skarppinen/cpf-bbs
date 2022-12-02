include("../../../config.jl");
import ArgParse
s = ArgParse.ArgParseSettings();
ArgParse.@add_arg_table! s begin
    "files"
        nargs = '*'
        arg_type = String
        required = true
    "--relpath", "-o"
        help = "folder path (relative) to folder where `files` are, where output should be placed"
        arg_type = String
        default = "summaries"
    "--verbose"
        help = "display messages?"
        action = :store_true
end;
args = ArgParse.parse_args(ARGS, s);
filenames = args["files"];
rel_path = args["relpath"];
verbose = args["verbose"];
verbose && println("Files to process are: $filenames");

include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));
using JLD2
using StatsBase: autocor

"""
Function writes a summary of file pointed by `ifile` to `ofile`.
"""
function summarise!(ofile, ifile)
    for stat in [iactBM, iactSV]# neffBM, neffSV]
        ofile[string(stat)] = mapslices(stat, ifile["s"], dims = 2)[:, 1];
    end
    #ofile["iact"] = mapslices(iact, ifile["s"], dims = 2)[:, 1];
    #ofile["neff"] = mapslices(neff, ifile["s"], dims = 2)[:, 1];
    ac1 = mapslices(x -> autocor(x, [1])[1], ifile["s"], dims = 2)[:, 1];
    ofile["ac1"] = ac1;
    ac1_IACT = (1.0 .+ ac1) ./ (1.0 .- ac1);
    ofile["iactAC1"] = ac1_IACT;
    #ofile["esjd"] = mapslices(esjd, ifile["s"], dims = 2)[:, 1];

    for key in ["args", "theta", "data", "intensity", "true-state", "tau",
                "local-ref-changes", "blocks", "state-var-times",
                "blocking-nsim"]
        if key in keys(ifile)
            ofile[key] = ifile[key];
        end
    end
    nothing;
end

input_folderpath = pwd();
verbose && println("Looking for files in folder $input_folderpath");
output_folderpath = abspath(joinpath(input_folderpath, rel_path))
verbose && println("Output folderpath is $output_folderpath")
verbose && println()
verbose && println("Starting processing..")
for filename in filenames
    input_filepath = joinpath(input_folderpath, filename);
    verbose && println("Processing file $input_filepath ...")
    output_filename = splitext(filename)[1] * "-summ" * ".jld2";
    output_filepath = joinpath(output_folderpath, output_filename)
    mkpath(output_folderpath)
    jldopen(input_filepath, "r") do ifile
        jldopen(output_filepath, "w") do ofile
            summarise!(ofile, ifile);
        end
    end
    verbose && println("Saved file $output_filepath");
end
verbose && println("Finished.")
