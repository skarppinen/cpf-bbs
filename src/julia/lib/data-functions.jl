include("../../../config.jl");
using CSV, DataFrames

"""
Load Corine 2018 as ArchGDAL.IDataset.
"""
function load_corine()
    corine_filename = string("clc", 2018, "_fi20m.tif");
    corine_filepath = joinpath(INPUT_PATH, "maps", corine_filename);
    corine = ArchGDAL.read(corine_filepath);
end

"""
Load mapping from UInt8's found in the Corine 2018 raster to land type description (String).
A `Dict{UInt8, String}` is returned.
"""
function load_corine_legend(; level::Int = 1)
    code_to_landtype_map = let level = level
        corine_legend_df = CSV.read(joinpath(INPUT_PATH, "corine2018-legend.csv"), DataFrame);
        not_missing = completecases(corine_legend_df);
        codes = Int.(corine_legend_df[not_missing, "Value"]);
        level_variable = string("Level", level, "Eng");
        desc = corine_legend_df[not_missing, level_variable];
        Dict(zip(UInt8.(codes), desc));
    end
    code_to_landtype_map;
end

function candidate_blockings(T::Integer)
    @assert T > 1 "`T` should be > 1.";

    # Find p such that 2 ^ pstar + 1 <= T.
    pstar = floor(Int, log(T - 1) / log(2));

    # Define output (in order of increasing coarseness).
    out = [Vector{Tuple{Int, Int}}(undef, 0) for i in 0:pstar];

    # Compute blockings.
    for (i, p) in enumerate(0:pstar)
        blocksize = 2 ^ p;
        l = 1; u = 0;
        while l < T
            u = l + blocksize;
            push!(out[i], (l, min(u, T)));
            l = u;
        end
    end
    out;
end
