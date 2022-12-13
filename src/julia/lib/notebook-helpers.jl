using RCall, DataFrames, JLD2

function set_plot_size!(width, height)
    RCall.rcall_p(:options, rcalljl_options = Dict(:width => width, :height => height))
end

"""
Return one DataFrame resulting from applying `f` to results in each
JLD2 file in `filepaths_jld2` and concatenating the results.
`f` should be a function taking a handle to a JLD2 file and returning
a DataFrame.
"""
function reduce_to_dataframe(filepaths_jld2, f::F) where F
    out = DataFrame();
    for filepath in filepaths_jld2
        d = jldopen(filepath, "r") do file
            f(file)
        end
        append!(out, d; cols = :setequal);
    end
    out;
end

"""
Return one DataFrame resulting from applying `f` to results in each
JLD2 file in `filepaths_jld2` and concatenating the results.
`f` should be a function taking a handle to a JLD2 file and returning
a DataFrame.
"""
function reduce_to_dataframe(f::F, filepaths_jld2) where F
    out = DataFrame();
    for filepath in filepaths_jld2
        d = jldopen(filepath, "r") do file
            f(file)
        end
        append!(out, d; cols = :setequal);
    end
    out;
end
