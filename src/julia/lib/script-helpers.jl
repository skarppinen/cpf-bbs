include("../../../config.jl");

function get_resampling(resampling_str::AString, npar::Integer)
    msg = "`resampling` should be one of `('killing', 'multinomial', 'systematic')`"
    @assert resampling_str in ("killing", "multinomial", "systematic") msg
    choices = Dict("multinomial" => MultinomialResampling(npar; randomisation = :shuffle),
                   "killing" => KillingResampling(npar; randomisation = :circular),
                   "systematic" => SystematicResampling(npar; order = :partition, randomisation = :circular)
                   );
    resampling = choices[resampling_str];
    @assert has_conditional(resampling) "`resampling = $resampling_str` does not implement conditional resampling";
    resampling;
end

function report_progress(i, total, time_sec; digits::Int = 3)
    time_formatted = round(time_sec, digits = digits);
    println(string("Finished iteration ", i, " / ", total, ", took ", time_formatted, "s."));
end

"""
`/path/to/file.zip` returns `/path/to/file`
`/path/to/file` returns `/path/to/file`
`/path/to/file.tar.gz` returns `/path/to/file`
`file` returns `file`
"""
function drop_postfix(path::AbstractString)
    pre = path;
    while true
        split = splitext(pre);
        pre = split[1];
        split[2] == "" && break;
    end
    pre
end
