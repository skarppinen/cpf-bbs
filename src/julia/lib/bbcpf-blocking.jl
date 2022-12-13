using DataFrames

"""
Return the `even blocking` for the BBCPF given time between observations `dt`,
`blocksize` and total length of time series `T`. Note that each block update of the
BBCPF updates the indices (l, u - 1) (for each returned block by this function).
Block should be thought of in 'physical time', one block updates a time interval
[t_l, t_u), where t_l and t_u are timepoints.

Examples:
`dt` = 1.0, `blocksize` = 5.0, `T` = 10.0 returns `[(1, 6), (6, 11)]`
`dt` = 2.0, `blocksize` = 2.0, `T` = 10.0 returns `[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]`
"""
function get_even_bbcpf_blocking(dt::AFloat, blocksize::AFloat, T::AFloat)
    blocksize > T && throw(ArgumentError("`blocksize` > `T`"));
    dt > blocksize && throw(ArgumentError("`dt` > `blocksize`"));
    block_bounds_from_u_indices(
        map(x -> searchsortedlast(collect(0.0:dt:T), x), blocksize:blocksize:T)
    )
end

"""
The function takes a vector of sorted integers with the first value equal to 1,
and returns a vector of tuples that give the lower and upper bounds of
a block.
"""
function block_bounds_from_u_indices(x::AbstractVector{Int})
    @assert issorted(x) "`x` must be sorted.";
    @assert length(x) >= 1 "`x` must be of length >= 1.";
    @assert x[1] != 1 "the first element should NOT be 1, give u indices only."
    @assert length(x) == length(unique(x)) "input should not have duplicate elements";

    block_bounds = Tuple{Int, Int}[];
    push!(block_bounds, (1, x[1]));
    for i in 2:length(x)
        push!(block_bounds, (x[i - 1], x[i]));
    end
    block_bounds;
end

"""
Function checks that the blocking given by `blocks` actually has correct
blocksizes for every block in terms of `dts` which should contain the lengths of
time discretisation steps. The `length of a block` is defined as the total time
spanned by indices l:(u-1) in each block (since these are actually updated in a
block update). The last block in time is a special case, since here also the uth
index is updated.
"""
function check_blocking_wrt_dts(blocks, dts, blocksize::AbstractFloat)
    @assert blocks[end][2] == length(dts) "`blocking doesn't match length of data dts`"
    for (i, b) in enumerate(blocks)
        msg = string("block ", i, " (l = ", b[1], ", u = ", b[2], ") is not of length `blocksize`.")
        @assert sum(dts[b[1]:(b[2] - 1)]) ≈ blocksize msg
    end
    nothing;
end

function check_blocking_wrt_dts(blocks, dts, blocksizes::AbstractVector{<: AbstractFloat})
    @assert blocks[end][2] == length(dts) "`blocking doesn't match length of data dts`"
    @assert length(blocks) == length(blocksizes) "length of `blocks` doesn't match that of `blocksizes`";
    for (i, b) in enumerate(blocks)
        msg = string("block ", i, " (l = ", b[1], ", u = ", b[2], ") is not of length `blocksizes[$i]`.")
        @assert sum(dts[b[1]:(b[2] - 1)]) ≈ blocksizes[i] msg
    end
    nothing;
end

"""
Function returns a dyadic blocking of a time interval based on input data
containing

`input` should be a dataframe with columns
`blocksize`, `block_l_time` (time of block lower bound) and `eff`
The dataframe should specify what `eff` resulted
from using `blocksize` in block starting from `block_l_time`.
The variable `eff` is an efficiency measure. The argument `fun` can be used
to determine how the best efficiency measure among a set of values for the measure
can be chosen.

The output of the function is a vector of 3-tuples, each of which has the form
`(lower_time, upper_time, blocksize)`, which indicates that the third value should
be used between times `lower_time` and `upper_time`.
"""
function determine_dyadic_blocking(input::DataFrame, fun::Function = maximum)
    nm = names(input);
    @assert "blocksize" in nm
    @assert "block_l_time" in nm
    @assert "eff" in nm
    @assert is_dyadic(unique(input[!, :blocksize]))

    # For each block lower bound, get dataframe
    # that specifies which blocksize was optimal in terms of
    # column `eff`.
    gdf = groupby(input, :block_l_time)
    cdf = combine(gdf, :eff => fun => :eff_proc);
    input = leftjoin(input, cdf, on = :block_l_time) |>
                x -> filter(r -> r[:eff] == r[:eff_proc], x);
    input = select(input, [:blocksize, :block_l_time]);

    # Make decision of blocksize by processing blocksizes from largest to smallest.
    out = Vector{NTuple{3, Float64}}(undef, 0);
    while nrow(input) > 0
        cur_blocksize = maximum(input[!, :blocksize]);
        sub = filter(r -> r[:blocksize] ≈ cur_blocksize, input);
        for b in sub[:, :block_l_time]
            lt = b;
            ut = b + cur_blocksize;
            push!(out, (lt, ut, cur_blocksize));
            filter!(r -> r[:block_l_time] < lt || r[:block_l_time] >= ut, input)
        end
    end
    sort!(out, by = x -> x[2]);
end

"""
Check that the values in `x` are powers of two of each other. (from smallest to largest)
Examples:
[1.0, 2.0, 4.0] is "dyadic".
[1.0, 3.0] is not "dyadic".
[1.0] is "dyadic".
[] is not "dyadic".
[6.0, 3.0, 24.0, 12.0] is "dyadic".

NOTE: Copies and sorts its output.
"""
function is_dyadic(x::AVec{<: Real})
    x = sort(x);
    isempty(x) && (return false;)
    min_value = log2(minimum(x))
    max_value = log2(maximum(x))
    for log_bs in min_value:1.0:max_value
        i = findfirst(v -> v ≈ 2 ^ log_bs, x);
        isnothing(i) && (return false;)
    end
    true;
end

function build_eff_dataframe(P, lbsp_m;
                             npar::Integer,
                             blocksizes::AbstractVector{<: AbstractFloat},
                             state_var_times::AbstractVector{<: AbstractFloat})
    @assert length(blocksizes) == length(lbsp_m)
    out = DataFrame();
    T = state_var_times[end];
    p_mult = npar / ((npar - 1) ^ 2);
    for (i, blocksize) in enumerate(blocksizes)
         nblocks = convert(Int, T / blocksize);
         utimes = blocksize * collect(1:nblocks);
         block_u_indices = map(ut -> searchsortedlast(state_var_times, ut), utimes);
         blocks = block_bounds_from_u_indices(block_u_indices);
         block_l_time = state_var_times[map(first, blocks)];
         mean_plu_est = map(enumerate(blocks)) do jb
             j, b = jb;
             l, u = b;
             lbsp_m_reps = lbsp_m[i][j, :];
             P_reps = exp.(mapslices(x -> sum(log.(1.0 .- p_mult * x)),
                           view(P, l:(u - 1), :), dims = 1)[1, :]);
             mean(lbsp_m_reps .* P_reps);
        end
        append!(out,
            DataFrame(
                block_l_time = block_l_time,
                blocksize = blocksize,
                eff = mean_plu_est
            )
        );
    end
    out;
end
