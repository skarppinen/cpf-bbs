include("../../../config.jl");
include("script-config.jl");
config_name = "block-metrics-sys-lgcpr";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[config_name]); # Get command line arguments.

## Simulation settings.
npar = args["npar"];
nreps = args["nreps"]; # How many times to repeat simulation.
blocksizes = args["blocksizes"]; # Size of blocks in model time.
verbose = args["verbose"];
outfolder = args["outfolder"];
data_seed = args["seed"];
jobid = args["jobid"]; # To avoid name clashes.
isnothing(outfolder) && (outfolder = pwd();)
verbose && println("Arguments are $args");

experiment_name = config_name;

using Statespace
using NLGSSM
using Distributions
using StatsBase: autocor
using JLD2
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));

## Load models.
include(joinpath(MODELS_PATH, "LGCP_REFLECTED", "LGCP_REFLECTED_model.jl"));
include(joinpath(MODELS_PATH, "BM_KALMAN", "BM_KALMAN_model.jl"));
pf_str =  "LGCPR_BM_PF";
kalman_str = "BM_MODEL";

# Pointers to models.
eval(Meta.parse("model_pf = $pf_str"));
eval(Meta.parse("model_kalman = $kalman_str"));

# Resampling.
resampling = get_resampling("systematic", npar);

## Load dataset.
using Random, JLD2
datafilepath = joinpath(INPUT_PATH, "lgcpr-data", string(data_seed) * ".jld2");
if !isfile(datafilepath)
    msg = "could not find datafile $datafilepath, run script `run-sim-lgcpr-data.jl` first"
    throw(ArgumentError(msg));
end
jldopen(datafilepath, "r") do file
    global data = file["data"];
    global Δ = file["args"]["base-dt"];
    global T = file["args"]["T"];
    global θ = file["theta"];
end;
bounds = (a = data.a, b = data.b);
if verbose
    println("Found datafile $datafilepath");
    println("Base Δ is $Δ")
    println("T is $T")
    println("Parameters are $θ")
end

# Check input blocksizes.
@assert issorted(blocksizes) "the input `blocksizes = $blocksizes` are not sorted.";
for blocksize in blocksizes
    try
        convert(Int, blocksize / Δ)
    catch InexactError
        local msg = "Base Δ = $Δ of datafile must divide `blocksize = $blocksize` evenly";
        throw(ArgumentError(msg));
    end
    try
        convert(Int, T / blocksize);
    catch InexactError
        local msg = "`blocksize = $blocksize` must divide `T` = $T of datafile evenly";
        throw(ArgumentError(msg));
    end
end
state_var_times = vcat([0.0], cumsum(data.dt)[1:(end - 1)]); # Times of state variables.
n_timepoints = length(state_var_times); # How many time points.

## Run kalman filter and smoother to easily obtain the M(x_u | x_l) distributions
# later for given blocking.
kalman_instance = let
    state = NLGSSMState(Float64; obsdim = model_kalman.obsdim,
                        statedim = model_kalman.statedim, n_obs = n_timepoints);
    instance = SSMInstance(model_kalman, state, data, n_timepoints);
    filter!(instance, θ, nodata = true);
    smooth!(instance, θ);
    instance;
end

## Create storage object for particle filtering.
ps = ParticleStorage(LGCPRParticle, npar, n_timepoints);
pf_instance = SSMInstance(model_pf, ps, data, n_timepoints);

## Allocate outputs.
#block_essprop_m = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
P = zeros(n_timepoints, nreps);
lbsp_M = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
npar_tmp = zeros(npar);
const X = pf_instance.storage.X; # Pointer to particles.
const W = pf_instance.storage.W; # Pointer to unnormalised log weights.
const ref = pf_instance.storage.ref; # Pointer to reference trajectory indices.
pf_runs = 0;

## Main loop.
for j in 1:nreps
    t1 = time();

    # Run particle filter and ancestor tracing.
    # (ensuring at least one pos weight at each time point, model has regions of zero potential)
    tries = pf_ensure_pos_weights!(pf_instance, θ, resampling = resampling);
    global pf_runs += tries;
    traceback!(pf_instance, AncestorTracing); # NOTE: Modifies `ref`.

    # Compute resampling rate / npar at each timepoint.
    for i in 1:n_timepoints
        # NOTE: Using npar_tmp as a temp here.
        npar_tmp .= view(W, :, i);
        normalise_logweights!(npar_tmp);
        @inbounds P[i, j] = sys_res_p(npar_tmp); # For metric based on resampling rate.
    end

    # Compute heuristic metrics for each candidate blocksize.
    for (h, blocksize) in enumerate(blocksizes)
        nblocks = convert(Int, T / blocksize);
        utimes = blocksize * collect(1:nblocks); # The desired u times.
        block_u_indices = map(ut -> searchsortedlast(state_var_times, ut), utimes)
        blocks = block_bounds_from_u_indices(block_u_indices);
        #blocks = get_even_bbcpf_blocking(dt, blocksize, T);
        check_blocking_wrt_dts(blocks, data.dt, blocksize);
        @assert blocks[end][2] == n_timepoints "something wrong. `blocks` has wrong end point.";

        # Compute all M(x_u | x_l) distributions.
        block_step_dists = build_jump_dists_from_ks!(kalman_instance, θ, blocks);
        for (i, b) in enumerate(blocks)
            l, u = b;
            # Compute block-ESS-M.
            for k in Base.OneTo(npar)
                npar_tmp[k] = logpdf(block_step_dists[i], SVector{1, Float64}(X[ref[u], u]),
                                                          SVector{1, Float64}(X[k, l]));
            end
            #block_essprop_m[h][i, j] = ess_unnorm_log(npar_tmp) / npar;

            # Compute lower boundary switch probability based on M(u | l).
            normalise_logweights!(npar_tmp);
            lbsp_M[h][i, j] = 1.0 - npar_tmp[ref[l]];
        end
    end
    t2 = time();
    if verbose
        msg = string("Finished one repetition in ", round(t2 - t1, digits = 3), "s.",
                     " (with ", tries, " PF iterations)")
        println(msg);
    end
end

## Get best inhomog. blocking.
block_eff_data, opt_blocks, opt_blocksizes = let
    d = build_eff_dataframe(P, lbsp_M; npar = npar,
                            blocksizes = blocksizes, state_var_times = state_var_times);
    blocking = determine_dyadic_blocking(d, maximum);
    blocks = let
        u = map(x -> x[2], blocking);
        u_indices = map(utime -> searchsortedlast(state_var_times, utime), u)
        block_bounds_from_u_indices(u_indices);
    end
    blocksizes_in_result = map(last, blocking);
    check_blocking_wrt_dts(blocks, data.dt, blocksizes_in_result);

    d, blocks, blocksizes_in_result;
end

## Save data.
folderpath = joinpath(outfolder, experiment_name);
mkpath(folderpath);
filename = string("jobid-", jobid, "-", randstring(20), ".jld2");
filepath = joinpath(folderpath, filename);

verbose && print("Saving output...")
t1 = time();
jldopen(filepath, "w") do file
    file["args"] = args;
    file["theta"] = θ;
    file["state-var-times"] = state_var_times;
    #file["block-essprop-M"] = block_essprop_m;
    file["block-eff-data"] = block_eff_data;
    file["opt-blocks"] = opt_blocks;
    file["opt-blocksizes"] = opt_blocksizes;
    file["lbsp-M"] = lbsp_M;
    file["P"] = P;
end
t2 = time();
verbose && println(string(" took ", round(t2 - t1, digits = 3), "s."))
verbose && println("Done.")
