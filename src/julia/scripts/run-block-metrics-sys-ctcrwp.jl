include("../../../config.jl");
include("script-config.jl");
config_name = "block-metrics-sys-ctcrwp";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[config_name]); # Get command line arguments.

## Simulation settings.
npar = args["npar"]; # Amount of particles.
blocksizes = args["blocksizes"]; # Size of blocks in model time.
T = args["T"]; # Total time of model.
dt = args["dt"]; # Time discretisation.
nreps = args["nreps"]; # How many times to repeat simulation.
par = args["par"]; # Parameters.
verbose = args["verbose"];
jobid = args["jobid"]; # To avoid name clashes.
outfolder = args["outfolder"];
isnothing(outfolder) && (outfolder = pwd();)

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

## Load model.
model = "CTCRWP_B";
pf_str = model * "_PF";
kalman_str = model * "_KALMAN";
include(joinpath(MODELS_PATH, pf_str, pf_str * "_model.jl"));
include(joinpath(MODELS_PATH, kalman_str, kalman_str * "_model.jl"));

# Pointers to models.
eval(Meta.parse("model_pf = $pf_str"));
eval(Meta.parse("model_kalman = $kalman_str"));

#par = Dict(:vmui => 0.0, :xmui => 0.0, :tau => 1.0, :sigma => 0.5, :betax => 0.5, :betav => 1.0);
θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
verbose && println(string("Got parameters ", θ));
resampling = get_resampling("systematic", npar);
verbose && println("Arguments are $args");

### Some argument checking.
@assert issorted(blocksizes) "the input `blocksizes = $blocksizes` are not sorted.";
for blocksize in blocksizes
    try
        convert(Int, T / blocksize)
    catch InexactError
        msg = "`blocksize` = $blocksize does not divide the total time series time, $T, evenly.";
        throw(ArgumentError(msg));
    end
    try
        convert(Int, blocksize / dt)
    catch InexactError
        msg = "`dt = $dt` does not divide `blocksize = $blocksize` evenly";
        throw(ArgumentError(msg));
    end
end
state_var_times = collect(0.0:dt:T); # Times of state variables in smoothing distribution.
n_timepoints = length(state_var_times); # How many time points.
data = let
    dt = vcat(diff(state_var_times), [0.0]);
    (dt = dt,);
end;

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
ps = ParticleStorage(CTCRWPBParticle, npar, n_timepoints);
pf_instance = SSMInstance(model_pf, ps, data, n_timepoints);

## Allocate outputs.
log_block_essprop_g = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
block_essprop_m = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
lbsp_M = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
P = zeros(n_timepoints, nreps);
npar_tmp = zeros(npar);
const X = pf_instance.storage.X; # Pointer to particles.
const W = pf_instance.storage.W; # Pointer to unnormalised log weights.
const ref = pf_instance.storage.ref; # Pointer to reference trajectory indices.

## Main loop.
for j in 1:nreps
    t1 = time();
    # Run particle filter and ancestor tracing.
    pf!(pf_instance, θ, resampling = resampling, conditional = false);
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
        blocks = get_even_bbcpf_blocking(dt, blocksize, T);
        nblocks = length(blocks);
        check_blocking_wrt_dts(blocks, data.dt, blocksize);
        @assert blocks[end][2] == n_timepoints "something wrong. `blocks` has wrong end point.";

        # Compute all M(x_u | x_l) distributions.
        block_step_dists = build_jump_dists_from_ks!(kalman_instance, θ, blocks);
        for (i, b) in enumerate(blocks)
            l, u = b;
            # Compute log of block-ESS-G.
            for t in l:(u - 1)
                v_logWt = view(W, :, t);
                log_block_essprop_g[h][i, j] += ess_unnorm_log(v_logWt, true) - log(npar); # Second arg = true returns log(ESS).
            end

            # Compute block-ESS-M.
            for k in Base.OneTo(npar)
                npar_tmp[k] = logpdf(block_step_dists[i], SVector{2, Float64}(X[ref[u], u]),
                                                          SVector{2, Float64}(X[k, l]));
            end
            block_essprop_m[h][i, j] = ess_unnorm_log(npar_tmp) / npar;

            # Compute lower boundary switch probability based on M(u | l).
            normalise_logweights!(npar_tmp);
            lbsp_M[h][i, j] = 1.0 - npar_tmp[ref[l]];
        end
    end
    t2 = time();
    verbose && println(string("Finished one repetition in ", round(t2 - t1, digits = 3), "s."));
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
    file["block-essprop-M"] = block_essprop_m;
    file["log-block-essprop-G"] = log_block_essprop_g;
    file["lbsp-M"] = lbsp_M;
    file["P"] = P;
end
t2 = time();
verbose && println(string(" took ", round(t2 - t1, digits = 3), "s."))
verbose && println("Done.")
