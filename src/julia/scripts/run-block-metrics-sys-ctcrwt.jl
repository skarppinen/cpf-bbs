include("../../../config.jl");
include("script-config.jl");
experiment_name = "block-metrics-sys-ctcrwt";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[experiment_name]);

## Arguments.
npar = args["npar"]; # 16
nreps = args["nreps"]; # 50
blocksizes = args["blocksizes"]; # 1.0
datafile = args["datafile"];
outfolder = args["outfolder"];
verbose = args["verbose"];
isnothing(outfolder) && (outfolder = pwd();)
outfolder = expanduser(outfolder);
datapath = expanduser(datafile);
verbose && println("Output folder is $outfolder");
@assert issorted(blocksizes)
verbose && println("Input data is fetched from $datapath");
verbose && println("Arguments are $args");

include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
using JLD2


jldopen(datapath, "r") do file
    global θ = file["theta"];
    global Δ = file["args"]["dt"]
    global T = file["T"]
    global data_ctcrw = file["data-ctcrw"];
    global data_ctcrwt = file["data-ctcrwt"];
    global raster = file["raster"];
    global state_var_times = file["state-var-times"];
    global uint_to_coef_lookup = file["uint-to-coef-lookup"];
    global watercoef = file["watercoef"];
end

## Construct potential raster.
raster_trans = transform(raster, uint_to_coef_lookup);
raster_mlog = transform(raster_trans, x -> -log(x));

## Check blocksizes.
for blocksize in blocksizes
    try
        convert(Int, blocksize / Δ)
    catch InexactError
        println("`dt` must divide each blocksize evenly");
        exit();
    end
end
for blocksize in blocksizes
    try
        convert(Int, T / blocksize)
    catch InexactError
        throw(ArgumentError("`blocksize` must divide the total time series time, " * string(T) * ", evenly."));
    end
end

## Get resampling.
resampling = get_resampling("systematic", npar);

## Run CTCRW model with found parameters.
n_timepoints = length(data_ctcrw.time);
include(joinpath(MODELS_PATH, "CTCRW_KALMAN", "CTCRW_KALMAN_model.jl"));
ss = NLGSSMState(CTCRW_MODEL, n_timepoints);
kalman_instance = SSMInstance(CTCRW_MODEL, ss, data_ctcrw, n_timepoints);
filter!(kalman_instance, θ);
smooth!(kalman_instance, θ);

## Build CTCRWH model.
include(joinpath(MODELS_PATH, "CTCRWH_PF", "CTCRWH_PF_model.jl"));
proposal = let
    out = compute_smooth_normal_conditionals!(kalman_instance, θ);
    (L = SMatrix{4, 4, Float64, 16}.(out.L),
     A = SMatrix{4, 4, Float64, 16}.(out.A),
     b = SVector{4, Float64}.(out.b));
end;
ctcrwh = CTCRWH_build(raster_mlog, proposal, CTCRWH_one_step_potential);
ps = ParticleStorage(CTCRWHParticle, npar, n_timepoints);
pf_instance = SSMInstance(ctcrwh, ps, data_ctcrwt, n_timepoints);

## Main loop.
#block_essprop_m = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
lbsp_M = [zeros(convert(Int, T / blocksize), nreps) for blocksize in blocksizes];
P = zeros(n_timepoints, nreps);
npar_tmp = zeros(npar);
const X = pf_instance.storage.X; # Pointer to particles.
const W = pf_instance.storage.W; # Pointer to unnormalised log weights.
const ref = pf_instance.storage.ref; # Pointer to reference trajectory indices.
npar_tmp = zeros(npar);
pf_runs = 0;
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
        check_blocking_wrt_dts(blocks, data_ctcrwt.dt, blocksize);
        @assert blocks[end][2] == n_timepoints "something wrong. `blocks` has wrong end point.";

        # Compute all M(x_u | x_l) distributions.
        block_step_dists = build_jump_dists_from_ks!(kalman_instance, θ, blocks);
        for (i, b) in enumerate(blocks)
            l, u = b;
            # Compute block-ESS-M.
            for k in Base.OneTo(npar)
                npar_tmp[k] = logpdf(block_step_dists[i], SVector{4, Float64}(X[ref[u], u]),
                                                          SVector{4, Float64}(X[k, l]));
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

## Figure out blocking.
d = build_eff_dataframe(P, lbsp_M; npar = npar,
                        blocksizes = blocksizes, state_var_times = state_var_times);
blocking = determine_dyadic_blocking(d, maximum);
blocks = let
   u = map(x -> x[2], blocking);
   u_indices = map(utime -> searchsortedlast(state_var_times, utime), u)
   block_bounds_from_u_indices(u_indices);
end
blocksizes_in_result = map(last, blocking);
check_blocking_wrt_dts(blocks, data_ctcrwt.dt, blocksizes_in_result);

filename = let
    "blocking-ctcrwt-dt$Δ-watercoef$watercoef-npar$npar-nreps$nreps.jld2";
end
filepath = joinpath(outfolder, filename);
mkpath(outfolder);
verbose && println("Saving to $filepath");
jldopen(filepath, "w") do file
    file["args"] = args;
    file["theta"] = θ;
    file["state-var-times"] = state_var_times;
    #file["block-essprop-M"] = block_essprop_m;
    file["lbsp-M"] = lbsp_M;
    file["P"] = P;
    file["block-eff-data"] = d;
    file["blocks"] = blocks;
    file["blocksizes"] = blocksizes_in_result;
    file["pf-runs"] = pf_runs;
end
