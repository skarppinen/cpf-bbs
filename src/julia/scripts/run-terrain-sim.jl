include("../../../config.jl");
include("script-config.jl");
experiment_name = "terrain-sim";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[experiment_name]);

## Arguments.
npar = args["npar"]; # 16
nsim = args["nsim"]; # 10000
burnin = args["burnin"]; # 1000
datafile = args["datafile"];
blocksize = args["blocksize"]; # 1.0
blockfile = args["blockfile"];
resampling_str = args["resampling"];
outfolder = args["outfolder"];
verbose = args["verbose"];
isnothing(outfolder) && (outfolder = pwd();)
outfolder = joinpath(abspath(outfolder));
verbose && println("Output folder is $outfolder");
verbose && println("Arguments are $args");

if !isnothing(blocksize) && !isnothing(blockfile)
    msg = "only specify `blocksize` (constant blocking) or `blockfile` (blocks loaded from file)";
    throw(ArgumentError(msg));
end

include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));
using JLD2

datapath = datafile;
jldopen(datapath, "r") do file
    global θ = file["theta"];
    global Δ = file["args"]["dt"]
    global T = file["T"]
    global data_ctcrw = file["data-ctcrw"];
    global data_ctcrwt = file["data-ctcrwt"];
    global raster = file["raster"];
    global state_var_times = file["state-var-times"];
    global uint_to_coef_lookup = file["uint-to-coef-lookup"];
end

blocks = let
    if !isnothing(blocksize)
        try
            convert(Int, blocksize / Δ)
        catch InexactError
            println("`dt` must divide `blocksize` evenly");
            exit();
        end
        try
            convert(Int, T / blocksize)
        catch InexactError
            throw(ArgumentError("`blocksize` must divide the total time series time, " * string(T) * ", evenly."));
        end
        blocks = get_even_bbcpf_blocking(Δ, blocksize, T);
        check_blocking_wrt_dts(blocks, data_ctcrwt.dt, blocksize);
        blocks;
    else
        blockfilepath = blockfile;
        blocks, _blocksizes = jldopen(blockfilepath, "r") do file
            file["blocks"], file["blocksizes"];
        end
        check_blocking_wrt_dts(blocks, data_ctcrwt.dt, _blocksizes);
        blocks;
    end
end

resampling = get_resampling(resampling_str, npar);
filename = let
    name = experiment_name;
    name *= string("-", resampling_str);
    name *= string("-npar", npar);
    if !isnothing(blocksize)
        name *= string("-blocksize", blocksize);
    else
        name *= string("-blockfile");
    end
    name *= string("-dt", Δ);
    name *= string("-nsim", nsim);
    name;
end
folderpath = joinpath(outfolder, experiment_name);
n_timepoints = length(data_ctcrw.time);

## CTCRW model.
include(joinpath(MODELS_PATH, "CTCRW_KALMAN", "CTCRW_KALMAN_model.jl"));
ss = NLGSSMState(CTCRW_MODEL, n_timepoints);
ctcrw_inst = SSMInstance(CTCRW_MODEL, ss, data_ctcrw, n_timepoints);
filter!(ctcrw_inst, θ);
smooth!(ctcrw_inst, θ);

## Build CTCRWH model.
include(joinpath(MODELS_PATH, "CTCRWH_PF", "CTCRWH_PF_model.jl"));
proposal = let
    out = compute_smooth_normal_conditionals!(ctcrw_inst, θ);
    (L = SMatrix{4, 4, Float64, 16}.(out.L),
     A = SMatrix{4, 4, Float64, 16}.(out.A),
     b = SVector{4, Float64}.(out.b));
end;
raster_trans = transform(raster, uint_to_coef_lookup);
raster_mlog = transform(raster_trans, x -> -log(x));
ctcrwh = CTCRWH_build(raster_mlog, proposal, CTCRWH_one_step_potential);

# Simulate from proposal (no potential map)
sim = [CTCRWHParticle() for i in 1:length(ctcrw_inst), j in 1:nsim];
for j in 1:size(sim, 2)
    v = view(sim, :, j);
    simulate_states!(v, ctcrwh, nothing, θ);
end
proposal_sim = map(p -> (p.mux, p.muy), sim);

## Run BBCPF.
verbose && println("Starting simulations...");
samplers = BBCPF_build_samplers(blocks, CTCRW_MODEL, θ, modeldata = data_ctcrw,
                                nodata = false);
ps = ParticleStorage(CTCRWHParticle, npar, n_timepoints);
pf_model_instance = SSMInstance(ctcrwh, ps, data_ctcrwt, n_timepoints);


verbose && println("Initialising...")
tries = pf_ensure_pos_weights!(pf_model_instance, θ, resampling = resampling);
traceback!(pf_model_instance, AncestorTracing);

verbose && print("Running BBCPF..")
for i in 1:burnin
    bbcpf!(pf_model_instance, θ, samplers;
           resampling = resampling);
end
for j in 1:nsim
    bbcpf!(pf_model_instance, θ, samplers; resampling = resampling);
    trace_reference!(view(sim, :, j), pf_model_instance.storage);
end
pmux = map(x -> x.mux, sim);
pmuy = map(x -> x.muy, sim);
pvx = map(x -> x.vx, sim);
pvy = map(x -> x.vy, sim);
verbose && println("done.");

## Save data.
folderpath = outfolder;
mkpath(folderpath);
filepath = joinpath(folderpath, filename) * ".jld2";

verbose && print("Saving output...");
jldopen(filepath, "w"; iotype = IOStream) do file
    file["args"] = args;
    file["theta"] = θ;
    file["data-ctcrw"] = data_ctcrw;
    file["data-ctcrwt"] = data_ctcrwt;
    file["mapraster"] = raster_trans;
    file["state-var-times"] = state_var_times;

    file["pmux"] = pmux;
    file["pmuy"] = pmuy;
    file["pvx"] = pvx;
    file["pvy"] = pvy;
    file["musmooth"] = ctcrw_inst.storage.mu_smooth;
    file["Psmooth"] = ctcrw_inst.storage.P_smooth;
    file["proposal-sim"] = proposal_sim;
end
verbose && println("done.");
