include("../../../config.jl");
include("script-config.jl");
experiment_name = "bbcpf-lgcpr";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[experiment_name]);

# Unload args.
npar = args["npar"];
nsim = args["nsim"];
burnin = args["burnin"];
blocksize = args["blocksize"];
backward_sampling = args["backward-sampling"];
resampling_str = args["resampling"];
verbose = args["verbose"];
outfolder = args["outfolder"];
data_seed = args["seed"];
blockfile = args["blockfile"];

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

## Check blocksize just in case.
if isnothing(blockfile)
    if isnothing(blocksize)
        local msg = "Either specify `blocksize` or set flag `blocks-from-file`"
        throw(ArgumentError(msg));
    end
    try
        convert(Int, blocksize / Δ)
    catch InexactError
        local msg = "Base Δ = $Δ of datafile must divide `blocksize` evenly";
        throw(ArgumentError(msg));
    end
    try
        convert(Int, T / blocksize);
    catch InexactError
        local msg = "`blocksize` must divide `T` = $T of datafile evenly";
        throw(ArgumentError(msg));
    end
end
isnothing(outfolder) && (outfolder = pwd();)
outfolder = joinpath(abspath(outfolder), experiment_name);
verbose && println("Output folder is $outfolder");

include(joinpath(LIB_PATH, "script-helpers.jl"));
filename = let
    name = experiment_name;
    name *= string("-", resampling_str);
    name *= string("-npar", npar);
    if isnothing(blockfile)
        if backward_sampling
            name *= string("-cpfbs");
        else
            name *= string("-blocksize", blocksize);
        end
    else
        block_spec = basename(drop_postfix(blockfile));
        name *= string("-", block_spec);
    end
    name *= string("-basedt", Δ);
    name *= string("-nsim", nsim);
    name *= string("-seed", data_seed);
    name;
end
folderpath = joinpath(outfolder, experiment_name);
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(MODELS_PATH, "LGCP_REFLECTED", "LGCP_REFLECTED_model.jl"));
include(joinpath(MODELS_PATH, "BM_KALMAN", "BM_KALMAN_model.jl"));

#npar = 8;
#resampling_str = "killing";

## Prepare for BBCPF.
resampling = get_resampling(resampling_str, npar);
state_var_times = vcat([0.0], cumsum(data.dt)[1:(end - 1)]); # Times of state variables.

# Get blocking.
if !isnothing(blockfile)
    block_filepath = joinpath(PROJECT_ROOT, blockfile);
    verbose && println("Attempting to read blockfile from $block_filepath");
    if !isfile(block_filepath)
        msg = "could not find blocking data from $block_filepath";
        throw(ArgumentError(msg));
    end
    global blocks, _blocksizes = jldopen(block_filepath, "r") do file
        if file["seed"] != data_seed
            throw(ArgumentError("seed in blockfile does not match seed passed to script (used to load dataset)."))
        end
        file["blocks"], file["blocksizes"];
    end
    check_blocking_wrt_dts(blocks, data.dt, _blocksizes);
    verbose && println("Loaded blocks from $block_filepath");
else
    global blocks = let
        # NOTE: there are differences in dt in this experiment.
        # Backward sampling does not have "dt" = "block size" from the perspective
        # of BBCPF. (hence the flag)
        if backward_sampling
            blocks = block_bounds_from_u_indices(2:length(state_var_times));
        else
            nblocks = convert(Int, T / blocksize);
            utimes = blocksize * collect(1:nblocks); # The desired u times.
            block_u_indices = map(ut -> searchsortedlast(state_var_times, ut), utimes)
            blocks = block_bounds_from_u_indices(block_u_indices);

            # Check that the blocking actually has right blocksize for every block.
            check_blocking_wrt_dts(blocks, data.dt, blocksize);
        end
        blocks;
    end
end
nblocks = length(blocks);
@assert blocks[end][2] == length(data.y) "invalid blocks, different size than data";
samplers = BBCPF_build_samplers(blocks, BM_MODEL, θ, nodata = true,
                                modeldata = data);
n_timepoints = length(data.y);
storage = ParticleStorage(LGCPRParticle, npar, n_timepoints);
pf_instance = SSMInstance(LGCPR_BM_PF, storage, data, n_timepoints);

# Object to store diagnostic information from each iteration.
bbcpf_monitors = (blockweight = zeros(npar, nblocks),
                  ptemp = LGCPRParticle(),
                  l_change_in_local_cpf = zeros(Bool, nblocks));

# Number of local reference changes in each block.
local_ref_changes = zeros(Int, nblocks);
#blockjump_ess = zeros(Float64, nblocks, nsim);

## Run BBCPF.
t1 = time();
Random.seed!();
sim = [LGCPRParticle() for i in 1:n_timepoints, j in 1:nsim];
pf!(pf_instance, θ, resampling = resampling);
traceback!(pf_instance, AncestorTracing);
for i in 1:burnin
    bbcpf!(pf_instance, θ, samplers; resampling = resampling);
end
for i in 1:nsim
    bbcpf!(pf_instance, θ, samplers, bbcpf_monitors; resampling = resampling);
    trace_reference!(view(sim, :, i), pf_instance.storage);
    for k in 1:nblocks
        @inbounds local_ref_changes[k] += bbcpf_monitors.l_change_in_local_cpf[k];
        w = view(bbcpf_monitors.blockweight, :, k);
        #@inbounds blockjump_ess[k, i] = ess_unnorm_log(w);
    end
end
s = map(x -> x.s, sim);
t2 = time();
verbose && report_progress(1, 1, t2 - t1);

folderpath = outfolder;
mkpath(folderpath);
filepath = joinpath(folderpath, string(filename, ".jld2"));
verbose && print("Saving output...");
jldopen(filepath, "w"; iotype = IOStream) do file
    file["args"] = args;
    file["theta"] = θ;
    file["s"] = s;
    file["state-var-times"] = state_var_times;
    file["blocks"] = blocks;
    file["local-ref-changes"] = local_ref_changes;
    #file["blockjump-ess"] = blockjump_ess;
end
verbose && println("done.");
