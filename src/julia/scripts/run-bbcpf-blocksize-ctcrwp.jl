include("../../../config.jl");
include("script-config.jl");
config_name = "bbcpf-blocksize-ctcrwp";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[config_name]); # Get command line arguments.

## Simulation settings.
npar = args["npar"]; # Amount of particles.
blocksize = args["blocksize"]; # Size of blocks in model time.
model = args["model"];
T = args["T"]; # Total time of model.
dt = args["dt"]; # Time discretisation.
#nreps = args["nreps"]; # How many times to repeat simulation.
burnin = args["burnin"]; # Amount of burnin iterations.
nsim = args["nsim"]; # Amount of iterations after burnin.
par = args["par"]; # Parameters.
resampling_str = args["resampling"]; # Resampling.
verbose = args["verbose"];
jobid = args["jobid"]; # To avoid name clashes.
outfolder = args["outfolder"];
isnothing(outfolder) && (outfolder = pwd();)

#verbose = true;
#model = "CTCRWP_B"; resampling_str = "killing";
experiment_name = string("bbcpf-blocksize-",
                         replace(lowercase(model), '_' => '-'));

using Statespace
using NLGSSM
using Distributions
using StatsBase: autocor
using JLD2
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "MeanAccumulator.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));

## Load models.
pf_str = model * "_PF";
kalman_str = model * "_KALMAN";
include(joinpath(MODELS_PATH, pf_str, pf_str * "_model.jl"));
include(joinpath(MODELS_PATH, kalman_str, kalman_str * "_model.jl"));

#par = ctcrwpb_par_statdist_fix(2.0, 0.0625);
#T = 64.0; blocksize = 16.0; npar = 32; nreps = 1; nsim = 10000; dt = 2 ^ -5; burnin = 100;
#Dict("betax" => 0.5, "betav" => 1.0, "tau" => .0, "sigma" => 2.0, "xmui" => 0.0, "vmui" => 0.0);

# Pointers to models.
eval(Meta.parse("model_pf = $pf_str"));
eval(Meta.parse("model_kalman = $kalman_str"));
ptype = particle_type(model_pf); # Type of particle used in model.

# Dimensions.
obsdim = model_kalman.obsdim;
statedim = model_kalman.statedim;

#par = Dict("betax" => 0.5, "betav" => 1.0, "tau" => 2.0, "sigma" => 0.5, "xmui" => 0.0, "vmui" => 0.0);
θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
verbose && println(string("Got parameters ", θ));
resampling = get_resampling(resampling_str, npar);
verbose && println("Arguments are $args");

### Some argument checking.
try
    convert(Int, T / blocksize)
catch InexactError
    println("`blocksize` must divide the total time series time, " * string(T) * ", evenly.");
    exit();
end
try
    convert(Int, blocksize / dt)
catch InexactError
    println("`dt` must divide `blocksize` evenly");
    exit();
end
state_var_times = collect(0.0:dt:T); # Times of state variables in smoothing distribution.
n_timepoints = length(state_var_times); # How many time points.

## Compute the analytical smoothing distribution using the Kalman filter and smoother.
ss = NLGSSMState(model_kalman, n_timepoints);
data = let
    y = zeros(Union{Missing, Float64}, 1, n_timepoints);
    y[1, end] = missing; # To avoid Kalman update at last time point.
    dt = vcat(diff(state_var_times), [0.0]);
    (y = y, dt = dt);
end;
#data = (y = permutedims(repeat([0.0], n_timepoints)),
#        dt = repeat([dt], n_timepoints));
kalman_instance = SSMInstance(model_kalman, ss, data, n_timepoints);
filter!(kalman_instance, θ);
smooth!(kalman_instance, θ);

## Run BBCPF.
# Create storage object for particle filtering.
ps = ParticleStorage(ptype, npar, n_timepoints);
pf_instance = SSMInstance(model_pf, ps, data, n_timepoints);

# Get samplers for BBCPF.
blocks = get_even_bbcpf_blocking(dt, blocksize, T);
check_blocking_wrt_dts(blocks, data.dt, blocksize);
@assert blocks[end][2] == n_timepoints "something wrong. `blocks` has wrong end point.";
samplers = BBCPF_build_samplers(blocks, model_kalman, θ;
                                modeldata = data, nodata = true);
block_ls = map(first, blocks); # Block lower bounds.
nblocks = length(block_ls);
@assert length(samplers) == nblocks "something wrong, `length(samplers) != nblocks`.";

# Allocate temporaries.
sim = [ptype() for i in 1:n_timepoints, j in 1:nsim];
outputs = (blockweight = zeros(npar, nblocks),
           ptemp = ptype(),
           l_change_in_local_cpf = zeros(Bool, nblocks));
#blockjump_ess = zeros(nblocks, nsim);
ref_traj_tmp = [ptype() for i in 1:n_timepoints];
esjds = [MeanAccumulator() for i in 1:n_timepoints]
mean_est_by_state = [zeros(n_timepoints) for i in 1:statedim];
var_est_by_state = [zeros(n_timepoints) for i in 1:statedim];
iact_by_state = [zeros(n_timepoints) for i in 1:statedim];
#global_ref_changes = zeros(Int, n_timepoints);
local_ref_changes = zeros(Int, nblocks);
ac1_est_by_state = [zeros(n_timepoints) for i in 1:statedim];
A = zeros(n_timepoints, nsim);

#pf!(pf_instance, θ, resampling = resampling);
#traceback!(pf_instance, AncestorTracing);
ptype_fields = fieldnames(ptype);

# Simulation loop.
t1 = time();
# Initialise.
pf!(pf_instance, θ, resampling = resampling);
traceback!(pf_instance, AncestorTracing);

# Burnin.
for j in 1:burnin
    bbcpf!(pf_instance, θ, samplers; resampling = resampling);
end
trace_reference!(ref_traj_tmp, pf_instance.storage);

# Do iterations.
for j in 1:nsim
    # Run BBCPF with some saved outputs.
    bbcpf!(pf_instance, θ, samplers, outputs; resampling = resampling);
    sim_col = view(sim, : , j);
    trace_reference!(sim_col, pf_instance.storage);

    # If first simulation (j = 1), last reference is last trajectory from burnin.
    # Otherwise last reference is from iteration j - 1.
    if j == 1
        sim_col_prev = view(ref_traj_tmp, :, 1);
    else
        sim_col_prev = view(sim, :, j - 1);
    end

    # - Time point level measures -
    # 1. Keep track where (global) reference changed.
    # 2. Accumulate expected square jump distances (ESJD) (a mean of squared distances).
    for k in 1:n_timepoints
        #@inbounds global_ref_changes[k] += !eq(sim_col_prev[k], sim_col[k]);
        @inbounds accumulate!(esjds[k], distsqr(sim_col_prev[k], sim_col[k]));
    end

    # - Block level measures -
    # 1. Keep track where (local) reference changed.
    # 2. Compute blockjump effective sample sizes.
    for k in 1:nblocks
        @inbounds local_ref_changes[k] += outputs.l_change_in_local_cpf[k];
        #w = view(outputs.blockweight, :, k);
        #@inbounds blockjump_ess[k, j] = ess_unnorm_log(w);
    end
end

# - Compute mean block-ESS's (so vector has one mean per block) -
#mean_block_ess = mapslices(mean, blockjump_ess; dims = 2)[:, 1];

# - Compute estimate of mean, variance, IACT and AC1 by state variable. -
for (k, field) in enumerate(ptype_fields)
    for h in eachindex(A)
        A[h] = getfield(sim[h], field);
    end
    mean_est_by_state[k] .= mapslices(mean, A; dims = 2)[:, 1];
    var_est_by_state[k] .= mapslices(var, A; dims = 2)[:, 1];
    iact_by_state[k] .= mapslices(iactBM, A; dims = 2)[:, 1];

    # First autocorrelations.
    for t in 1:n_timepoints
        μ = kalman_instance.storage.mu_smooth[t][k];
        σ² = kalman_instance.storage.P_smooth[t][k, k];
        ac1_est_by_state[k][t] = autocor_known_meanvar(view(A, t, :), μ, σ², 1:1)[1];
    end
end
t2 = time();
verbose && report_progress(1, 1, t2 - t1);
verbose && print("Computing summaries...")
t1 = time();

## Compute summaries that are saved.
ksmoother_mean_by_state = map(1:statedim) do i
    map(x -> x[i], kalman_instance.storage.mu_smooth);
end
ksmoother_var_by_state = map(1:statedim) do i
    map(x -> x[i, i], kalman_instance.storage.P_smooth);
end
t2 = time();
verbose && println(string(" took ", round(t2 - t1, digits = 3), "s."))

## Save data.
folderpath = joinpath(outfolder, experiment_name);
mkpath(folderpath);
filename = string("jobid-", jobid, "-", randstring(20), ".jld2");
filepath = joinpath(folderpath, filename);

verbose && print("Saving output...")
t1 = time();
jldopen(filepath, "w"; iotype = IOStream) do file
    file["args"] = args;
    file["blocks"] = blocks;
    file["theta"] = θ;
    file["state-var-times"] = state_var_times;
    file["iactBM-by-state"] = iact_by_state;
    file["mean-est-by-state"] = mean_est_by_state;
    file["var-est-by-state"] = var_est_by_state;
    file["ac1-est-by-state"] = ac1_est_by_state;
    file["esjd"] = value.(esjds);
    #file["global-ref-changes"] = global_ref_changes;
    file["local-ref-changes"] = local_ref_changes;
    #file["mean-block-ess"] = mean_block_ess;
    file["ksmoother-mean-by-state"] = ksmoother_mean_by_state;
    file["ksmoother-var-by-state"] = ksmoother_var_by_state;
end
t2 = time();
verbose && println(string(" took ", round(t2 - t1, digits = 3), "s."))
verbose && println("Done.")
