include("../../../config.jl");
include("script-config.jl");
script_name = "sim-lgcpr-data";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[script_name]);

T = args["T"];
Δ = args["base-dt"];
a = args["a"];
b = args["b"];
sigma = args["sigma"];
sigmai = args["sigmai"];
alpha = args["alpha"];
beta = args["beta"];
seed = args["seed"];
outfolder = args["outfolder"];

@assert a < b "a < b not satisfied";
try
    convert(Int, T / Δ);
catch InexactError
    println("`T` must divide `base-dt` evenly");
    exit();
end
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(MODELS_PATH, "LGCP_REFLECTED", "LGCP_REFLECTED_model.jl"));
using Random, JLD2

data_dts = diff(0.0:Δ:T);
θ = (sigma = sigma,
     alpha = alpha,
     beta = beta,
     sigmai = sigmai);
bounds = (a = a, b = b);
Random.seed!(seed);
λs, true_state, τ, state_var_times, obs = simulate_lgcpr_data(data_dts, θ, bounds);

# NOTE: Here we add zero to dts, since we want to also sample the state variable
# at the last time point (T). (the last interval starting from T is length 0.0)
# Also add one observation to y, just to make dt and y of same length,
# the last value of y is however never read.
data = merge((y = vcat(obs, [false]),
              dt = vcat(diff(state_var_times), [0.0])),
              bounds);
@assert length(data.y) == length(state_var_times) "length of observations does not match length of state variable times";
@assert length(data.dt) == length(data.y) "invalid observations, dt and obscount don't match";

mkpath(outfolder);
filename = string(seed) * ".jld2";
outfilepath = joinpath(outfolder, filename)
jldopen(outfilepath, "w") do file
    file["args"] = args;
    file["data"] = data;
    file["tau"] = τ;
    file["intensity"] = λs;
    file["true-state"] = map(x -> x.s, true_state);
    file["theta"] = θ;
end
println("Simulated data to $outfilepath");
