include("../../../config.jl");
include("script-config.jl");
experiment_name = "build-ctcrwt-data";
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG[experiment_name]);

## Arguments.
Δ = args["dt"];
watercoef = args["watercoef"];
outfolder = args["outfolder"];
verbose = args["verbose"];
isnothing(outfolder) && (outfolder = pwd();)
outfolder = joinpath(abspath(outfolder));
verbose && println("Output folder is $outfolder");
verbose && println("Arguments are $args");

using JLD2
include(joinpath(LIB_PATH, "bbcpf.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "statespace-functions.jl"));
include(joinpath(LIB_PATH, "script-helpers.jl"));
include(joinpath(LIB_PATH, "math-functions.jl"));
include(joinpath(LIB_PATH, "data-functions.jl"));

## Load Corine as a SimpleRaster.
corine = load_corine();
map_bbox = BoundingBox(648720.0, 662688.0, 6908040.0, 6922440.0) |>
           x -> add_margin(x, 1000.0);
raster = SimpleRaster(corine, 1, map_bbox);

## Construct potential raster.
code_to_landtype_lookup = load_corine_legend(level = 1);
landtype_to_coef_lookup = Dict("Artificial surfaces" => 0.2,
                                 "Agricultural areas" => 0.6,
                                 "Forests and semi-natural areas" => 0.5,
                                 "Water bodies" => watercoef,#0.0,
                                 "Wetlands" => 0.5);
code_to_coef_lookup = Dict{UInt8, Float64}();
for k in keys(code_to_landtype_lookup)
    code_to_coef_lookup[k] = landtype_to_coef_lookup[code_to_landtype_lookup[k]];
end
code_to_coef_lookup[raster.mis] = 0.0;

## Data points.
xy = [(658960.0, 6919928.0), (658960.0, 6919928.0 - 1500.0),
      (661000.0, 6.91685 * 10 ^ 6), (658960.0, 6919928.0 - 5 * 1500.0),
      (658960.0, 6919928.0 - 6 * 1500.0), (658960.0 - 3200.0, 6919928.0 - 7.3 * 1500.0),
      (658960.0 - 6500.0, 6919928.0 - 5000.0),
      (658960.0 - 10000.0, 6919928.0 - 2000.0),
      (658960.0 - 9000.0, 6919928.0 + 500.0),
      (658960.0 - 8000.0, 6919928.0 + 2750.0),
      (658960.0 - 7000.0, 6919928.0 - 300.0),
      (658960.0 - 4000.0, 6919928.0 - 800.0),
      (658960.0 - 6000.0, 6919928.0 - 1800.0),
      (658960.0 - 5400.0, 6919928.0 - 3500.0),
      (658960.0 - 3250.0, 6919928.0 - 1200.0),
      (658960.0 - 500.0, 6919928.0)];
T = length(xy) * 1.0;
xy = map(x -> [x[1], x[2]], xy);

# Construct dataset for CTCRW model.
state_var_times = collect(0.0:Δ:T);
data_ctcrw = let time_end = T
    otimes = collect(range(Δ, time_end, length = length(xy)));
    y = Matrix{Union{Float64, Missing}}(missing, 2, length(state_var_times));
    ind = map(t -> findfirst(x -> x >= t, state_var_times), otimes);
    for (i, index) in enumerate(ind)
        y[:, index] .= xy[i];
    end
    (dt = diff(state_var_times),
     time = copy(state_var_times),
     y = y);
end;
n_timepoints = length(data_ctcrw.time);
data_ctcrwt = let
    # Special case at last time point to step to timepoint T.
    dt = vcat(copy(data_ctcrw.dt), [0.0]);
    (dt = dt,)
end;

function build_set_param(data)
    imuxi = findfirst(x -> !ismissing(x), data.y[1, :]);
    imuyi = findfirst(x -> !ismissing(x), data.y[2, :]);
    muxi = data.y[1, imuxi];
    muyi = data.y[2, imuyi];
    let muxi = muxi, muyi = muyi
        function(x)
            (muxi = muxi,
             vxi = 0.0,
             muyi = muyi,
             vyi = 0.0,
             musigmai = 50.0,
             beta = exp(x[1]),
             sigma = exp(x[2]),
             tau = 50.0);
        end
    end
end
function construct_objective(model, data)
    set_param = build_set_param(data);
    T = size(data.y, 2);
    ss = NLGSSMState(model, T);
    instance = SSMInstance(model, ss, data, T);
    let instance = instance, set_param = set_param
        function(x)
            θ = set_param(x);
            -loglik!(instance, θ);
        end
    end
end

## Find theta by maximum likelihood.
include(joinpath(MODELS_PATH, "CTCRW_KALMAN", "CTCRW_KALMAN_model.jl"));
using Optim
obj = construct_objective(CTCRW_MODEL, data_ctcrw);
res = optimize(obj, [-1.0, -2.0], NelderMead());
if !Optim.converged(res)
    throw(ErrorException("Estimation of CTCRW parameters did not converge!"));
end
get_param = build_set_param(data_ctcrw);
θ = get_param(Optim.minimizer(res));

## Save data.
folderpath = outfolder;
mkpath(folderpath);
filename = "ctcrwt-data-dt$Δ-watercoef$watercoef";
filepath = joinpath(folderpath, filename) * ".jld2";

verbose && print("Saving output to $filepath...");
jldopen(filepath, "w"; iotype = IOStream) do file
    file["args"] = args;
    file["theta"] = θ;
    file["T"] = T;
    file["state-var-times"] = state_var_times;
    file["data-ctcrw"] = data_ctcrw;
    file["data-ctcrwt"] = data_ctcrwt;
    file["raster"] = raster;
    file["uint-to-coef-lookup"] = code_to_coef_lookup;
    file["watercoef"] = landtype_to_coef_lookup["Water bodies"];
end
verbose && println(" done.");
