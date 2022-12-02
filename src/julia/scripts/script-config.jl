import ArgParse

# A function to allow ArgParse to parse a Float64 vector passed from the command line.
function ArgParse.parse_item(::Type{Array{T, 1}}, x::AbstractString) where T <: Real
    strip_chars = ['[', ']', ','];
    split_x = split(x, [' ', ',']) |> (x -> filter(y -> y != "", x));
    char_array = map(y -> strip(y, strip_chars), split_x);
    return parse.(T, char_array)
end

# Parse a Pair from command line.
function ArgParse.parse_item(::Type{Pair{String, T}}, x::AbstractString) where T <: Real
    str = replace(x, "=>" => " ")
    str = split(str, ' ') |> (x -> filter(y -> y != "", x)) |> x -> string.(x);
    variable = str[1];
    value = str[2];
    p = Pair(variable, parse(Float64, value));
    p;
end

# Parse a Dict with numeric values from command line.
function ArgParse.parse_item(::Type{Dict{String, T}}, x::AbstractString) where T <: Real
    str_split = split(x, ',');
    pair_strs = map(x -> strip(x, ['(', ')', ' ']), str_split) |>
        x -> filter(y -> y != "", x) |> x -> string.(x);
    pair_strs[1] = replace(pair_strs[1], "Dict(" => "");
    pairs = ArgParse.parse_item.(Pair{String, Float64}, pair_strs);
    Dict(pairs...);
end

ARGUMENT_CONFIG = Dict();
let
    s = ArgParse.ArgParseSettings(prog = "run-bbcpf-blocksize-ctcrwp.jl");
    ArgParse.@add_arg_table! s begin
        "--model", "-m"
            help = "the model to run. CTCRWP or CTCRWP_B."
            arg_type = String
            default = "CTCRWP_B"
        "--npar", "-p"
            help = "number of particles"
            arg_type = Int
            default = 16
        "--blocksize"
            help = "size of blocks in the CPF-BBS"
            arg_type = Float64
            required = true
        "--T"
            help = "total length of time series"
            arg_type = Float64
            default = 64.0
        "--dt"
            help = "length of one time step"
            arg_type = Float64
            default = 1.0
        "--nsim", "-s"
            help = "number of iterations of the CPF-BBS"
            arg_type = Int
            default = 1000
        "--burnin", "-b"
            help = "amount of burnin iterations of the CPF-BBS"
            arg_type = Int
            default = 100
        "--par"
            help = "the parameter values for model specified by argument `model`."
            arg_type = Dict{String, Float64}
            required = true
        "--resampling"
            help = string("resampling algorithm to use. should be ",
                          "`multinomial`, `killing` or `systematic`");
            arg_type = String
            default = "systematic"
        "--outfolder"
            help = "folder where to place results. default uses 'pwd()' within script."
            arg_type = String
        "--jobid"
            help = "used in avoiding naming clashes"
            arg_type = Int
            default = 1
        "--verbose"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["bbcpf-blocksize-ctcrwp"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-block-metrics-sys-ctcrwp.jl");
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "number of particles"
            arg_type = Int
            default = 16
        "--blocksizes"
            help = "a sorted vector of blocksizes to consider. each must divide `T` evenly and must be divisible by `dt`."
            arg_type = Vector{Float64}
            required = true
        "--T"
            help = "total length of time series"
            arg_type = Float64
            default = 64.0
        "--dt"
            help = "length of one time step"
            arg_type = Float64
            default = 1.0
        "--nreps"
            help = "number of times to repeat simulation"
            arg_type = Int
            default = 50
        "--par"
            help = "the parameter values for model specified by argument `model`."
            arg_type = Dict{String, Float64}
            required = true
        "--outfolder"
            help = "folder where to place results. default uses 'pwd()' within script."
            arg_type = String
        "--jobid"
            help = "used in avoiding naming clashes"
            arg_type = Int
            default = 1
        "--verbose"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["block-metrics-sys-ctcrwp"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-block-metrics-sys-lgcpr.jl");
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "number of particles"
            arg_type = Int
            default = 8 
        "--blocksizes"
            help = string("a sorted vector of blocksizes to consider. each must divide `T` ",
                          "of dataset evenly and must be divisible by base `dt` of dataset.")
            arg_type = Vector{Float64}
            required = true
        "--nreps"
            help = "number of times to repeat simulation"
            arg_type = Int
            default = 50
        "--seed"
            help = string("seed used in data simulation. used to find dataset.")
            arg_type = Int
            default = 12345
        "--outfolder"
            help = "folder where to place results. default uses `pwd()` within script."
            arg_type = String
        "--jobid"
            help = "used in avoiding naming clashes"
            arg_type = Int
            default = 1
        "--verbose"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["block-metrics-sys-lgcpr"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-build-ctcrwt-data.jl");
    ArgParse.@add_arg_table! s begin
        "--dt"
            help = "length of one time step"
            arg_type = Float64
            default = 2.0 ^ (-7) # 0.0078125
        "--watercoef"
            help = "terrain coefficient of water"
            arg_type = Float64
            default = 0.0
        "--outfolder"
            help = "folder where to place results."
            default = joinpath(INPUT_PATH, "ctcrwt-data")
            arg_type = String
        "--verbose"
            help = "display messages?"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["build-ctcrwt-data"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-block-metrics-sys-ctcrwt.jl");
    ArgParse.@add_arg_table! s begin
    "--npar", "-p"
        help = "number of particles"
        arg_type = Int
        default = 16
    "--blocksizes"
        help = string("a sorted vector of blocksizes to consider. each must divide `T` ",
                      "of dataset evenly and must be divisible by base `dt` of dataset.")
        arg_type = Vector{Float64}
        required = true
    "--nreps"
        help = "number of times to run pf (successfully). with -Inf potentials repeats might be made"
        arg_type = Int
        default = 50
    "--datafile"
        help = string("datafile used to get data from (some file from `input` folder)");
        arg_type = String
        required = true
    "--outfolder"
        help = "folder where to place results. default uses 'pwd()' within script."
        arg_type = String
        default = joinpath(INPUT_PATH, "ctcrwt-data")
    "--verbose"
        action = :store_true
    end;
    global ARGUMENT_CONFIG["block-metrics-sys-ctcrwt"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-terrain-sim.jl");
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "number of particles"
            arg_type = Int
            default = 16
        "--blocksize"
            help = "size of blocks in the CPF-BBS. can be used to specify constant blocking."
            arg_type = Float64
        "--datafile"
            help = string("datafile used to get data from (some file from `input` folder)");
            arg_type = String
            required = true
        "--blockfile"
            help = string("filepath to load blocking from. if set `blocksize` is ignored.");
            arg_type = String
        "--nsim", "-s"
            help = "number of iterations of the CPF-BBS"
            arg_type = Int
            default = 1000
        "--burnin", "-b"
            help = "amount of burnin iterations of the CPF-BBS"
            arg_type = Int
            default = 100
        "--resampling"
            help = string("resampling algorithm to use. should be ",
                          "`multinomial`, `killing` or `systematic`.");
            arg_type = String
            default = "systematic"
        "--outfolder"
            help = "folder where to place results. default uses 'pwd()' within script."
            arg_type = String
        "--verbose"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["terrain-sim"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-bbcpf-lgcpr.jl");
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "number of particles"
            arg_type = Int
            default = 16
        "--blocksize"
            help = "size of blocks in the CPF-BBS"
            arg_type = Float64
        "--nsim", "-s"
            help = "number of iterations of the CPF-BBS"
            arg_type = Int
            default = 1000
        "--burnin"
            help = "amount of burnin iterations of the CPF-BBS"
            arg_type = Int
            default = 100
        "--backward-sampling"
            help = "if set, disregard `blocksize` argument and run with backward sampling"
            action = :store_true
        "--resampling"
            help = string("resampling algorithm to use. should be ",
                          "`multinomial`, `killing` or `systematic`");
            arg_type = String
            default = "systematic"
        "--seed"
            help = string("seed used in data simulation. used to find dataset.")
            arg_type = Int
            default = 12345
        "--blockfile"
            help = string("optional. disregard all blocking specification and load blocking from data ",
                          "specified by this path. the path should be specified ",
                          "relative to the root folder of the project. the blockfile is checked ",
                          "to contain the same `seed` as the data used with the model.");
            arg_type = String
        "--outfolder"
            help = "folder where to place results. default uses 'pwd()' within script."
            arg_type = String
        "--verbose"
            action = :store_true
    end;
    global ARGUMENT_CONFIG["bbcpf-lgcpr"] = s;
end

let
    s = ArgParse.ArgParseSettings(prog = "run-sim-lgcpr-data.jl");
    ArgParse.@add_arg_table! s begin
        "--base-dt"
            help = "length of one time step used in simulating piecewise constant intensity."
            arg_type = Float64
            default = 1/64
        "--T"
            help = "length of time series in physical time"
            arg_type = Float64
            default = 256.0
        "--a"
            help = "lower bound of reflected BM"
            arg_type = Float64
            default = 0.0
        "--b"
            help = "upper bound of reflected BM"
            arg_type = Float64
            default = 3.0
        "--sigma"
            help = "standard deviation of the Brownian motion"
            arg_type = Float64
            default = 0.3
        "--sigmai"
            help = "standard deviation of initial normal distribution for latent state"
            arg_type = Float64
            default = 1.0
        "--beta"
            help = "parameter affecting piecewise intensity"
            arg_type = Float64
            default = 0.5
        "--alpha"
            help = "parameter affecting piecewise intensity"
            arg_type = Float64
            default = 1.0
        "--seed"
            help = string("seed used in data simulation")
            arg_type = Int
            default = 12345
        "--outfolder"
            help = "folder where to place results."
            default = joinpath(INPUT_PATH, "lgcpr-data")
            arg_type = String
    end;
    global ARGUMENT_CONFIG["sim-lgcpr-data"] = s;
end


let
    s = ArgParse.ArgParseSettings(prog = "run-lgcpr-determine-dyadic-blocking.jl");
    ArgParse.@add_arg_table! s begin
        "--seed"
            help = string("seed of dataset. used to load datafile.")
            arg_type = Int
            default = 12345
        "--infolder"
            help = string("folder to look for input data")
            arg_type = String
            default = joinpath(OUTPUT_PATH, "simulation-experiments", "lgcpr-find-blocksizes")
        "--outfolder"
            help = "folder where to place results"
            default = joinpath(INPUT_PATH, "lgcpr-data")
            arg_type = String
    end;
    global ARGUMENT_CONFIG["lgcpr-determine-dyadic-blocking"] = s;
end
