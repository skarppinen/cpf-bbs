## Script for generating task lists for running scripts at CSC.
include("../../../config.jl");
include(joinpath(LIB_PATH, "build-task-list.jl"));
include(joinpath(LIB_PATH, "find-par.jl"));
include(joinpath(LIB_PATH, "model-helpers.jl"));
include(joinpath(SCRIPTS_PATH, "script-config.jl"));
task_list_outpath = joinpath(SRC_PATH, "bash");

format_dict(d) = replace(string(d), '"' => "");

let experiment = "blocksize-vs-param-ctcrwp-b"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * "bbcpf-blocksize-ctcrwp");

    fn = "";
    model = "CTCRWP_B";
    n_job_repeats = 1;
    #nreps = 2;
    nsim = 20000;
    burnin = 1000;
    T_power = 6;
    jobid = 0;
    dt_power = -7;
    tau = 1.0;
    for blocksize in 2.0 .^ (dt_power:T_power),
        npar in [2, 4, 8, 16, 32],
        sigma in [0.5 / 4, 0.5, 2.0],
        resampling in ["killing", "systematic", "multinomial"]

        dt = 2.0 ^ dt_power;
        T = 2.0 ^ T_power;

        s11 = s22 = 1.0;
        betav = sigma * sigma * inv(2.0 * s11);
        betax = solve_βx(sigma, s11, s22);
        @assert !isinf(betax) && !isnan(betax) "problem with solved value of `betax`";
        par = Dict("betav" => betav, "betax" => betax, "tau" => tau,
                   "sigma" => sigma, "xmui" => 0.0, "vmui" => 0.0);
        θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
        asymp_cov = ctcrwpb_statcov(θ)
        @assert asymp_cov[1, 1] ≈ s11 "problem: asymptotic variance not equal to requested"
        @assert asymp_cov[2, 2] ≈ s22 "problem: asymptotic variance not equal to requested"

        for j in 1:n_job_repeats
            jobid += 1;
            args = Dict("par" => string(par),
                        "npar" => npar,
                        "model" => model,
                        "dt" => dt,
                        "T" => T,
                        "blocksize" => blocksize,
                        "nsim" => nsim,
                        "burnin" => burnin,
                        "resampling" => resampling,
                        "verbose" => true,
                        "jobid" => jobid);
            fn *= generate_call(script_name, args, ARGUMENT_CONFIG["bbcpf-blocksize-ctcrwp"]);
        end
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "blocksize-vs-potential-ctcrwp-b"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-bbcpf-blocksize-ctcrwp");

    fn = "";
    model = "CTCRWP_B";
    n_job_repeats = 1;
    #nreps = 2;
    nsim = 20000;
    sigma = 1.0;
    burnin = 1000;
    T_power = 6;
    jobid = 0;
    dt_power = -7;
    for blocksize in 2.0 .^ (dt_power:T_power),
        tau in [0.5 / 4, 0.5, 2.0],
        npar in [2, 4, 8, 16, 32],
        resampling in ["killing", "systematic", "multinomial"]

        dt = 2.0 ^ dt_power;
        T = 2.0 ^ T_power;

        s11 = s22 = 1.0;
        betav = sigma * sigma * inv(2.0 * s11);
        betax = solve_βx(sigma, s11, s22);
        @assert !isinf(betax) && !isnan(betax) "problem with solved value of `betax`";
        par = Dict("betav" => betav, "betax" => betax, "tau" => tau,
                   "sigma" => sigma, "xmui" => 0.0, "vmui" => 0.0);
        θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
        asymp_cov = ctcrwpb_statcov(θ)
        @assert asymp_cov[1, 1] ≈ s11 "problem: asymptotic variance not equal to requested"
        @assert asymp_cov[2, 2] ≈ s22 "problem: asymptotic variance not equal to requested"

        for j in 1:n_job_repeats
            jobid += 1;
            args = Dict("par" => string(par),
                        "npar" => npar,
                        "model" => model,
                        "dt" => dt,
                        "T" => T,
                        "blocksize" => blocksize,
                        "nsim" => nsim,
                        "burnin" => burnin,
                        "resampling" => resampling,
                        "verbose" => true,
                        "jobid" => jobid);
            fn *= generate_call(script_name, args, ARGUMENT_CONFIG["bbcpf-blocksize-ctcrwp"]);
        end
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "block-metrics-sys-ctcrwp"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * "block-metrics-sys-ctcrwp");

    fn = "";
    nreps = 50;
    T_power = 6;
    jobid = 0;
    dt_power = -7;
    blocksizes = collect(2.0 .^ (dt_power:T_power));
    for npar in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        sigma in [0.03125, 0.125, 0.5, 2.0, 4.0]

        dt = 2.0 ^ dt_power;
        T = 2.0 ^ T_power;
        tau = 1.0;

        s11 = s22 = 1.0;
        betav = sigma * sigma * inv(2.0 * s11);
        betax = solve_βx(sigma, s11, s22);
        @assert !isinf(betax) && !isnan(betax) "problem with solved value of `betax`";
        par = Dict("betav" => betav, "betax" => betax, "tau" => tau,
                   "sigma" => sigma, "xmui" => 0.0, "vmui" => 0.0);
        θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
        asymp_cov = ctcrwpb_statcov(θ)
        @assert asymp_cov[1, 1] ≈ s11 "problem: asymptotic variance not equal to requested"
        @assert asymp_cov[2, 2] ≈ s22 "problem: asymptotic variance not equal to requested"

        jobid += 1;
        args = Dict("par" => string(par),
                    "npar" => npar,
                    "dt" => dt,
                    "T" => T,
                    "nreps" => nreps,
                    "blocksizes" => replace(string(blocksizes), ' ' => ""),
                    "verbose" => true,
                    "jobid" => jobid);
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG["block-metrics-sys-ctcrwp"]);
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "ctcrwp-plu-est"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * "bbcpf-blocksize-ctcrwp");

    fn = "";
    model = "CTCRWP_B";
    n_job_repeats = 1;
    #nreps = 2;
    nsim = 1000;
    burnin = 200;
    T_power = 6;
    jobid = 0;
    dt_power = -7;
    tau = 1.0;
    for blocksize in 2.0 .^ (dt_power:T_power),
        npar in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        sigma in [0.03125, 0.125, 0.5, 2.0, 4.0],
        resampling in ["systematic"]

        dt = 2.0 ^ dt_power;
        T = 2.0 ^ T_power;

        s11 = s22 = 1.0;
        betav = sigma * sigma * inv(2.0 * s11);
        betax = solve_βx(sigma, s11, s22);
        @assert !isinf(betax) && !isnan(betax) "problem with solved value of `betax`";
        par = Dict("betav" => betav, "betax" => betax, "tau" => tau,
                   "sigma" => sigma, "xmui" => 0.0, "vmui" => 0.0);
        θ = (; Dict(Symbol.(keys(par)) .=> values(par))...);
        asymp_cov = ctcrwpb_statcov(θ)
        @assert asymp_cov[1, 1] ≈ s11 "problem: asymptotic variance not equal to requested"
        @assert asymp_cov[2, 2] ≈ s22 "problem: asymptotic variance not equal to requested"

        for j in 1:n_job_repeats
            jobid += 1;
            args = Dict("par" => string(par),
                        "npar" => npar,
                        "model" => model,
                        "dt" => dt,
                        "T" => T,
                        "blocksize" => blocksize,
                        "nsim" => nsim,
                        "burnin" => burnin,
                        "resampling" => resampling,
                        "verbose" => true,
                        "jobid" => jobid);
            fn *= generate_call(script_name, args, ARGUMENT_CONFIG["bbcpf-blocksize-ctcrwp"]);
        end
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "block-metrics-sys-lgcpr"
    script_id = experiment;
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * script_id);
    fn = "";
    npar = 8;
    verbose = true;
    nreps = 50;
    blocksizes = 2.0 .^ (-6:5)
    jobid = 0;

    jobid += 1;
    args = Dict("npar" => npar,
                "nreps" => nreps,
                "blocksizes" => replace(string(blocksizes), ' ' => ""),
                "seed" => 12345,
                "verbose" => verbose,
                "jobid" => jobid);
    fn *= generate_call(script_name, args, ARGUMENT_CONFIG[script_id]);
    open(joinpath(task_list_outpath, outfile), "w") do io
         write(io, fn);
    end
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "lgcpr-opt-vs-hom"
    script_id = "bbcpf-lgcpr"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * script_id);

    fn = "";
    burnin = 1000;
    nsim = 25000;
    seed = 12345;
    npar = 8;
    verbose = true;
    resampling = "systematic";
    cpf_bs_with_blocksize = 2.0 ^ -6;

    # Get paths to desired blockfiles relative to project root.
    blockfiles_basefolder = joinpath(PROJECT_ROOT, "input", "lgcpr-data");
    blockfilenames = readdir(blockfiles_basefolder);
    blockfilenames = blockfilenames[startswith.(blockfilenames, "blocking")];
    blockfilenames = blockfilenames[contains.(blockfilenames, "cpf-metrics")]
    blockfiles = joinpath.("input", "lgcpr-data", blockfilenames);

    # Write tasks for experiments with blockfiles.
    for blockfile in blockfiles
        args = Dict("npar" => npar,
                    "nsim" => nsim,
                    "burnin" => burnin,
                    "resampling" => resampling,
                    "seed" => seed,
                    "blockfile" => blockfile,
                    "verbose" => verbose
                    );
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG[script_id]);
    end

    # Write tasks for experiments with constant blocksize.
    for blocksize_exponent in -6:5
        blocksize = 2.0 ^ blocksize_exponent
        backward_sampling = false;
        if blocksize == cpf_bs_with_blocksize
            backward_sampling = true;
        end
        args = Dict("npar" => npar,
                    "blocksize" => blocksize,
                    "nsim" => nsim,
                    "burnin" => burnin,
                    "backward-sampling" => backward_sampling,
                    "resampling" => resampling,
                    "seed" => seed,
                    "verbose" => verbose);
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG[script_id]);
    end

    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "block-metrics-sys-ctcrwt"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}", "run-" * "block-metrics-sys-ctcrwt");
    #script_name = joinpath(SCRIPTS_PATH, "run-" * "block-metrics-sys-ctcrwt");

    fn = "";
    #nreps = 50;
    jobid = 0;
    dt_power = -7;
    watercoef = 0.0;

    for npar in [512],
        nreps in [25],
        dt_power in [-7, -3]

        dt = 2.0 ^ dt_power;
        blocksizes = collect(2.0 .^ (dt_power:2));
        datafile = joinpath(INPUT_PATH, "ctcrwt-data", "ctcrwt-data-dt$dt-watercoef$watercoef.jld2");
        if !isfile(datafile)
            throw(ArgumentError("did not find file $datafile"));
        end
        jobid += 1;
        args = Dict("npar" => npar,
                    "nreps" => nreps,
                    "blocksizes" => replace(string(blocksizes), ' ' => ""),
                    "datafile" => datafile,
                    "verbose" => true);
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG["block-metrics-sys-ctcrwt"]);
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end

let experiment = "terrain-sim"
    script_id = "terrain-sim"
    outfile = experiment * "-tasklist";
    script_name = joinpath(raw"${BBCPF_JL_SCRIPT_FOLDER}",  "run-" * script_id); 
    #script_name = joinpath(SCRIPTS_PATH, "run-" * script_id);
    fn = "";
    burnin = 1000;
    nsim = 10000;
    npar = 16;
    verbose = true;
    resampling = "systematic";

    for alg in ["bbcpf", "cpfbs"],
        Δ in [0.0078125, 0.125]

        datafile = joinpath(INPUT_PATH, "ctcrwt-data", "ctcrwt-data-dt$Δ-watercoef0.0.jld2");
        if !isfile(datafile)
            throw("file not found: $datafile");
        end
        if alg == "cpfbs"
            blocksize = Δ;
        else
            blocksize = 1.0;
        end

        args = Dict("npar" => npar,
                    "blocksize" => blocksize,
                    "nsim" => nsim,
                    "datafile" => datafile,
                    "burnin" => burnin,
                    "resampling" => resampling,
                    "verbose" => verbose);
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG[script_id]);
    end

    for Δ in [0.0078125, 0.125]
        datafile = joinpath(INPUT_PATH, "ctcrwt-data", "ctcrwt-data-dt$Δ-watercoef0.0.jld2");
        if !isfile(datafile)
            throw("file not found: $datafile");
        end
        blockfile = joinpath(INPUT_PATH, "ctcrwt-data", "blocking-ctcrwt-dt$Δ-watercoef0.0-npar512-nreps25.jld2");
        if !isfile(blockfile)
            throw("file not found: $blockfile");
        end
        args = Dict("npar" => npar,
                    "nsim" => nsim,
                    "datafile" => datafile,
                    "blockfile" => blockfile,
                    "burnin" => burnin,
                    "resampling" => resampling,
                    "verbose" => verbose);
        fn *= generate_call(script_name, args, ARGUMENT_CONFIG[script_id]);
    end
    open(joinpath(task_list_outpath, outfile), "w") do io
           write(io, fn);
    end;
    msg = string("Wrote ", countlines(joinpath(task_list_outpath, outfile)), " jobs",
                 " for experiment ", experiment);
    println(msg);
end


