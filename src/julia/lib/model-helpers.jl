using StaticArrays
include("find-par.jl");

function ctcrwpb_T(Δ, θ)
    βx = θ.betax;
    βv = θ.betav;
    v = if βx == βv
        Δ * exp(-βv * Δ);
    else
        (exp(-βx * Δ) - exp(-βv * Δ)) / (βv - βx);
    end
    @SMatrix [exp(-βv * Δ)   0.0;
              v   exp(-βx * Δ)];
end

function ctcrwpb_RxRt(Δ, θ)
    βv = θ.betav;
    βx = θ.betax;
    σ = θ.sigma;
    σ² = σ * σ;

    var_v = σ² * inv(2.0 * βv) * (1.0 - exp(-2.0 * βv * Δ));
    if βx == βv
        cov_x_v = σ² * inv(4.0 * βv * βv) * (1.0 + exp(-2.0 * βv * Δ) * (-2.0 * βv * Δ - 1.0));

        var_x_1 = exp(-2.0 * βv * Δ);
        var_x_2 = 1.0 + 2.0 * βv * Δ * (βv * Δ + 1.0);
        var_x = σ² * inv(4.0 * βv * βv * βv) * (1.0 - var_x_1 * var_x_2);
    else
        var_x_1 = inv(2.0 * βx) * (1.0 - exp(-2.0 * βx * Δ));
        var_x_2 = inv(2.0 * βv) * (1.0 - exp(-2.0 * βv * Δ));
        var_x_3 = -2.0 * inv(βx + βv) * (1.0 - exp(-(βx + βv) * Δ))
        var_x = σ² * inv((βv - βx) * (βv - βx)) * (var_x_1 + var_x_2 + var_x_3);

        cov_x_v_1 = inv(βv + βx) * (1.0 - exp(-(βv + βx) * Δ));
        cov_x_v_2 = -inv(2.0 * βv) * (1.0 - exp(-2.0 * βv * Δ));
        cov_x_v = σ² * inv(βv - βx) * (cov_x_v_1 + cov_x_v_2);
    end
    @SMatrix [var_v cov_x_v;
              cov_x_v var_x];
end

function ctcrwpb_statcov(θ)
    βx = θ.betax;
    βv = θ.betav;
    σ = θ.sigma;
    σ² = σ * σ;

    var_v = σ² * inv(2.0 * βv);
    if βx == βv
        cov_x_v = σ² * inv(4.0 * βv * βv);
        var_x = σ² * inv(4.0 * βv * βv * βv);
    else
        cov_x_v = σ² * inv(βv - βx) * (inv(βv + βx) - inv(2.0 * βv));
        var_x = σ² * inv((βv - βx) * (βv - βx)) * (inv(2.0 * βx) + inv(2.0 * βv) - 2.0 * inv(βx + βv));
    end
    @SMatrix [var_v cov_x_v;
              cov_x_v var_x];
end

function ctcrwT(Δ, θ)
    v = (1.0 - exp(-θ.beta * Δ)) / θ.beta;
    @SMatrix [1.0 v;
              0.0  exp(-θ.beta * Δ)];
end
function ctcrwRxRt(Δ, θ)
    sigma = θ.sigma; beta = θ.beta;
    var_mu_1 = exp(log(2.0) - log(beta) + logcdf(Exponential(1.0 / beta), Δ));
    var_mu_2 = exp(logcdf(Exponential(1.0 / (2.0 * beta)), Δ) - log(2.0) - log(beta));
    var_mu = (sigma / beta) ^ 2.0 * (Δ - var_mu_1 + var_mu_2);
    var_v = exp(2.0 * log(sigma) - log(2.0) - log(beta) + logcdf(Exponential((1.0 / (2.0 * beta))), Δ));
    cov_mu_v = 0.5 * (sigma / beta) ^ 2.0 * (1.0 - 2.0 * exp(-beta * Δ) + exp(-2.0 * beta * Δ));
    @SMatrix [var_mu cov_mu_v;
              cov_mu_v var_v];
end


function ctcrwpb_par_statdist_fix(sigma, tau; s11 = 1.0, s22 = 1.0, vmui = 0.0, xmui = 0.0)
    betav = sigma * sigma * inv(2.0 * s11);
    betax = solve_βx(sigma, s11, s22);
    @assert !isinf(betax) && !isnan(betax) "problem with solved value of `betax`";
    Dict("betav" => betav, "betax" => betax, "tau" => tau,
         "sigma" => sigma, "xmui" => xmui, "vmui" => vmui);
end
