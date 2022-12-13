include("../../../../config.jl");
include(joinpath(LIB_PATH, "model-helpers.jl"));
using Statespace: ModelIdentifier
using NLGSSM
using Distributions
struct CTCRW <: ModelIdentifier end

CTCRW_MODEL = let MU_X = 1, V_X = 2, MU_Y = 3, V_Y = 4, MU_X_OBS = 1, MU_Y_OBS = 2#, PARAMETERISATION = :paper

    function mu_1!(mu_pred::AbstractArray{<: Real, 1}, data, θ)
        #Δ = data.firstshift; #data.dt[1];
        #T = ctcrwT(Δ, θ);
        out1 = SVector{2, Float64}(θ.muxi, θ.vxi);
        out2 = SVector{2, Float64}(θ.muyi, θ.vyi);
        mu_pred .= vcat(out1, out2);

        #mu_pred[MU_X] = p[:mu_x_init_mean];
        #mu_pred[MU_Y] = p[:mu_y_init_mean];
        #mu_pred[V_X] = exp(p[:log_v_x_init_mean]);
        #mu_pred[V_Y] = exp(p[:log_v_y_init_mean]);
        nothing;
    end

    function P_1!(P_pred::AbstractArray{<: Real, 2}, data, θ)
        P_pred .= 0.0;
        #Δ = data.firstshift; #data.dt[1];
        #T = ctcrwT(Δ, θ);
        #RxRt = ctcrwRxRt(Δ, θ);
        P_init = @SMatrix [(θ.musigmai * θ.musigmai) 0.0;
                            0.0 (θ.sigma * θ.sigma / (2.0 * θ.beta))];
        out = P_init;# * transpose(T) .+ RxRt;
        view(P_pred, 1:2, 1:2) .= out;
        view(P_pred, 3:4, 3:4) .= out;
        nothing;
        #P_pred[MU_X, MU_X] = exp(p[:log_x_var]);
        #P_pred[MU_Y, MU_Y] = exp(p[:log_y_var]);

        #if PARAMETERISATION == :paper
            # This matches with crawl if the parameterisation is 'sigma' and 'beta' in the Johnson (2008) paper.
        #    P_pred[V_X, V_X] = exp(2.0 * p[:log_sigma]) / exp(p[:log_beta]);
        #    P_pred[V_Y, V_Y] = exp(2.0 * p[:log_sigma]) / exp(p[:log_beta]);
        #else
        #    P_pred[V_X, V_X] = exp(2.0 * p[:log_sigma]) * exp(p[:log_beta]);
        #    P_pred[V_Y, V_Y] = exp(2.0 * p[:log_sigma]) * exp(p[:log_beta]);
        #end
        #nothing
    end

    function T!(mu_pred::AbstractArray{<: Real, 1}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        # Current filtered values.
        Δ = data.dt[t];
        T = ctcrwT(Δ, θ);
        out1 = T * SVector{2, Float64}(mu_filt[MU_X], mu_filt[V_X]);
        out2 = T * SVector{2, Float64}(mu_filt[MU_Y], mu_filt[V_Y]);
        mu_pred .= vcat(out1, out2);
        nothing;

        #MU_X_OLD = mu_filt[MU_X];
        #MU_Y_OLD = mu_filt[MU_Y];
        #V_X_OLD = mu_filt[V_X];
        #V_Y_OLD = mu_filt[V_Y];

        #beta = exp(p[:log_beta]);
        #dt = data[:dt, t];

        # Propagate.
        #mu_pred[MU_X] = MU_X_OLD + V_X_OLD * (1.0 - exp(-beta * dt)) / beta;
        #mu_pred[MU_Y] = MU_Y_OLD + V_Y_OLD * (1.0 - exp(-beta * dt)) / beta;
        #mu_pred[V_X] = V_X_OLD * exp(-beta * dt);
        #mu_pred[V_Y] = V_Y_OLD * exp(-beta * dt);
        #nothing
    end

    function R!(R::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        R .= 0.0;
        Δ = data.dt[t]; #[:dt, t];
        Δ <= 0.0 && (return nothing;)
        RxRt = ctcrwRxRt(Δ, θ);
        view(R, 1:2, 1:2) .= RxRt;
        view(R, 3:4, 3:4) .= RxRt;
        #println("theta is $θ");
        Rf = cholesky!(R);
        R .= Rf.L;
        nothing;

        # sigma = exp(p[:log_sigma]);
        # beta = exp(p[:log_beta]);
        #
        # var_mu_1 = exp(log(2.0) - log(beta) + logcdf(Exponential(1.0 / beta), dt));
        # var_mu_2 = exp(logcdf(Exponential(1.0 / (2.0 * beta)), dt) - log(2.0) - log(beta));
        #
        # # Like in the paper:
        # if PARAMETERISATION == :paper
        #     var_mu = (sigma / beta) ^ 2.0 * (dt - var_mu_1 + var_mu_2);
        #     var_v = exp(2.0 * log(sigma) - log(2.0) - log(beta) + logcdf(Exponential((1.0 / (2.0 * beta))), dt));
        #     cov_mu_v = 0.5 * (sigma / beta) ^ 2.0 * (1.0 - 2.0 * exp(-beta * dt) + exp(-2.0 * beta * dt));
        # else
        #     # In crawl parameterisation:
        #     var_mu = (sigma ^ 2.0) * (dt - var_mu_1 + var_mu_2);
        #     var_v = (sigma ^ 2.0 / 2.0) * exp(log(beta) + logcdf(Exponential(1.0 / (2.0 * beta)), dt));
        #     cov_mu_v = 0.5 * (sigma) ^ 2.0 * (1.0 - 2.0 * exp(-beta * dt) + exp(-2.0 * beta * dt));
        # end
        #
        # R[MU_X, MU_X] = R[MU_Y, MU_Y] = var_mu;
        # R[V_X, V_X] = R[V_Y, V_Y] = var_v;
        # R[MU_X, V_X] = R[MU_Y, V_Y] = cov_mu_v;
        # R[V_X, MU_X] = R[V_Y, MU_Y] = cov_mu_v;
        #
        # Rf = cholesky!(R); # Compute Cholesky factorization in place.
        # R .= Rf.L; # Extract lower diagonal matrix.
        # nothing
    end

    function Tj!(Tj::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        #mu_pred[MU_X] = MU_X_OLD + V_X_OLD * (1.0 - exp(-beta * dt)) / beta;
        #mu_pred[MU_Y] = MU_Y_OLD + V_Y_OLD * (1.0 - exp(-beta * dt)) / beta;
        #mu_pred[V_X] = V_X_OLD * exp(-beta * dt);
        #mu_pred[V_Y] = V_Y_OLD * exp(-beta * dt);

        Tj .= 0.0;
        Δ = data.dt[t];
        T = ctcrwT(Δ, θ);
        view(Tj, 1:2, 1:2) .= T;
        view(Tj, 3:4, 3:4) .= T;
        # w.r.t MU_X:
        #Tj[MU_X, MU_X] = 1.0;

        # w.r.t MU_Y:
        #Tj[MU_Y, MU_Y] = 1.0;

        # w.r.t V_X:
        #Tj[MU_X, V_X] = (1.0 - exp(-beta * dt)) / beta;
        #Tj[V_X, V_X] = exp(-beta * dt);

        # w.r.t V_Y:
        #Tj[MU_Y, V_Y] = (1.0 - exp(-beta * dt)) / beta;
        #Tj[V_Y, V_Y] = exp(-beta * dt);
        nothing;
    end

    function Z!(Z::AbstractArray{<: Real, 1}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Z .= 0.0;
        Z[MU_X_OBS] = mu_pred[MU_X];
        Z[MU_Y_OBS] = mu_pred[MU_Y];
        nothing
    end

    function Zj!(Zj::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Zj .= 0.0;
        Zj[MU_X_OBS, MU_X] = 1.0;
        Zj[MU_Y_OBS, MU_Y] = 1.0;
        nothing
    end

    function H!(H::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        H .= 0.0;
        H[MU_X_OBS, MU_X_OBS] = H[MU_Y_OBS, MU_Y_OBS] = θ.tau; #exp(p[:log_tau]);
        nothing
    end

    NonLinearGaussianSSM(CTCRW,
                         statenames = [:MU_X, :V_X, :MU_Y, :V_Y],
                         obsnames = [:X, :Y],
                         mu_1! = mu_1!, P_1! = P_1!,
                         T! = T!, Tj! = Tj!, R! = R!,
                         Z! = Z!, Zj! = Zj!, H! = H!);
end;
println("Model named CTCRW_MODEL loaded.");
#include("CTCRW_functions.jl");
