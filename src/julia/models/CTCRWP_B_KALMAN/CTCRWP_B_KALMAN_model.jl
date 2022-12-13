using Statespace: ModelIdentifier
using NLGSSM
using Distributions
using StaticArrays
struct CTCRWP_B <: ModelIdentifier end
include("../../../../config.jl");
include(joinpath(LIB_PATH, "model-helpers.jl"));


# The model we want to infer has form
# p(x0) p(x1 | x0) p(y1 | x1) p(x2 | x1) p(y2 | x2) ...
CTCRWP_B_KALMAN = let V_X = 1, MU_X = 2, MU_X_OBS = 1

    function mu_1!(mu_pred::AbstractArray{<: Real, 1}, data, θ)
        #Δ = data.dt[1];

        # Mean at time 0.0.
        mui = SVector{2, Float64}(θ.vmui, θ.xmui);
        out = mui;
        #out = ctcrwpb_T(Δ, θ) * mui;

        mu_pred[MU_X] = out[MU_X];
        mu_pred[V_X] = out[V_X];
        nothing;
    end

    function P_1!(P_pred::AbstractArray{<: Real, 2}, data, θ)
        #Δ = data.dt[1];

        #T = ctcrwpb_T(Δ, θ)
        #RxRt = ctcrwpb_RxRt(Δ, θ);
        P_init = ctcrwpb_statcov(θ); # Stationary covariance matrix.

        # Covariance at time 0.0.
        #P_pred .= T * P_init * transpose(T) .+ RxRt;
        P_pred .= P_init;
        nothing;
    end

    # NOTE: Transition from t to t + 1. (called for t in [1, K - 1])
    function T!(mu_pred::AbstractArray{<: Real, 1}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        cur = SVector{2, Float64}(mu_filt[V_X], mu_filt[MU_X]);
        Δ = data.dt[t];
        out = ctcrwpb_T(Δ, θ) * cur;

        mu_pred[MU_X] = out[MU_X];
        mu_pred[V_X] = out[V_X];
        nothing;
    end

    # NOTE: Related to transition from t to t + 1. (called for t in [1, K - 1])
    function R!(R::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        R .= 0.0;
        Δ = data.dt[t];
        Δ <= 0.0 && (return nothing;)

        R .= ctcrwpb_RxRt(Δ, θ);
        Rf = cholesky!(R); # Compute Cholesky factorization in place.
        R .= Rf.L; # Extract lower diagonal matrix.
        nothing;
    end

    # NOTE: Related to transition from t to t + 1. (called for t in [1, K - 1])
    function Tj!(Tj::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Δ = data.dt[t];
        Tj .= ctcrwpb_T(Δ, θ);
        nothing;
    end

    function Z!(Z::AbstractArray{<: Real, 1}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Z[MU_X_OBS] = mu_pred[MU_X];
        nothing;
    end

    function Zj!(Zj::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Zj[MU_X_OBS, MU_X] = 1.0;
        nothing;
    end

    # NOTE: Called for t in [1, K]). Note that if observation is missing, this does _not_ get called.
    function H!(H::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, t::Integer, data, θ)
        Δ = data.dt[t];
        H[MU_X_OBS, MU_X_OBS] = θ.tau / sqrt(Δ);
        nothing;
    end
    NonLinearGaussianSSM(CTCRWP_B,
                         statenames = [:V_X, :MU_X],
                         obsnames = [:X],
                         mu_1! = mu_1!, P_1! = P_1!,
                         T! = T!, Tj! = Tj!, R! = R!,
                         Z! = Z!, Zj! = Zj!, H! = H!);
end;
println("Model named CTCRWP_B_KALMAN loaded.");
