include("../../../../config.jl");
using Statespace: ModelIdentifier
using NLGSSM
using Distributions
struct BrownianMotionWithObs <: ModelIdentifier end

BM_MODEL = let
    function mu_1!(mu_pred::AbstractArray{<: Real, 1}, data, θ)
        @inbounds mu_pred[1] = 0.0;
    end

    function P_1!(P_pred::AbstractArray{<: Real, 2}, data, θ)
        @inbounds P_pred[1, 1] = θ.sigmai * θ.sigmai;
    end

    function T!(mu_pred::AbstractArray{<: Real, 1}, mu_filt::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        @inbounds mu_pred[1] = mu_filt[1];
    end

    function R!(R::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        Δ = data.dt[i];
        @inbounds R[1, 1] = θ.sigma * √(Δ);
        nothing;
    end

    function Tj!(Tj::AbstractArray{<: Real, 2}, mu_filt::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        @inbounds Tj[1, 1] = 1.0;
    end

    function Z!(Z::AbstractArray{<: Real, 1}, mu_pred::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        @inbounds Z[1] = mu_pred[1];
        nothing;
    end

    function Zj!(Zj::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        @inbounds Zj[1, 1] = 1.0;
        nothing
    end

    function H!(H::AbstractArray{<: Real, 2}, mu_pred::AbstractArray{<: Real, 1}, i::Integer, data, θ)
        @inbounds H[1, 1] = θ.tau;
        nothing
    end

    NonLinearGaussianSSM(BrownianMotionWithObs,
                         statenames = [:MU],
                         obsnames = [:Y],
                         mu_1! = mu_1!, P_1! = P_1!,
                         T! = T!, Tj! = Tj!, R! = R!,
                         Z! = Z!, Zj! = Zj!, H! = H!);
end;
println("Model named BM_MODEL loaded.");
#include("CTCRW_functions.jl");
