using StaticArrays
using Random
using Distributions
include("../../../../config.jl");
include(joinpath(LIB_PATH, "model-helpers.jl"));
include(joinpath(LIB_PATH, "Gaussians.jl"));

mutable struct CTCRWPBParticle <: Particle
    v::Float64
    x::Float64
end
function CTCRWPBParticle()
    CTCRWPBParticle(NaN, NaN);
end
import Base.copy!
function copy!(dest::CTCRWPBParticle, src::CTCRWPBParticle)
    dest.x = src.x;
    dest.v = src.v;
    dest;
end
function copy!(dest::CTCRWPBParticle, src::SVector{2, <: Real})
    dest.v = src[1];
    dest.x = src[2];
    dest;
end
function SVector{2, T}(p::CTCRWPBParticle) where T
    SVector{2, T}(p.v, p.x);
end

# Check if two particles have exactly the same value.
function eq(p1::CTCRWPBParticle, p2::CTCRWPBParticle)
    #isapprox(p1.x, p2.x) && isapprox(p1.v, p2.v);
    p1.x == p2.x && p1.v == p2.v;
end
function dist(p1::CTCRWPBParticle, p2::CTCRWPBParticle)
    norm(SVector{2, Float64}(p1) - SVector{2, Float64}(p2));
end


CTCRWP_B_PF = let
    function Mi!(p::CTCRWPBParticle, data, θ)
        #Δ = data.dt[1];
        #T = ctcrwpb_T(Δ, θ);
        #RxRt = ctcrwpb_RxRt(Δ, θ);
        #P_init = ctcrwpb_statcov(θ); # Stationary covariance matrix.

        #μ = T * SVector{2, Float64}(θ.vmui, θ.xmui);
        #Σ = Hermitian(T * P_init * transpose(T) .+ RxRt);

        μ = SVector{2, Float64}(θ.vmui, θ.xmui);
        Σ = ctcrwpb_statcov(θ);
        dist = Gaussian(μ, LowerTriCholesky(SMatrix{2, 2, Float64, 4}(cholesky(Σ).L)));
        sample = rand(dist);
        copy!(p, sample);
        nothing
    end
    function M!(pnext::CTCRWPBParticle, pcur::CTCRWPBParticle, t::Int, data, θ)
        Δ = data.dt[t - 1];
        if Δ <= 0.0
            copy!(pnext, pcur);
            return nothing;
        end
        T = ctcrwpb_T(Δ, θ);
        v = SVector{2, Float64}(pcur);
        μ = T * v;
        Σ = Hermitian(ctcrwpb_RxRt(Δ, θ));

        dist = Gaussian(μ, LowerTriCholesky(SMatrix{2, 2, Float64, 4}(cholesky(Σ).L)));
        sample = rand(dist);
        copy!(pnext, sample);
        nothing;
    end
    function lM(pnext::CTCRWPBParticle, pcur::CTCRWPBParticle, t::Int, data, θ)
        Δ = data.dt[t - 1];
        #if Δ <= 0.0
        #    copy!(pnext, pcur);
        #    return nothing;
        #end
        T = ctcrwpb_T(Δ, θ);
        v = SVector{2, Float64}(pcur);
        μ = T * v;
        Σ = Hermitian(ctcrwpb_RxRt(Δ, θ));
        dist = Gaussian(μ, LowerTriCholesky(SMatrix{2, 2, Float64, 4}(cholesky(Σ).L)));

        logpdf(dist, SVector{2, Float64}(pnext));
    end

    function lG(pprev::CTCRWPBParticle, pcur::CTCRWPBParticle, t::Int, data, θ)
        dt = data.dt[t];
        dt <= 0.0 && (return 0.0;)
        -dt * pcur.x * pcur.x / (2.0 * θ.tau * θ.tau);
    end
    function lGi(p::CTCRWPBParticle, data, θ)
        dt = data.dt[1];
        -dt * p.x * p.x  / (2.0 * θ.tau * θ.tau);
    end
    GenericSSM(CTCRWPBParticle, Mi!, nothing, M!, lM, lGi, lG);
end

println("Model named CTCRWP_B_PF loaded.");
