using StaticArrays
using Distributions
include("../../lib/math-functions.jl");

mutable struct LGCPRParticle <: Particle
    s::Float64
end
LGCPRParticle() = LGCPRParticle(NaN);
import Base.copy!
function copy!(dest::LGCPRParticle, src::LGCPRParticle)
    dest.s = src.s;
    dest;
end
function copy!(dest::LGCPRParticle, src::SVector{1, <: Real})
    @inbounds dest.s = src[1];
    dest;
end
function SVector{1, Float64}(p::LGCPRParticle)
    SVector{1, Float64}(p.s);
end
function eq(p1::LGCPRParticle, p2::LGCPRParticle)
    p1.s == p2.s;
end

function logpdf_folded_normal(d::Normal, x::Real, a::Real, b::Real, order::Int = 10)
    if x <= a || x >= b
        return -Inf;
    end
    xa, xb = x, x;
    lp = logpdf(d, x);
    for i in 1:order
        xa_, xb_ = xa, xb;
        xa = 2.0 * a - xb_;
        xb = 2.0 * b - xa_;
        lp = logsumexp(lp, logpdf(d, xa))
        lp = logsumexp(lp, logpdf(d, xb))
    end
    lp;
end

LGCPR_BM_PF = let
    function Mi!(p::LGCPRParticle, data, θ)
        p.s = θ.sigmai * randn(); # Draw from distribution at physical time 0.0.
        nothing;
    end
    function M!(pnext::LGCPRParticle, pcur::LGCPRParticle, i, data, θ)
        Δ = data.dt[i - 1];
        pnext.s = pcur.s + √(Δ) * θ.sigma * randn();
        nothing;
    end
    function lGi(p::LGCPRParticle, data, θ)
        if p.s <= data.a || p.s >= data.b
            return -Inf;
        end
        @inbounds Δ = data.dt[1];
        lg = -Δ * θ.beta * exp(-θ.alpha * p.s);
        if Δ > 0.0 && @inbounds data.y[1]
            # There was an observation in the first interval.
            lg += log(θ.beta) - θ.alpha * p.s;
        end

        # Reflected BM proposal density.
        dist = Normal(0.0, θ.sigmai);
        log_M = logpdf_folded_normal(dist, p.s, data.a, data.b, 10);

        # BM proposal density.
        log_prop_M = logpdf(dist, p.s);

        # Correcting for proposal being BM.
        log_M - log_prop_M + lg;
    end
    function lG(pprev::LGCPRParticle, pcur::LGCPRParticle, i, data, θ)
        if pcur.s <= data.a || pcur.s >= data.b
            return -Inf;
        end
        @inbounds Δ = data.dt[i];
        lg = -Δ * θ.beta * exp(-θ.alpha * pcur.s);
        if Δ > 0.0 && @inbounds data.y[i]
            # There was an observation in the ith interval.
            lg += log(θ.beta) - θ.alpha * pcur.s;
        end

        # Reflected BM proposal density.
        Δprev = data.dt[i - 1];
        dist = Normal(pprev.s, √(Δprev) * θ.sigma);
        log_M = logpdf_folded_normal(dist, pcur.s, data.a, data.b, 10);

        # BM proposal density.
        log_prop_M = logpdf(dist, pcur.s);

        # Correcting for proposal being BM.
        log_M - log_prop_M + lg;
    end
    GenericSSM(LGCPRParticle, Mi!, nothing, M!, nothing, lGi, lG);
end
println("Model named LGCPR_BM_PF loaded.");

## Some functions for simulation.

# A 'reflection' of x onto (a,b)
function reflect(x::Real, a::Real, b::Real)
    if a < x < b
        return x
    else
        ba = b-a
        y = abs(x - a)
        d, r = divrem(y, ba)
        if rem(d, 2) == 1
            r = ba-r
        end
        return a + r
    end
end

function Mi_RBM!(p::LGCPRParticle, data, θ)
    p.s = reflect(θ.sigmai * randn(), data.a, data.b);
    nothing;
end
function M_RBM!(pnext::LGCPRParticle, pcur::LGCPRParticle, i, data, θ)
    Δ = data.dt[i - 1];
    pnext.s = reflect(pcur.s + √(Δ) * θ.sigma * randn(), data.a, data.b);
    nothing;
end

"""
Simulate from an inhomogeneous Poisson process defined by a piecewise constant
intensity function. Input is intensity values and lengths of constant intervals.
"""
function sample_pwc_ipp(λs, dt)
    @assert length(λs) == length(dt) "input vectors need to have same length.";
    λ_tot = sum(dt .* λs);
    nobs = rand(Poisson(λ_tot));

    # Sample from density defined by λs / λ_tot and dt.
    # First draw indices of intervals where observations came,
    # then draw times within intervals. Time is uniform within interval
    # since density is constant within an interval.
    prob = λs / λ_tot;
    intervals_w_obs = wsample(Base.OneTo(length(prob)), prob, nobs);
    t = vcat([0.0], cumsum(dt));
    τ = zeros(nobs); # Observation times.
    for (k, i) = enumerate(intervals_w_obs)
        τ[k] = t[i] + dt[i] * rand();
    end
    sort!(τ);
    τ;
end

"""
Simulate data from the LGCPR model.
"""
function simulate_lgcpr_data(base_dt, θ, bounds)
    states = [LGCPRParticle() for i in 1:length(base_dt)];
    simulate_states!(states, Mi_RBM!, M_RBM!, merge((dt = base_dt,), bounds), θ);

    # Intensity values at each interval.
    λs = θ.beta * exp.(-θ.alpha * map(x -> x.s, states));

    # Sample from IPP defined by intensities.
    τ = sample_pwc_ipp(λs, base_dt);

    # Consolidate observation times and discretisation points to one vector and
    # compute observations.
    times = sort!(union([0.0], cumsum(base_dt), τ));
    obs = zeros(Bool, length(times) - 1);
    for (i, t) in enumerate(times[1:end - 1])
        if t in τ
            obs[i] = true;
        end
    end
    λs, states, τ, times, obs;
end
