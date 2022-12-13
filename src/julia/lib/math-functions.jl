include("asymptotic-variance.jl");

"""
Normalise a vector of weight logarithms, `log_weights`, in place.
After normalisation, the weights are in the linear scale.
Additionally, the logarithm of the linear scale mean weight is returned.
"""
function normalise_logweights!(log_weights::AbstractVector{<: Real})
  m = maximum(log_weights);
  if isapprox(m, -Inf) # To avoid NaN in case that all values are -Inf.
    log_weights .= zero(eltype(log_weights));
    return -Inf;
  end
  log_weights .= exp.(log_weights .- m);
  log_mean_weight = m + log(mean(log_weights));
  normalize!(log_weights, 1);
  log_mean_weight;
end

"""
Compute log(sum(exp.(`x`))) in a numerically stable way.
"""
function logsumexp(x::AbstractArray{<: Real})
  m = maximum(x);
  isapprox(m, -Inf) && (return -Inf;) # If m is -Inf, without this we would return NaN.
  s = 0.0;
  for i in eachindex(x)
    @inbounds s += exp(x[i] - m);
  end
  m + log(s);
end

"""
Return `log(a + b)` given `log(a)` and `log(b)`.
"""
function logsumexp(loga::Real, logb::Real)
    m = max(loga, logb);
    isapprox(m, -Inf) && (return -Inf;)
    s = exp(loga - m) + exp(logb - m);
    m + log(s);
end

"""
Compute log(sum(exp.(`mul` * `x`))) in a numerically stable way.
"""
function logsumexp(x::AbstractArray{<: Real}, mul::Real)
  m = mul * maximum(x);
  isapprox(m, -Inf) && (return -Inf;) # If m is -Inf, without this we would return NaN.
  s = 0.0;
  for i in eachindex(x)
    @inbounds s += exp(mul * x[i] - m);
  end
  m + log(s);
end

"""
Return `log(a + b)` given `log(a)` and `log(b)`.
"""
function logsumexp(loga::Real, logb::Real)
    m = max(loga, logb);
    isapprox(m, -Inf) && (return -Inf;)
    s = exp(loga - m) + exp(logb - m);
    m + log(s);
end

"""
Compute the effective sample size (ESS) of weights from unnormalised log weights.
If `log = true`, return log of ESS instead.
"""
function ess_unnorm_log(x::AbstractArray{<: Real}, log::Bool = false)
    log_ess = 2.0 * logsumexp(x) - logsumexp(x, 2.0);
    if log
        return log_ess;
    end
    exp(log_ess);
end

function esjd(x::AVec{<: Real})
    esjd = zero(eltype(x)); # Current mean.
    for i in 2:length(x)
        j = i - 1; # Index of distance computed at this iteration.
        d2 = (x[i] - x[i - 1]) * (x[i] - x[i - 1]);
        esjd = (j - 1) * esjd / j + d2 / j;
    end
    esjd;
end

"""
    iact(x)

Calculate integrated autocorrelation of the sequence 'x' using an adaptive window
truncated autocorrelation sum estimator.
"""
function iact(x::AVec{<: Real})
    n = length(x);

    # Calculate standardised X.
    x_ = (x .- mean(x)) / sqrt(var(x));

    # The value C is suggested by Sokal according to
    # http://dfm.io/posts/autocorr/
    C = max(5.0, log10(n));

    # Compute the IACT by summing the autocorrelations
    # up to an index dependent on C.
    tau = 1.0;
    for k in 1:(n-1)
        tau += 2.0 * acf_standardised_x(x_, k);
        if k > C * tau
            break;
        end
    end
    tau;
end

function neff(x::AVec{<: Real})
    length(x) / iact(x);
end

function iactBM(x::AVec{<: Real})
    estimateBM(x) / var(x);
end

function neffBM(x::AVec{<: Real})
    length(x) / iactBM(x);
end

function iactSV(x::AVec{<: Real})
    estimateSV(x) / var(x);
end

function neffSV(x::AVec{<: Real})
    length(x) / iactSV(x);
end

using StatsBase: autocov
function autocor_known_meanvar(x::AVec{<: Real}, μ::Real, σ²::Real,
                               lags = collect(0:(min(size(x,1)-1, 10*log10(size(x,1))))))
    autocov(x .- μ, convert.(Int, lags); demean = false) ./ σ²;
end

function distsqr(p1, p2)
    n = dist(p1, p2);
    n * n;
end

using Statistics: mean, var
using LinearAlgebra

"""
Symmetrise a matrix in place. Useful for dealing with small numerical
inconsistencies that make the Cholesky decomposition fail, for instance.
"""
function symmetrise!(X::AMat{<: Real})
    for j in 2:size(X)[2]
        for i in 1:(j-1)
            m = (X[i, j] + X[j, i]) / 2.0;
            X[i, j] = X[j, i] = m;
        end
    end
    nothing;
end

"""
Compute `"resampling rate" / N` for systematic resampling.
"""
@inline function sys_res_p(w::AbstractVector{<: AbstractFloat})
    wm = inv(length(w));
    mapreduce(w_i -> abs(w_i - wm), +, w) / 2.0;
end

"""
Compute the autocorrelation at lag `lag` for a univariate series `x`.
The series `x` is assumed standardised, eg. the mean has been subtracted
and the values have been divided by the standard deviation.
"""
function acf_standardised_x(x::AVec{<: Real}, lag::Int)
      n = length(x);
      lag < n || return 0.0
      dot(x, 1:(n - lag), x, (1 + lag):n) / (n - lag);
end
