include("../../../config.jl");

using Statespace
using NLGSSM
using LinearAlgebra
using Distributions: randn!
using Random
#include("Parameter.jl");

abstract type EstimationMethod end
struct MAP <: EstimationMethod end
struct ML <: EstimationMethod end

function init!(ssm::SSMInstance{<: NonLinearGaussianSSM}, p)
    m = ssm.model; data = ssm.data; s = ssm.storage;

    # Set filtered mean and covariance to initial values.
    s.filtered_index[] = 0;
    m.mu_1!(s.mu_pred[1], data, p);
    m.P_1!(s.P_pred[1], data, p);
    nothing
end

function loglik!(ssm::SSMInstance{<: NonLinearGaussianSSM}, p;
                 n_obs::Int = length(ssm)) 
    return filter!(ssm, p; n_obs = n_obs);# , smooth_iter = smooth_iter);
end

import Base.filter!
function filter!(ssm::SSMInstance{<: NonLinearGaussianSSM}, p;
                 n_obs::Int = length(ssm),# smooth_iter::Int = 0,
                 nodata::Bool = false) 

    # Run standard EKF.
    logLik = ekf!(ssm, p; n_obs = n_obs, nodata = nodata);
    #isfinite(logLik) || (return logLik;) # If loglik has become -Inf, break out and return -Inf.

    # Run 'smooth_iter' iterations with improving linearisation points, if smooth_iter > 0.
    #for j in 1:smooth_iter
    #    smooth!(ssm, p); # Populate s with new smoothed estimates.
    #    logLik = ekf!(ssm, p; n_obs = n_obs, linearisation = :smooth, nodata = nodata);
    #    isfinite(logLik) || (return logLik;) # If loglik has become -Inf, break out and return -Inf.
    #end
    return logLik;
end

function ekf!(ssm::SSMInstance{<: NonLinearGaussianSSM}, p;
              n_obs::Int = length(ssm), #linearisation::Symbol = :standard,
              nodata::Bool = false) 
    s = ssm.storage;
    logLik = 0.0;
    init!(ssm, p);
    for i in 1:(n_obs - 1)
        logLik += update_step!(ssm, i, p; nodata = nodata); # Update, compute mu_[i | i] and P_...
        isfinite(logLik) || (return logLik;) # If loglik has become -Inf, break out and return -Inf.
        s.filtered_index[] += 1;
        predict_step!(ssm, i, p); #) ; linearisation = linearisation); # Prediction, compute mu_[i + 1 | i] and P_...
    end
    # NOTE: No predict for last time step.
    logLik += update_step!(ssm, n_obs, p; nodata = nodata); # Update, compute mu_[i | i] and P_...
    s.filtered_index[] += 1;

    logLik;
end

function zero_if_missing_minus!(v::AbstractArray{T1, 1},
                                y::AbstractArray{T2, 1},
                                z::AbstractArray{T1, 1}) where
                                {T1 <: Real, T2 <: Union{<: Real, Missing}}
    for j in eachindex(v)
        v[j] = ismissing(y[j]) ? 0.0 : y[j] - z[j];
    end
    nothing;
end

function get_missing!(mis, x)
    for i in eachindex(x)
        ismissing(x[i]) ? mis[i] = true : mis[i] = false;
    end
    nothing
end

function stateindices(model::NonLinearGaussianSSM)
    NamedTuple{Tuple(model.statenames)}(collect(1:length(model.statenames)));
end

function obsindices(model::NonLinearGaussianSSM)
    NamedTuple{Tuple(model.obsnames)}(collect(1:length(model.obsnames)));
end

function smooth!(ssm::SSMInstance{<: NonLinearGaussianSSM}, p)
    m = ssm.model; data = ssm.data; s = ssm.storage;

    # Pointers.
    mu_smooth = s.mu_smooth; mu_filt = s.mu_filt; mu_pred = s.mu_pred;
    P_pred = s.P_pred; P_smooth = s.P_smooth;
    Tj = s.Tj; K = s.K; Imat = s.Imat; Zj = s.Zj;
    Zj_t_invF = s.Zj_t_invF; v = s.v;
    tmp_k = s.tmp_k; tmp_kxk1 = s.tmp_kxk1;
    tmp_kxk2 = s.tmp_kxk2;
    n_filtered = s.filtered_index[];

    N = s.N; N .= 0.0;
    r = s.r; r .= 0.0;
    L = s.L; L .= 0.0;

    # Smoothing loop.
    # NOTE: i == n_filtered is somewhat of a special case,
    # equivalent to setting L = 0. See p. 91 of Durbin & Koopman.
    for i in n_filtered:-1:1
        # NOTE: Do this because Tj is not necessarily define for i = n_filtered.
        # (related to transition to timepoint n_filtered + 1!)
        if i != n_filtered
            m.Tj!(Tj, mu_filt[i], i, data, p);

            #L = Tj * (Imat - K[i] * Zj[i]);
            mul!(tmp_kxk1, K[i], Zj[i]);
            tmp_kxk2 .= Imat .- tmp_kxk1;
            mul!(L, Tj, tmp_kxk2); # L is computed.
        end
        #r = Zj' * inv(F) * v + L' * r;
        Lt = transpose(L);
        mul!(tmp_k, Lt, r);
        r .= tmp_k;
        mul!(tmp_k, Zj_t_invF[i], v[i]);
        r .= r .+ tmp_k; # r is computed.

        #N = Zj[i]' * inv(F[i]) * Zj[i] + L' * N * L;
        mul!(tmp_kxk2, N, L);
        mul!(N, Lt, tmp_kxk2);
        mul!(tmp_kxk1, Zj_t_invF[i], Zj[i]);
        N .= N .+ tmp_kxk1 ; # N is computed.

        #mu_smooth[i] .= mu_pred[i] .+ P_pred[i] * r;
        mul!(tmp_k, P_pred[i], r);
        mu_smooth[i] .= mu_pred[i] .+ tmp_k; # mu_smooth[i] is computed.

        #P_smooth[i] .= P_pred[i] .- P_pred[i] * N * P_pred[i];
        mul!(tmp_kxk1, P_pred[i], N);
        mul!(tmp_kxk2, tmp_kxk1, P_pred[i]);
        P_smooth[i] .= P_pred[i] .- tmp_kxk2; # P_smooth[i] is computed.
    end
    nothing
end

"""
Return (a copy) of the smoothed means and covariances for all time points.
"""
function smoothed(s::NLGSSMState)
    n_filtered = s.filtered_index[];
    n_filtered == 0 && throw(ArgumentError("filtering has not been done yet."));
    (mean = deepcopy(s.mu_smooth[1:n_filtered]),
     cov = deepcopy(s.P_smooth[1:n_filtered]));
end

function smoothed(ssm::SSMInstance{<: NonLinearGaussianSSM})
    smoothed(ssm.storage);
end

"""
Return (a copy) of the filtered means and covariances for all time points.
"""
function filtered(s::NLGSSMState)
    n_filtered = s.filtered_index[];
    n_filtered == 0 && throw(ArgumentError("filtering has not been done yet."));
    (mean = deepcopy(s.mu_filt[1:n_filtered]),
     cov = deepcopy(s.P_filt[1:n_filtered]));
end

function filtered(ssm::SSMInstance{<: NonLinearGaussianSSM})
    filtered(ssm.storage);
end


function update_step!(ssm::SSMInstance{<: NonLinearGaussianSSM}, i::Integer, p;
                      linearisation::Symbol = :standard,
                      nodata::Bool = false) 
    m = ssm.model; data = ssm.data; s = ssm.storage;

    # Get references to preallocated objects.
    mu_pred = s.mu_pred; P_pred = s.P_pred;
    mu_filt = s.mu_filt; P_filt = s.P_filt;
    mu_smooth = s.mu_smooth;
    Zj = s.Zj; H = s.H; F = s.F;
    Z = s.Z; v = s.v; K = s.K;
    Zj_t_invF = s.Zj_t_invF;

    tmp_pxp1 = s.tmp_pxp1; tmp_pxp2 = s.tmp_pxp2;
    tmp_pxk = s.tmp_pxk; tmp_kxp1 = s.tmp_kxp1;
    tmp_kxk1 = s.tmp_kxk1; tmp_kxk2 = s.tmp_kxk2;
    tmp_p = s.tmp_p; tmp_k = s.tmp_k; Imat = s.Imat; obsdim = s.obsdim;
    mis = s.mis;
    logLik_increment = 0.0;
    num_na = 0;
    do_update = !nodata; # If do_update = false, force skipping of update step.

    # If do_update is true, it might still change to false (if there is a fully missing obs).
    if do_update
        obs = view(data.y, :, i);
        get_missing!(mis, obs);
        num_na = sum(mis); # Amount of missing values at current iteration.
        do_update = num_na < obsdim;
    end

    if do_update #if num_na < obsdim && !nodata
        obs = view(data.y, :, i);

        if linearisation == :smooth
            m.Zj!(Zj[i], mu_smooth[i], i, data, p);
        else
            m.Zj!(Zj[i], mu_pred[i], i, data, p);
        end
        m.H!(H, mu_pred[i], i, data, p);

        # Compute the full covariance matrix (H * H')
        # H .= H * transpose(H); This is what we want.
        mul!(tmp_pxp2, H, transpose(H));
        H .= tmp_pxp2; # The full covariance.

        # Handle partial missingness. The dimension remains the same,
        # no dimensions are dropped. Hence the covariance matrix H needs the eye matrix to missing indexes.
        if num_na > 0
            Zj[i][mis, :] .= 0.0;
            H[mis, :] .= 0.0;
            H[:, mis] .= 0.0;
            H[mis, mis] .= view(Imat, 1:num_na, 1:num_na);
        end

        # Compute F matrix.
        # F .= Zj * P_pred * transpose(Zj) .+ H; This is what we want.
        mul!(tmp_pxk, Zj[i], P_pred[i]);
        transpose!(tmp_kxp1, Zj[i]);
        mul!(F, tmp_pxk, tmp_kxp1);
        F .= F .+ H; # F is computed.

        if !isposdef(F) # If F is not positive definite calculate F = 0.5 * (F + F^T). This is an attempt to fix numerics.
            transpose!(tmp_pxp1, F);
            tmp_pxp2 .= F .+ tmp_pxp1;
            mul!(F, 0.5, tmp_pxp2);
        end
        chol_F = cholesky!(F, check = false); # Compute Cholesky factorization object, reuse F.
        issuccess(chol_F) || (return -Inf); # If Cholesky can't be calculated (even after symmetrising), return -Inf.

        # Compute prediction error.
        # v = y - Z(smooth[i]) - Zj(smooth[i])^T * (mu_pred[i] - mu_smooth[i]);
        #if linearisation == :smooth
        #    m.Z!(Z, mu_smooth[i], i, data, p);
        #    tmp_k .= mu_pred[i] .- mu_smooth[i];
        #    mul!(tmp_p, Zj[i], tmp_k);
        #    Z .= Z .+ tmp_p;
        #else
        m.Z!(Z, mu_pred[i], i, data, p);
        #end
        # v .= y .- Z; v[mis] .= 0.0;
        zero_if_missing_minus!(v[i], obs, Z);

        # Compute filtered mean.
        # The computations in readable code look like this:
        # K .= P_pred * transpose(Zj) * F^-1;
        # mu_filt .= mu_pred .+ K * v;
        transpose!(tmp_kxp1, Zj[i]);
        rdiv!(tmp_kxp1, chol_F.U);
        rdiv!(tmp_kxp1, chol_F.L);
        Zj_t_invF[i] .= tmp_kxp1; # Saved for smoothing.

        mul!(K[i], P_pred[i], tmp_kxp1); # K is computed.
        mul!(mu_filt[i], K[i], v[i]);
        mu_filt[i] .= mu_filt[i] .+ mu_pred[i]; # mu_filt is computed.

        # Compute P_filt.
        # tmp_kxk1 .= Imat .- K * Zj;
        # P_filt .= tmp_kxk1 * P_pred * transpose(tmp) .+ K * H * transpose(K);
        mul!(tmp_kxk1, K[i], Zj[i]);
        tmp_kxk1 .= Imat .- tmp_kxk1;
        transpose!(tmp_kxk2, tmp_kxk1);
        mul!(P_filt[i], tmp_kxk1, P_pred[i]);
        mul!(tmp_kxk1, P_filt[i], tmp_kxk2);
        transpose!(tmp_pxk, K[i]);
        mul!(tmp_kxp1, K[i], H);
        mul!(tmp_kxk2, tmp_kxp1, tmp_pxk);
        P_filt[i] .= tmp_kxk1 .+ tmp_kxk2; # P_filt is computed.

        # Calculate loglikelihood increment.
        tmp_p .= v[i];
        ldiv!(chol_F.L, tmp_p); # Here we calculate (chol_F.L)^-1 * v
        p = obsdim - num_na;
        logLik_increment = -0.5 * (log(2 * pi) * p + 2.0 * sum(log.(diag(F))) + dot(tmp_p, tmp_p)); # F has diagonal values of Cholesky of F.
        tmp_pxp1 .= chol_F.U; # F is butchered, so return it to it's real value.
        mul!(F, chol_F.L, tmp_pxp1);

    else # If all observed values are missing, then skip measurement update.
        # Set Zj to zero if all values are missing.
        Zj[i] .= 0.0;
        Zj_t_invF[i] .= 0.0;
        v[i] .= 0.0;
        K[i] .= 0.0;

        mu_filt[i] .= mu_pred[i];
        P_filt[i] .= P_pred[i];
    end
    logLik_increment
end

function predict_step!(ssm::SSMInstance{<: NonLinearGaussianSSM}, i::Integer, p)#; linearisation::Symbol = :standard)
    m = ssm.model; data = ssm.data; s = ssm.storage;

    # Get references to preallocated objects:
    mu_filt = s.mu_filt; mu_pred = s.mu_pred;
    mu_smooth = s.mu_smooth;
    P_filt = s.P_filt; P_pred = s.P_pred;

    Tj = s.Tj; R = s.R;
    tmp_kxk1 = s.tmp_kxk1;
    tmp_k = s.tmp_k;

    # Compute Jacobian of propagation function.
    #if linearisation == :smooth
    #    m.Tj!(Tj, mu_smooth[i], i, data, p);
    #else
    m.Tj!(Tj, mu_filt[i], i, data, p);
    #end

    m.R!(R, mu_filt[i], i, data, p); # R is calculated in place.

    # Compute predicted mean estimate.
    #if linearisation == :smooth
        # mu_pred[i + 1] = T!(mu_smooth[i]) + Tj(mu_smooth[i]) * (mu_filt[i] - mu_smooth[i]);
    #    m.T!(mu_pred[i + 1], mu_smooth[i], i, data, p);
    #    mul!(tmp_k, Tj, mu_filt[i]);
    #    mu_pred[i + 1] .= mu_pred[i + 1] .+ tmp_k;
    #    mul!(tmp_k, Tj, mu_smooth[i]);
    #    mu_pred[i + 1] .= mu_pred[i + 1] .- tmp_k;
    #else
    m.T!(mu_pred[i + 1], mu_filt[i], i, data, p);
    #end

    # Compute predicted covariance estimate:
    # P_pred .= Tj * P_filt * transpose(Tj) .+ R * transpose(R);
    mul!(tmp_kxk1, Tj, P_filt[i]);
    mul!(P_pred[i + 1], tmp_kxk1, transpose(Tj)); # m.s.P_pred = m.s.Tj * m.s.P_filt * transpose(m.s.Tj);
    mul!(tmp_kxk1, R, transpose(R)); # m.s.R * transpose(m.s.R);
    P_pred[i + 1] .= P_pred[i + 1] .+ tmp_kxk1;

    nothing
end

function _sample_initial_statevec!(x::AbstractVector{<: Real}, ssm::SSMInstance{<: NonLinearGaussianSSM}, p;
                                   rng::MersenneTwister = Random.GLOBAL_RNG)
    tv = ssm.data.tv; s = ssm.storage; m = ssm.model;

    # Temporaries.
    x_tmp1 = s.tmp_k; x_tmp2 = view(s.tmp_kxp1, :, 1);
    P_tmp1 = s.tmp_kxk1; P_tmp2 = s.tmp_kxk2;

    ### State simulation.
    # Initialisation.
    m.mu_1!(x, tv, p);
    m.P_1!(P_tmp1, tv, p);

    # Sample from N(mu[1], P[1]). Here SVD instead of Cholesky, because first covariance can be semidefinite.
    svd_fact = svd(P_tmp1);
    randn!(rng, x_tmp1);
    mul!(P_tmp2, svd_fact.U, Diagonal(sqrt.(svd_fact.S)));
    mul!(x_tmp2, P_tmp2, x_tmp1);
    x .= x .+ x_tmp2; # Initial state sampled from initial distribution.
    nothing;
end

function _sample_statevec!(x_next::AbstractVector{<: Real}, x_cur::AbstractVector{<: Real}, ssm::SSMInstance{<: NonLinearGaussianSSM}, t::Int, p;
                           rng::MersenneTwister = Random.GLOBAL_RNG)
    tv = ssm.data.tv; s = ssm.storage; m = ssm.model;
    x_tmp1 = s.tmp_k; x_tmp2 = view(s.tmp_kxp1, :, 1);
    P_tmp1 = s.tmp_kxk1; P_tmp2 = s.tmp_kxk2;

    # Advance state.
    m.T!(x_next, x_cur, t - 1, tv, p); # Compute mean.
    m.R!(P_tmp1, x_cur, t - 1, tv, p); # Cholesky of state covariance.
    randn!(rng, x_tmp1); mul!(x_tmp2, P_tmp1, x_tmp1); # Sample noise vector.
    x_next .= x_next .+ x_tmp2; # Next sample.
    nothing
end

function simulate(ssm::SSMInstance, p, t::Union{Val{:observation}, Val{:state}};
                  rng::MersenneTwister = Random.GLOBAL_RNG)
    if typeof(t) == Val{:state}
        x = zeros(eltype(ssm.storage.R), ssm.model.statedim, length(ssm.data));
    else
        x = zeros(eltype(ssm.storage.R), ssm.model.obsdim, length(ssm.data));
    end
    simulate!(x, ssm, p, t; rng = rng);
    x;
end

function simulate!(x::AbstractMatrix{<: Real}, ssm::SSMInstance{<: NonLinearGaussianSSM}, p, ::Val{:state};
                   rng::MersenneTwister = Random.GLOBAL_RNG)
    m = ssm.model; tv = ssm.data.tv; s = ssm.storage;
    @assert size(x, 1) == m.statedim;
    timepoints = size(x, 2);

    # Sample states.
    _sample_initial_statevec!(view(x, :, 1), ssm, p; rng = rng);
    for t in 2:timepoints
        x_cur = view(x, :, t - 1); x_next = view(x, :, t);
        _sample_statevec!(x_next, x_cur, ssm, t, p; rng = rng);
    end
    nothing
end

function simulate!(y::AbstractMatrix{<: Real}, ssm::SSMInstance{<: NonLinearGaussianSSM}, p, ::Val{:observation};
                   rng::MersenneTwister = Random.GLOBAL_RNG)
    s = ssm.storage; tv = ssm.data.tv; m = ssm.model;
    timepoints = size(y, 2);

    x_cur = zeros(m.statedim); x_next = zeros(m.statedim); # Three allocations.
    y_tmp1 = zeros(m.obsdim);
    y_tmp2 = s.tmp_p; P_tmp = s.tmp_pxp1;

    _sample_initial_statevec!(x_cur, ssm, p; rng = rng);
    m.Z!(view(y, :, 1), x_cur, 1, tv, p);
    m.H!(P_tmp, x_cur, 1, tv, p);
    randn!(rng, y_tmp1); mul!(y_tmp2, P_tmp, y_tmp1);
    y[:, 1] .= view(y, :, 1) .+ y_tmp2;

    for t in 2:timepoints
        # Simulate new observation:
        _sample_statevec!(x_next, x_cur, ssm, t, p; rng = rng);
        m.Z!(view(y, :, t), x_next, t, tv, p);
        m.H!(P_tmp, x_next, t, tv, p);
        randn!(rng, y_tmp1); mul!(y_tmp2, P_tmp, y_tmp1);
        y[:, t] .= view(y, :, t) .+ y_tmp2; # Observation sampled.
        x_cur .= x_next;
    end
    nothing
end

function predict(s::NLGSSMState, m::NonLinearGaussianSSM, data, p, ::Val{:state}, n_ahead::Int)
    filtered_index = s.filtered_index[];

    # Allocate temporaries and returned arrays.
    mu_pred = copy(s.mu_pred[filtered_index + 1]);
    P_pred = copy(s.P_pred[filtered_index + 1]);
    mu_filt = Array{Float64, 2}(undef, m.statedim, n_ahead);
    P_filt = Array{Float64, 3}(undef, m.statedim, m.statedim, n_ahead);
    Tj = similar(s.Tj); R = similar(s.R);

    tmp_kxk1 = s.tmp_kxk1;
    tmp_kxk2 = s.tmp_kxk2;

    for i in 1:n_ahead
        # Save prediction. Note that after fitting the one step prediction already exists.
        mu_filt[:, i] .= mu_pred;
        P_filt[:, :, i] .= P_pred;

        #predict_step!(m, filtered_index + i, data, p; linearisation = :standard);

        # Compute predicted mean estimate.
        m.Tj!(Tj, mu_pred, filtered_index + i, data, p);
        m.R!(R, mu_pred, filtered_index + i, data, p);
        m.T!(mu_pred, view(mu_filt, :, i), filtered_index + i, data, p);

        # Compute predicted covariance estimate:
        # P_pred .= Tj * P_filt * transpose(Tj) .+ R * transpose(R);
        mul!(tmp_kxk1, Tj, P_pred);
        transpose!(tmp_kxk2, Tj);
        mul!(P_pred, tmp_kxk1, tmp_kxk2); # m.s.P_pred = m.s.Tj * m.s.P_filt * transpose(m.s.Tj);
        transpose!(tmp_kxk1, R);
        mul!(tmp_kxk2, R, tmp_kxk1); # m.s.R * transpose(m.s.R);
        P_pred .= P_pred .+ tmp_kxk2;

    end
    # Return the predicted values and time + time indexes at predicted values.
    (index = (filtered_index + 1):(filtered_index + n_ahead),
     mu_filt = mu_filt, P_filt = P_filt);
end

function predict(s::NLGSSMState, m::NonLinearGaussianSSM, data, p, ::Val{:state}; predict_horizon::T) where T <: Real
    predict_horizon <= 0.0 && throw(ArgumentError("'predict_horizon' must be > 0.0"));
    filtered_index = s.filtered_index[]; # The index at which the last filtered mean has been calculated.
    data.time[filtered_index] + predict_horizon > maximum(data.time) && throw(ArgumentError("'predict_horizon' is too long, not enough explanatory data points."))
    n_steps = findall(data.time .== data.time[filtered_index] + predict_horizon)[] - filtered_index;
    pred = predict(s, m, data, p, Val(:state), n_steps);
    (time = data.time[(filtered_index + 1):(filtered_index + n_ahead)], index = pred.index,
     mu_filt = pred.mu_filt, P_filt = pred.P_filt)
end

function predict(s::NLGSSMState, m::NonLinearGaussianSSM, data, p, ::Val{:observation}; predict_horizon::T) where T <: Real
    predict_horizon <= 0.0 && throw(ArgumentError("'predict_horizon' must be > 0.0"));
    filtered_index = s.filtered_index[]; # The index at which the last filtered mean has been calculated.
    data.time[filtered_index] + predict_horizon > maximum(data.time) && throw(ArgumentError("'predict_horizon' is too long, not enough explanatory data points."))
    n_ahead = findall(data.time .== data.time[filtered_index] + predict_horizon)[] - filtered_index;
    pred = predict(s, m, data, p, Val(:observation), n_ahead);
    (time = data.time[(filtered_index + 1):(filtered_index + n_ahead)], mean = pred.mean, covariance = pred.covariance)
end

function predict(s::NLGSSMState, m::NonLinearGaussianSSM, data, p, ::Val{:observation}, n_ahead::Int) 

    # Compute filtered state estimates.
    state_pred = predict(s, m, data, p, Val(:state), n_ahead);
    n_steps = length(state_pred.index);

    mu_filt = state_pred.mu_filt;
    P_filt = state_pred.P_filt;

    # Allocate objects for results.
    mean = Matrix{Float64}(undef, m.obsdim, n_ahead);
    cov = Array{Float64, 3}(undef, m.obsdim, m.obsdim, n_ahead);
    Zj = similar(s.Zj[1]); # Preallocate matrix for Jacobian of observation propagation equation.
    H = similar(s.H); # Preallocate matrix for Cholesky of observation noise covariance.
    tmp_pxk = s.tmp_pxk; tmp_kxp1 = s.tmp_kxp1;
    tmp_pxp1 = s.tmp_pxp1; tmp_pxp2 = s.tmp_pxp2;

    # Compute observation level predictive distribution.
    filtered_index = s.filtered_index[];
    for i in 1:n_ahead
        # Compute observation level mean estimate.
        m.Z!(view(mean, :, i), view(mu_filt, :, i), filtered_index + i, data, p);

        # Compute observation level covariance estimate.
        m.H!(H, view(mu_filt, :, i), filtered_index + i, data, p);
        m.Zj!(Zj, view(mu_filt, :, i), filtered_index + i, data, p);

        #cov[:, :, i] .= Zj * view(P_filt, :, :, i) * transpose(Zj) + H * transpose(H);
        mul!(tmp_pxk, Zj, view(P_filt, :, :, i));
        transpose!(tmp_kxp1, Zj);
        mul!(view(cov, :, :, i), tmp_pxk, tmp_kxp1);
        transpose!(tmp_pxp1, H);
        mul!(tmp_pxp2, H, tmp_pxp1);
        cov[:, :, i] .= view(cov, :, :, i) .+ tmp_pxp2;
    end
    # Return predictive distribution as well as time and time indexes.
    (index = state_pred.index, mean = mean, covariance = cov)
end

"""
Check validity of Jacobian function 'candidate' at point 'x' using finite differences.
'f' should be the vector-valued function whose Jacobian 'candidate' is
supposed to return.
Difference of the finite difference Jacobian and Jacobian calculated using 'candidate' is returned.
"""
function check_jacobian(x, f, candidate; delta::Float64)
    candidate_jac = candidate(x);
    finite_diff_jac = similar(candidate_jac);
    e = similar(x);
    for i in eachindex(x)
        e .= zero(eltype(x));
        e[i] = one(eltype(x));
        finite_diff_jac[:, i] = (f(x .+ 0.5 * delta .* e) - f(x .- 0.5 * delta .* e)) / delta;
    end
    candidate_jac .- finite_diff_jac;
end

function check_jacobian(ssm::SSMInstance{<: NonLinearGaussianSSM}, p; x = ones(Float64, ssm.model.statedim), t::Int = 1, delta::Float64 = 1e-03)
    model = ssm.model; data = ssm.data.tv;
    statedim = model.statedim;
    obsdim = model.obsdim;
    @assert length(x) == statedim;
    T = function(x)
        r = similar(x);
        model.T!(r, x, t, data, p);
        return r;
    end
    Tj = function(x)
        r = Matrix{eltype(x)}(undef, statedim, statedim);
        model.Tj!(r, x, t, data, p);
        return r;
    end
    Z = function(x)
        r = Vector{eltype(x)}(undef, obsdim);
        model.Z!(r, x, t, data, p);
        return r;
    end
    Zj = function(x)
        r = Matrix{eltype(x)}(undef, obsdim, statedim);
        model.Zj!(r, x, t, data, p);
        return r;
    end
    println("Tj - finitediff Tj:");
    println(check_jacobian(x, T, Tj, delta = delta));
    println("Zj - finitediff Zj:");
    println(check_jacobian(x, Z, Zj, delta = delta));
end

"""
The function computes the ingredients needed to build the normal distributions
p(X_[k] | X_[k-1], Y_[1:T]), for k = 2, ..., T, and p(X_[1] | Y_[1:T]) (for k = 1).
A NamedTuple with elements A, b and L is returned.
The distributions are then Normal(A[k] * X_[k-1] + b[k], L[k] * L[k]^T), where L is the lower diagonal Cholesky factor.
Specifying X_[k-1] will define each distribution completely.
"""
function compute_smooth_normal_conditionals!(ssm::SSMInstance{<: NonLinearGaussianSSM}, θ)
    m = ssm.model; ss = ssm.storage; data = ssm.data;

   # Setup pointers.
   mu_filt = ss.mu_filt; mu_smooth = ss.mu_smooth;
   P_pred = ss.P_pred; P_filt = ss.P_filt; P_smooth = ss.P_smooth;
   tmp_kxk1 = ss.tmp_kxk1; tmp_kxk2 = ss.tmp_kxk2;
   Tj = ss.Tj; Imat = ss.Imat;
   filt_end = ss.filtered_index[];

   # Allocate return values.
   # TODO: Later want to make an object with these and compute in place?
   A = [zeros(eltype(ss), ss.statedim, ss.statedim) for i in 1:filt_end];
   b = [zeros(eltype(ss), ss.statedim) for i in 1:filt_end];
   L = [zeros(eltype(ss), ss.statedim, ss.statedim) for i in 1:filt_end];

   # Compute A, b and L at each time index.
   for k in filt_end:-1:2
      #C_k_1 = P_filt[k - 1] * transpose(Tj) * inv(P_pred[k]);
      m.Tj!(Tj, mu_filt[k - 1], k - 1, data, θ);
      mul!(tmp_kxk2, P_filt[k - 1], transpose(Tj));
      tmp_kxk1 .= P_pred[k];
      !issymmetric(tmp_kxk1) && symmetrise!(tmp_kxk1);
      chol = cholesky!(tmp_kxk1);
      rdiv!(tmp_kxk2, chol.U); rdiv!(tmp_kxk2, chol.L); # C_k-1 is computed.

      #A[k] .= P_smooth[k] * transpose(C_k_1) * inv(P_smooth[k - 1]);
      mul!(tmp_kxk1, P_smooth[k], transpose(tmp_kxk2));
      tmp_kxk2 .= P_smooth[k - 1];
      !issymmetric(tmp_kxk2) && symmetrise!(tmp_kxk2);
      chol = cholesky!(tmp_kxk2);
      rdiv!(tmp_kxk1, chol.U);
      L[k] .= tmp_kxk1; # Precompute some of P to avoid recomputing C_k-1.
      rdiv!(tmp_kxk1, chol.L);
      A[k] .= tmp_kxk1; # A is computed.

      # b = mu_smooth[k] - A * mu_smooth[k - 1]
      mul!(b[k], A[k], mu_smooth[k - 1]);
      b[k] .= mu_smooth[k] .- b[k]; # b is computed.

      # P = P_smooth[k] - A * C_k-1 * P_smooth[k], L = cholesky(P).
      # First way to compute P (need to make modifications upstream ^)
      # Note that L contains a matrix P_smooth[k] * C_k-1 * sqrt(inv(P_smooth[k-1])) here.
      mul!(tmp_kxk1, L[k], transpose(L[k]));
      L[k] .= P_smooth[k] .- tmp_kxk1; # P is computed.
      !issymmetric(L[k]) && symmetrise!(L[k]);
      chol = cholesky!(L[k]);
      L[k] .= chol.L;
   end
   # Set the first proposal distribution to be just the smoothed distribution.
   A[1] .= 0.0;
   b[1] .= mu_smooth[1];
   tmp_kxk2 .= P_smooth[1];
   !issymmetric(tmp_kxk2) && symmetrise!(tmp_kxk2);
   chol = cholesky!(tmp_kxk2);
   L[1] .= chol.L;

   (A = A, b = b, L = L);
end

function confidence_ellipse!(P::AbstractMatrix{<: Real}, mu::AbstractVector{<: Real}, sigma::AbstractMatrix{<: Real},
                             level::Float64 = 0.95)
    @assert length(mu) == size(P, 1);
    n_points = size(P, 2);
    q = sqrt(quantile(Chisq(length(mu)), level));
    !issymmetric(sigma) && symmetrise!(sigma);
    C = cholesky(sigma);
    angles = range(0.0; stop = 2.0 * pi, length = n_points);
    ellps = transpose([cos.(angles) sin.(angles)]);
    for i in 1:(n_points - 1)
        P[:, i] .= mu .+ q * C.L * view(ellps, :, i);
    end
    P[:, n_points] .= view(P, :, 1);
    nothing;
end


