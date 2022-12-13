module NLGSSM
export NonLinearGaussianSSM, NLGSSMState, storagetype, forward_density

using LinearAlgebra
using Distributions
using Statespace: SSM, ModelIdentifier, SSMStorage

### Storage object for non linear gaussian statespace model.
### Preallocated matrices and vectors.
struct NLGSSMState{T <: Real} <: SSMStorage
    obsdim::Int
    n_obs::Int
    statedim::Int
    filtered_index::Base.RefValue{Int} # By keeping this, storage can be used only partially.

    mu_filt::Array{Vector{T}, 1}
    mu_pred::Array{Vector{T}, 1}
    P_filt::Array{Matrix{T}, 1}
    P_pred::Array{Matrix{T}, 1}
    Tj::Matrix{T}
    R::Matrix{T}
    Z::Vector{T}
    Zj::Array{Matrix{T}, 1}
    H::Matrix{T}

    r::Vector{T}
    N::Matrix{T}
    L::Matrix{T}
    mu_smooth::Array{Vector{T}, 1}
    P_smooth::Array{Matrix{T}, 1}

    # Temporaries.
    tmp_kxk1::Matrix{T} # k x k temporary matrix. k is dimension of state vector.
    tmp_kxk2::Matrix{T}
    tmp_pxp1::Matrix{T} # p x p temporary matrix. p is dimension of observation vector.
    tmp_pxp2::Matrix{T}
    tmp_kxp1::Matrix{T}
    tmp_kxp2::Matrix{T}
    tmp_pxk::Matrix{T}
    tmp_k::Vector{T}
    tmp_p::Vector{T}

    # Preallocated for calculations.
    v::Array{Vector{T}, 1}
    F::Matrix{T}
    Zj_t_invF::Array{Matrix{T}, 1}
    Imat::Matrix{T}
    mis::BitArray{1}
    K::Array{Matrix{T}, 1}

    NLGSSMState(::Type{T}; obsdim::Integer, n_obs::Integer, statedim::Integer) where {T <: Real} = begin

        # Main matrices and vectors needed in calculations.
        mu_filt = [zeros(T, statedim) for i in 1:n_obs];
        P_filt = [zeros(T, statedim, statedim) for i in 1:n_obs]; # Filtered estimates.
        mu_pred = [zeros(T, statedim) for i in 1:n_obs];
        P_pred = [zeros(T, statedim, statedim) for i in 1:n_obs]; # Predicted estimates.

        filtered_index = Base.RefValue(0);
        Tj = zeros(T, statedim, statedim); # Jacobian of state vector.
        R = zeros(T, statedim, statedim); # Cholesky matrix of state covariance.
        Z = zeros(T, obsdim);
        Zj = [zeros(T, obsdim, statedim) for i in 1:n_obs]; # Jacobian of observation vector.
        H = zeros(T, obsdim, obsdim);
        r = zeros(T, statedim);
        N = zeros(T, statedim, statedim);
        L = zeros(T, statedim, statedim);

        mu_smooth = [zeros(T, statedim) for i in 1:n_obs];
        P_smooth = [zeros(T, statedim, statedim) for i in 1:n_obs];

        # Preallocated temporaries. k = statedim, p = obsdim.
        tmp_kxk1 = zeros(T, statedim, statedim);
        tmp_kxk2 = zeros(T, statedim, statedim);
        tmp_pxp1 = zeros(T, obsdim, obsdim);
        tmp_pxp2 = zeros(T, obsdim, obsdim);
        tmp_kxp1 = zeros(T, statedim, obsdim);
        tmp_kxp2 = zeros(T, statedim, obsdim);
        tmp_pxk = zeros(T, obsdim, statedim);
        tmp_k = zeros(T, statedim);
        tmp_p = zeros(T, obsdim);

        v = [zeros(T, obsdim) for i in 1:n_obs];
        F = zeros(T, obsdim, obsdim);
        Zj_t_invF = [zeros(T, statedim, obsdim) for i in 1:n_obs];
        Imat = Matrix{T}(1.0LinearAlgebra.I, statedim, statedim);
        mis = BitArray{1}(undef, obsdim);
        K = [zeros(T, statedim, obsdim) for i in 1:n_obs];

        new{T}(obsdim, n_obs, statedim, filtered_index, mu_filt, mu_pred, P_filt, P_pred,
            Tj, R, Z, Zj, H, r, N, L, mu_smooth, P_smooth, tmp_kxk1, tmp_kxk2, tmp_pxp1, tmp_pxp2, tmp_kxp1, tmp_kxp2, tmp_pxk, tmp_k, tmp_p,
            v, F, Zj_t_invF, Imat, mis, K);
    end
end

import Base.eltype
function eltype(ss::NLGSSMState{T}) where T
    return T;
end

struct NonLinearGaussianSSM{M <: ModelIdentifier, μ₁ <: Function, Σ₁ <: Function,
                            Tfun <: Function, Tjfun <: Function, Rfun <: Function,
                            Zfun <: Function, Zjfun <: Function, Hfun <: Function} <: SSM
    id::Type{M} # ID of model.
    statedim::Int
    obsdim::Int
    statenames::Vector{Symbol}
    obsnames::Vector{Symbol}

    ### The functions that define the model:
    # Initialisation.
    mu_1! :: μ₁
    P_1! :: Σ₁

    # State equation.
    T! :: Tfun
    Tj! :: Tjfun
    R! :: Rfun

    # Observation equation.
    Z! :: Zfun
    Zj! :: Zjfun
    H! :: Hfun

end

function NonLinearGaussianSSM(T::Type{<: ModelIdentifier};
                              statenames::Vector{Symbol}, obsnames::Vector{Symbol},
                              mu_1!::Function, P_1!::Function, T!::Function, Tj!::Function, R!::Function,
                              Z!::Function, Zj!::Function, H!::Function)
    obsdim = length(obsnames);
    statedim = length(statenames);
    @assert obsdim > 0 "'obsdim' must be greater than zero.";
    @assert statedim > 0 "'statedim' must be greater than zero."
    NonLinearGaussianSSM(T, statedim, obsdim, statenames, obsnames, mu_1!, P_1!, T!, Tj!, R!, Z!, Zj!, H!);
end

import Statespace.storagetype
function storagetype(m::NonLinearGaussianSSM)
    NLGSSMState;
end

function NLGSSMState(model::NonLinearGaussianSSM, T::Int)
    NLGSSMState(Float64; statedim = model.statedim, obsdim = model.obsdim, n_obs = T);
end

function forward_density(model::NonLinearGaussianSSM)
  TMP_VEC1 = zeros(model.statedim);
  TMP_MAT1 = zeros(model.statedim, model.statedim);
  TMP_MAT2 = zeros(model.statedim, model.statedim);

  f = let TMP_VEC1 = TMP_VEC1, TMP_MAT1 = TMP_MAT1, TMP_MAT2 = TMP_MAT2
      function m(x_next::AbstractVector{<: Real}, x_cur::AbstractVector{<: Real}, t_next::Int, data, p)
          model.T!(TMP_VEC1, x_cur, t_next - 1, data, p);
          model.R!(TMP_MAT1, x_cur, t_next - 1, data, p);
          mul!(TMP_MAT2, TMP_MAT1, transpose(TMP_MAT1));
          C = PDMat(TMP_MAT2, Cholesky(TMP_MAT1, :L, 0));
          logpdf(MvNormal(TMP_VEC1, C), x_next);
      end
  end
  f;
end


end
