## Some code for working with Gaussian distributions with StaticArrays.
# Mainly, the Gaussian object implements `logpdf` and `rand`
# methods that do not allocate like the methods `logpdf` and `rand`
# for MvNormal in Distributions.jl.

using Distributions
using LinearAlgebra
using LinearAlgebra: norm_sqr
using StaticArrays

import Base: rand
import Distributions: pdf, logpdf, sqmahal, cdf, quantile
import Base: size

"""
An abstract type for an MVector or SVector with length N and element type T.
"""
const VecND{N, T} = StaticArray{Tuple{N}, T, 1};

"""
An abstract type for an N * N MMatrix or SMatrix with element type T.
"""
const MatND{N, T} = StaticArray{Tuple{N, N}, T, 2};

"""
A type representing a lower triangular Cholesky factor.
"""
struct LowerTriCholesky{S}
    L::S
    function LowerTriCholesky(σ::T) where {T}
        !istril(σ) && throw(ArgumentError("Argument not lower triangular"));
        new{T}(σ);
    end
end

cholupper(ltc::LowerTriCholesky) = ltc.L';
whiten(ltc::LowerTriCholesky, z) = ltc.L\z;
_logdet(ltc::LowerTriCholesky, d) = 2.0 * sumlogdiag(ltc.L, d);

"""
    Gaussian(μ, √Σ) -> P
Gaussian distribution with mean `μ`` and lower triangular Cholesky factor L.
Designed to work with MVectors and SVectors.
"""
struct Gaussian{T, S <: LowerTriCholesky}
    μ::T
    L::S
    function Gaussian(μ::T, L::LowerTriCholesky) where T
        @assert length(μ) == size(L.L, 1) == size(L.L, 2) "dimension mismatch";
        new{T,typeof(L)}(μ, L);
    end
end

function Gaussian(μ::VecND{N, T}, Σ::MatND{N, T}) where {N, T}
    M = SMatrix(Σ);
    Gaussian(μ, LowerTriCholesky(cholesky(M).L));
end

dimension(P::Gaussian) = length(P.μ);
whiten(Σ, z) = cholupper(Σ)'\z;
whiten(Σ::UniformScaling, z) = z/sqrt(Σ.λ);
sumlogdiag(Σ::Float64, d = 1) = log(Σ);
sumlogdiag(Σ, d) = sum(log.(diag(Σ)));
sumlogdiag(J::UniformScaling, d) = log(J.λ) * d;

_logdet(Σ, d) = logdet(Σ);
_logdet(J::UniformScaling, d) = log(J.λ) * d;

function sqmahal(P::Gaussian{<: VecND{N, T}}, x) where {N, T}
    norm_sqr(whiten(P.L, SVector{N, T}(x) - SVector{N, T}(P.μ)));
end

"""
Non-allocating `logpdf`.
"""
function logpdf(P::Gaussian, x)
    -(sqmahal(P, x) + _logdet(P.L, dimension(P)) + dimension(P) * log(2.0 * pi)) / 2.0;
end

"""
Non-allocating `rand`, returning an SVector.
"""
function rand(P::Gaussian{<: VecND{N, T}}) where {T, N}
    SVector{N, T}(P.μ) + P.L.L * randn(SVector{N, T})
end

"""
Update the lower triangular Cholesky of the Gaussian distribution `g`.
The replacement matrix is assumed to be proper and not checked.
"""
function update_chol!(g::Gaussian{<: VecND{N, T}}, L::MatND{N, T}) where {T, N}
    g.L.L .= zero(T);
    g.L.L .= L;
    nothing;
end

"""
Update the lower triangular Cholesky of the Gaussian distribution `g`
by taking the upper triangle of the matrix `M`.
"""
function update_chol_utri!(g::Gaussian{<: VecND{N, T}}, M::MatND{N, T}) where {T, N}
    L = g.L.L;
    L .= zero(T);
    for j in 1:size(M, 2)
        for i in 1:j
            @inbounds L[j, i] = M[i, j];
        end
    end
    nothing;
end
