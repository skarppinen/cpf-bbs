include("../../../config.jl");
include("Gaussians.jl");

using Distributions
using Random
using LinearAlgebra
using StaticArrays

"""
A distribution object representing a conditional normal distribution.
The purpose of the type is to provide a structure that allows for allocation-free
sampling and logdensity calculation for Y | X = x when x is allowed to vary,
e.g it will be given as a parameter after the construction of the object itself.

For reference, Y | X has mean and covariance:
μ_{y | x} = μ_y + Σ_{yx} * Σ_x^{-1} (x - μ_x)
Σ_{y | x} = Σ_y - Σ_{yx} * Σ_x^{-1} * Σ_{xy}

The type parameters:
`yN`: Dimension of Y.
`xN`: Dimension of X.
`yL`: Number of elements in covariance matrix of Y.
`xL`: Number of elements in covariance matrix of X.
`yxL`: Number of elements in covariance matrix Σyx.

Sadly, Julia doesn't allow computation of `yL`, `xL` and `yxL` based on `yN` and `xN` (yet?).
"""
struct ConditionalNormal{yN, xN, yL, xL, yxL}
    # Base dist has a normal distribution with same covariance as above,
    # but with adjustment (terms containing x) missing from mean.
    base_dist::Gaussian{SVector{yN, Float64}, LowerTriCholesky{SMatrix{yN, yN, Float64, yL}}}
    Σx::SMatrix{xN, xN, Float64, xL}
    Σyx::SArray{Tuple{yN, xN}, Float64, 2, yxL}
    function ConditionalNormal(μy::AVec{<: Real}, μx::AVec{<: Real},
                               Σy::AMat{<: Real}, Σx::AMat{<: Real}, Σyx::AMat{<: Real})
        ydim = length(μy);
        xdim = length(μx);
        μy_type = SVector{ydim, Float64};
        Σy_type = SMatrix{ydim, ydim, Float64, ydim * ydim};

        Σx_chol = cholesky(Hermitian(Σx));
        Wt = Σx_chol.L \ transpose(Σyx);
        W = transpose(Wt);

        base_dist_μ = μy_type(μy - W * (Σx_chol.L \ μx));
        base_dist_cholmat = Σy_type(Matrix(cholesky(Hermitian(Σy - W * Wt)).L));
        base_dist_L = LowerTriCholesky(base_dist_cholmat);
        base_dist = Gaussian(base_dist_μ, base_dist_L);
        new{ydim, xdim, ydim * ydim, xdim * xdim, ydim * xdim}(base_dist, Σx, Σyx);
    end
end

function adjust_mean(dn::ConditionalNormal{yN, xN}, x::SVector{xN, <: Real}) where {yN, xN}
    dn.Σyx * (dn.Σx \ x);
end

# """
# Sample a random vector to `y` conditional on `x`.
# """
# function Random.rand!(d::ConditionalNormal, y::AbstractVector{<: Real}, x::AbstractVector{<: Real})
#     # Compute conditional mean.
#     d._VEC2 .= x .- d.mean[2];
#     mul!(y, d.ccinv, d._VEC2);
#     d._VEC1 .= d.mean[1] .+ y;
#
#     # Sample.
#     Random.rand!(MvNormal(d._VEC1, d.cov), y);
#     nothing
# end

function Random.rand(dn::ConditionalNormal{yN, xN}, x::SVector{xN, <: Real}) where {yN, xN}
    rand(dn.base_dist) + adjust_mean(dn, x);
end


"""
Compute the logpdf value of the distribution at `y` conditional on `x`.
"""
function Distributions.logpdf(dn::ConditionalNormal{yN, xN}, y::SVector{yN, <: Real},
                              x::SVector{xN, <: Real}) where {yN, xN}
    logpdf(Gaussian(dn.base_dist.μ + adjust_mean(dn, x), dn.base_dist.L), y);

    # Compute conditional mean. (y is preserved)
    #d._VEC2 .= x .- d.mean[2];
    #mul!(d._VEC1, d.ccinv, d._VEC2);
    #d._VEC1 .= d.mean[1] .+ d._VEC1;

    # Compute logpdf.
    #logpdf(MvNormal(d._VEC1, d.cov), y);
end
