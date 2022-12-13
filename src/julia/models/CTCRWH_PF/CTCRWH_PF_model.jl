include("../../../../config.jl");
#include(joinpath(LIB_PATH, "utils.jl"));
#include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
#include(joinpath(LIB_PATH, "bresenham_summing.jl"));
using Distributions

mutable struct CTCRWHParticle <: Particle
    mux::Float64
    vx::Float64
    muy::Float64
    vy::Float64
end
CTCRWHParticle() = CTCRWHParticle(NaN, NaN, NaN, NaN);

import Base.copy!
function copy!(dest::CTCRWHParticle, src::CTCRWHParticle)
    dest.mux = src.mux;
    dest.vx = src.vx;
    dest.muy = src.muy;
    dest.vy = src.vy;
    nothing;
end
function copy!(dest::CTCRWHParticle, src::SVector{4, Float64})
    @inbounds dest.mux = src[1];
    @inbounds dest.vx = src[2];
    @inbounds dest.muy = src[3];
    @inbounds dest.vy = src[4];
    nothing;
end
function SVector{4, Float64}(p::CTCRWHParticle)
    SVector{4, Float64}(p.mux, p.vx, p.muy, p.vy);
end

"""
Build the CTCRWH model.
"""
function CTCRWH_build(raster::SimpleRaster, proposal, potential::Function)
    model = let raster = raster, proposal = proposal, potential = potential

        # Simulate from the initial state distribution.
        function Mi!(p::CTCRWHParticle, data, θ)
            b = proposal[:b][1]; L = proposal[:L][1];
            out = b + L * randn(SVector{4, Float64});
            copy!(p, out);
            #randn!(TMP_VEC);
            #mul!(x, L, TMP_VEC);
            #x .= x .+ b;
            nothing;
        end

        # Simulate from the state vector at time t > 1.
        function M!(pnext::CTCRWHParticle, pcur::CTCRWHParticle, t::Int, data, θ)
            A = proposal[:A][t]; L = proposal[:L][t];
            b = proposal[:b][t];
            xcur = SVector{4, Float64}(pcur);
            out = A * xcur + L * randn(SVector{4, Float64}) + b;
            copy!(pnext, out);
            #randn!(TMP_VEC); mul!(x_next, L, TMP_VEC);
            #mul!(TMP_VEC, A, x_cur);
            #x_next .= x_next .+ TMP_VEC .+ b;
            nothing
        end

        # V is computed as -log(v_i) where v_i is the "land type preference" in
        # cell of type i. Land type preferences are in [0, Inf).
        function lGi(p::CTCRWHParticle, data, θ)
            v_prop = potential(nothing, p, raster);
            dt = data.dt[1];
            -dt * v_prop;
        end

        # V is computed as -log(v_i) where v_i is the "land type preference" in
        # cell of type i. Land type preferences are in [0, Inf).
        function lG(pprev::CTCRWHParticle, pcur::CTCRWHParticle, t::Int, data, θ)
            dt = data.dt[t];
            dt <= 0.0 && (return 0.0;)
            v_prop = potential(pprev, pcur, raster);
            return -dt * v_prop;
        end
        GenericSSM(CTCRWHParticle, Mi!, nothing, M!, nothing, lGi, lG);
    end
    model;
end

###########################
### Potential functions ###
###########################

"""
Potential function that only takes into account the proposed location.
"""
function CTCRWH_one_step_potential(pprev,
                                   pcur::CTCRWHParticle,
                                   raster::SimpleRaster{<: AbstractFloat})
    extract(raster, pcur.mux, pcur.muy);
end
#
# """
# Potential function that uses the Bresenham line drawing algorithm (sum) in the
# potential computation.
# """
# function CTCRWH_bh_sum_potential(prev::AbstractVector{<: AbstractFloat},
#                                     prop::AbstractVector{<: AbstractFloat},
#                                     raster::SimpleRaster{<: AbstractFloat})
#     y1, x1 = get_pixel_coordinates(raster, prev[1], prev[2]);
#     y2, x2 = get_pixel_coordinates(raster, prop[1], prop[2]);
#     bh_tally(raster.r, x1, y1, x2, y2, false);
# end
#
# function CTCRWH_bh_mean_potential(prev::AbstractVector{<: AbstractFloat},
#                                   prop::AbstractVector{<: AbstractFloat},
#                                   raster::SimpleRaster{<: AbstractFloat})
#     y1, x1 = get_pixel_coordinates(raster, prev[1], prev[2]);
#     y2, x2 = get_pixel_coordinates(raster, prop[1], prop[2]);
#     bh_tally(raster.r, x1, y1, x2, y2, true);
# end
