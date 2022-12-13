include("../../../config.jl");
include(joinpath(LIB_PATH, "pfilter.jl"));
include(joinpath(LIB_PATH, "ConditionalNormal.jl"));
include(joinpath(LIB_PATH, "bbcpf-blocking.jl"));
using NLGSSM
using SparseArrays

"""
A type for the (in practice, Gaussian) block samplers used in the BBCPF algorithm.
A BlockSampler is a structure that contains the distributions required
to sample through a block in the BBCPF algorithm.

To achieve this two types of distributions are required:
1. M(X[k] | X[k - 1], X[u]) for k = l + 1, ..., u - 1,
2. M(X[u] | X[l]),
where l and u denote the lower and upper bounds of the block, respectively.

A BlockSampler implements the following methods:

rand(bcs::BlockSampler, x_i-1, x_u, i::Integer)
which samples a new sample (SVector) from the "one step distribution"
M(X[i] | X[i - 1], X[u]).

logweight(bcs::BlockSampler, x_u, x_l)
which returns the logpdf of the "block weight distribution" M(x_u | x_l).
"""
struct BlockSampler{yN, xN, yL, xL, yxL}
    bounds::Tuple{Int, Int} # Block bounds.
    weightdist::ConditionalNormal{yN, yN, yL, yL, yL} # The distribution M[u | l].
    onestepdists::Vector{ConditionalNormal{yN, xN, yL, xL, yxL}} # Vector of distributions M[x_k | x_{k-1}, x_u],
                                                                 # k = l + 1, ..., u - 1.
    function BlockSampler(bounds::Tuple{Int, Int},
                          weightdist::ConditionalNormal{yN, yN, yL, yL, yL},
                          onestepdists::Vector{ConditionalNormal{yN, xN, yL, xL, yxL}}) where {yN, xN, yL, xL, yxL}
        @assert length(onestepdists) == (bounds[2] - bounds[1] - 1) "`length(onestepdists)` must equal `bounds[2] - bounds[1] - 1`";
        new{yN, xN, yL, xL, yxL}(bounds, weightdist, onestepdists);
    end
end

function Random.rand(bs::BlockSampler{yN}, x_prev::AbstractArray{<: Real},
                     x_u::AbstractArray{<: Real}, i::Integer) where {yN}
    # NOTE: Assumed that x_prev is first in the conditioning vector!
    x_cond = vcat(SVector{yN}(x_prev), SVector{yN}(x_u));
    Random.rand(bs.onestepdists[i], x_cond);
end

function logweight(bs::BlockSampler{yN},
                   x_u::AVec{<: Real}, x_l::AVec{<: Real}) where yN
    logpdf(bs.weightdist, SVector{yN}(x_u), SVector{yN}(x_l));
end

"""
The block backward conditional particle filter.
Note that it is assumed that the particle filter (function `pf!`) has been run
and a reference trajectory has been set using `traceback!` before invoking
the function.

Arguments:
`ssm`: An object of type SSMInstance{<: GenericSSM}.
`θ`: Parameters that the model can optionally depend on.
`samplers`: Samplers for each block in the algorithm.
"""
function bbcpf!(ssm::SSMInstance{<: GenericSSM}, θ,
                samplers::Vector{<: BlockSampler{yN}},
                outputs::Any = nothing;
                resampling::Resampling) where yN
    model = ssm.model; ps = ssm.storage; data = ssm.data;
    X = ps.X; A = ps.A; W = ps.W; ref = ps.ref; wnorm = ps.wnorm;

    n_particles = particle_count(ps);
    all_particle_indices = Base.OneTo(n_particles);

    # Run the conditional particle filter.
    pf!(ssm, θ; resampling = resampling, conditional = true);

    # Use the last index in the weight storage to keep the "blockweight".
    T = capacity(ps, 2); #ps.filtered_index[]; # Number of timepoints.
    blockweight = view(W, :, T);

    # Sample a particle index at time T proportional on the weights at time T.
    # Note: after running `pf!', `wnorm` are the last normalised weights here.
    @inbounds ref[T] = b = wsample(all_particle_indices, wnorm);

    # Start main loop.
    for i in length(samplers):-1:1
        sampler = samplers[i];
        l, u = sampler.bounds; # Get bounds of current block.

        # Set the reference trajectory for the block forward pass based on the
        # initial forward pass.
        #@inbounds set_reference!(ps; l = l, u = u - 1, index = A[b, u - 1]);
        @inbounds traceback!(ssm, AncestorTracing;
                             l = l, u = u - 1, index = A[b, u - 1]);

        # Compute the "blockweight".
        for j in all_particle_indices
            @inbounds blockweight[j] = inv(u - l) * logweight(sampler, SVector{yN, Float64}(X[b, u]),
                                                                       SVector{yN, Float64}(X[j, l]));
        end
        # NOTE: Not part of the BBCPF, just saving:
        # 1. the blockweights for analysis of the algorithm.
        # 2. particle at index X[ref[l], l].
        if !isnothing(outputs)
            outputs.blockweight[:, i] .= blockweight .* (u - l);
            copy!(outputs.ptemp, X[ref[l], l]);
        end

        # Forward CPF through block.
        for t in (l + 1):(u - 1)
            x_cur = view(X, :, t - 1); x_next = view(X, :, t);
            w_cur = view(W, :, t - 1); w_next = view(W, :, t);
            a_cur = view(A, :, t - 1);

            # Add the blockweight to the weights at time t - 1.
            # Resample conditionally with the latest (blockweighted) weights.
            wnorm .= w_cur .+ blockweight; normalise_logweights!(wnorm);
            @inbounds conditional_resample!(resampling, a_cur, wnorm, ref[t], ref[t - 1]);
            #@inbounds resample!(a_cur, wnorm, resampling; ref_prev = ref[t - 1], ref_cur = ref[t]);

            # Sample from M[X[k] | X[k - 1], X[u]].
            for j in OneToNWithout(n_particles, ref[t])
                s = rand(sampler, SVector{yN, Float64}(x_cur[a_cur[j]]),
                                  SVector{yN, Float64}(X[b, u]), t - l);
                copy!(x_next[j], s);
                #@inbounds rand!(sampler, x_next[j], x_cur[a_cur[j]], X[b, u], t - l);
            end

            # Compute the unnormalised logweights at time t.
            for j in all_particle_indices
                @inbounds w_next[j] = model.lG(x_cur[a_cur[j]], x_next[j], t, data, θ);
            end

            # Update the blockweight (using wnorm as a temporary).
            for j in all_particle_indices
                @inbounds wnorm[j] = blockweight[a_cur[j]];
            end
            for j in all_particle_indices
                @inbounds blockweight[j] = wnorm[j];
            end
        end

        ## Draw an index b at time u - 1 and trace back until time l.
        # Compute weights for "backward sampling step".
        w_cur = view(W, :, u - 1); w_next = view(W, :, u);
        wnorm .= w_cur .+ blockweight; # log(G[t-1] * W[t-1])
        for j in all_particle_indices
            @inbounds w_next[j] = model.lG(X[j, u - 1], X[b, u], u, data, θ);
            @inbounds wnorm[j] = wnorm[j] + w_next[j];
        end
        normalise_logweights!(wnorm);

        # Draw indice and trace back.
        #set_reference!(ps; l = l, u = u - 1, index = wsample(all_particle_indices, wnorm));
        @inbounds traceback!(ssm, AncestorTracing;
                             l = l, u = u - 1, index = wsample(all_particle_indices, wnorm));
        #@inbounds traceback!(ssm, AncestorTracing;
        #                     l = l, u = u - 1, index = wsample_one(wnorm));

        # NOTE: Not part of the algorithm, saving if sampled particle at index l
        # is different from block reference (for analysis of algorithm).
        if !isnothing(outputs)
            @inbounds outputs.l_change_in_local_cpf[i] = !eq(outputs.ptemp, X[ref[l], l]);
        end
        @inbounds b = ref[l];
    end
    nothing
end

"""
The function computes the smoothed covariance P[s, t] = Cov(X[s], X[t] | Y[1:n])
based on P[s + 1, t] = Cov(X[s + 1], X[t] | Y[1:n]) by using a backward
recursive formula by De Jong and Mackinnon (1988). P[s + 1, t] is passed as an
argument, as well as the output from the Kalman filter and smoother, which
are used in the computation.
"""
function _backward_smooth_cov!(x::AbstractMatrix{<: Real}, psplus1::AbstractMatrix{<: Real},
                              ssm::SSMInstance{<: NonLinearGaussianSSM}, s::Integer, θ)
    model = ssm.model; data = ssm.data; ss = ssm.storage;
    tmp_kxk1 = ss.tmp_kxk1; tmp_kxk2 = ss.tmp_kxk2;
    Tj = ss.Tj;

    # K = inv(mu_pred[s + 1]) * P[s + 1, t]
    tmp_kxk2 .= ss.P_pred[s + 1];
    #println(tmp_kxk2);
    !issymmetric(tmp_kxk2) && symmetrise!(tmp_kxk2);
    #println(tmp_kxk2);
    chol = cholesky!(tmp_kxk2);
    tmp_kxk1 .= psplus1;
    ldiv!(chol, tmp_kxk1);

    # mu_filt[s] * Tj' * K
    model.Tj!(Tj, ss.mu_filt[s], s, data, θ);
    mul!(tmp_kxk2, transpose(Tj), tmp_kxk1);
    mul!(x, ss.P_filt[s], tmp_kxk2);
    nothing
end



"""
The function initialises a SparseArray with matrix elements that
contains the individual covariance matrices that are needed to compute
the distributions M(X[k] | X[k - 1], X[u]) and M(X[u] | X[l]) for each block
in the BBCPF algorithm.
The returned object has the required entries set to zero matrices.
To populate the SparseArray with the actual covariances, call
`BBCPF_populate_normal_sampler_covariance!`.
"""
function BBCPF_init_normal_sampler_covariance(block_bounds::AbstractVector{Tuple{Int, Int}}, statedim::Integer)
    cind = CartesianIndex[];
    for b in block_bounds
        for i in (b[2] - 2):-1:b[1]
            push!(cind, CartesianIndex(i, b[2]));
        end
        for i in b[1]:(b[2] - 1)
            push!(cind, CartesianIndex(i, i + 1));
        end
    end
    for i in 1:block_bounds[end][2]
        push!(cind, CartesianIndex(i, i));
    end
    sparse(map(x -> x.I[1], cind), map(x -> x.I[2], cind),
           map(x -> zeros(statedim, statedim), 1:length(cind)));
end

"""
The function populates the collection of covariance matrices returned
by `BBCPF_init_normal_sampler_covariance` based on Kalman filter and smoother output.
In particular, the function can be used to construct the sampler covariances
for models where the smoothing distribution is used as the proposal distribution
M. However, the function can also be of use when the Kalman filter and smoother
is run without data.
"""
function BBCPF_populate_sampler_covariance_kalman!(C::SparseMatrixCSC, blocks::AVec{Tuple{Int, Int}},
                             ssm::SSMInstance{<: NonLinearGaussianSSM}, θ)
    ss = ssm.storage;

    # Populate diagonal of matrix.
    for i in 1:size(C, 2)
        C[i, i] .= ss.P_smooth[i];
    end
    for b in blocks
        u = b[2]; l = b[1];

        # Populate last column inside block.
        for i in (u - 1):-1:l
            _backward_smooth_cov!(C[i, u], C[i + 1, u],
                                  ssm, i, θ);
        end
        # Populate offdiagonal inside block.
        for j in (l + 1):(u - 1)
            _backward_smooth_cov!(C[j - 1, j], C[j, j],
                                  ssm, j - 1, θ);
        end
    end
    nothing
end

"""
The function builds smoother block samplers based on the smoothing distribution returned
by the Kalman smoother. The block samplers are to be used with the BBCPF algorithm.
"""
function build_normal_block_samplers(C::SparseMatrixCSC, blocks::AVec{Tuple{Int, Int}},
                                     mu_smooth::AVec{<: Vector{<: Real}},
                                     P_smooth::AVec{<: Matrix{<: Real}})
    @assert length(mu_smooth) > 0 "`mu_smooth` can't be length 0.";
    ydim = length(mu_smooth[1]);
    xdim = ydim * 2;
    onestepdist_type = ConditionalNormal{ydim, xdim, ydim * ydim, xdim * xdim, ydim * xdim};
    block_samplers = Vector{BlockSampler{ydim, xdim, ydim * ydim,
                                         xdim * xdim, ydim * xdim}}(undef, length(blocks));
    for (i, b) in enumerate(blocks)
        l, u = b;
        weightdist = ConditionalNormal(mu_smooth[u], mu_smooth[l], P_smooth[u], P_smooth[l],
                                       transpose(C[l, u]));
        onestepdists = Vector{onestepdist_type}(undef, u - l - 1);
        for (k, j) in enumerate((l + 1):(u - 1))
            Bk_mean = vcat(mu_smooth[j - 1], mu_smooth[u]);
            Bk_var = [P_smooth[j - 1] C[j - 1, u]; transpose(C[j - 1, u]) P_smooth[u]];
            c = [transpose(C[j - 1, j]) C[j, u]];
            onestepdists[k] = ConditionalNormal(mu_smooth[j], Bk_mean,
                                                P_smooth[j], Bk_var, c);
        end
        block_samplers[i] = BlockSampler(b, weightdist, onestepdists);
    end
    block_samplers;
end

"""
Function returns the distributions `M(x_u | x_l)` for each block defined by
`blocks`. It is assumed that Kalman smoother has been run for `instance`.
"""
function build_jump_dists_from_ks!(instance::SSMInstance, θ,
                                   blocks::AVec{Tuple{Int, Int}})
    @assert length(blocks) >= 1 "`blocks` must not be empty.";
    boundaries = Vector{Int}(undef, 0);
    push!(boundaries, blocks[1][1]);
    for i in 1:length(blocks)
       push!(boundaries, blocks[i][2]);
    end
    @assert allunique(boundaries) "problem with `blocks`, duplicate boundary values.";
    @assert issorted(boundaries) "problem with `blocks`. boundary values not increasing."
    @assert blocks[1][1] == 1 "first element of first block of `blocks` must be 1."

    C = BBCPF_init_normal_sampler_covariance(blocks, instance.model.statedim);
    BBCPF_populate_sampler_covariance_kalman!(C, blocks, instance, θ);
    mu_smooth = instance.storage.mu_smooth;
    P_smooth = instance.storage.P_smooth;
    map(blocks) do b
        l, u = b;
        ConditionalNormal(mu_smooth[u], mu_smooth[l], P_smooth[u], P_smooth[l],
                          transpose(C[l, u]));
    end
end

"""
Build the block samplers in one go for the BBCPF by running Kalman filter and
smoother for a particular LinearGaussianSSM.
The `nodata = true` argument is equivalent to running the KF with all data missing.
The `modeldata` argument can be used to specify some external data that the LinearGaussian
model is using.
"""
function BBCPF_build_samplers(blocks::AVec{Tuple{Int, Int}},
                              model::NonLinearGaussianSSM,
                              θ;
                              modeldata::Any = nothing,
                              nodata::Bool = true)
    @assert length(blocks) >= 1 "`blocks` must not be empty.";
    boundaries = Vector{Int}(undef, 0);
    push!(boundaries, blocks[1][1]);
    for i in 1:length(blocks)
        push!(boundaries, blocks[i][2]);
    end
    @assert allunique(boundaries) "problem with `blocks`, duplicate boundary values.";
    @assert issorted(boundaries) "problem with `blocks`. boundary values not increasing."
    @assert blocks[1][1] == 1 "first element of first block of `blocks` must be 1."

    # Run Kalman filter and smoother to get data to build block samplers for
    # the BBCPF.
    T = blocks[end][2];
    state = NLGSSMState(Float64; obsdim = model.obsdim,
                        statedim = model.statedim, n_obs = T);
    instance = SSMInstance(model, state, modeldata, T);
    filter!(instance, θ, nodata = nodata);
    smooth!(instance, θ); # This is actually redundant if nodata = true..

    # Build block samplers for the BBCPF.
    #blocks = block_bounds_from_u_indices(boundaries);
    C = BBCPF_init_normal_sampler_covariance(blocks, model.statedim);
    BBCPF_populate_sampler_covariance_kalman!(C, blocks, instance, θ);
    build_normal_block_samplers(C, blocks, instance.storage.mu_smooth,
                                instance.storage.P_smooth);

end
