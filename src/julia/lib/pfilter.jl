include("../../../config.jl");

using Statespace
using LinearAlgebra
using Distributions
using Random
using Resamplings
using Resamplings: _wsample_one as wsample_one;
import Base
include("math-functions.jl");
include("OneToNWithout.jl");

"""
An abstract type representing an user defined particle type used
with GenericSSMs.
To qualify as a Particle, a type UserParticle <: Particle should
implement the interface (TODO: Write something to check this):

1. copy!(dest::UserParticle, src::UserParticle).
   - transfers values from `src` to `dest`.
2. UserParticle() returning a "null" particle.
   - This is used in allocating storage for the algorithms.

Additionally, the user may specify:
* statenames(::Type{UserParticle}), which should return the names of the
  state variables as a tuple of Symbols. There is a fallback, which simply calls
  `fieldnames(UserParticle)`, which should be sufficient for most uses.

Some algorithms may require additional methods that must be defined for
the particle type for the algorithm to work.
"""
abstract type Particle end

"""
Get statenames for a generic particle.
"""
function statenames(::Type{P}) where P <: Particle
    fieldnames(P);
end

"""
An abstract type representing a traceback algorithm.
Instances of this abstract type can be passed to filtering algorithms to
run the tracebacking in a certain way.
"""
abstract type Traceback end
struct AncestorTracing <: Traceback end
struct BackwardSampling <: Traceback end


"""
An object for storing values computed during particle filtering.
A ParticleStorage object must be allocated before particle filtering can be done.
A ParticleStorage object mainly stores multiple matrices of dimension (N, T),
where N is the number of particles, and T is the number of time steps.
However, T maybe longer than the number of time points in the data.

The fields in the ParticleStorage object are as follows:

`X`: A matrix of particles of dimension `particle_dim`. The simulated particles
are saved to this matrix during particle filtering.

`W`: A matrix of unnormalised weights. The weights are computed to this matrix
during particle filtering.

`A`: A matrix of resampling indices. The vector A[:, t] contains the indices
of the particles that survived to time point t + 1. The particle X[A[i, t], t]
is the ancestor of the particle X[i, t + 1].

`V`: A vector of normalised weights at the last timepoint of the data.
This vector is stored to obtain the likelihood approximation.

`wnorm`: A temporary vector used for weight normalisation. Allocating this
vector allows for saving the unnormalised weights in W.

`ref`: A vector of indices which marks a reference trajectory in the particle
matrix X. Only used for the conditional particle filter.
X[ref[t], t] gives the t:th particle in the reference trajectory.

`filtered_index`: UNUSED. An integer denoting how many time points have passed in
particle filtering. Keeping this value allows the use of a single storage object
for multiple time series which are not of equal length. After filtering,
this value matches with the number of time points in the data.

"""
struct ParticleStorage{P <: Particle, T <: AFloat} <: SSMStorage
  X::Matrix{P}
  W::Matrix{T}
  A::Matrix{Int}
  V::Vector{T}
  wnorm::Vector{T}
  ref::Vector{Int}
  #filtered_index::Base.RefValue{Int}
  function ParticleStorage(::Type{P}, npar::Integer, ts_length::Integer) where {P <: Particle}
    @assert npar > 0 "`npar` must be > 0."
    @assert ts_length > 0 "`ts_length` must be > 0."
    X = [P() for i in 1:npar, j in 1:ts_length];
    W = zeros(Float64, npar, ts_length);
    V = zeros(Float64, npar);
    wnorm = zeros(Float64, npar);
    A = zeros(typeof(npar), npar, ts_length); # The additional last column is used in BS.
    ref = ones(typeof(npar), ts_length);
    #filtered_index = Ref(0);
    new{P, Float64}(X, W, A, V, wnorm, ref);#, filtered_index);
  end
end
capacity(ps::ParticleStorage) = size(ps.X);
capacity(ps::ParticleStorage, dim::Integer) = size(ps.X, dim);
particle_count(ps::ParticleStorage) = capacity(ps, 1);
particle_dimension(ps::ParticleStorage) = length(ps.X[1, 1]);

"""
A type representing a generic SSM model.

A GenericSSM is constructed by specifying six functions:

`Mi!`: A function with the signature
(x <: Particle, data, par),
drawing a sample from the initial distribution of the state to `x`.

`lMi`: A function with the signature
(x <: Particle, data, par), returning the log density of
the initial distribution wrt `x`.

`M!`: A function propagating the state with the signature
(xnext <: Particle, xcur <: Particle, t::Integer, data, par).
This function should sample from M_t(. | `xcur`),
and place the sampled value in place to `xnext`.

`lM`: A function with the signature
(xnext <: Particle, xcur <: Particle, t::Integer, data, par). This function
should return the log density of M[t](xnext | xcur).

`lGi`: A function returning the log weight for a particle at time t = 1.
The signature of the function should be lGi(xinit <: Particle, data, par).

`lG`: A function returning the log weight for a particle at time t >= 2.
By most, lG is allowed to depend on the two latest particles.
The signature for lG should be of the form
lG(xprev <: Particle, xcur <: Particle, t::Integer, data, par).

Note that lMi and lM must be defined, even though they are not required by
some of the algorithms.
Currently, it is fine `lMi` and `lM` to the function identity, for example,
if the algorithm used does not require them.

The argument `data` is reserved for data of any type required for any of the previous
functions.
Similarly, the argument `par` is reserved for the model parameters of any type in
the previous functions.
"""
struct GenericSSM{P <: Particle, InitSampler <: Function, LogInitDens <: Union{Nothing, Function},
                  ForwardSampler <: Function, LogForwardDens <: Union{Nothing, Function},
                  LogInitPotFun <: Function, LogPotFun <: Function} <: SSM
  Mi!::InitSampler # Function sampling from the initial distribution of particles.
  lMi::LogInitDens # Function returning log density of the initial distribution.

  M!::ForwardSampler # Function sampling from the propagating model.
  lM::LogForwardDens # Function returning log density of the propagating model.

  lGi::LogInitPotFun # Potential function returning log-weight for a particle, t = 1.
  lG::LogPotFun # Potential function returning log-weight for a particle, t >= 2
  function GenericSSM(::Type{P}, Mi!::Function, lMi::Union{Nothing, Function},
                      M!::Function, lM::Union{Nothing, Function},
                      lGi::Function, lG::Function) where {P <: Particle}
    new{P, typeof(Mi!), typeof(lMi), typeof(M!),
        typeof(lM), typeof(lGi), typeof(lG)}(Mi!, lMi, M!, lM, lGi, lG);
  end
end

function dimension(model::GenericSSM{P}) where P <: Particle
  length(statenames(P));
end
function particle_type(ps::GenericSSM{P}) where P
    P;
end


"""
Simulate `length(y)` amount of observations from a generic state space model.
(GenericSSM)

Arguments:
* `y`: A preallocated vector of vectors where simulated observations/states should be
computed to.
* `model`: A GenericSSM object.
* `sim_obs`: A function (f) with the signature f(y[k], x[k], k, data, θ)
which simulates from p(y[k] | x[k]) where y[k] and x[k] are the kth observation vector
and latent vector, respectively. `data` can be any object containing values needed
to simulate, and `θ` is reserved for the parameters of the model. If `sim_obs` is nothing,
then the function simulates states, not observations.
* `data`: Data or quantities required by model or `sim_obs`.
* `θ`: Parameters of the model.
"""
function simulate_observations!(y::AVec{<: AVec{<: Real}}, model::GenericSSM{P},
                                sim_obs::Function, data, θ) where P <: Particle
    @assert !isempty(y) "the vector `y` must not be empty.";
    N = length(y);
    xcur = P(); xnext = P();

    model.Mi!(xcur, data, θ);
    sim_obs(y[1], xcur, 1, data, θ);
    for i in 2:N
        model.M!(xnext, xcur, i, data, θ);
        sim_obs(y[i], xnext, i, data, θ);
        copy!(xcur, xnext);
    end
    nothing;
end

function simulate_states!(x::AVec{P}, Mi!::Function, M!::Function, data, θ) where P <: Particle
    @assert !isempty(x) "the vector `x` must not be empty.";
    N = length(x);
    xcur = P(); xnext = P();
    Mi!(xcur, data, θ);
    copy!(x[1], xcur);
    for i in 2:N
        M!(xnext, xcur, i, data, θ);
        copy!(x[i], xnext);
        copy!(xcur, xnext);
    end
    nothing;
end

function simulate_states!(x::AVec{P}, model::GenericSSM{P}, data, θ) where P <: Particle
    simulate_states!(x, model.Mi!, model.M!, data, θ);
end


"""
Simulate both the observations and the latent state.
"""
function simulate_observations_and_states!(y::AVec{<: AVec{<: Real}},
                   x::AVec{P},
                   model::GenericSSM{P},
                   sim_obs::Function, data, θ) where {P <: Particle}
    @assert !isempty(y) "the vector `y` must not be empty.";
    @assert !isempty(x) "the vector `x` must not be empty.";
    N = length(y);
    @assert length(x) == N "`the lengths of `x` and `y` do not match.";

    model.Mi!(x[1], data, θ);
    sim_obs(y[1], x[1], 1, data, θ);
    for i in 2:N
        model.M!(x[i], x[i - 1], i, data, θ);
        sim_obs(y[i], x[i], i, data, θ);
    end
    nothing;
end

import Statespace.storagetype;
function storagetype(model::GenericSSM)
  ParticleStorage;
end

isneginf(x::Real) = isinf(x) && x < 0.0;

"""
The particle filter.

Arguments:
`ssm`: An SSMInstance{<: GenericSSM} object. See documentation for
SSMInstance for details.

`θ`: Parameters which the model object can depend on. Can be of any type.

`resampling`: A Resampling object determining which kind of resampling is
used during the algorithm.

`conditional`: A boolean value stating whether the conditional or the standard
particle filter should be run.
If the value is true, a reference trajectory is assumed to be initialised
in `ssm.storage`. To initialise the reference trajectory,
use the function `traceback!`.

"""
function pf!(ssm::SSMInstance{<: GenericSSM}, θ;
             resampling::Resampling,
             conditional::Bool = false)

    model = ssm.model; ps = ssm.storage; data = ssm.data;

    # Initialisation.
    _init!(ps);
    X = ps.X; W = ps.W; A = ps.A; V = ps.V; ref = ps.ref;
    wnorm = ps.wnorm;
    ts_length = length(ssm);
    n_particles = particle_count(ps);
    all_particle_indices = Base.OneTo(n_particles);

    # Simulate from initial distribution.
    if conditional
        for j in OneToNWithout(n_particles, @inbounds ref[1])
            @inbounds model.Mi!(X[j, 1], data, θ);
        end
    else
        for j in all_particle_indices
            @inbounds model.Mi!(X[j, 1], data, θ);
        end
    end
    # Compute initial weights.
    for j in all_particle_indices
        @inbounds W[j, 1] = model.lGi(X[j, 1], data, θ);
        @inbounds wnorm[j] = W[j, 1];
    end
    V .= V .+ normalise_logweights!(wnorm);

    for t in 2:ts_length
        a = view(A, :, t - 1);
        # Resample and propagate surviving particles.
        if conditional
            @inbounds conditional_resample!(resampling, a, wnorm, ref[t], ref[t - 1]);
            for j in OneToNWithout(n_particles, ref[t])
                @inbounds model.M!(X[j, t], X[a[j], t - 1], t, data, θ);
            end
        else
            resample!(resampling, a, wnorm);
            for j in all_particle_indices
                @inbounds model.M!(X[j, t], X[a[j], t - 1], t, data, θ);
            end
        end
        # Compute weights.
        for j in all_particle_indices
            @inbounds W[j, t] = model.lG(X[a[j], t - 1], X[j, t], t, data, θ);
            @inbounds wnorm[j] = W[j, t];
        end
        V .= V .+ normalise_logweights!(wnorm);
    end
    V .= V .+ log.(wnorm);
    nothing
end

function pf_ensure_pos_weights!(ssm::SSMInstance{<: GenericSSM}, θ;
                                resampling::Resampling,
                                max_tries::Int = 1000)
    valid_weights = false;
    fail_msg = string("could not run the particle filter with nonzero weights ",
                      " at all timepoints (tried $max_tries times)");
    iter = 0;
    n_timepoints = size(ssm.storage.W, 2);
    while !valid_weights
        iter += 1;
        iter > max_tries && (throw(ArgumentError(fail_msg));)
        pf!(ssm, θ; resampling = resampling, conditional = false);
        ok = true;
        for i in 1:n_timepoints
            ok = any(x -> !isneginf(x), view(ssm.storage.W, :, i));
            !ok && (break;)
        end
        valid_weights = ok;
    end
    iter;
end

function loglik(ps::ParticleStorage)
  logsumexp(ps.V);
end

"""
Backward sampling, assuming last normalised weights are in `ssm.ps.wnorm`.
After calling the function, the sampled trajectory indices are in
`ssm.storage.ref`. See `trace_reference!` for reference trajectory tracing.

Arguments:
`model`: The model object of type GenericSSM.
`θ`: Parameters of the model.
"""
@inline function traceback!(ssm::SSMInstance{<: GenericSSM}, θ, ::Type{BackwardSampling})
    model = ssm.model; ps = ssm.storage; data = ssm.data;
    X = ps.X; W = ps.W; ref = ps.ref; wnorm = ps.wnorm;

    npar = particle_count(ps);
    ts_length = capacity(ps, 2); #ps.filtered_index[];
    ref[ts_length] = aj = wsample_one(wnorm);

    # Sample indices and save to `ref`.
    for t in (ts_length - 1):-1:1
      for j in 1:npar
        @inbounds wnorm[j] = W[j, t] +
                             model.lG(X[j, t], X[aj, t + 1], t + 1, data, θ) +
                             model.lM(X[aj, t + 1], X[j, t], t + 1, data, θ);
      end
      normalise_logweights!(wnorm);
      @inbounds ref[t] = aj = wsample_one(wnorm);
    end
    nothing;
end

"""
Ancestor tracing, assuming last normalised weights in `ssm.storage.wnorm`,
if `index` is not given.
After calling the function, the sampled trajectory indices are in
`ssm.storage.ref`. See `trace_reference!` for reference trajectory tracing.
The parameters are passed so that the signature of this function coincides with
that of backward sampling.

Arguments:
`model`: The model object of type GenericSSM.
`θ`: Parameters of the model.
"""
@inline function traceback!(ssm::SSMInstance{<: GenericSSM}, θ, ::Type{AncestorTracing};
                            index::Int = wsample_one(ssm.storage.wnorm),
                            l::Int = 1, u::Int = capacity(ssm.storage, 2)) #.filtered_index[])
    ps = ssm.storage; ref = ps.ref;
    n_particles = particle_count(ssm.storage);
    @assert index <= n_particles string("invalid particle index ",  index,
                                        ", storage holds ",
                                        n_particles, " particles.");
    #ts_length = ps.filtered_index[];
    A = ps.A; #wnorm = ps.wnorm;
    #@inbounds ref[ts_length] = wsample_one(wnorm);
    @inbounds ref[u] = a = index;
    for t in (u - 1):-1:l
        @inbounds ref[t] = a = A[a, t]; #A[ref[t + 1], t];
    end
    nothing;
end


"""
Convenience method for ancestor tracing. (no need for parameters)
"""
@inline function traceback!(ssm::SSMInstance{<: GenericSSM}, ::Type{AncestorTracing};
                            index::Int = wsample_one(ssm.storage.wnorm),
                            l::Int = 1, u::Int = capacity(ssm.storage, 2)) #.filtered_index[])
    traceback!(ssm, nothing, AncestorTracing; index = index, l = l, u = u);
end

function _init!(ps::ParticleStorage)
  #ps.filtered_index[] = 0;
  ps.V .= 0.0;
  nothing
end

"""
Function traces the reference trajectory to the preallocated array `path`.

Arguments:
`path`: A vector of particles. The reference trajectory is computed in place
to this vector.
`ps`: A ParticleStorage object containing the results from particle filtering.
"""
@inline function trace_reference!(path::AVec{<: Particle},
                                  ps::ParticleStorage)
  X = ps.X; ref = ps.ref;
  ts_length = capacity(ps)[2]; #ps.filtered_index[];
  for t in Base.OneTo(ts_length)
    @inbounds copy!(path[t], X[ref[t], t]);
  end
  nothing
end

"""
Function sets a trajectory in the ParticleStorage object as the reference
trajectory.
The indices of the reference trajectory are recorded to the field `ps.ref`.
It is assumed that the ParticleStorage object contains the output from
particle filtering before this function is called.

When called with only the first argument, a particle at time T is chosen
proportional on the latest (normalised) weights.
The particle's full trajectory (timepoints 1:T) is then traced to `ps.ref`.
The arguments index, l and u can be used to control the tracing:
For example, setting l = 30, u = 35 and index = 1,
the particle at index 1 at time u = 35 is traced to the indices
30:35 in `ps.ref`.

Arguments:
`ps`: A ParticleStorage object containing the results from particle filtering,
e.g calling the function `pf!`.
`index`: The index of the particle at time point `u` whose trajectory is set as
reference. The default is random.
`l`: The lower timepoint for tracing, default = 1.
`u`: The upper bound for tracing, default = T.
"""
function set_reference!(ps::ParticleStorage;
                        index::Int = wsample_one(ps.wnorm),
                        l::Int = 1, u::Int = capacity(ps, 2))#ps.filtered_index[])
  A = ps.A; ref = ps.ref;
  n_particles = particle_count(ps);
  @assert index <= n_particles string("invalid particle index ",  index,
                                      ", storage holds ",
                                      n_particles, " particles.");
  ref[u] = a = index;
  for t in (u - 1):-1:l
    ref[t] = a = A[a, t];
  end
  nothing;
end

"""
Run the conditional particle filter with backward sampling to fill
`sim` with trajectories.

Arguments:
`sim`: A matrix of particles with the row number corresponding to the
number of timepoints, and column number corresponding to the number of simulations.
`ssm`: The state space model instance.
`par`: Parameters of the model.
`resampling`: Resampling object.
`init`: Whether the forward pass should initially be run to get the first reference.
"""
function cpf_bs!(cpfout::Matrix{<: Particle}, ssm::SSMInstance{<: GenericSSM}, θ;
                 resampling::Resampling,
                 init::Bool = true,
                 burnin::Int = 200,
                 thin::Int = 1)
   @assert has_conditional(resampling) "the input `resampling` is not capable for conditional resampling."
   @assert burnin >= 0 "`burnin` must be >= 0";
   @assert thin >= 1 "`thin` must be >= 1";
   if init
      pf!(ssm, θ, resampling = resampling);
      set_reference!(ssm.storage);
   end
   iter = size(cpfout, 2) * thin + burnin;
   for i in 1:iter
      pf!(ssm, θ; resampling = resampling, conditional = true);
      traceback!(ssm, θ, BackwardSampling);
      if i > burnin
          j = i - burnin;
          if j % thin == 0
              j = Int(j / thin);
              @inbounds trace_reference!(cpfout, j, ssm.storage);
          end
      end
   end
   nothing;
end
