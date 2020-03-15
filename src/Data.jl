module Data

using BNP2, Parameters
using Random: AbstractRNG, GLOBAL_RNG

function rotate(θ, x)
    # Rotation matrix
    R = [
        cos(θ) -sin(θ); 
        sin(θ)  cos(θ)
    ]
    return R * x
end

add_gaussiannoise(xs, sigma) = add_gaussiannoise(GLOBAL_RNG, xs, sigma)
function add_gaussiannoise(
    rng::AbstractRNG, 
    xs::AbstractVector{<:AbstractVector}, 
    sigma,
)
    return map(x -> x + sigma * randn(rng, size(x)...), xs)
end

export rotate, add_gaussiannoise

###

function sim_traj(ms, Q, P, dt, T)
    env = Space(Particle.(ms, Q, P))
    traj = simulate(env, DiffEqSimulator(dt), T)
    return [env, traj...]
end

abstract type AbstractDataConfig end

sim(cfg::AbstractDataConfig) = sim(GLOBAL_RNG, cfg)

"""Three body probelm simulation """
@with_kw struct RandInitThreeBodyCfg <: AbstractDataConfig
    ms
    Q
    P
    n_train
    n_valid
    n_test
    T
    dt
end

function sim(rng, cfg::RandInitThreeBodyCfg)
    @unpack ms, Q, P, n_train, n_valid, n_test, dt, T = cfg

    n_rand_init = 4
    n_train_per, n_valid_per, n_test_per = 
        div(n_train, n_rand_init), div(n_valid, n_rand_init), div(n_test, n_rand_init)
    n_data = n_train_per + n_valid_per + n_test_per

    trajs = []

    # Trajectories with different initial conditions
    σ = 1f-2
    for i in 1:n_data
        traj = sim_traj(
            ms, 
            Q + σ * randn(rng, size(Q)), 
            P + σ * randn(rng, size(P)), 
            dt, 
            T
        )
        push!(trajs, traj)
    end

    # Trajectories with different speed
    for i in 1:n_data
        traj = sim_traj(ms, Q, P * (1 + rand(rng)), dt, T)
        push!(trajs, traj)
    end

    # Trajectories with different initial directions
    for i in 1:n_data
        traj = sim_traj(ms, Q, Data.rotate(rand(rng) * π, P), dt, T)
        push!(trajs, traj)
    end

    # Trajectories with different moving directions
    for i in 1:n_data
        traj = sim_traj(ms, Q, P .+ randn(rng, 2), dt, T)
        push!(trajs, traj)
    end

    trajs_train, trajs_valid, trajs_test = 
        trajs[1:n_train], trajs[n_train+1:n_train+n_valid], trajs[n_train+n_valid+1:end]
    
    return (train=trajs_train, valid=trajs_valid, test=trajs_test)
end

preprocess(trajs; kwargs...) = preprocess(Float32, trajs; kwargs...)
function preprocess(T, trajs; σ_obs=0)
    # Convert trajectories to a single tensor
    S_list = []
    for traj in trajs
        states = stateof.(traj)                     # Vec{Mat{2, d, n}}{T}
        S = cat(states...; dims=4)                  # Mat{2, d, n, T}
        S = cat(S[1,:,:,:], S[2,:,:,:]; dims=1)     # Mat{2d, n, T}
        push!(S_list, S)
    end
    S = Array{T,4}(cat(S_list...; dims=4))          # Mat(2d, n, T, N)
    !iszero(σ_obs) && (S += σ_obs * randn(size(S))) # add Gaussian noise
    return S
end

export Data, RandInitThreeBodyCfg, sim, preprocess

end # module
