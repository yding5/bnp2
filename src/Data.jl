module Data

using Parameters
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

using BNP2

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
    n_init
    n_speed
    n_direction
    n_moving
    n_tests_per
    dt
    T
end

function sim(rng, cfg::RandInitThreeBodyCfg)
    @unpack ms, Q, P, n_init, n_speed, n_direction, n_moving, n_tests_per, dt, T = cfg

    trajs, trajs_test = [], []

    # Trajectories with different initial conditions
    σ = 1f-2
    for i in 1:n_init+n_tests_per
        traj = sim_traj(
            ms, 
            Q + σ * randn(rng, size(Q)), 
            P + σ * randn(rng, size(P)), 
            dt, 
            T
        )
        push!(i <= n_init ? trajs : trajs_test, traj)
    end

    # Trajectories with different speed
    for i in 1:n_speed+n_tests_per
        traj = sim_traj(ms, Q, P * (1 + rand(rng)), dt, T)
        push!(i <= n_direction ? trajs : trajs_test, traj)
    end

    # Trajectories with different initial directions
    for i in 1:n_direction+n_tests_per
        traj = sim_traj(ms, Q, Data.rotate(rand(rng) * π, P), dt, T)
        push!(i <= n_speed ? trajs : trajs_test, traj)
    end

    # Trajectories with different moving directions
    for i in 1:n_moving+n_tests_per
        traj = sim_traj(ms, Q, P .+ randn(rng, 2), dt, T)
        push!(i <= n_moving ? trajs : trajs_test, traj)
    end
    
    return (trajs=trajs, trajs_test=trajs_test)
end

function preprocess(trajs; σ_obs=0, is_paired=false)
    # Convert trajectories to a single tensor
    S_list = []
    for traj in trajs
        states = stateof.(traj)                     # Vec{Mat{2, d, n}}{T}
        S = cat(states...; dims=4)                  # Mat{2, d, n, T}
        S = cat(S[1,:,:,:], S[2,:,:,:]; dims=1)     # Mat{2d, n, T}
        push!(S_list, S)
    end
    S = Array{Float32,4}(cat(S_list...; dims=4))    # Mat(2d, n, T, N)
    !iszero(σ_obs) && (S += σ_obs * randn(size(S))) # add Gaussian noise
    if is_paired
        s = S[:,:,1:end-1,:]
        s′ = S[:,:,2:end,:]
        return (s=s, s′=s′)
    else
        return S
    end
end

end # module
