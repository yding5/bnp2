abstract type AbstractSimulator end

function simulate(env::T, sim::AbstractSimulator, n_steps::Int) where {T<:AbstractEnvironment}
    traj = Vector{T}(undef, n_steps)
    for i in 1:n_steps
        traj[i] = transition(i == 1 ? env : traj[i-1], sim)
    end
    return traj
end

transition(env::AbstractEnvironment, dt::AbstractFloat) = transition(env, SimpleSimulator(dt))
simulate(env::AbstractEnvironment, dt::AbstractFloat, n_steps::Int) =
    simulate(env, SimpleSimulator(dt), n_steps)

### Simple

struct SimpleSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(env::AbstractEnvironment, sim::SimpleSimulator)
    q, p = positionof(env), velocityof(env)
    p += sim.dt / 2 * accelerationof(env)
    q += sim.dt * p
    env = reconstruct(env, q, p)
    p += sim.dt / 2 * accelerationof(env)
    return reconstruct(env, q, p)
end

### OrdinaryDiffEq

using OrdinaryDiffEq: DynamicalODEProblem, Tsit5, init, step!

struct DiffEqSimulator{T} <: AbstractSimulator
    dt::T
end

function init_prob(env, sim, n_steps=1)
    dqdt(q, p, args...) = p                                         # d position d t = velocity
    dpdt(q, p, args...) = accelerationof(reconstruct(env, q, p))    # d velocity d t = acceleration
    prob = DynamicalODEProblem(dqdt, dpdt, positionof(env), velocityof(env), (0.0, sim.dt * n_steps))
    return init(prob, Tsit5())  # NOTE: `Tsit5` is much better than `VerletLeapfrog`.
end

function transition(env::AbstractEnvironment, sim::DiffEqSimulator)
    int = init_prob(env, sim)
    step!(int, sim.dt)
    return reconstruct(env, int.u.x[1], int.u.x[2])
end

function simulate(env::T, sim::DiffEqSimulator, n_steps::Int) where {T<:AbstractEnvironment}
    int = init_prob(env, sim, n_steps)
    traj = Vector{T}(undef, n_steps)
    for i in 1:n_steps
        step!(int, sim.dt, true)
        traj[i] = reconstruct(env, int.u.x[1], int.u.x[2])
    end
    return traj
end

### Pymunk

struct PymunkSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(obj, env::EarthWithForce, sim::PymunkSimulator)
    @unpack mass, position, velocity = obj
    @unpack dt = sim
    
    space = pymunk.Space()
    space.gravity = (0, -g)
    radius = 0.25
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Circle(body, radius, (0, 0))
    space.add(body, shape)
    space.shapes[1].body.apply_force_at_world_point(
        force=tuple((forceof(env.force, obj))...), 
        point=(0, 0),
    )

    space.step(dt)
    position = [get.(Ref(space.shapes[1].body.position), 0:1)...]
    velocity = [get.(Ref(space.shapes[1].body.velocity), 0:1)...]
    
    return Particle(mass, position, velocity)
end

function simulate(obj::T, env::EarthWithObjects, sim::PymunkSimulator, n_steps::Int) where {T<:AbstractObject}
    @unpack mass, position, velocity = obj
    @unpack dt = sim

    space = pymunk.Space()
    space.gravity = (0.0, -g)
    radius = 0.5
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.elasticity = 0.9
    space.add(body, shape)
    
    body_static = pymunk.Body(body_type=pymunk.Body.STATIC)
    for obj in env.objs
        space.add(pymunkobj(body_static, obj))
    end
    
    n_steps == 0 && return (space, sim)

    traj = []
    for t in 1:n_steps
        space.step(dt)
        position = [get.(Ref(space.shapes[1].body.position), 0:1)...]
        velocity = [get.(Ref(space.shapes[1].body.velocity), 0:1)...]
        push!(traj, Particle(mass, position, velocity))
    end
    
    return Vector{typeof(first(traj))}(traj)
end

;