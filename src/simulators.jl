abstract type AbstractSimulator end

function simulate(obj::T, env::AbstractEnvironment, sim::AbstractSimulator, n_steps::Int) where {T<:AbstractObject}
    traj = T[]
    for i in 1:n_steps
        obj′ = transition(obj, env, sim)
        push!(traj, obj′)
        obj = obj′
    end
    return traj
end

function simulate(env::T, sim::AbstractSimulator, n_steps::Int) where {T<:AbstractEnvironment}
    traj = T[]
    for i in 1:n_steps
        env′ = transition(env, sim)
        push!(traj, env′)
        env = env′
    end
    return traj
end

### Simple

struct SimpleSimulator{T} <: AbstractSimulator
    dt::T
end

transition(env, dt::AbstractFloat) = transition(env, SimpleSimulator(dt))

simulate(env::AbstractEnvironment, dt::AbstractFloat, n_steps::Int) =
    simulate(env, SimpleSimulator(dt), n_steps)

function transition(obj, env, sim::SimpleSimulator)
    @unpack mass, position, velocity = obj
    @unpack dt = sim
    velocity += dt / 2 * accelerationof(env, obj)
    position += dt * velocity
    obj = Particle(massof(obj), position, velocity)
    velocity += dt / 2 * accelerationof(env, obj)
    return Particle(massof(obj), position, velocity)
end

function transition(env, sim::SimpleSimulator)
    q, p = positionof(env), velocityof(env)
    @unpack dt = sim
    p += dt / 2 * accelerationof(env)
    q += dt * p
    env = Space(Particle.(massof(env), vec2list(q), vec2list(p)))
    p += dt / 2 * accelerationof(env)
    return Space(Particle.(massof(env), vec2list(q), vec2list(p)))
end

### OrdinaryDiffEq

using OrdinaryDiffEq: DynamicalODEProblem, VerletLeapfrog, init, step!

struct DiffEqSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(obj, env, sim::DiffEqSimulator)
    @unpack mass = obj
    @unpack dt = sim
    
    # d position d t = velocity
    dpdt(pos, vel, p, t) = vel
    # d velocity d t = acceleration
    dvdt(pos, vel, p, t) = accelerationof(env, Particle(mass, pos, vel))
    
    prob = DynamicalODEProblem(dpdt, dvdt, positionof(obj), velocityof(obj), (0.0, 1.0))
    int = init(prob, VerletLeapfrog(); dt=dt)
    step!(int, dt)
    return Particle(mass, int.u.x[1], int.u.x[2])
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

function simulate(obj::T, env::EarthWithObjects, sim::PymunkSimulator, n_steps::Int) where {T}
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

### DynamicalSystems

using DynamicalSystems

struct DynSysSimulator{T} <: AbstractSimulator
    dt::T
end

function pqof(u)
    dim = div(length(u), 2)
    return u[1:dim], u[dim+1:end]
end
vec2list(v) = [v[i:i+1] for i in 1:2:length(v)]

function simulate(env::AbstractEnvironment, sim::DynSysSimulator, n_steps::Int)
    @unpack dt = sim
    
    state = stateof(env)
    parameters = nothing
    system = ContinuousDynamicalSystem(state, parameters) do du, u, parameters, t
        q, p = pqof(u)
        dim = length(q)
        space = Space(Particle.(massof(env), vec2list(q), vec2list(p)))
        dq, dp = p, accelerationof(space)
        du[1:dim], du[dim+1:end] = dq, dp
    end
    
    return trajectory(system, n_steps * dt; dt=dt)
end

;