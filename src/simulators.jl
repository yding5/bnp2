using Parameters

abstract type AbstractSimulator end

function simulate(obj::T, env, sim::AbstractSimulator, n_steps::Int) where {T}
    traj = T[]
    for i in 1:n_steps
        obj′ = transition(obj, env, sim)
        push!(traj, obj′)
        obj = obj′
    end
    return traj
end

### Simple

struct SimpleSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(obj, env, sim::SimpleSimulator)
    @unpack mass, position, velocity = obj
    @unpack dt = sim
    velocity += dt / 2 * getacceleration(env, obj)
    position += dt * velocity
    obj = Particle(massof(obj), position, velocity)
    velocity += dt / 2 * getacceleration(env, obj)
    return Particle(massof(obj), position, velocity)
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
    dvdt(pos, vel, p, t) = getacceleration(env, Particle(mass, pos, vel))
    
    prob = DynamicalODEProblem(dpdt, dvdt, positionof(obj), velocityof(obj), (0.0, 1.0))
    int = init(prob, VerletLeapfrog(); dt=dt)
    step!(int, dt)
    return Particle(mass, int.u.x[1], int.u.x[2])
end

### Pymunk

struct PymunkSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(obj, env::Earth, sim::PymunkSimulator)
    @unpack mass, position, velocity = obj
    @unpack dt = sim
    
    space = pymunk.Space()
    space.gravity = (0, -GRAV)
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
