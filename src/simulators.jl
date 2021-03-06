abstract type AbstractSimulator end

transition(env::AbstractEnvironment, dt::AbstractFloat) = transition(env, SimpleSimulator(dt))
simulate(env::AbstractEnvironment, dt::AbstractFloat, n_steps::Int) = 
    simulate(env, SimpleSimulator(dt), n_steps)

function simulate(env::T, sim::AbstractSimulator, n_steps::Int) where {T<:AbstractEnvironment}
    traj = Vector{T}(undef, n_steps)
    for i in 1:n_steps
        traj[i] = transition(i == 1 ? env : traj[i-1], sim)
    end
    return traj
end

### Simple

struct SimpleSimulator{T} <: AbstractSimulator
    dt::T
end

function transition(env::AbstractEnvironment, sim::SimpleSimulator)
    q, p = positionof(env), velocityof(env)
    p += sim.dt / 2 * accelerationof(env)
    q += sim.dt * p
    p += sim.dt / 2 * accelerationof(reconstruct(env, q, p))
    return reconstruct(env, q, p)
end

### OrdinaryDiffEq

using OrdinaryDiffEq: DynamicalODEProblem, VerletLeapfrog, Tsit5, init, step!

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

pymunkobj(::AbstractObject) = nothing

function pymunkobj(p::Particle)
    radius = 0.25
    elasticity = 0.9
    mass = massof(p)
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    body.position = positionof(p)
    body.velocity = velocityof(p)
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.elasticity = elasticity
    return body, shape
end

pymunkobj(f::Forced) = pymunkobj(f.obj)

function pymunkobj(bar::Bar)
    body_static = pymunk.Body(body_type=pymunk.Body.STATIC)
    v = bar.p1 - bar.p2
    u = orthonormalvecof(v) * bar.thickness / 2
    shape = pymunk.Poly(body_static, [bar.p1 - u, bar.p1 + u, bar.p2 + u, bar.p2 - u])
    shape.elasticity = bar.elasticity
    return shape
end

function pymunkobj(env::AbstractEnvironment)
    space = pymunk.Space()
    for (i, obj) in enumerate(objectsof(env))
        body, shape = pymunkobj(obj)
        f = massof(obj) * accelerationof(env, i)
        body.apply_force_at_world_point(force=tuple(f...), point=(0, 0))
        space.add(body, shape)
    end
    for static in staticof(env)
        obj = pymunkobj(static)
        if !isnothing(obj)
            space.add(obj)
        end
    end
    return space
end

function transition(env::T, sim::PymunkSimulator) where {T<:AbstractEnvironment}
    space = pymunkobj(env)
    space.step(sim.dt)
    objects = objectsof(env)
    objects = map(1:length(objectsof(env))) do i
        position = [get.(Ref(space.shapes[i].body.position), 0:1)...]
        velocity = [get.(Ref(space.shapes[i].body.velocity), 0:1)...]
        reconstruct(objects[i], position, velocity)
    end
    return reconstruct(env, objects)
end
