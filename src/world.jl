### Object

abstract type AbstractObject end

struct Particle{M, P, V} <: AbstractObject
    mass::M
    position::P
    velocity::V
end

massof(p::Particle) = p.mass
positionof(p::Particle) = p.position
velocityof(p::Particle) = p.velocity

struct Segment{T<:Tuple, R} <: AbstractObject
    s0::T
    s1::T
    elasticity::R
    radius::R
end

Segment(s0, s1) = Segment(s0, s1, 0.9, 0.25)

function pymunkobj(body, s::Segment)
    obj = pymunk.Segment(body, s.s0, s.s1, s.radius)
    obj.elasticity = s.elasticity
    return obj
end

### Force

abstract type AbstractForce end

struct Force{V<:AbstractVector{<:Real}} <: AbstractForce
    F::V
end

forceof(f::Force, ::Particle) = f.F

### Environment

abstract type AbstractEnvironment end

const GRAV = 9.81

struct EarthWithForce{F<:AbstractForce} <: AbstractEnvironment
    force::F
end

getacceleration(earth::EarthWithForce, p::Particle) = forceof(earth.force, p) / massof(p) + [0.0, -GRAV]

### Complex environments

struct EarthWithObjects{Os<:AbstractVector{<:AbstractObject}} <: AbstractEnvironment
    objs::Os
end
