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

### Force

abstract type AbstractForce end

struct Force{V<:AbstractVector{<:Real}} <: AbstractForce
    F::V
end

forceof(f::Force, ::Particle) = f.F

### Environment

abstract type AbstractEnvironment end

const GRAV = 9.81

struct Earth{F<:AbstractForce} <: AbstractEnvironment
    force::F
end

getacceleration(earth::Earth, p::Particle) = forceof(earth.force, p) / massof(p) + [0.0, -GRAV]
