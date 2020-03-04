### Object

abstract type AbstractObject end

## Particle

struct Particle{D, M, P, V} <: AbstractObject
    mass::M     # kg
    position::P # m
    velocity::V # m^2 s^-1
end

function Particle(m::M, p::P, v::V) where {M, P, V}
    @argcheck length(p) == length(v)
    return Particle{length(p), M, P, V}(m, p, v)
end
Particle(mass, dim::Int) = Particle(mass, zeros(dim), zeros(dim))
Particle(mass, position::AbstractVector) = 
    Particle(mass, position, zeros(eltype(position), length(position)))
Particle(mass, position::AbstractVector, ::Nothing) = Particle(mass, position)

reconstruct(p::Particle, pos, vel) = reconstruct(p; position=pos, velocity=vel)

massof(p::Particle) = p.mass
positionof(p::Particle) = p.position
velocityof(p::Particle) = p.velocity
dimensionof(p::Particle{D}) where {D} = 2D

function forceof(p1::Particle, p2::Particle)
    p1 === p2 && return 0
    Δp = p2.position - p1.position
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_force(p1.mass, p2.mass, r²) * u
end

## Bar

struct Bar{T<:Tuple, R} <: AbstractObject
    s0::T
    s1::T
    elasticity::R
    depth::R
end

Bar(s0, s1) = Bar(s0, s1, 0.9, 0.3)
Bar(s0::T, s1::T, elasticity, depth) where {T<:AbstractArray} = 
    Bar(tuple(s0...), tuple(s1...), elasticity, depth)

function pymunkobj(body, s::Bar)
    v = [(s.s0 .- s.s1)...]
    u = tuple(([v[2], -v[1]] ./ sqrt(sum(v.^2)))...) .* s.depth
    obj = pymunk.Poly(body, [s.s0 .- u, s.s0 .+ u, s.s1 .+ u, s.s1 .- u])
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

## Earth

struct EarthWithForce{F<:AbstractForce} <: AbstractEnvironment
    force::F
end

accelerationof(earth::EarthWithForce, p::Particle) = forceof(earth.force, p) / massof(p) + [0.0, -g]

## Complex environments

struct EarthWithObjects{Os<:AbstractVector{<:AbstractObject}} <: AbstractEnvironment
    objs::Os
end

## Space

struct Space{O<:Tuple, A} <: AbstractEnvironment
    objects::O
    acceleration::A
end

Space(os::AbstractVector) = Space(tuple(os...))
Space(os::Tuple) = Space(os, acceleration_by_interaction(os))

reconstruct(s::Space, pvec::T, vvec::T) where {T<:AbstractVector{<:Real}} = 
    reconstruct(s, _tolist.((pvec, vvec))...)
reconstruct(s::Space, plist, vlist) = Space(reconstruct.(s.objects, plist, vlist))

particles(mvec, pvec::T, vvec::T) where {T<:AbstractVector{<:Real}} = 
    particles(mvec, _tolist.((pvec, vvec)))
particles(mlist, plist, plist) = Particle.(mlist, plist, plist)

massof(s::Space) = massof.(s.objects)
positionof(s::Space) = vcat(positionof.(s.objects)...)
velocityof(s::Space) = vcat(velocityof.(s.objects)...)
dimensionof(s::Space) = sum(dimensionof.(s.objects))
accelerationof(s::Space) = s.acceleration

function stateof(s::Space)
    position = vcat(positionof.(s.objects)...)
    velocity = vcat(velocityof.(s.objects)...)
    l1, l2 = length.((position, velocity))
    return @LArray vcat(position, velocity) (q=1:l1, p=l1+1:l1+l2)
end

function acceleration_by_interaction(objects::Tuple)
    fs = []
    for o1 in objects
        f1 = mapreduce(o2 -> forceof(o1, o2), +, filter(o2 -> !(o1 === o2), objects))
        push!(fs, f1 / massof(o1))
    end
    return vcat(fs...)
end
