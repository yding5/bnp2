### Object

abstract type AbstractObject end

"""Particle"""
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
Particle(mass, position::AbstractVector) = Particle(mass, position, zeros(eltype(position), length(position)))
Particle(mass, position::AbstractVector, ::Nothing) = Particle(mass, position)

function Base.Broadcast.broadcasted(
    ::typeof(Particle), ms::AbstractVector, pvec::AbstractVector{<:Real}, vvecornothing=nothing
)
    d = div(length(pvec), length(ms))
    return Particle.(ms, _tolist(d, pvec), _tolist(d, vvecornothing))
end

reconstruct(p::Particle, pos, vel) = reconstruct(p; position=pos, velocity=vel)

massof(p::Particle) = p.mass
positionof(p::Particle) = p.position
velocityof(p::Particle) = p.velocity
dimensionof(p::Particle{D}) where {D} = D

"""Bar"""
struct Bar{T1, T2} <: AbstractObject
    pstart::T1
    pend::T1
    elasticity::T2
    tickness::T2
end

Bar(pstart, pend) = Bar(pstart, pend, 0.9, 0.3)

function pymunkobj(body, bar::Bar)
    v = bar.pstart - bar.pend
    u = [v[2], -v[1]] / sqrt(sum(v.^2)) * bar.tickness / 2
    obj = pymunk.Poly(body, [bar.pstart - u, bar.pstart + u, bar.pend + u, bar.pend - u])
    obj.elasticity = bar.elasticity
    return obj
end

### Environment

abstract type AbstractEnvironment end

# TODO: introduce D
"""Space"""
struct Space{D, O<:Tuple, A} <: AbstractEnvironment
    objects::O
    acceleration::A
end

Space(os::O, acc::A) where {O, A} = Space{dimensionof(first(os)), O, A}(os, acc)
Space(os::AbstractVector) = Space(tuple(os...))
function Space(os::Tuple)
    @argcheck all(dimensionof(first(os)) .== dimensionof.(os))
    return Space(os, acceleration_by_interaction(os))
end

reconstruct(s::Space{D}, pvec::T, vvec::T) where {D, T<:AbstractVector{<:Real}} = 
    reconstruct(s, _tolist.(D, (pvec, vvec))...)
reconstruct(s::Space, plist, vlist) = Space(reconstruct.(s.objects, plist, vlist))

massof(s::Space) = massof.(s.objects)
positionof(s::Space) = vcat(positionof.(s.objects)...)
velocityof(s::Space) = vcat(velocityof.(s.objects)...)
dimensionof(s::Space{D}) where {D} = D
accelerationof(s::Space) = s.acceleration

function stateof(s::Space)
    position = vcat(positionof.(s.objects)...)
    velocity = vcat(velocityof.(s.objects)...)
    l1, l2 = length.((position, velocity))
    return @SLVector((q=1:l1, p=l1+1:l1+l2))(vcat(position, velocity))
end

function acceleration_by_interaction(objects::Tuple)
    fs = []
    for o1 in objects
        f1 = mapreduce(o2 -> forceof(o1, o2), +, filter(o2 -> !(o1 === o2), objects))
        push!(fs, f1 / massof(o1))
    end
    return vcat(fs...)
end

"""Earth"""
struct Earth <: AbstractEnvironment
    space
end

### Force

function forceof(p1::Particle, p2::Particle)
    p1 === p2 && return 0
    Δp = p2.position - p1.position
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_force(p1.mass, p2.mass, r²) * u
end

abstract type AbstractForce end

struct Force{V<:AbstractVector{<:Real}} <: AbstractForce
    F::V
end

forceof(f::Force, ::Particle) = f.F
