### Object

abstract type AbstractObject end

forceof(::AbstractObject) = nothing

"""Particle"""
struct Particle{D, M, S} <: AbstractObject
    mass::M     # (kg)
    state::S    # position (m) & velocity (m^2 s^-1)
end

massof(p::Particle) = p.mass
stateof(p::Particle) = p.state
positionof(p::Particle) = p.state[1,:]
velocityof(p::Particle) = p.state[2,:]
dimensionof(p::Particle{D}) where {D} = D

Particle(m::M, s::S) where {M, S} = Particle{size(s, 2), M, S}(m, s)
Particle(mass::T, dim::Int) where {T} = Particle(mass, zeros(T, 2, dim))
Particle(mass, p::AbstractVector{T}) where {T} = Particle(mass, p, zeros(T, length(p)))
Particle(mass, p::AbstractVector, ::Nothing) = Particle(mass, p)
Particle(mass, p::T, v::T) where {T<:AbstractVector} = Particle(mass, vcat(p', v'))
reconstruct(p::Particle, args...) = Particle(p.mass, args...)

function Base.Broadcast.broadcasted(
    ::typeof(Particle), ms::AbstractVector, states::AbstractMatrix
)
    return Particle.(ms, _tolist(div(size(states, 2), length(ms)), states))
end

function Base.Broadcast.broadcasted(
    ::typeof(Particle), ms::AbstractVector, ps::AbstractVector{<:Real}, vs=nothing
)
    return Particle.(ms, _tolist.(div(length(ps), length(ms)), (ps, vs))...)
end

"""Forced"""
struct Forced{O, F} <: AbstractObject
    object::O
    f::F
end

massof(f::Forced) = massof(f.object)
stateof(f::Forced) = stateof(f.object)
positionof(f::Forced) = positionof(f.object)
velocityof(f::Forced) = velocityof(f.object)
dimensionof(f::Forced) = dimensionof(f.object)
forceof(f::Forced) = f.f

reconstruct(f::Forced, args...) = Forced(reconstruct(f.object, args...), f.f)

"""Bar"""
struct Bar{T1, T2} <: AbstractObject
    pstart::T1
    pend::T1
    tickness::T2
    elasticity::T2
end

Bar(pstart, pend) = Bar(pstart, pend, 0.3, 0.9)

function pymunkobj(body, bar::Bar)
    v = bar.pstart - bar.pend
    u = orthonormalvecof(v) * bar.tickness / 2
    obj = pymunk.Poly(body, [bar.pstart - u, bar.pstart + u, bar.pend + u, bar.pend - u])
    obj.elasticity = bar.elasticity
    return obj
end

### Environment

abstract type AbstractEnvironment end

"""Space"""
struct Space{O<:Tuple, A} <: AbstractEnvironment
    objects::O
    acceleration::A
end

stateof(s::Space) = hcat(stateof.(s.objects)...)
positionof(s::Space) = vcat(positionof.(s.objects)...)
velocityof(s::Space) = vcat(velocityof.(s.objects)...)
dimensionof(s::Space) = dimensionof(first(s.objects))
objectsof(s::Space) = s.objects
accelerationof(s::Space) = s.acceleration

Space(os::AbstractVector, args...) = Space(tuple(os...), args...)
function Space(os::Tuple)
    @argcheck all(dimensionof(first(os)) .== dimensionof.(os))
    return Space(os, acceleration_by_interaction(os))
end

forceadd(::Nothing, f2) = f2
forceadd(f1, ::Nothing) = f1
forceadd(f1, f2) = f1 + f2

function acceleration_by_interaction(objects::Tuple)
    fs = []
    for o1 in objects
        objects′ = filter(o2 -> !(o1 === o2), objects)
        if length(objects′) > 0
            f1 = mapreduce(o2 -> forceof(o1, o2), +, objects′)
        else
            f1 = nothing
        end
        push!(fs, forceadd(f1, forceof(o1)) / massof(o1))
    end
    return vcat(fs...)
end

reconstruct(s::Space, objects) = Space(objects)
reconstruct(s::Space, states::AbstractMatrix) = 
    Space(reconstruct.(s.objects, _tolist(dimensionof(s), states)))
reconstruct(s::Space, ps::AbstractVector{<:Real}, vs=nothing) = 
    Space(reconstruct.(s.objects, _tolist.(dimensionof(s), (ps, vs))...))

"""WithStatic"""
struct WithStatic{S<:AbstractEnvironment, F} <: AbstractEnvironment
    space::S
    static::F
end

### Force

function forceof(p1::Particle, p2::Particle)
    p1 === p2 && return nothing
    Δp = positionof(p2) - positionof(p1)
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_force(p1.mass, p2.mass, r²) * u
end

abstract type AbstractForce end

struct Force{V<:AbstractVector{<:Real}} <: AbstractForce
    F::V
end
