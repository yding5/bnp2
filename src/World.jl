using Parameters, ArgCheck
import Parameters: reconstruct

### Object

abstract type AbstractObject end

objectof(obj::AbstractObject) = obj

"""Particle"""
struct Particle{D, M, S} <: AbstractObject
    mass::M     # (kg)
    state::S    # position (m) & velocity (m s^-1)
end

massof(p::Particle) = p.mass
stateof(p::Particle) = p.state
positionof(p::Particle) = p.state[1,:]
velocityof(p::Particle) = p.state[2,:]
dimensionof(p::Particle{D}) where {D} = D

Particle(m::M, s::S) where {M, S} = Particle{size(s, 2), M, S}(m, s)
Particle(mass::T, D::Int) where {T} = Particle(mass, zeros(T, 2, D))
Particle(mass, pos::AbstractVector{T}) where {T} = Particle(mass, pos, zeros(T, length(pos)))
Particle(mass, pos::AbstractVector, ::Nothing) = Particle(mass, pos)
Particle(mass, pos::T, vel::T) where {T<:AbstractVector} = Particle(mass, vcat(pos', vel'))
reconstruct(p::Particle, args...) = Particle(p.mass, args...)

function Base.Broadcast.broadcasted(
    ::typeof(Particle),
    ms::AbstractVector, S::AbstractArray{<:Real, 3}
)
    return Particle.(ms, [S[:,:,i] for i in 1:length(ms)])
end

function Base.Broadcast.broadcasted(
    ::typeof(Particle), 
    ms::AbstractVector, P::AbstractMatrix, V::Union{Nothing, AbstractMatrix}=nothing
)
    n = length(ms)
    ps = [P[:,i] for i in 1:n]
    vs = isnothing(V) ? fill(nothing, n) : [V[:,i] for i in 1:n]
    return Particle.(ms, ps, vs)
end

"""Forced"""
struct Forced{O<:AbstractObject, F} <: AbstractObject
    obj::O
    force::F
end

objectof(f::Forced) = f.obj
massof(f::Forced) = massof(f.obj)
stateof(f::Forced) = stateof(f.obj)
positionof(f::Forced) = positionof(f.obj)
velocityof(f::Forced) = velocityof(f.obj)
dimensionof(f::Forced) = dimensionof(f.obj)

reconstruct(f::Forced, args...) = Forced(reconstruct(f.obj, args...), f.force)

"""Bar"""
struct Bar{P, T} <: AbstractObject
    p1::P
    p2::P
    thickness::T
    elasticity::T
end

Bar(p1, p2) = Bar(p1, p2, 0.3, 0.9)

"""GravitationalField"""
struct GravitationalField{D, S} <: AbstractObject
    direction::D
    strength::S
end

function accelerationof(::AbstractObject, f::GravitationalField)
    u = f.direction / sqrt(sum(abs2, f.direction))
    return f.strength * u
end

const EARTH = GravitationalField([0, -1], g)

### Environment

abstract type AbstractEnvironment end

stateof(env::AbstractEnvironment) = cat(stateof.(objectsof(env))...; dims=3)
positionof(env::AbstractEnvironment) = cat(positionof.(objectsof(env))...; dims=2)
velocityof(env::AbstractEnvironment) = cat(velocityof.(objectsof(env))...; dims=2)
dimensionof(env::AbstractEnvironment) = dimensionof(first(objectsof(env)))
accelerationof(env::AbstractEnvironment, i::Int) = accelerationof(env)[:,i]

"""WithStatic"""
struct WithStatic{E, S<:Tuple, C} <: AbstractEnvironment
    env::E
    static::S
    _cache::C
end

envof(w::WithStatic) = w.env
objectsof(w::WithStatic) = objectsof(envof(w))
accelerationof(w::WithStatic) = accelerationof(envof(w)) + w._cache
staticof(ws::WithStatic) = ws.static

WithStatic(env, static::AbstractObject) = WithStatic(env, tuple(static))
function WithStatic(env, static)
    as = map(objectsof(env)) do obj
        a = zeros(dimensionof(obj))
        for s in static
            if s isa GravitationalField
                a += accelerationof(obj, s)
            end
        end
        a
    end
    _cache = cat(as...; dims=2)
    return WithStatic(env, static, _cache)
end

reconstruct(ws::WithStatic, args...) = WithStatic(reconstruct(envof(ws), args...), ws.static)

"""Space"""
struct Space{O<:Tuple, C} <: AbstractEnvironment
    objects::O
    _cache::C
end

envof(s::Space) = s
objectsof(s::Space) = s.objects
accelerationof(s::Space) = s._cache
staticof(s::Space) = tuple()

Space(obj::AbstractObject, args...) = Space(tuple(obj), args...)
Space(objs::AbstractVector, args...) = Space(tuple(objs...), args...)
function Space(objs::Tuple)
    @argcheck all(dimensionof(first(objs)) .== dimensionof.(objs))
    _cache = acceleration_by_interaction(objs)
    return Space(objs, _cache)
end

"""
    accelerationof_by(obj1, obj2)

Acceleration of `obj1` caused by `obj2`.
"""
function accelerationof_by(obj1, obj2)
    Δp = positionof(obj2) - positionof(obj1)
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_acceleration(massof(obj2), r²) * u
end

function acceleration_by_interaction(objects::Tuple)
    as = map(objects) do obj1
        objects′ = filter(obj2 -> !(obj1 === obj2), objects)
        a = zeros(dimensionof(obj1))
        if length(objects′) > 0
            for obj2 in objects′
                a += accelerationof_by(obj1, obj2)
            end
        end
        obj1 isa Forced ? a + obj1.force / massof(obj1) : a
    end
    return cat(as...; dims=2)
end

reconstruct(s::Space, objects) = Space(objects)
function reconstruct(s::Space, S::AbstractArray{<:Real, 3})
    objs = objectsof(s)
    return Space(reconstruct.(objs, [S[:,:,i] for i in 1:length(objs)]))
end
function reconstruct(s::Space, P::T, V::Union{T, Nothing}=nothing) where {T<:AbstractMatrix{<:Real}}
    objs = objectsof(s)
    n = length(objs)
    ps = [P[:,i] for i in 1:n]
    vs = isnothing(V) ? fill(nothing, n) : [V[:,i] for i in 1:n]
    return Space(reconstruct.(objs, ps, vs))
end

export AbstractObject, Particle, Forced, Bar, GravitationalField, EARTH
export objectof, forceof, massof, stateof, positionof, velocityof, dimensionof
export AbstractEnvironment, WithStatic, Space
export envof, objectsof, accelerationof, staticof
